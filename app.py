import argparse
import json
import os
import pickle
import time
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
from cryptography.fernet import Fernet

WELCOME_MESSAGES: Dict[str, str] = {
    "alice": "Welcome back, Alice 👋",
    "bob": "Hi Bob, access granted ✅",
    "charlie": "Great to see you, Charlie!",
}

FACE_SIZE = (200, 200)
ENROLL_STEPS = [
    "Look straight",
    "Turn slightly left",
    "Turn slightly right",
    "Tilt slightly up",
    "Tilt slightly down",
    "Move a bit closer",
    "Move a bit farther",
    "Look straight again",
]


@dataclass
class MatchResult:
    location: Tuple[int, int, int, int]
    predicted_label: str
    final_label: str
    confidence: float


class SecureFaceStore:
    def __init__(self, secure_dir: Path, key_file: Optional[Path] = None) -> None:
        self.secure_dir = secure_dir
        self.secure_dir.mkdir(parents=True, exist_ok=True)
        self.key_file = key_file or (self.secure_dir / "face_store.key")
        self.samples_file = self.secure_dir / "face_samples.enc"
        self.audit_file = self.secure_dir / "recognition_audit.log.enc"
        self.cipher = Fernet(self._load_or_create_key())

    def _load_or_create_key(self) -> bytes:
        if self.key_file.exists():
            return self.key_file.read_bytes()
        key = Fernet.generate_key()
        self.key_file.write_bytes(key)
        os.chmod(self.key_file, 0o600)
        return key

    def load_samples(self) -> Dict[str, List[np.ndarray]]:
        if not self.samples_file.exists():
            return {}
        payload = self.cipher.decrypt(self.samples_file.read_bytes())
        data = pickle.loads(payload)
        return {
            name: [np.array(sample, dtype=np.uint8) for sample in sample_list]
            for name, sample_list in data.items()
        }

    def save_samples(self, samples: Dict[str, List[np.ndarray]]) -> None:
        serializable = {
            name: [sample.tolist() for sample in sample_list]
            for name, sample_list in samples.items()
        }
        self.samples_file.write_bytes(self.cipher.encrypt(pickle.dumps(serializable)))
        os.chmod(self.samples_file, 0o600)

    def add_sample(self, name: str, face_image: np.ndarray) -> None:
        normalized = name.strip().lower()
        samples = self.load_samples()
        samples.setdefault(normalized, []).append(face_image)
        self.save_samples(samples)

    def append_audit_event(self, label: str, confidence: float, unlocked: bool) -> None:
        event = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "label": label,
            "confidence": round(float(confidence), 3),
            "unlocked": unlocked,
        }
        enc = self.cipher.encrypt(json.dumps(event).encode("utf-8"))
        with self.audit_file.open("ab") as f:
            f.write(enc + b"\n")
        os.chmod(self.audit_file, 0o600)


class FaceEngine:
    def __init__(self, samples: Dict[str, List[np.ndarray]], threshold: float) -> None:
        if not hasattr(cv2, "face"):
            raise RuntimeError("OpenCV face module missing. Install opencv-contrib-python.")

        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=16, grid_x=8, grid_y=8)
        self.id_to_name: Dict[int, str] = {}

        self._train(samples)
        self.threshold = threshold if threshold > 0 else self._calibrate_threshold()

    @staticmethod
    def _preprocess(face_gray: np.ndarray) -> np.ndarray:
        face = cv2.resize(face_gray, FACE_SIZE)
        face = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(face)
        return cv2.GaussianBlur(face, (3, 3), 0)

    def _detect_faces(self, gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
        return self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=6,
            minSize=(90, 90),
        )

    def _train(self, samples: Dict[str, List[np.ndarray]]) -> None:
        faces: List[np.ndarray] = []
        labels: List[int] = []
        for idx, (name, sample_list) in enumerate(sorted(samples.items())):
            self.id_to_name[idx] = name
            for sample in sample_list:
                faces.append(self._preprocess(sample))
                labels.append(idx)

        if not faces:
            raise RuntimeError("No enrolled faces found. Run: python app.py enroll --name <person>")

        self.recognizer.train(faces, np.array(labels, dtype=np.int32))
        self.training_faces = faces

    def _calibrate_threshold(self) -> float:
        scores: List[float] = []
        for face in self.training_faces:
            _, conf = self.recognizer.predict(face)
            scores.append(float(conf))
        p95 = np.percentile(scores, 95) if scores else 80.0
        threshold = float(np.clip(p95 + 20.0, 60.0, 130.0))
        print(f"[INFO] Auto threshold selected: {threshold:.1f}")
        return threshold

    def recognize(self, frame: np.ndarray) -> List[MatchResult]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = self._detect_faces(gray)
        results: List[MatchResult] = []

        for x, y, w, h in boxes:
            crop = gray[y : y + h, x : x + w]
            processed = self._preprocess(crop)
            label_id, confidence = self.recognizer.predict(processed)
            predicted = self.id_to_name.get(label_id, "unknown")
            final = predicted if confidence <= self.threshold else "unknown"
            results.append(
                MatchResult(
                    location=(y, x + w, y + h, x),
                    predicted_label=predicted,
                    final_label=final,
                    confidence=float(confidence),
                )
            )

        return results


class FaceUnlockUI:
    def __init__(
        self,
        secure_store: SecureFaceStore,
        camera_index: int,
        confidence_threshold: float,
        cooldown_seconds: float,
        stable_frames: int,
    ) -> None:
        self.secure_store = secure_store
        self.camera_index = camera_index
        self.cooldown_seconds = cooldown_seconds
        self.stable_frames = max(stable_frames, 3)
        self.engine = FaceEngine(secure_store.load_samples(), threshold=confidence_threshold)

        self.history: Deque[str] = deque(maxlen=self.stable_frames)
        self.last_unlock_time = 0.0
        self.last_person: Optional[str] = None
        self.last_audit_at = 0.0

    @staticmethod
    def _personalized(name: str) -> str:
        return WELCOME_MESSAGES.get(name, f"Welcome, {name.title()}!")

    @staticmethod
    def _panel(frame: np.ndarray, y1: int, y2: int) -> None:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, y1), (frame.shape[1], y2), (20, 20, 30), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    def _draw_ui(self, frame: np.ndarray, state: str, greeting: str, faces: int, best_score: float) -> None:
        color = (0, 200, 0) if state == "UNLOCKED" else (0, 0, 230)
        self._panel(frame, 0, 82)
        self._panel(frame, frame.shape[0] - 56, frame.shape[0])

        cv2.putText(frame, "Secure Face Unlock", (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (236, 236, 236), 2)
        cv2.putText(frame, f"Status: {state}", (14, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2)
        cv2.putText(frame, greeting, (295, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (248, 248, 248), 2)

        metrics = f"thr={self.engine.threshold:.1f} best={best_score:.1f}"
        cv2.putText(frame, metrics, (14, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (225, 225, 225), 1)
        cv2.putText(frame, "q/ESC quit | e enroll", (250, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (225, 225, 225), 1)
        cv2.putText(frame, f"Detected: {faces}", (frame.shape[1] - 150, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (225, 225, 225), 1)

    def run(self) -> bool:
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {self.camera_index}")

        enroll_requested = False
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("[WARN] Could not read frame")
                    break

                results = self.engine.recognize(frame)
                now = time.time()
                best_score = 999.0
                best_label = "unknown"

                for result in results:
                    top, right, bottom, left = result.location
                    if result.confidence < best_score:
                        best_score = result.confidence
                        best_label = result.final_label

                    known = result.final_label != "unknown"
                    shown = result.final_label if known else f"unknown ({result.predicted_label})"
                    color = (0, 190, 0) if known else (0, 0, 230)
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    cv2.putText(
                        frame,
                        f"{shown} | score {result.confidence:.1f}",
                        (left, max(16, top - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.54,
                        color,
                        2,
                    )

                self.history.append(best_label)
                known_hist = [x for x in self.history if x != "unknown"]

                unlocked = False
                person = None
                if len(known_hist) >= self.stable_frames:
                    label, votes = Counter(known_hist).most_common(1)[0]
                    if votes >= self.stable_frames:
                        unlocked = True
                        person = label

                if unlocked and person:
                    self.last_person = person
                    self.last_unlock_time = now
                    state = "UNLOCKED"
                    greeting = self._personalized(person)
                    audit_label = person
                elif (now - self.last_unlock_time) < self.cooldown_seconds and self.last_person:
                    state = "UNLOCKED"
                    greeting = self._personalized(self.last_person)
                    audit_label = self.last_person
                else:
                    state = "LOCKED"
                    greeting = "Locked: align your face inside camera frame"
                    audit_label = "unknown"

                self._draw_ui(frame, state, greeting, len(results), best_score)

                if now - self.last_audit_at >= 1.0:
                    self.secure_store.append_audit_event(audit_label, best_score, state == "UNLOCKED")
                    self.last_audit_at = now

                cv2.imshow("Secure Face Lock", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("e"):
                    enroll_requested = True
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

        return enroll_requested


def _draw_face_grid(frame: np.ndarray, progress: float, step_text: str, status: str) -> Tuple[int, int, int, int]:
    h, w = frame.shape[:2]
    grid_w = int(w * 0.42)
    grid_h = int(h * 0.58)
    x1 = (w - grid_w) // 2
    y1 = (h - grid_h) // 2
    x2 = x1 + grid_w
    y2 = y1 + grid_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (30, 140, 80), -1)
    cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 120), 2)

    # grid lines
    for i in (1, 2):
        gx = x1 + (grid_w * i) // 3
        gy = y1 + (grid_h * i) // 3
        cv2.line(frame, (gx, y1), (gx, y2), (0, 190, 105), 1)
        cv2.line(frame, (x1, gy), (x2, gy), (0, 190, 105), 1)

    # progress bar
    bar_y = y2 + 14
    cv2.rectangle(frame, (x1, bar_y), (x2, bar_y + 12), (90, 90, 90), 1)
    fill = int((x2 - x1 - 2) * max(0.0, min(1.0, progress)))
    cv2.rectangle(frame, (x1 + 1, bar_y + 1), (x1 + 1 + fill, bar_y + 11), (0, 210, 120), -1)

    cv2.putText(frame, "Face Scan", (x1, y1 - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (240, 240, 240), 2)
    cv2.putText(frame, step_text, (x1, y2 + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (240, 240, 240), 2)
    cv2.putText(frame, status, (x1, y2 + 66), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 210, 130), 2)
    cv2.putText(frame, "q cancel | s manual capture", (x1, y2 + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (210, 210, 210), 1)

    return x1, y1, x2, y2


def _extract_enroll_face(gray: np.ndarray, guide_rect: Tuple[int, int, int, int]) -> Tuple[Optional[np.ndarray], str]:
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    boxes = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6, minSize=(90, 90))

    if len(boxes) != 1:
        return None, "Keep exactly one face visible"

    x1, y1, x2, y2 = guide_rect
    x, y, w, h = boxes[0]
    face_x2, face_y2 = x + w, y + h

    # overlap with guide region
    inter_x1 = max(x, x1)
    inter_y1 = max(y, y1)
    inter_x2 = min(face_x2, x2)
    inter_y2 = min(face_y2, y2)
    inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area = w * h
    overlap = (inter / area) if area else 0
    if overlap < 0.7:
        return None, "Align full face inside grid"

    guide_w = x2 - x1
    if not (guide_w * 0.45 <= w <= guide_w * 0.90):
        return None, "Move slightly closer/farther"

    crop = gray[y:face_y2, x:face_x2]
    blur = cv2.Laplacian(crop, cv2.CV_64F).var()
    if blur < 60:
        return None, "Hold still (blurry frame)"

    face = cv2.resize(crop, FACE_SIZE)
    face = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(face)
    return face, "Good capture"


def enroll_new_face_with_grid(store: SecureFaceStore, name: str, camera_index: int, samples_count: int) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}")

    saved = 0
    last_capture = 0.0
    print("Face grid scan started. Follow prompts on screen.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Could not read frame")
                break

            step_idx = min(saved, len(ENROLL_STEPS) - 1)
            step_text = f"Step {saved + 1}/{samples_count}: {ENROLL_STEPS[step_idx]}"
            progress = saved / max(samples_count, 1)
            guide_rect = _draw_face_grid(frame, progress, step_text, "Position your face in the grid")

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face, status = _extract_enroll_face(gray, guide_rect)
            x1, y1, _, y2 = guide_rect
            cv2.putText(frame, status, (x1, y2 + 66), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 210, 130), 2)

            if face is not None and (time.time() - last_capture) > 0.6:
                store.add_sample(name, face)
                saved += 1
                last_capture = time.time()
                print(f"Captured sample {saved}/{samples_count}")

            cv2.imshow("Face Enrollment", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Enrollment cancelled.")
                break
            if key == ord("s") and face is not None:
                store.add_sample(name, face)
                saved += 1
                print(f"Manual capture {saved}/{samples_count}")
                time.sleep(0.15)

            if saved >= samples_count:
                store.append_audit_event(name.lower(), 0.0, True)
                print(f"Enrolled '{name}' with {samples_count} guided samples.")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Secure face lock/unlock UI")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument(
        "--threshold",
        type=float,
        default=-1,
        help="LBPH threshold; <=0 enables auto-calibration (recommended)",
    )
    parser.add_argument("--cooldown", type=float, default=2.0)
    parser.add_argument("--stable-frames", type=int, default=3)
    parser.add_argument("--secure-dir", type=Path, default=Path("secure_data"))
    parser.add_argument("--key-file", type=Path, default=None)

    sub = parser.add_subparsers(dest="command", required=False)
    enroll = sub.add_parser("enroll", help="Guided face enrollment with face grid")
    enroll.add_argument("--name", required=True)
    enroll.add_argument("--samples", type=int, default=12)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = SecureFaceStore(args.secure_dir, args.key_file)

    if args.command == "enroll":
        enroll_new_face_with_grid(store, args.name, args.camera_index, args.samples)
        return

    while True:
        app = FaceUnlockUI(
            secure_store=store,
            camera_index=args.camera_index,
            confidence_threshold=args.threshold,
            cooldown_seconds=args.cooldown,
            stable_frames=args.stable_frames,
        )
        enroll_requested = app.run()

        if enroll_requested:
            person = input("Enter person name to enroll: ").strip()
            if person:
                enroll_new_face_with_grid(store, person, args.camera_index, samples_count=12)
            continue
        break


if __name__ == "__main__":
    main()
