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
    label: str
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
        data = pickle.loads(self.cipher.decrypt(self.samples_file.read_bytes()))
        return {
            name: [np.array(sample, dtype=np.uint8) for sample in samples]
            for name, samples in data.items()
        }

    def save_samples(self, samples: Dict[str, List[np.ndarray]]) -> None:
        serializable = {
            name: [sample.tolist() for sample in face_list]
            for name, face_list in samples.items()
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
        enc_line = self.cipher.encrypt(json.dumps(event).encode("utf-8"))
        with self.audit_file.open("ab") as f:
            f.write(enc_line + b"\n")
        os.chmod(self.audit_file, 0o600)


class FaceEngine:
    def __init__(self, samples: Dict[str, List[np.ndarray]], threshold: float) -> None:
        if not hasattr(cv2, "face"):
            raise RuntimeError("OpenCV face module missing. Install opencv-contrib-python.")

        self.threshold = threshold
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=16, grid_x=8, grid_y=8)

        self.id_to_name: Dict[int, str] = {}
        self._train(samples)

    @staticmethod
    def _preprocess_face(gray_face: np.ndarray) -> np.ndarray:
        resized = cv2.resize(gray_face, FACE_SIZE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(resized)
        return cv2.GaussianBlur(clahe, (3, 3), 0)

    def _has_eyes(self, gray_face: np.ndarray) -> bool:
        eyes = self.eye_detector.detectMultiScale(gray_face, scaleFactor=1.1, minNeighbors=4, minSize=(18, 18))
        return len(eyes) >= 1

    def _detect_faces(self, gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
        eq = cv2.equalizeHist(gray)
        faces1 = self.face_detector.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=6, minSize=(80, 80))
        faces2 = self.face_detector.detectMultiScale(eq, scaleFactor=1.12, minNeighbors=5, minSize=(80, 80))

        merged: List[Tuple[int, int, int, int]] = []
        for faces in (faces1, faces2):
            for x, y, w, h in faces:
                duplicate = False
                for mx, my, mw, mh in merged:
                    inter_x1 = max(x, mx)
                    inter_y1 = max(y, my)
                    inter_x2 = min(x + w, mx + mw)
                    inter_y2 = min(y + h, my + mh)
                    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                    union = w * h + mw * mh - inter_area
                    iou = inter_area / union if union else 0
                    if iou > 0.35:
                        duplicate = True
                        break
                if not duplicate:
                    merged.append((x, y, w, h))
        return merged

    def _train(self, samples: Dict[str, List[np.ndarray]]) -> None:
        faces: List[np.ndarray] = []
        labels: List[int] = []

        for idx, (name, sample_list) in enumerate(sorted(samples.items())):
            self.id_to_name[idx] = name
            for sample in sample_list:
                faces.append(self._preprocess_face(sample))
                labels.append(idx)

        if not faces:
            raise RuntimeError("No enrolled faces found. Run: python app.py enroll --name <person>")

        self.recognizer.train(faces, np.array(labels, dtype=np.int32))

    def recognize(self, frame: np.ndarray) -> List[MatchResult]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = self._detect_faces(gray)
        results: List[MatchResult] = []

        for (x, y, w, h) in boxes:
            crop = gray[y : y + h, x : x + w]
            processed = self._preprocess_face(crop)
            label_id, confidence = self.recognizer.predict(processed)

            label = "unknown"
            if confidence <= self.threshold and self._has_eyes(processed):
                label = self.id_to_name.get(label_id, "unknown")

            results.append(MatchResult((y, x + w, y + h, x), label, confidence))

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

        self.label_history: Deque[str] = deque(maxlen=self.stable_frames)
        self.last_unlock_time = 0.0
        self.last_person: Optional[str] = None
        self.last_audit_at = 0.0

    @staticmethod
    def _personalized_message(name: str) -> str:
        return WELCOME_MESSAGES.get(name, f"Welcome, {name.title()}!")

    @staticmethod
    def _panel(frame: np.ndarray, y1: int, y2: int) -> None:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, y1), (frame.shape[1], y2), (20, 20, 30), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    def _draw_ui(self, frame: np.ndarray, state: str, greeting: str, faces: int) -> None:
        state_color = (0, 200, 0) if state == "UNLOCKED" else (0, 0, 230)
        self._panel(frame, 0, 78)
        self._panel(frame, frame.shape[0] - 54, frame.shape[0])

        cv2.putText(frame, "Secure Face Unlock", (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (235, 235, 235), 2)
        cv2.putText(frame, f"Status: {state}", (14, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.72, state_color, 2)
        cv2.putText(frame, greeting, (290, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (250, 250, 250), 2)

        cv2.putText(frame, "q/ESC quit | e enroll new face", (14, frame.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (225, 225, 225), 1)
        cv2.putText(frame, f"Detected: {faces}", (frame.shape[1] - 150, frame.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (225, 225, 225), 1)

    def run(self) -> None:
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {self.camera_index}")

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("[WARN] Could not read frame")
                    break

                results = self.engine.recognize(frame)
                now = time.time()
                best_conf = 999.0
                best_label = "unknown"

                for res in results:
                    top, right, bottom, left = res.location
                    if res.confidence < best_conf:
                        best_conf, best_label = res.confidence, res.label

                    known = res.label != "unknown"
                    color = (0, 190, 0) if known else (0, 0, 230)
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    cv2.putText(
                        frame,
                        f"{res.label} | score {res.confidence:.1f}",
                        (left, max(16, top - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                    )

                self.label_history.append(best_label)
                known_labels = [x for x in self.label_history if x != "unknown"]
                unlocked = False
                current_person = None

                if len(known_labels) >= self.stable_frames:
                    most_common, votes = Counter(known_labels).most_common(1)[0]
                    if votes >= self.stable_frames:
                        unlocked = True
                        current_person = most_common

                if unlocked and current_person:
                    self.last_person = current_person
                    self.last_unlock_time = now
                    state = "UNLOCKED"
                    greeting = self._personalized_message(current_person)
                    audit_label = current_person
                elif (now - self.last_unlock_time) < self.cooldown_seconds and self.last_person:
                    state = "UNLOCKED"
                    greeting = self._personalized_message(self.last_person)
                    audit_label = self.last_person
                else:
                    state = "LOCKED"
                    greeting = "Locked: hold face steady in front of camera"
                    audit_label = "unknown"

                self._draw_ui(frame, state=state, greeting=greeting, faces=len(results))

                if now - self.last_audit_at >= 1.0:
                    self.secure_store.append_audit_event(
                        label=audit_label,
                        confidence=best_conf,
                        unlocked=(state == "UNLOCKED"),
                    )
                    self.last_audit_at = now

                cv2.imshow("Secure Face Lock", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("e"):
                    cap.release()
                    cv2.destroyAllWindows()
                    return
        finally:
            cap.release()
            cv2.destroyAllWindows()


def _draw_scan_ring(frame: np.ndarray, progress: float, status: str, step_text: str) -> None:
    h, w = frame.shape[:2]
    center = (w // 2, h // 2 + 20)
    radius = min(w, h) // 5

    overlay = frame.copy()
    cv2.circle(overlay, center, radius + 22, (25, 25, 35), -1)
    cv2.addWeighted(overlay, 0.52, frame, 0.48, 0, frame)

    cv2.circle(frame, center, radius, (100, 100, 100), 2)
    end_angle = int(360 * max(0.0, min(1.0, progress)))
    cv2.ellipse(frame, center, (radius, radius), -90, 0, end_angle, (0, 220, 120), 6)

    cv2.putText(frame, "Face Enrollment", (center[0] - 110, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (240, 240, 240), 2)
    cv2.putText(frame, step_text, (center[0] - 140, h - 72), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 240, 240), 2)
    cv2.putText(frame, status, (center[0] - 140, h - 44), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 210, 130), 2)
    cv2.putText(frame, "Press q to cancel", (center[0] - 140, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 210, 210), 1)


def _extract_enroll_face(gray_frame: np.ndarray, ring_center: Tuple[int, int], ring_radius: int) -> Tuple[Optional[np.ndarray], str]:
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    boxes = detector.detectMultiScale(gray_frame, scaleFactor=1.15, minNeighbors=6, minSize=(90, 90))
    if len(boxes) != 1:
        return None, "Keep exactly one face visible"

    x, y, w, h = boxes[0]
    face_cx, face_cy = x + w // 2, y + h // 2
    dist = ((face_cx - ring_center[0]) ** 2 + (face_cy - ring_center[1]) ** 2) ** 0.5
    if dist > ring_radius * 0.55:
        return None, "Center your face in the ring"

    size_ok = ring_radius * 1.05 <= max(w, h) <= ring_radius * 1.95
    if not size_ok:
        return None, "Move slightly closer/farther"

    crop = gray_frame[y : y + h, x : x + w]
    blur_score = cv2.Laplacian(crop, cv2.CV_64F).var()
    if blur_score < 80:
        return None, "Hold still, image too blurry"

    face = cv2.resize(crop, FACE_SIZE)
    face = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(face)
    return face, "Good capture"


def enroll_new_face_android_style(
    secure_store: SecureFaceStore,
    name: str,
    camera_index: int,
    samples_count: int,
) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}")

    print("Android-style scan started. Follow on-screen prompts.")
    saved = 0
    last_capture = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Could not read frame")
                break

            h, w = frame.shape[:2]
            ring_center = (w // 2, h // 2 + 20)
            ring_radius = min(w, h) // 5

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face, status = _extract_enroll_face(gray, ring_center, ring_radius)

            step_idx = min(saved, len(ENROLL_STEPS) - 1)
            step_text = f"Step {saved + 1}/{samples_count}: {ENROLL_STEPS[step_idx]}"
            progress = saved / max(samples_count, 1)
            _draw_scan_ring(frame, progress=progress, status=status, step_text=step_text)

            if face is not None and (time.time() - last_capture) > 0.6:
                secure_store.add_sample(name, face)
                saved += 1
                last_capture = time.time()
                status = f"Captured {saved}/{samples_count}"

            cv2.imshow("Face Enrollment (Android Style)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Enrollment cancelled.")
                break

            if saved >= samples_count:
                secure_store.append_audit_event(label=name.lower(), confidence=0.0, unlocked=True)
                print(f"Enrolled '{name}' with {samples_count} guided samples.")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Secure face lock/unlock UI (OpenCV only)")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=50.0, help="LBPH threshold (lower=stricter, typical 45-60)")
    parser.add_argument("--cooldown", type=float, default=2.0)
    parser.add_argument("--stable-frames", type=int, default=4, help="Consistent known frames required before unlock")
    parser.add_argument("--secure-dir", type=Path, default=Path("secure_data"))
    parser.add_argument("--key-file", type=Path, default=None)

    sub = parser.add_subparsers(dest="command", required=False)
    enroll = sub.add_parser("enroll", help="Android-style guided face enrollment")
    enroll.add_argument("--name", required=True)
    enroll.add_argument("--samples", type=int, default=8)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = SecureFaceStore(secure_dir=args.secure_dir, key_file=args.key_file)

    if args.command == "enroll":
        enroll_new_face_android_style(store, args.name, args.camera_index, args.samples)
        return

    while True:
        app = FaceUnlockUI(
            secure_store=store,
            camera_index=args.camera_index,
            confidence_threshold=args.threshold,
            cooldown_seconds=args.cooldown,
            stable_frames=args.stable_frames,
        )
        app.run()

        choice = input("Press 'e' + Enter to enroll new face, or just Enter to exit: ").strip().lower()
        if choice == "e":
            name = input("Enter person name: ").strip()
            if name:
                enroll_new_face_android_style(store, name, args.camera_index, samples_count=8)
            continue
        break


if __name__ == "__main__":
    main()
