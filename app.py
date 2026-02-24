import argparse
import json
import os
import pickle
import time
from collections import deque
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


@dataclass
class MatchResult:
    location: Tuple[int, int, int, int]
    label: str
    confidence: float


class SecureFaceStore:
    """Encrypted storage for face samples and recognition audit events."""

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

        encrypted = self.samples_file.read_bytes()
        payload = self.cipher.decrypt(encrypted)
        data = pickle.loads(payload)
        return {
            name: [np.array(sample, dtype=np.uint8) for sample in samples]
            for name, samples in data.items()
        }

    def save_samples(self, samples: Dict[str, List[np.ndarray]]) -> None:
        serializable = {
            name: [sample.tolist() for sample in sample_list]
            for name, sample_list in samples.items()
        }
        encrypted = self.cipher.encrypt(pickle.dumps(serializable))
        self.samples_file.write_bytes(encrypted)
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
        encrypted_line = self.cipher.encrypt(json.dumps(event).encode("utf-8"))
        with self.audit_file.open("ab") as f:
            f.write(encrypted_line + b"\n")
        os.chmod(self.audit_file, 0o600)


class FaceEngine:
    def __init__(self, samples: Dict[str, List[np.ndarray]], threshold: float) -> None:
        if not hasattr(cv2, "face"):
            raise RuntimeError("OpenCV face module missing. Install opencv-contrib-python.")

        self.threshold = threshold
        self.detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=16, grid_x=8, grid_y=8)
        self.id_to_name: Dict[int, str] = {}
        self._train(samples)

    @staticmethod
    def _preprocess_face(gray_face: np.ndarray) -> np.ndarray:
        norm = cv2.resize(gray_face, FACE_SIZE)
        norm = cv2.equalizeHist(norm)
        return cv2.GaussianBlur(norm, (3, 3), 0)

    def _train(self, samples: Dict[str, List[np.ndarray]]) -> None:
        faces: List[np.ndarray] = []
        labels: List[int] = []

        for idx, (name, sample_list) in enumerate(sorted(samples.items())):
            self.id_to_name[idx] = name
            for sample in sample_list:
                faces.append(self._preprocess_face(sample))
                labels.append(idx)

        if not faces:
            raise RuntimeError("No enrolled faces found. Run enrollment first.")

        self.recognizer.train(faces, np.array(labels, dtype=np.int32))

    def recognize(self, frame: np.ndarray) -> List[MatchResult]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = self.detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6, minSize=(90, 90))
        results: List[MatchResult] = []

        for (x, y, w, h) in boxes:
            crop = gray[y : y + h, x : x + w]
            processed = self._preprocess_face(crop)
            label_id, confidence = self.recognizer.predict(processed)
            label = self.id_to_name.get(label_id, "unknown") if confidence <= self.threshold else "unknown"
            results.append(MatchResult(location=(y, x + w, y + h, x), label=label, confidence=confidence))

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
        self.stable_frames = stable_frames
        self.engine = FaceEngine(secure_store.load_samples(), threshold=confidence_threshold)

        self.history: Deque[str] = deque(maxlen=max(stable_frames, 3))
        self.last_unlock_time = 0.0
        self.last_person: Optional[str] = None
        self.last_audit_at = 0.0

    @staticmethod
    def _personalized_message(name: str) -> str:
        return WELCOME_MESSAGES.get(name, f"Welcome, {name.title()}!")

    @staticmethod
    def _draw_glass_panel(frame: np.ndarray, y1: int, y2: int) -> None:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, y1), (frame.shape[1], y2), (18, 18, 26), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    def _draw_ui(self, frame: np.ndarray, state: str, greeting: str, people_count: int) -> None:
        state_color = (0, 200, 0) if state == "UNLOCKED" else (0, 0, 230)
        self._draw_glass_panel(frame, 0, 74)
        self._draw_glass_panel(frame, frame.shape[0] - 52, frame.shape[0])

        cv2.putText(frame, "Face Access Control", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (230, 230, 230), 2)
        cv2.putText(frame, f"Status: {state}", (14, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.72, state_color, 2)
        cv2.putText(frame, greeting, (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (250, 250, 250), 2)

        hint = "q/ESC quit | stable unlock filter enabled"
        cv2.putText(frame, hint, (14, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (220, 220, 220), 1)
        cv2.putText(frame, f"Detected faces: {people_count}", (frame.shape[1] - 240, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (220, 220, 220), 1)

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
                best_label = "unknown"
                best_conf = 999.0

                for result in results:
                    top, right, bottom, left = result.location
                    if result.confidence < best_conf:
                        best_label = result.label
                        best_conf = result.confidence

                    is_known = result.label != "unknown"
                    color = (0, 190, 0) if is_known else (0, 0, 230)
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    cv2.putText(
                        frame,
                        f"{result.label} | score {result.confidence:.1f}",
                        (left, max(16, top - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                    )

                self.history.append(best_label)
                stable_known = [x for x in self.history if x != "unknown"]
                unlocked = bool(stable_known) and len(stable_known) >= self.stable_frames

                if unlocked:
                    label_counts: Dict[str, int] = {}
                    for item in stable_known:
                        label_counts[item] = label_counts.get(item, 0) + 1
                    person = max(label_counts, key=label_counts.get)
                    self.last_person = person
                    self.last_unlock_time = now
                    greeting = self._personalized_message(person)
                    state = "UNLOCKED"
                    audit_label = person
                elif (now - self.last_unlock_time) < self.cooldown_seconds and self.last_person:
                    greeting = self._personalized_message(self.last_person)
                    state = "UNLOCKED"
                    audit_label = self.last_person
                else:
                    greeting = "Locked: authorized face not confirmed"
                    state = "LOCKED"
                    audit_label = "unknown"

                self._draw_ui(frame, state=state, greeting=greeting, people_count=len(results))

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
        finally:
            cap.release()
            cv2.destroyAllWindows()


def _extract_single_face(gray_frame: np.ndarray) -> Tuple[Optional[np.ndarray], str]:
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    boxes = detector.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=6, minSize=(90, 90))
    if len(boxes) != 1:
        return None, "Need exactly one face"

    x, y, w, h = boxes[0]
    crop = gray_frame[y : y + h, x : x + w]
    blur_score = cv2.Laplacian(crop, cv2.CV_64F).var()
    if blur_score < 70:
        return None, "Image is blurry, stay steady"

    processed = cv2.equalizeHist(cv2.resize(crop, FACE_SIZE))
    return processed, "Ready to capture"


def enroll_new_face(secure_store: SecureFaceStore, name: str, camera_index: int, samples_count: int) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}")

    saved = 0
    print("Press 's' to capture sample, vary expression/angle slightly, 'q' to cancel.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Could not read frame")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face, status = _extract_single_face(gray)
            color = (0, 190, 0) if face is not None else (0, 0, 230)

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 58), (20, 20, 28), -1)
            cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
            cv2.putText(frame, f"Enroll {name} | samples {saved}/{samples_count}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (240, 240, 240), 2)
            cv2.putText(frame, status, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)
            cv2.imshow("Enroll New Face", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Enrollment cancelled.")
                break
            if key == ord("s"):
                if face is None:
                    print("Capture skipped:", status)
                    continue

                secure_store.add_sample(name, face)
                saved += 1
                print(f"Captured sample {saved}/{samples_count}")
                time.sleep(0.2)

                if saved >= samples_count:
                    secure_store.append_audit_event(label=name.lower(), confidence=0.0, unlocked=True)
                    print(f"Enrolled '{name}' successfully into encrypted store.")
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Secure face lock/unlock UI (OpenCV only)")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=52.0, help="LBPH threshold, lower is stricter")
    parser.add_argument("--cooldown", type=float, default=2.0)
    parser.add_argument("--stable-frames", type=int, default=4, help="Known predictions needed before unlock")
    parser.add_argument("--secure-dir", type=Path, default=Path("secure_data"))
    parser.add_argument("--key-file", type=Path, default=None)

    subparsers = parser.add_subparsers(dest="command", required=False)
    enroll_parser = subparsers.add_parser("enroll", help="Add a new authorized face")
    enroll_parser.add_argument("--name", required=True)
    enroll_parser.add_argument("--samples", type=int, default=12, help="Samples to capture for better accuracy")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = SecureFaceStore(secure_dir=args.secure_dir, key_file=args.key_file)

    if args.command == "enroll":
        enroll_new_face(store, args.name, args.camera_index, samples_count=args.samples)
        return

    app = FaceUnlockUI(
        secure_store=store,
        camera_index=args.camera_index,
        confidence_threshold=args.threshold,
        cooldown_seconds=args.cooldown,
        stable_frames=args.stable_frames,
    )
    app.run()


if __name__ == "__main__":
    main()
