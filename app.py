import argparse
import json
import os
import pickle
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
        payload = pickle.dumps(serializable)
        encrypted = self.cipher.encrypt(payload)
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
            raise RuntimeError(
                "OpenCV face module not found. Install opencv-contrib-python."
            )

        self.threshold = threshold
        self.detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        self.id_to_name: Dict[int, str] = {}
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self._train(samples)

    def _train(self, samples: Dict[str, List[np.ndarray]]) -> None:
        faces: List[np.ndarray] = []
        labels: List[int] = []
        next_id = 0

        for name, sample_list in sorted(samples.items()):
            self.id_to_name[next_id] = name
            for sample in sample_list:
                faces.append(sample)
                labels.append(next_id)
            next_id += 1

        if not faces:
            raise RuntimeError("No enrolled faces found. Use enroll command first.")

        self.recognizer.train(faces, np.array(labels, dtype=np.int32))

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))

    def recognize(self, frame: np.ndarray) -> List[MatchResult]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = self.detect_faces(frame)
        results: List[MatchResult] = []

        for (x, y, w, h) in boxes:
            crop = gray[y : y + h, x : x + w]
            crop = cv2.resize(crop, FACE_SIZE)
            label_id, confidence = self.recognizer.predict(crop)
            # Lower confidence is better in LBPH.
            if confidence <= self.threshold:
                label = self.id_to_name.get(label_id, "unknown")
            else:
                label = "unknown"

            results.append(
                MatchResult(location=(y, x + w, y + h, x), label=label, confidence=confidence)
            )

        return results


class FaceUnlockUI:
    def __init__(
        self,
        secure_store: SecureFaceStore,
        camera_index: int = 0,
        confidence_threshold: float = 55.0,
        cooldown_seconds: float = 2.0,
    ) -> None:
        self.secure_store = secure_store
        self.camera_index = camera_index
        self.cooldown_seconds = cooldown_seconds
        self.engine = FaceEngine(secure_store.load_samples(), threshold=confidence_threshold)

        self.last_unlock_time = 0.0
        self.last_person: Optional[str] = None
        self.last_audit_at = 0.0

    @staticmethod
    def _personalized_message(name: str) -> str:
        return WELCOME_MESSAGES.get(name, f"Welcome, {name.title()}!")

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
                unlocked = False
                welcome_text = "LOCKED: Authorized face not detected"
                status_color = (0, 0, 255)
                audit_label = "unknown"
                audit_conf = 999.0

                for result in results:
                    top, right, bottom, left = result.location
                    label = result.label

                    if label != "unknown":
                        unlocked = True
                        self.last_unlock_time = now
                        self.last_person = label
                        status_color = (0, 180, 0)
                        welcome_text = self._personalized_message(label)
                        audit_label = label
                        audit_conf = result.confidence

                    box_color = (0, 255, 0) if label != "unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
                    cv2.putText(
                        frame,
                        f"{label} (lbph:{result.confidence:.1f})",
                        (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        box_color,
                        2,
                    )

                if not unlocked and (now - self.last_unlock_time) < self.cooldown_seconds and self.last_person:
                    status_color = (0, 180, 0)
                    welcome_text = self._personalized_message(self.last_person)

                if now - self.last_audit_at >= 1.0:
                    self.secure_store.append_audit_event(
                        label=audit_label, confidence=audit_conf, unlocked=unlocked
                    )
                    self.last_audit_at = now

                cv2.rectangle(frame, (0, 0), (frame.shape[1], 45), (20, 20, 20), -1)
                cv2.putText(frame, welcome_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                cv2.imshow("Face Lock / Unlock UI", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()


def _extract_single_face(gray_frame: np.ndarray) -> Optional[np.ndarray]:
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    boxes = detector.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))
    if len(boxes) != 1:
        return None
    x, y, w, h = boxes[0]
    crop = gray_frame[y : y + h, x : x + w]
    return cv2.resize(crop, FACE_SIZE)


def enroll_new_face(secure_store: SecureFaceStore, name: str, camera_index: int, samples_count: int) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}")

    saved = 0
    print("Press 's' to capture sample face, 'q' to cancel.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Could not read frame")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            preview_face = _extract_single_face(gray)
            status = "Ready" if preview_face is not None else "Need exactly 1 face"

            cv2.putText(
                frame,
                f"Enroll: {name} | saved {saved}/{samples_count} | {status}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.imshow("Enroll New Face", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Enrollment cancelled.")
                break
            if key == ord("s"):
                if preview_face is None:
                    print("Ensure exactly one face is visible before capturing.")
                    continue

                secure_store.add_sample(name, preview_face)
                saved += 1
                print(f"Captured sample {saved}/{samples_count}")

                if saved >= samples_count:
                    secure_store.append_audit_event(label=name.lower(), confidence=0.0, unlocked=True)
                    print(f"Enrolled '{name}' successfully into encrypted face store.")
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Secure face lock/unlock UI (OpenCV only)")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument(
        "--threshold",
        type=float,
        default=55.0,
        help="LBPH threshold (lower is stricter, typical 40-70)",
    )
    parser.add_argument("--cooldown", type=float, default=2.0)
    parser.add_argument("--secure-dir", type=Path, default=Path("secure_data"))
    parser.add_argument("--key-file", type=Path, default=None)

    subparsers = parser.add_subparsers(dest="command", required=False)
    enroll_parser = subparsers.add_parser("enroll", help="Add a new authorized face")
    enroll_parser.add_argument("--name", required=True, help="Person name to enroll")
    enroll_parser.add_argument("--samples", type=int, default=8, help="Captured samples count")

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
    )
    app.run()


if __name__ == "__main__":
    main()
