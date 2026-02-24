import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import face_recognition
import numpy as np
from cryptography.fernet import Fernet


WELCOME_MESSAGES: Dict[str, str] = {
    "alice": "Welcome back, Alice 👋",
    "bob": "Hi Bob, access granted ✅",
    "charlie": "Great to see you, Charlie!",
}


@dataclass
class MatchResult:
    location: Tuple[int, int, int, int]
    label: str
    distance: float


class SecureFaceStore:
    """Encrypted storage for known face encodings and recognition events."""

    def __init__(self, secure_dir: Path, key_file: Optional[Path] = None) -> None:
        self.secure_dir = secure_dir
        self.secure_dir.mkdir(parents=True, exist_ok=True)
        self.key_file = key_file or (self.secure_dir / "face_store.key")
        self.db_file = self.secure_dir / "known_faces.enc"
        self.audit_file = self.secure_dir / "recognition_audit.log.enc"
        self.cipher = Fernet(self._load_or_create_key())

    def _load_or_create_key(self) -> bytes:
        if self.key_file.exists():
            return self.key_file.read_bytes()

        key = Fernet.generate_key()
        self.key_file.write_bytes(key)
        os.chmod(self.key_file, 0o600)
        return key

    def load_encodings(self) -> List[Tuple[str, np.ndarray]]:
        if not self.db_file.exists():
            return []

        encrypted = self.db_file.read_bytes()
        payload = self.cipher.decrypt(encrypted)
        data = json.loads(payload.decode("utf-8"))

        loaded: List[Tuple[str, np.ndarray]] = []
        for item in data:
            loaded.append((item["name"], np.array(item["encoding"], dtype=np.float64)))
        return loaded

    def save_encodings(self, rows: List[Tuple[str, np.ndarray]]) -> None:
        serializable = [
            {"name": name, "encoding": encoding.tolist()} for name, encoding in rows
        ]
        payload = json.dumps(serializable).encode("utf-8")
        encrypted = self.cipher.encrypt(payload)
        self.db_file.write_bytes(encrypted)
        os.chmod(self.db_file, 0o600)

    def add_encoding(self, name: str, encoding: np.ndarray) -> None:
        rows = self.load_encodings()
        rows.append((name.strip().lower(), encoding))
        self.save_encodings(rows)

    def append_audit_event(self, label: str, distance: float, unlocked: bool) -> None:
        event = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "label": label,
            "distance": round(float(distance), 5),
            "unlocked": unlocked,
        }
        encrypted_line = self.cipher.encrypt(json.dumps(event).encode("utf-8"))
        with self.audit_file.open("ab") as f:
            f.write(encrypted_line + b"\n")
        os.chmod(self.audit_file, 0o600)


class FaceUnlockUI:
    def __init__(
        self,
        secure_store: SecureFaceStore,
        camera_index: int = 0,
        match_threshold: float = 0.45,
        cooldown_seconds: float = 2.0,
    ) -> None:
        self.secure_store = secure_store
        self.camera_index = camera_index
        self.match_threshold = match_threshold
        self.cooldown_seconds = cooldown_seconds

        self.known_faces = self.secure_store.load_encodings()
        if not self.known_faces:
            raise RuntimeError(
                "No authorized faces found. Enroll a person first using: "
                "python app.py enroll --name <person_name>"
            )

        self.last_unlock_time = 0.0
        self.last_person: Optional[str] = None
        self.last_audit_at = 0.0

    def _recognize(self, frame: np.ndarray) -> List[MatchResult]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb, model="hog")
        encodings = face_recognition.face_encodings(rgb, locations)

        known_encodings = [encoding for _, encoding in self.known_faces]
        known_labels = [name for name, _ in self.known_faces]

        results: List[MatchResult] = []
        for loc, enc in zip(locations, encodings):
            distances = face_recognition.face_distance(known_encodings, enc)
            best_idx = int(np.argmin(distances))
            best_dist = float(distances[best_idx])
            label = known_labels[best_idx] if best_dist <= self.match_threshold else "unknown"
            results.append(MatchResult(location=loc, label=label, distance=best_dist))
        return results

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

                results = self._recognize(frame)
                now = time.time()
                unlocked_for_this_frame = False
                welcome_text = "LOCKED: Authorized face not detected"
                status_color = (0, 0, 255)
                audit_label = "unknown"
                audit_distance = 1.0

                for result in results:
                    top, right, bottom, left = result.location
                    label = result.label
                    dist = result.distance

                    if label != "unknown":
                        unlocked_for_this_frame = True
                        self.last_unlock_time = now
                        self.last_person = label
                        status_color = (0, 180, 0)
                        welcome_text = self._personalized_message(label)
                        audit_label = label
                        audit_distance = dist

                    box_color = (0, 255, 0) if label != "unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
                    cv2.putText(
                        frame,
                        f"{label} ({dist:.2f})",
                        (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        box_color,
                        2,
                    )

                if (
                    not unlocked_for_this_frame
                    and (now - self.last_unlock_time) < self.cooldown_seconds
                    and self.last_person
                ):
                    status_color = (0, 180, 0)
                    welcome_text = self._personalized_message(self.last_person)

                # Write encrypted audit events at most once per second
                if now - self.last_audit_at >= 1.0:
                    self.secure_store.append_audit_event(
                        label=audit_label,
                        distance=audit_distance,
                        unlocked=unlocked_for_this_frame,
                    )
                    self.last_audit_at = now

                cv2.rectangle(frame, (0, 0), (frame.shape[1], 45), (20, 20, 20), -1)
                cv2.putText(
                    frame,
                    welcome_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    status_color,
                    2,
                )
                cv2.imshow("Face Lock / Unlock UI", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()


def enroll_new_face(secure_store: SecureFaceStore, name: str, camera_index: int) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}")

    print("Press 's' to save detected face for enrollment, 'q' to cancel.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Could not read frame")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            locations = face_recognition.face_locations(rgb, model="hog")

            for top, right, bottom, left in locations:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)

            cv2.putText(
                frame,
                f"Enroll: {name} | Faces detected: {len(locations)}",
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
                if len(locations) != 1:
                    print("Please make sure exactly one face is visible before saving.")
                    continue

                encodings = face_recognition.face_encodings(rgb, locations)
                if not encodings:
                    print("Face detected but encoding failed. Try again.")
                    continue

                secure_store.add_encoding(name, encodings[0])
                secure_store.append_audit_event(label=name.lower(), distance=0.0, unlocked=True)
                print(f"Enrolled '{name}' successfully into encrypted face store.")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Secure face lock/unlock UI")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.45, help="Smaller = stricter")
    parser.add_argument("--cooldown", type=float, default=2.0)
    parser.add_argument(
        "--secure-dir",
        type=Path,
        default=Path("secure_data"),
        help="Directory for encrypted face data and encrypted audit logs",
    )
    parser.add_argument(
        "--key-file",
        type=Path,
        default=None,
        help="Optional path for encryption key file",
    )

    subparsers = parser.add_subparsers(dest="command", required=False)

    enroll_parser = subparsers.add_parser("enroll", help="Add a new authorized face")
    enroll_parser.add_argument("--name", required=True, help="Person name to enroll")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = SecureFaceStore(secure_dir=args.secure_dir, key_file=args.key_file)

    if args.command == "enroll":
        enroll_new_face(store, args.name, args.camera_index)
        return

    app = FaceUnlockUI(
        secure_store=store,
        camera_index=args.camera_index,
        match_threshold=args.threshold,
        cooldown_seconds=args.cooldown,
    )
    app.run()


if __name__ == "__main__":
    main()
