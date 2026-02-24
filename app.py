import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import face_recognition
import numpy as np


@dataclass
class KnownFace:
    name: str
    encoding: np.ndarray


WELCOME_MESSAGES: Dict[str, str] = {
    "alice": "Welcome back, Alice 👋",
    "bob": "Hi Bob, access granted ✅",
    "charlie": "Great to see you, Charlie!",
}


class FaceUnlockUI:
    def __init__(
        self,
        known_faces_dir: Path,
        camera_index: int = 0,
        match_threshold: float = 0.45,
        cooldown_seconds: float = 2.0,
    ) -> None:
        self.known_faces_dir = known_faces_dir
        self.camera_index = camera_index
        self.match_threshold = match_threshold
        self.cooldown_seconds = cooldown_seconds
        self.known_faces: List[KnownFace] = self._load_known_faces()
        self.last_unlock_time = 0.0
        self.last_person: Optional[str] = None

    def _load_known_faces(self) -> List[KnownFace]:
        if not self.known_faces_dir.exists():
            raise FileNotFoundError(
                f"Known faces directory not found: {self.known_faces_dir}. "
                "Create folders like known_faces/alice/alice1.jpg"
            )

        loaded: List[KnownFace] = []
        image_paths = sorted(
            [
                p
                for p in self.known_faces_dir.rglob("*")
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]
        )

        for image_path in image_paths:
            name = image_path.parent.name.strip().lower()
            image = face_recognition.load_image_file(str(image_path))
            encodings = face_recognition.face_encodings(image)
            if not encodings:
                print(f"[WARN] No face found in training image: {image_path}")
                continue
            loaded.append(KnownFace(name=name, encoding=encodings[0]))

        if not loaded:
            raise RuntimeError(
                "No encodings loaded. Add face images in subfolders under known_faces/."
            )

        print(f"Loaded {len(loaded)} face encodings from {self.known_faces_dir}")
        return loaded

    def _recognize(self, frame: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], str, float]]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb, model="hog")
        encodings = face_recognition.face_encodings(rgb, locations)

        results: List[Tuple[Tuple[int, int, int, int], str, float]] = []
        for loc, enc in zip(locations, encodings):
            distances = face_recognition.face_distance(
                [known.encoding for known in self.known_faces], enc
            )
            best_idx = int(np.argmin(distances))
            best_dist = float(distances[best_idx])
            if best_dist <= self.match_threshold:
                label = self.known_faces[best_idx].name
            else:
                label = "unknown"
            results.append((loc, label, best_dist))
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

                for (top, right, bottom, left), label, dist in results:
                    if label != "unknown":
                        unlocked_for_this_frame = True
                        self.last_unlock_time = now
                        self.last_person = label
                        status_color = (0, 180, 0)
                        welcome_text = self._personalized_message(label)

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

                if not unlocked_for_this_frame and (now - self.last_unlock_time) < self.cooldown_seconds:
                    status_color = (0, 180, 0)
                    if self.last_person:
                        welcome_text = self._personalized_message(self.last_person)

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Face detection + person-specific unlock UI"
    )
    parser.add_argument(
        "--known-faces-dir",
        type=Path,
        default=Path("known_faces"),
        help="Folder with subfolders per person containing face images",
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.45,
        help="Smaller = stricter matching",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=2.0,
        help="Keep unlocked message for this many seconds after last recognized frame",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = FaceUnlockUI(
        known_faces_dir=args.known_faces_dir,
        camera_index=args.camera_index,
        match_threshold=args.threshold,
        cooldown_seconds=args.cooldown,
    )
    app.run()


if __name__ == "__main__":
    main()
