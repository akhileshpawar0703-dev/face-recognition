import argparse
import json
import os
import pickle
import time
import urllib.request
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

YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"


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
        self.model_dir = self.secure_dir / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)
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


class FaceDetector:
    def __init__(self, model_dir: Path, detector_mode: str = "auto") -> None:
        self.detector_mode = detector_mode
        self.yunet = None
        self.haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if detector_mode in ("auto", "yunet"):
            self.yunet = self._load_yunet(model_dir)
        if detector_mode == "yunet" and self.yunet is None:
            raise RuntimeError("YuNet detector requested but could not be loaded.")

    def _load_yunet(self, model_dir: Path):
        try:
            model_path = model_dir / "face_detection_yunet_2023mar.onnx"
            if not model_path.exists():
                urllib.request.urlretrieve(YUNET_URL, model_path)
            detector = cv2.FaceDetectorYN_create(
                str(model_path),
                "",
                (320, 320),
                score_threshold=0.7,
                nms_threshold=0.3,
                top_k=5000,
            )
            print("[INFO] Detection backend: YuNet")
            return detector
        except Exception:
            print("[WARN] YuNet unavailable, fallback to Haar cascade")
            return None

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        h, w = frame.shape[:2]
        boxes: List[Tuple[int, int, int, int]] = []

        if self.yunet is not None:
            self.yunet.setInputSize((w, h))
            _, faces = self.yunet.detect(frame)
            if faces is not None:
                for row in faces:
                    x, y, bw, bh = row[:4]
                    x, y, bw, bh = int(x), int(y), int(bw), int(bh)
                    if bw > 20 and bh > 20:
                        boxes.append((x, y, bw, bh))
                return boxes

        min_size = max(60, min(h, w) // 8)
        faces = self.haar.detectMultiScale(frame, scaleFactor=1.15, minNeighbors=5, minSize=(min_size, min_size))
        for x, y, bw, bh in faces:
            boxes.append((int(x), int(y), int(bw), int(bh)))
        return boxes


class FaceEngine:
    def __init__(self, samples: Dict[str, List[np.ndarray]], threshold: float, detector: FaceDetector) -> None:
        if not hasattr(cv2, "face"):
            raise RuntimeError("OpenCV face module missing. Install opencv-contrib-python.")

        self.detector = detector
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=16, grid_x=8, grid_y=8)
        self.id_to_name: Dict[int, str] = {}

        self._train(samples)
        self.threshold = threshold if threshold > 0 else self._calibrate_threshold()

    @staticmethod
    def _preprocess(face_gray: np.ndarray) -> np.ndarray:
        face = cv2.resize(face_gray, FACE_SIZE)
        face = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(face)
        return cv2.GaussianBlur(face, (3, 3), 0)

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
        scores = [float(self.recognizer.predict(face)[1]) for face in self.training_faces]
        p95 = np.percentile(scores, 95) if scores else 80.0
        threshold = float(np.clip(p95 + 25.0, 65.0, 130.0))
        print(f"[INFO] Auto threshold selected: {threshold:.1f}")
        return threshold

    def detect_faces(self, frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return self.detector.detect(gray if self.detector.yunet is None else frame_bgr)

    def recognize(self, frame_bgr: np.ndarray) -> List[MatchResult]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        boxes = self.detect_faces(frame_bgr)
        results: List[MatchResult] = []

        for x, y, w, h in boxes:
            x = max(0, x)
            y = max(0, y)
            w = min(w, gray.shape[1] - x)
            h = min(h, gray.shape[0] - y)
            if w <= 0 or h <= 0:
                continue

            crop = gray[y : y + h, x : x + w]
            processed = self._preprocess(crop)
            label_id, confidence = self.recognizer.predict(processed)
            predicted = self.id_to_name.get(label_id, "unknown")
            final = predicted if confidence <= self.threshold else "unknown"
            results.append(MatchResult((y, x + w, y + h, x), predicted, final, float(confidence)))

        return results


class FaceUnlockUI:
    def __init__(
        self,
        secure_store: SecureFaceStore,
        camera_index: int,
        confidence_threshold: float,
        cooldown_seconds: float,
        stable_frames: int,
        detector_mode: str,
    ) -> None:
        self.secure_store = secure_store
        self.camera_index = camera_index
        self.cooldown_seconds = cooldown_seconds
        self.stable_frames = max(stable_frames, 2)
        detector = FaceDetector(secure_store.model_dir, detector_mode=detector_mode)
        self.engine = FaceEngine(secure_store.load_samples(), threshold=confidence_threshold, detector=detector)

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


def _draw_android_face_guide(frame: np.ndarray, progress: float, step_text: str, status: str) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Android-like enrollment overlay with oval guide and progress."""
    h, w = frame.shape[:2]
    center = (w // 2, h // 2)
    axes = (int(w * 0.17), int(h * 0.30))

    # dim outside for focus
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (20, 20, 28), -1)
    cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)

    # oval face frame
    cv2.ellipse(frame, center, axes, 0, 0, 360, (0, 215, 120), 2)

    # top/bottom/left/right alignment ticks
    tick = 24
    cv2.line(frame, (center[0] - tick, center[1] - axes[1]), (center[0] + tick, center[1] - axes[1]), (0, 215, 120), 2)
    cv2.line(frame, (center[0] - tick, center[1] + axes[1]), (center[0] + tick, center[1] + axes[1]), (0, 215, 120), 2)
    cv2.line(frame, (center[0] - axes[0], center[1] - tick), (center[0] - axes[0], center[1] + tick), (0, 215, 120), 2)
    cv2.line(frame, (center[0] + axes[0], center[1] - tick), (center[0] + axes[0], center[1] + tick), (0, 215, 120), 2)

    # progress ring around oval
    end_angle = int(360 * max(0.0, min(1.0, progress)))
    cv2.ellipse(frame, center, (axes[0] + 24, axes[1] + 24), -90, 0, end_angle, (0, 230, 140), 5)

    # header/footer text
    cv2.putText(frame, "Android-style Face Setup", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.88, (240, 240, 240), 2)
    cv2.putText(frame, step_text, (24, h - 66), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (240, 240, 240), 2)
    cv2.putText(frame, status, (24, h - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 220, 130), 2)
    cv2.putText(frame, "q cancel | s manual capture", (24, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (210, 210, 210), 1)

    return center, axes


def _extract_enroll_face(detector: FaceDetector, frame_bgr: np.ndarray, guide: Tuple[Tuple[int, int], Tuple[int, int]]) -> Tuple[Optional[np.ndarray], str]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector.detect(frame_bgr if detector.yunet is not None else gray)
    if not faces:
        return None, "No face detected"

    # choose largest face
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    cx, cy = x + w // 2, y + h // 2
    center, axes = guide

    # check face center inside oval
    nx = (cx - center[0]) / max(axes[0], 1)
    ny = (cy - center[1]) / max(axes[1], 1)
    if (nx * nx + ny * ny) > 0.72:
        return None, "Center your face in the oval"

    # relative size constraints for typical webcam distance
    target_w = axes[0] * 1.35
    if not (target_w * 0.65 <= w <= target_w * 1.45):
        return None, "Move slightly closer/farther"

    crop = gray[max(0, y):max(0, y) + h, max(0, x):max(0, x) + w]
    if crop.size == 0:
        return None, "Face crop failed"

    blur = cv2.Laplacian(crop, cv2.CV_64F).var()
    if blur < 35:
        return None, "Hold still (blurry frame)"

    face = cv2.resize(crop, FACE_SIZE)
    face = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(face)
    return face, "Good capture"


def enroll_new_face_android_style(
    store: SecureFaceStore,
    name: str,
    camera_index: int,
    samples_count: int,
    detector_mode: str,
) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}")

    detector = FaceDetector(store.model_dir, detector_mode=detector_mode)
    saved = 0
    last_capture = 0.0
    print("Android-like face setup started. Follow prompts on screen.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Could not read frame")
                break

            step_idx = min(saved, len(ENROLL_STEPS) - 1)
            step_text = f"Step {saved + 1}/{samples_count}: {ENROLL_STEPS[step_idx]}"
            progress = saved / max(samples_count, 1)
            guide = _draw_android_face_guide(frame, progress, step_text, "Align face with oval")

            face, status = _extract_enroll_face(detector, frame, guide)
            cv2.putText(frame, status, (24, frame.shape[0] - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 220, 130), 2)

            if face is not None and (time.time() - last_capture) > 0.45:
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
                time.sleep(0.12)

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
    parser.add_argument("--threshold", type=float, default=-1, help="LBPH threshold; <=0 enables auto-calibration")
    parser.add_argument("--cooldown", type=float, default=2.0)
    parser.add_argument("--stable-frames", type=int, default=2)
    parser.add_argument("--detector", choices=["auto", "yunet", "haar"], default="auto")
    parser.add_argument("--secure-dir", type=Path, default=Path("secure_data"))
    parser.add_argument("--key-file", type=Path, default=None)

    sub = parser.add_subparsers(dest="command", required=False)
    enroll = sub.add_parser("enroll", help="Android-like guided face enrollment")
    enroll.add_argument("--name", required=True)
    enroll.add_argument("--samples", type=int, default=12)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = SecureFaceStore(args.secure_dir, args.key_file)

    if args.command == "enroll":
        enroll_new_face_android_style(store, args.name, args.camera_index, args.samples, detector_mode=args.detector)
        return

    while True:
        app = FaceUnlockUI(
            secure_store=store,
            camera_index=args.camera_index,
            confidence_threshold=args.threshold,
            cooldown_seconds=args.cooldown,
            stable_frames=args.stable_frames,
            detector_mode=args.detector,
        )
        enroll_requested = app.run()
        if enroll_requested:
            person = input("Enter person name to enroll: ").strip()
            if person:
                enroll_new_face_android_style(store, person, args.camera_index, samples_count=12, detector_mode=args.detector)
            continue
        break


if __name__ == "__main__":
    main()
