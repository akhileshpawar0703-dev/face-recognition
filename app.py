import argparse
import base64
import hashlib
import json
import os
import pickle
import tempfile
import time
import urllib.request
import webbrowser
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
from cryptography.fernet import Fernet, InvalidToken

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
    """Encrypted storage with minimization + tamper-evident audit chain."""

    def __init__(self, secure_dir: Path, key_file: Optional[Path] = None, max_samples_per_person: int = 20) -> None:
        self.secure_dir = secure_dir
        self.secure_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir = self.secure_dir / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.key_file = key_file or (self.secure_dir / "face_store.key")
        self.samples_file = self.secure_dir / "face_samples.enc"
        self.audit_file = self.secure_dir / "recognition_audit.log.enc"
        self.audit_state_file = self.secure_dir / "audit_chain.state"
        self.file_registry_file = self.secure_dir / "file_registry.enc"
        self.max_samples_per_person = max_samples_per_person
        self.cipher = Fernet(self._load_or_create_key())

    def _load_or_create_key(self) -> bytes:
        if self.key_file.exists():
            return self.key_file.read_bytes()
        key = Fernet.generate_key()
        self.key_file.write_bytes(key)
        os.chmod(self.key_file, 0o600)
        return key

    def _read_audit_chain_state(self) -> str:
        if not self.audit_state_file.exists():
            return "GENESIS"
        return self.audit_state_file.read_text(encoding="utf-8").strip() or "GENESIS"

    def _write_audit_chain_state(self, chain_hash: str) -> None:
        self.audit_state_file.write_text(chain_hash, encoding="utf-8")
        os.chmod(self.audit_state_file, 0o600)

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
        # Privacy/minimization: cap stored samples per person.
        minimized: Dict[str, List[np.ndarray]] = {}
        for name, sample_list in samples.items():
            minimized[name] = sample_list[-self.max_samples_per_person :]

        serializable = {
            name: [sample.tolist() for sample in sample_list]
            for name, sample_list in minimized.items()
        }
        self.samples_file.write_bytes(self.cipher.encrypt(pickle.dumps(serializable)))
        os.chmod(self.samples_file, 0o600)

    def add_sample(self, name: str, face_image: np.ndarray) -> None:
        normalized = name.strip().lower()
        samples = self.load_samples()
        samples.setdefault(normalized, []).append(face_image)
        self.save_samples(samples)

    def delete_person(self, name: str) -> bool:
        normalized = name.strip().lower()
        samples = self.load_samples()
        if normalized not in samples:
            return False
        del samples[normalized]
        self.save_samples(samples)
        self.append_audit_event(label=normalized, confidence=0.0, unlocked=False, event_type="delete_identity")
        return True

    def append_audit_event(self, label: str, confidence: float, unlocked: bool, event_type: str = "recognition") -> None:
        ts = datetime.now(timezone.utc).isoformat()
        core = {
            "timestamp_utc": ts,
            "event_type": event_type,
            "label": label,
            "confidence": round(float(confidence), 3),
            "unlocked": unlocked,
        }
        prev_hash = self._read_audit_chain_state()
        payload_for_hash = json.dumps(core, sort_keys=True) + "|" + prev_hash
        chain_hash = hashlib.sha256(payload_for_hash.encode("utf-8")).hexdigest()
        record = {**core, "prev_hash": prev_hash, "chain_hash": chain_hash}

        enc = self.cipher.encrypt(json.dumps(record).encode("utf-8"))
        with self.audit_file.open("ab") as f:
            f.write(enc + b"\n")
        os.chmod(self.audit_file, 0o600)
        self._write_audit_chain_state(chain_hash)

    def verify_audit_chain(self) -> bool:
        if not self.audit_file.exists():
            return True
        prev_hash = "GENESIS"
        for line in self.audit_file.read_bytes().splitlines():
            if not line:
                continue
            try:
                rec = json.loads(self.cipher.decrypt(line).decode("utf-8"))
            except (InvalidToken, json.JSONDecodeError):
                return False
            expected = hashlib.sha256(
                (json.dumps(
                    {
                        "timestamp_utc": rec.get("timestamp_utc"),
                        "event_type": rec.get("event_type", "recognition"),
                        "label": rec.get("label"),
                        "confidence": rec.get("confidence"),
                        "unlocked": rec.get("unlocked"),
                    },
                    sort_keys=True,
                )
                + "|"
                + prev_hash).encode("utf-8")
            ).hexdigest()
            if rec.get("prev_hash") != prev_hash or rec.get("chain_hash") != expected:
                return False
            prev_hash = rec["chain_hash"]
        return True

    def prune_audit_older_than(self, days: int) -> int:
        if days <= 0 or not self.audit_file.exists():
            return 0

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        kept: List[bytes] = []
        removed = 0
        prev_hash = "GENESIS"

        for line in self.audit_file.read_bytes().splitlines():
            if not line:
                continue
            try:
                rec = json.loads(self.cipher.decrypt(line).decode("utf-8"))
            except Exception:
                continue

            ts = datetime.fromisoformat(rec["timestamp_utc"].replace("Z", "+00:00"))
            if ts >= cutoff:
                core = {
                    "timestamp_utc": rec["timestamp_utc"],
                    "event_type": rec.get("event_type", "recognition"),
                    "label": rec["label"],
                    "confidence": rec["confidence"],
                    "unlocked": rec["unlocked"],
                }
                payload = json.dumps(core, sort_keys=True) + "|" + prev_hash
                chain_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
                new_rec = {**core, "prev_hash": prev_hash, "chain_hash": chain_hash}
                kept.append(self.cipher.encrypt(json.dumps(new_rec).encode("utf-8")))
                prev_hash = chain_hash
            else:
                removed += 1

        with self.audit_file.open("wb") as f:
            for enc in kept:
                f.write(enc + b"\n")
        os.chmod(self.audit_file, 0o600)
        self._write_audit_chain_state(prev_hash)
        return removed

    def _load_file_registry(self) -> Dict[str, Dict[str, str]]:
        if not self.file_registry_file.exists():
            return {}
        payload = self.cipher.decrypt(self.file_registry_file.read_bytes())
        data = json.loads(payload.decode("utf-8"))
        return data if isinstance(data, dict) else {}

    def _save_file_registry(self, registry: Dict[str, Dict[str, str]]) -> None:
        payload = json.dumps(registry).encode("utf-8")
        self.file_registry_file.write_bytes(self.cipher.encrypt(payload))
        os.chmod(self.file_registry_file, 0o600)

    def register_encrypted_pdf(self, encrypted_path: Path, owner: str, file_key: bytes, original_name: str) -> None:
        registry = self._load_file_registry()
        registry[str(encrypted_path.resolve())] = {
            "owner": owner.strip().lower(),
            "key_b64": base64.urlsafe_b64encode(file_key).decode("utf-8"),
            "original_name": original_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save_file_registry(registry)

    def get_encrypted_pdf_record(self, encrypted_path: Path) -> Optional[Dict[str, str]]:
        registry = self._load_file_registry()
        return registry.get(str(encrypted_path.resolve()))


def encrypt_pdf_with_face(store: SecureFaceStore, pdf_path: Path, output_path: Optional[Path], owner: str) -> Path:
    if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"PDF not found or invalid extension: {pdf_path}")

    samples = store.load_samples()
    owner_norm = owner.strip().lower()
    if owner_norm not in samples:
        raise ValueError(f"Owner '{owner}' is not enrolled. Enroll first.")

    out = output_path or pdf_path.with_suffix(".facepdf")
    raw_pdf = pdf_path.read_bytes()
    file_key = Fernet.generate_key()
    encrypted_pdf = Fernet(file_key).encrypt(raw_pdf)
    out.write_bytes(encrypted_pdf)

    store.register_encrypted_pdf(out, owner=owner_norm, file_key=file_key, original_name=pdf_path.name)
    store.append_audit_event(label=owner_norm, confidence=0.0, unlocked=False, event_type="encrypt_pdf")
    return out


def authenticate_face_once(
    store: SecureFaceStore,
    camera_index: int,
    threshold: float,
    detector_mode: str,
    timeout_seconds: int,
) -> Optional[str]:
    detector = FaceDetector(store.model_dir, detector_mode=detector_mode)
    engine = FaceEngine(store.load_samples(), threshold=threshold, detector=detector)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}")

    history: Deque[str] = deque(maxlen=3)
    deadline = time.time() + timeout_seconds
    try:
        while time.time() < deadline:
            ok, frame = cap.read()
            if not ok:
                continue

            results = engine.recognize(frame)
            best = "unknown"
            best_score = 999.0
            for result in results:
                t, r, b, l = result.location
                candidate = result.final_label
                if result.final_label == "unknown" and result.predicted_label != "unknown" and result.confidence <= engine.soft_threshold:
                    candidate = result.predicted_label

                if result.confidence < best_score:
                    best_score, best = result.confidence, candidate

                color = (0, 190, 0) if candidate != "unknown" else (0, 0, 230)
                cv2.rectangle(frame, (l, t), (r, b), color, 2)
                cv2.putText(frame, f"{candidate} | {result.confidence:.1f}", (l, max(16, t - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            history.append(best)
            known = [x for x in history if x != "unknown"]
            if len(known) >= 2:
                label, votes = Counter(known).most_common(1)[0]
                if votes >= 2:
                    cv2.putText(frame, f"Authenticated: {label}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 120), 2)
                    cv2.imshow("Face PDF Auth", frame)
                    cv2.waitKey(250)
                    return label

            remaining = max(0, int(deadline - time.time()))
            cv2.putText(frame, f"Face auth for PDF unlock | timeout {remaining}s", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (240, 240, 240), 2)
            cv2.imshow("Face PDF Auth", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return None


def unlock_pdf_with_face(
    store: SecureFaceStore,
    encrypted_pdf: Path,
    camera_index: int,
    threshold: float,
    detector_mode: str,
    timeout_seconds: int,
    output_pdf: Optional[Path],
) -> Path:
    if not encrypted_pdf.exists():
        raise FileNotFoundError(f"Encrypted PDF not found: {encrypted_pdf}")

    rec = store.get_encrypted_pdf_record(encrypted_pdf)
    if not rec:
        raise RuntimeError("Encrypted PDF metadata not found in secure store. Encrypt via this app first.")

    person = authenticate_face_once(
        store=store,
        camera_index=camera_index,
        threshold=threshold,
        detector_mode=detector_mode,
        timeout_seconds=timeout_seconds,
    )
    if not person:
        store.append_audit_event(label="unknown", confidence=999.0, unlocked=False, event_type="pdf_unlock_failed")
        raise RuntimeError("Face authentication failed or timed out.")

    if person.strip().lower() != rec["owner"]:
        store.append_audit_event(label=person, confidence=999.0, unlocked=False, event_type="pdf_unlock_denied")
        raise PermissionError(f"Authenticated as '{person}', but this PDF is owned by '{rec['owner']}'.")

    file_key = base64.urlsafe_b64decode(rec["key_b64"].encode("utf-8"))
    decrypted = Fernet(file_key).decrypt(encrypted_pdf.read_bytes())

    if output_pdf:
        out = output_pdf
    else:
        tmp = Path(tempfile.gettempdir()) / f"face_unlock_{int(time.time())}_{rec['original_name']}"
        out = tmp

    out.write_bytes(decrypted)
    store.append_audit_event(label=person, confidence=0.0, unlocked=True, event_type="pdf_unlock_success")
    return out


class FaceDetector:
    def __init__(self, model_dir: Path, detector_mode: str = "auto") -> None:
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
            detector = cv2.FaceDetectorYN_create(str(model_path), "", (320, 320), score_threshold=0.7, nms_threshold=0.3, top_k=5000)
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
                    x, y, bw, bh = map(int, row[:4])
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
        self.soft_threshold = float(min(self.threshold + 45.0, 170.0))

    @staticmethod
    def _preprocess(face_gray: np.ndarray) -> np.ndarray:
        face = cv2.resize(face_gray, FACE_SIZE)
        face = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(face)
        return cv2.GaussianBlur(face, (3, 3), 0)

    def _train(self, samples: Dict[str, List[np.ndarray]]) -> None:
        faces, labels = [], []
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
        out: List[MatchResult] = []
        for x, y, w, h in boxes:
            x, y = max(0, x), max(0, y)
            w, h = min(w, gray.shape[1] - x), min(h, gray.shape[0] - y)
            if w <= 0 or h <= 0:
                continue
            crop = gray[y : y + h, x : x + w]
            processed = self._preprocess(crop)
            label_id, confidence = self.recognizer.predict(processed)
            predicted = self.id_to_name.get(label_id, "unknown")
            final = predicted if confidence <= self.threshold else "unknown"
            out.append(MatchResult((y, x + w, y + h, x), predicted, final, float(confidence)))
        return out


class FaceUnlockUI:
    def __init__(self, secure_store: SecureFaceStore, camera_index: int, confidence_threshold: float, cooldown_seconds: float, stable_frames: int, detector_mode: str) -> None:
        self.secure_store = secure_store
        self.camera_index = camera_index
        self.cooldown_seconds = cooldown_seconds
        self.stable_frames = max(stable_frames, 2)
        detector = FaceDetector(secure_store.model_dir, detector_mode=detector_mode)
        self.engine = FaceEngine(secure_store.load_samples(), threshold=confidence_threshold, detector=detector)

        self.history: Deque[str] = deque(maxlen=self.stable_frames)
        self.score_history: Deque[float] = deque(maxlen=self.stable_frames)
        self.last_unlock_time = 0.0
        self.last_person: Optional[str] = None
        self.last_audit_at = 0.0

        # Runtime safety: rate-limit repeated failures with temporary lockouts.
        self.fail_streak = 0
        self.locked_until = 0.0

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
        metrics = f"thr={self.engine.threshold:.1f} soft={self.engine.soft_threshold:.1f} best={best_score:.1f}"
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
                best_score, best_label = 999.0, "unknown"

                for result in results:
                    top, right, bottom, left = result.location
                    candidate_label = result.final_label
                    soft_match = (
                        result.final_label == "unknown"
                        and result.predicted_label != "unknown"
                        and result.confidence <= self.engine.soft_threshold
                    )
                    if soft_match:
                        candidate_label = result.predicted_label

                    if result.confidence < best_score:
                        best_score = result.confidence
                        best_label = candidate_label

                    if result.final_label != "unknown":
                        shown, color = result.final_label, (0, 190, 0)
                    elif soft_match:
                        shown, color = f"maybe {result.predicted_label}", (0, 180, 255)
                    else:
                        shown, color = f"unknown ({result.predicted_label})", (0, 0, 230)

                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    cv2.putText(frame, f"{shown} | score {result.confidence:.1f}", (left, max(16, top - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.54, color, 2)

                self.history.append(best_label)
                self.score_history.append(best_score)
                known_hist = [x for x in self.history if x != "unknown"]

                unlocked, person = False, None
                if len(known_hist) >= self.stable_frames:
                    label, votes = Counter(known_hist).most_common(1)[0]
                    if votes >= self.stable_frames:
                        scores_for_label = [sc for lb, sc in zip(self.history, self.score_history) if lb == label]
                        avg_score = float(np.mean(scores_for_label)) if scores_for_label else 999.0
                        if avg_score <= self.engine.soft_threshold:
                            unlocked, person = True, label

                if now < self.locked_until:
                    state = "LOCKED"
                    secs = int(self.locked_until - now)
                    greeting = f"Temporarily locked ({secs}s) after repeated failures"
                    audit_label = "unknown"
                elif unlocked and person:
                    self.fail_streak = 0
                    self.last_person = person
                    self.last_unlock_time = now
                    state, greeting, audit_label = "UNLOCKED", self._personalized(person), person
                elif (now - self.last_unlock_time) < self.cooldown_seconds and self.last_person:
                    state, greeting, audit_label = "UNLOCKED", self._personalized(self.last_person), self.last_person
                else:
                    self.fail_streak += 1
                    if self.fail_streak in (10, 20, 30):
                        backoff = min(60, 5 * (2 ** (self.fail_streak // 10 - 1)))
                        self.locked_until = now + backoff
                    state, greeting, audit_label = "LOCKED", "Locked: align your face inside camera frame", "unknown"

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
    h, w = frame.shape[:2]
    center, axes = (w // 2, h // 2), (int(w * 0.17), int(h * 0.30))
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (20, 20, 28), -1)
    cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)
    cv2.ellipse(frame, center, axes, 0, 0, 360, (0, 215, 120), 2)
    tick = 24
    cv2.line(frame, (center[0] - tick, center[1] - axes[1]), (center[0] + tick, center[1] - axes[1]), (0, 215, 120), 2)
    cv2.line(frame, (center[0] - tick, center[1] + axes[1]), (center[0] + tick, center[1] + axes[1]), (0, 215, 120), 2)
    cv2.line(frame, (center[0] - axes[0], center[1] - tick), (center[0] - axes[0], center[1] + tick), (0, 215, 120), 2)
    cv2.line(frame, (center[0] + axes[0], center[1] - tick), (center[0] + axes[0], center[1] + tick), (0, 215, 120), 2)
    end_angle = int(360 * max(0.0, min(1.0, progress)))
    cv2.ellipse(frame, center, (axes[0] + 24, axes[1] + 24), -90, 0, end_angle, (0, 230, 140), 5)
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

    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    cx, cy = x + w // 2, y + h // 2
    center, axes = guide
    nx, ny = (cx - center[0]) / max(axes[0], 1), (cy - center[1]) / max(axes[1], 1)
    if (nx * nx + ny * ny) > 0.72:
        return None, "Center your face in the oval"

    target_w = axes[0] * 1.35
    if not (target_w * 0.65 <= w <= target_w * 1.45):
        return None, "Move slightly closer/farther"

    crop = gray[max(0, y) : max(0, y) + h, max(0, x) : max(0, x) + w]
    if crop.size == 0:
        return None, "Face crop failed"

    blur = cv2.Laplacian(crop, cv2.CV_64F).var()
    if blur < 35:
        return None, "Hold still (blurry frame)"

    face = cv2.resize(crop, FACE_SIZE)
    face = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(face)
    return face, "Good capture"


def enroll_new_face_android_style(store: SecureFaceStore, name: str, camera_index: int, samples_count: int, detector_mode: str) -> None:
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
                store.append_audit_event(name.lower(), 0.0, True, event_type="enrollment")
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
    parser.add_argument("--max-samples-per-person", type=int, default=20)
    parser.add_argument("--audit-retention-days", type=int, default=30)

    sub = parser.add_subparsers(dest="command", required=False)
    enroll = sub.add_parser("enroll", help="Android-like guided face enrollment")
    enroll.add_argument("--name", required=True)
    enroll.add_argument("--samples", type=int, default=12)
    enroll.add_argument("--detector", choices=["auto", "yunet", "haar"], default=None)

    delete = sub.add_parser("delete", help="Delete an enrolled identity (privacy right-to-delete)")
    delete.add_argument("--name", required=True)

    sub.add_parser("verify-audit", help="Verify tamper-evident audit chain")

    encrypt_pdf = sub.add_parser("encrypt-pdf", help="Encrypt a PDF and bind unlock to an enrolled face")
    encrypt_pdf.add_argument("--pdf", type=Path, required=True, help="Source PDF path")
    encrypt_pdf.add_argument("--owner", required=True, help="Enrolled owner name required for unlock")
    encrypt_pdf.add_argument("--out", type=Path, default=None, help="Output encrypted PDF path (.facepdf)")

    unlock_pdf = sub.add_parser("unlock-pdf", help="Face-auth unlock an encrypted PDF and open it")
    unlock_pdf.add_argument("--file", type=Path, required=True, help="Encrypted PDF path (.facepdf)")
    unlock_pdf.add_argument("--timeout", type=int, default=25, help="Face auth timeout in seconds")
    unlock_pdf.add_argument("--out", type=Path, default=None, help="Optional decrypted PDF output path")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = SecureFaceStore(args.secure_dir, args.key_file, max_samples_per_person=args.max_samples_per_person)

    # privacy minimization: prune old audit records on startup
    pruned = store.prune_audit_older_than(args.audit_retention_days)
    if pruned:
        print(f"[INFO] Pruned {pruned} old audit events")

    if args.command == "verify-audit":
        ok = store.verify_audit_chain()
        print("Audit chain OK" if ok else "Audit chain FAILED")
        return

    if args.command == "encrypt-pdf":
        out = encrypt_pdf_with_face(store, pdf_path=args.pdf, output_path=args.out, owner=args.owner)
        print(f"Encrypted PDF created: {out}")
        print("Use unlock-pdf to trigger camera auth and open the file.")
        return

    if args.command == "unlock-pdf":
        out = unlock_pdf_with_face(
            store=store,
            encrypted_pdf=args.file,
            camera_index=args.camera_index,
            threshold=args.threshold,
            detector_mode=args.detector,
            timeout_seconds=args.timeout,
            output_pdf=args.out,
        )
        print(f"PDF unlocked to: {out}")
        webbrowser.open(out.resolve().as_uri())
        print("Opened with default PDF handler (Chrome/reader depending on your OS defaults).")
        return

    if args.command == "delete":
        deleted = store.delete_person(args.name)
        print(f"Deleted: {args.name}" if deleted else f"Identity not found: {args.name}")
        return

    if args.command == "enroll":
        enroll_detector = args.detector if args.detector is not None else "auto"
        enroll_new_face_android_style(store, args.name, args.camera_index, args.samples, detector_mode=enroll_detector)
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
