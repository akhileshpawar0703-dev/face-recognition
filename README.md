# Secure Face Detection Lock/Unlock UI (Android-Style)

This update gives a **proper Android mobile-like face setup flow** while also improving detection reliability.

## What changed
- Added a detector backend selector:
  - `--detector auto` (default): YuNet first, fallback to Haar
  - `--detector yunet`: force YuNet
  - `--detector haar`: force Haar
- Enrollment UI now uses an **Android-style oval guide** with:
  - alignment ticks,
  - outer progress arc,
  - step-by-step prompts,
  - auto capture + manual capture (`s`).
- Enrollment and runtime use the same detector backend for consistency.
- Recognition still uses encrypted face storage + adaptive threshold support.

## Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Enroll (Android-like)

```bash
python app.py --detector auto enroll --name "akhilesh" --samples 12
# also supported:
# python app.py enroll --name "akhilesh" --samples 12 --detector auto
```

## Run unlock UI

```bash
python app.py --camera-index 0 --threshold -1 --stable-frames 2 --detector auto
```

## Tuning if still failing
1. Try `--detector yunet`.
2. If YuNet model download is blocked, use `--detector haar`.
3. Try higher threshold values like `--threshold 95` or `--threshold 115`.
4. Re-enroll in brighter frontal lighting.

## Secure data
Stored in `secure_data/`:
- `face_store.key`
- `face_samples.enc`
- `recognition_audit.log.enc`
- `models/face_detection_yunet_2023mar.onnx` (if YuNet used)
