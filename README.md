# Secure Face Detection Lock/Unlock UI (Android-Style Enrollment)

This project provides a better-looking, user-friendly face lock/unlock app with:
- improved face recognition stability,
- secure encrypted storage,
- Android-phone-like guided face scanning UI for enrollment.

## What is improved
- Cleaner lock screen UI with status panels and clear instructions.
- Better face detection/recognition robustness:
  - histogram equalization + denoise preprocessing,
  - dual-pass face detection merge,
  - eye-presence validation,
  - stable-frame voting before unlock.
- Android-style enrollment flow:
  - center ring guide,
  - progress arc,
  - auto-capture when face quality and position are valid,
  - step-by-step prompts (left/right/up/down/near/far).

## Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Enroll first (recommended)

```bash
python app.py enroll --name "alice" --samples 8
```

## Run lock/unlock UI

```bash
python app.py --camera-index 0 --threshold 50 --stable-frames 4 --cooldown 2.0
```

Inside unlock UI:
- Press `e` to close and then enroll a new face from terminal prompt.
- Press `q` or `ESC` to quit.

## Accuracy tuning
- Lower `--threshold` => stricter matching (fewer false unlocks).
- Higher `--stable-frames` => more consistent recognition required before unlock.
- Enroll in good lighting and collect diverse samples (angles/distances).

## Secure data
Stored in `secure_data/`:
- `face_store.key`
- `face_samples.enc`
- `recognition_audit.log.enc`

All sensitive files are encrypted; key/data files are set with restricted permissions where supported.
