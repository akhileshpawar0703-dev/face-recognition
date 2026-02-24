# Secure Face Detection Lock/Unlock UI (No dlib)

A clean OpenCV-only desktop UI for **face-based lock/unlock** with secure encrypted storage.

## What improved
- Better visual UI with glass-style top/bottom info panels.
- More user-friendly enrollment feedback.
- Better recognition stability and accuracy using:
  - image preprocessing (equalization + denoise),
  - stricter detection params,
  - temporal stability filter (`--stable-frames`) before unlocking,
  - blur check during enrollment.

## Why this avoids your old install issue
Your previous failure was because `face-recognition` needs `dlib`, and `dlib` often fails to compile on Windows without full Visual C++ setup.  
This project uses only:
- `opencv-contrib-python`
- `numpy`
- `cryptography`

## Features
- Real-time lock/unlock face UI.
- Person-specific welcome messages.
- Add new face/person with webcam (`enroll` mode).
- Encrypted storage for face samples and audit logs.

## Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Enroll (required before unlock)

```bash
python app.py enroll --name "alice" --samples 12
```

Enrollment tips:
- Keep exactly one face visible.
- Press `s` to capture.
- Slightly vary angle/expression for each sample.
- Avoid blurry frames.

## Run

```bash
python app.py --camera-index 0 --threshold 52 --stable-frames 4 --cooldown 2.0
```

- Lower `--threshold` = stricter recognition.
- Higher `--stable-frames` = fewer false unlocks.

## Secure files
Generated in `secure_data/` by default:
- `face_store.key`
- `face_samples.enc`
- `recognition_audit.log.enc`

## Personal greetings
Update `WELCOME_MESSAGES` in `app.py`.
