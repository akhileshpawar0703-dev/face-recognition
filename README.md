# Secure Face Detection Lock/Unlock UI (No dlib)

This version uses **OpenCV only** for detection + recognition, so it avoids the `dlib` build failure you hit on Windows.

## Why this fixes your error
Your error came from `face-recognition -> dlib`, which needs a C++ toolchain/Visual Studio build setup on Windows.  
This project now uses:
- Haar Cascade for face detection
- LBPH recognizer (`cv2.face`) for person recognition

No `face-recognition` and no `dlib` dependency.

## Features
- Lock/Unlock UI from webcam.
- Unlock only for enrolled people.
- Personalized welcome messages per person.
- `enroll` command to add new people from webcam.
- Secure encrypted storage for:
  - enrolled face samples (`secure_data/face_samples.enc`)
  - recognition audit (`secure_data/recognition_audit.log.enc`)
- Encryption key generated automatically with restricted permission (`chmod 600` where supported).

## Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Enroll a person first

```bash
python app.py enroll --name "alice" --samples 8
```

- Keep exactly one face in frame.
- Press `s` to capture each sample.
- Press `q` to cancel.

## Run UI

```bash
python app.py --camera-index 0 --threshold 55 --cooldown 2.0
```

- LBPH threshold: lower is stricter (typical range `40-70`).
- Press `q` or `ESC` to quit.

## Personalized greetings

Edit `WELCOME_MESSAGES` in `app.py`.

## Notes
- `opencv-contrib-python` is required because LBPH is under `cv2.face`.
- If camera index `0` does not work, try `1` or `2`.
