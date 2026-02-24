# Secure Face Detection Lock/Unlock UI

This app provides a webcam-based lock/unlock UI with **person-specific face recognition** and **encrypted data storage**.

## What is implemented
- Real-time face detection using OpenCV + `face_recognition`.
- Unlock only when the detected face matches an enrolled authorized person.
- Person-specific welcome text after successful unlock.
- **Enroll mode** to add a new face from webcam.
- Encrypted storage for:
  - known face encodings (`secure_data/known_faces.enc`)
  - detection audit log (`secure_data/recognition_audit.log.enc`)
- Encryption key file created with restricted permissions (`chmod 600`).

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Enroll a new face (required first)

```bash
python app.py enroll --name "alice"
```

During enrollment:
- keep exactly one face visible.
- press `s` to save that face.
- press `q` to cancel.

## Run lock/unlock UI

```bash
python app.py --camera-index 0 --threshold 0.45 --cooldown 2.0
```

Press `q` or `ESC` to quit.

## Optional config

```bash
python app.py \
  --secure-dir secure_data \
  --key-file secure_data/face_store.key \
  --camera-index 0
```

## Personalized greetings

Edit `WELCOME_MESSAGES` in `app.py`:

```python
WELCOME_MESSAGES = {
    "alice": "Welcome back, Alice 👋",
    "bob": "Hi Bob, access granted ✅",
}
```

Anyone not listed receives:

```text
Welcome, <Name>!
```
