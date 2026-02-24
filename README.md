# Face Detection Lock/Unlock UI (Person Specific)

This project implements a webcam-based lock/unlock UI using **OpenCV + face_recognition**.

## Features
- Detects faces from live camera feed.
- Matches only against known/authorized people.
- Shows **LOCKED** when no authorized face is detected.
- Shows **UNLOCKED** state and personalized welcome message when a known face is matched.
- Supports different welcome messages for different people.

## Folder structure

```text
face-recognition/
├── app.py
├── requirements.txt
└── known_faces/
    ├── alice/
    │   ├── 1.jpg
    │   └── 2.jpg
    ├── bob/
    │   └── 1.jpg
    └── charlie/
        └── 1.jpg
```

Each subfolder name under `known_faces/` is treated as the person's name.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> Note: `face-recognition` depends on `dlib`. On Linux you may need build tools (`cmake`, `build-essential`, python headers).

## Run

```bash
python app.py --known-faces-dir known_faces --camera-index 0 --threshold 0.45 --cooldown 2.0
```

Press `q` or `ESC` to quit.

## Custom welcome messages

Edit `WELCOME_MESSAGES` in `app.py`:

```python
WELCOME_MESSAGES = {
    "alice": "Welcome back, Alice 👋",
    "bob": "Hi Bob, access granted ✅",
}
```

Any recognized person not listed there gets the default:

```text
Welcome, <Name>!
```
