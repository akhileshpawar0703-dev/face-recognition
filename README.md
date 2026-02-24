# Secure Face Detection Lock/Unlock UI (Enhanced)

This build focuses on two things you requested:
1) stronger recognition (fixes "samples captured but still unknown"),
2) **face grid** enrollment UI (instead of circular ring).

## What changed
- Reverted detection behavior closer to the previously working setup:
  - single-pass Haar detection (`scaleFactor=1.2`, `minNeighbors=6`).
- Improved recognition reliability:
  - auto threshold calibration from your enrolled samples when `--threshold <= 0`,
  - stronger preprocessing (CLAHE + slight blur),
  - clearer overlay showing `unknown (predicted_name)` and score.
- Replaced ring scanner with **face grid scanner**:
  - 3x3 face grid region,
  - overlap/size checks against grid,
  - blur quality check,
  - auto capture + manual capture (`s`).

## Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Enroll face first

```bash
python app.py enroll --name "akhilesh" --samples 12
```

## Run unlock UI

```bash
python app.py --camera-index 0 --threshold -1 --stable-frames 3 --cooldown 2.0
```

### Tuning
- Start with `--threshold -1` (auto threshold).
- If still not recognizing, try manual threshold: `80`, `90`, `100`.
- Increase `--stable-frames` for stricter unlock behavior.

## Runtime keys
- `e`: enroll new face.
- `q` / `ESC`: quit.

## Secure data
Stored under `secure_data/`:
- `face_store.key`
- `face_samples.enc`
- `recognition_audit.log.enc`

All face samples and audit records are encrypted.
