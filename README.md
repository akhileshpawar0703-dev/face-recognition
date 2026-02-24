# Secure Face Detection Lock/Unlock UI (Enhanced)

This version improves recognition reliability and provides an Android-like guided face enrollment experience.

## Key fixes for your issue (samples taken but face not recognized)
- Added **auto threshold calibration** (`--threshold <= 0`) from enrolled samples.
- Improved preprocessing (CLAHE + denoise) and detection robustness.
- Added relaxed eye-check logic to avoid false unknown labels in low light.
- Added stability voting across frames before unlock.
- Added debug info on screen: `thr=<threshold> best=<score>` so you can tune quickly.

## Android-style enrollment UI
- Circular face scan ring and progress arc.
- Guided prompts (left/right/up/down/near/far).
- Auto capture on good frame quality.
- Optional manual capture key `s`.
- Face quality checks: single face, center alignment, distance/size, blur.

## Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Enroll first

```bash
python app.py enroll --name "akhilesh" --samples 12
```

## Run lock/unlock

```bash
python app.py --camera-index 0 --threshold -1 --stable-frames 4 --cooldown 2.0
```

- `--threshold -1` (or any `<=0`) enables auto calibration (recommended).
- If still too strict, try manual threshold values: `70`, `80`, `90`.
- If false unlocks happen, increase `--stable-frames`.

## During runtime
- `e` inside unlock window -> close and start new enrollment prompt.
- `q` or `ESC` -> quit.

## Secure data
Stored in `secure_data/`:
- `face_store.key`
- `face_samples.enc`
- `recognition_audit.log.enc`

All stored samples/audit entries are encrypted.
