# Secure Face Detection Lock/Unlock UI (Detection-Fixed)

This update specifically targets your latest issue: **face not being detected reliably**.

## What changed for detection reliability
- Added a new detector layer with selectable backend:
  - `--detector auto` (default): tries **YuNet** first, then falls back to Haar
  - `--detector yunet`: force YuNet
  - `--detector haar`: force Haar
- YuNet model auto-downloads once into `secure_data/models/`.
- Enrollment now uses the same detector backend as runtime.
- Enrollment grid checks were relaxed to avoid rejecting valid face captures.
- Stable unlock default lowered to `--stable-frames 2` to avoid over-strict gating.

## Face-grid enrollment (requested)
- Ring is removed.
- 3x3 face grid guide + progress bar.
- Auto capture + manual capture (`s`).

## Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Enroll

```bash
python app.py enroll --name "akhilesh" --samples 12 --detector auto
```

## Run

```bash
python app.py --camera-index 0 --threshold -1 --stable-frames 2 --detector auto
```

## Tuning if still not detecting
1. Try `--detector yunet` first.
2. If network blocks model download, use `--detector haar`.
3. Increase threshold manually: `--threshold 90` or `--threshold 110`.
4. Ensure bright frontal lighting and re-enroll.

## Secure data
Stored in `secure_data/`:
- `face_store.key`
- `face_samples.enc`
- `recognition_audit.log.enc`
- `models/face_detection_yunet_2023mar.onnx` (if YuNet used)
