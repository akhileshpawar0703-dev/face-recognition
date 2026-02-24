# Secure Face Detection Lock/Unlock UI (Android-Style)

This build includes Android-style enrollment, stronger unlock logic, **privacy minimization**, and **tamper/runtime safety** enhancements.

## New security/privacy additions

### 8) Privacy & data minimization
- No raw camera frames are persisted; only processed face templates are stored.
- Per-user sample cap: `--max-samples-per-person` (default `20`) keeps only recent samples.
- Right-to-delete identity:
  ```bash
  python app.py delete --name "akhilesh"
  ```
- Audit retention pruning on startup: `--audit-retention-days` (default `30`).

### 9) Tamper resistance / runtime safety
- Tamper-evident audit chain (`prev_hash` + `chain_hash`) for each encrypted audit record.
- Audit integrity verification command:
  ```bash
  python app.py verify-audit
  ```
- Runtime lockout/backoff after repeated failures (temporary lock window).

## Detection backend
- `--detector auto` (default): YuNet first, fallback to Haar
- `--detector yunet`
- `--detector haar`

YuNet model is auto-downloaded to `secure_data/models/` when available.

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
python app.py \
  --camera-index 0 \
  --threshold -1 \
  --stable-frames 2 \
  --detector auto \
  --max-samples-per-person 20 \
  --audit-retention-days 30
```

## Tuning if still failing
1. Try `--detector yunet`.
2. If YuNet model download is blocked, use `--detector haar`.
3. If it still shows `unknown (your_name)` with high score, try `--threshold 95` or `--threshold 115`.
4. Re-enroll in brighter frontal lighting.

## Secure data files
Stored in `secure_data/`:
- `face_store.key`
- `face_samples.enc`
- `recognition_audit.log.enc`
- `audit_chain.state`
- `models/face_detection_yunet_2023mar.onnx` (if YuNet used)
