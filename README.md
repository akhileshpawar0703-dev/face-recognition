# Secure Face Detection Lock/Unlock UI (Android-Style)

This build includes Android-style enrollment, stronger unlock logic, **privacy minimization**, **tamper/runtime safety**, and **face-protected PDF unlocking**.

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

## Face-locked PDF feature (new)

You can encrypt a PDF and require face authentication to unlock/open it.

### 1) Encrypt PDF (bind to enrolled owner)

```bash
python app.py encrypt-pdf --pdf "C:/docs/secret.pdf" --owner "akhilesh" --out "C:/docs/secret.facepdf"
```

### 2) Unlock PDF with face (camera auto-triggers)

```bash
python app.py unlock-pdf --file "C:/docs/secret.facepdf" --timeout 25
```

- This command opens the camera, performs face authentication, decrypts the PDF, and opens it with the **default OS PDF handler** (Chrome or any reader depending on your system).
- Optional:
  ```bash
  python app.py unlock-pdf --file "C:/docs/secret.facepdf" --out "C:/docs/decrypted_secret.pdf"
  ```

> Note: Directly intercepting *any random* double-click/open in third-party PDF readers is OS-level integration. This app provides a secure launcher flow (`unlock-pdf`) that triggers camera automatically before opening.

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

## Simple Desktop UI (new)

If you prefer not to type commands, run:

```bash
python ui.py
```

This opens a Tkinter-based control panel for:
- starting lock/unlock camera UI,
- enrolling users,
- encrypting/unlocking PDFs,
- deleting identities,
- verifying audit chain.

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
- `file_registry.enc`
- `models/face_detection_yunet_2023mar.onnx` (if YuNet used)
