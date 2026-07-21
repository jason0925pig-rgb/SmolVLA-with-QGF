# SmolVLA with QGF - Encrypted Research Handoff

This repository stores an encrypted handoff package for the SmolVLA + Guided
Action Flow (QGF) research project. The private package contains research notes,
experiment results, server runbooks, and a source-code snapshot.

The repository is public, but the handoff payload is encrypted with AES-256-GCM.
The password is intentionally not stored in Git, in the README, or in the
decryption script.

## Decrypt

Requirements:

- Python 3.10 or newer
- `cryptography`

```bash
python -m pip install cryptography
python handoff_crypto.py unpack SmolVLA_QGF_Handoff_2026-07-21.qgfpack --output handoff
```

The script asks for the password interactively, so the password does not enter
the shell history. On success, begin with `handoff/docs/00_START_HERE.md`.

## Verify Before Decrypting

Windows PowerShell:

```powershell
Get-FileHash .\SmolVLA_QGF_Handoff_2026-07-21.qgfpack -Algorithm SHA256
```

Linux:

```bash
sha256sum SmolVLA_QGF_Handoff_2026-07-21.qgfpack
```

Compare the output with `SHA256SUMS.txt`.

## Security Notes

- Do not commit the extracted `handoff/` directory.
- Do not put SSH private keys, API tokens, Hugging Face tokens, or passwords in
  this repository.
- Transfer server credentials separately through a secure channel.
- Anyone who knows the package password can decrypt the payload. Rotate the
  password and replace the encrypted file if access must be revoked.

