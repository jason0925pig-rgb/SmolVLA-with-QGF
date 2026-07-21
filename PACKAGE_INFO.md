# Package Information

- Created: 2026-07-21
- Payload: `SmolVLA_QGF_Handoff_2026-07-21.qgfpack`
- Size: 3,085,079 bytes
- Encryption: AES-256-GCM
- Key derivation: Scrypt (`N=32768`, `r=8`, `p=1`, 16-byte random salt)
- Nonce: 12 random bytes
- Encrypted contents: gzip-compressed tar archive
- Files in payload: 395, plus the internal hash index
- Password storage: not stored in this repository

Validation performed before publishing:

1. A deliberately incorrect password was rejected before extraction.
2. The intended password extracted the package successfully.
3. All 395 files matched the internal SHA-256 index.
4. The payload was scanned for common private-key, GitHub token, Hugging Face
   token, OpenAI key, and plaintext project-password patterns.

