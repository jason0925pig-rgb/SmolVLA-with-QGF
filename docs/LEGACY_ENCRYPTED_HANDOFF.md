# Legacy encrypted handoff

The repository originally contained only the encrypted July 2026 research
handoff (`SmolVLA_QGF_Handoff_2026-07-21.qgfpack`). It is retained for audit
and recovery, but the repository now also contains the maintained source,
tests, experiment tools, and Armstrong baseline runtime in readable form.

To decrypt the legacy archive, install `cryptography` and run:

```bash
python handoff_crypto.py unpack SmolVLA_QGF_Handoff_2026-07-21.qgfpack --output handoff
```

The password is intentionally not stored in Git.
