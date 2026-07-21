# Security Policy

The encrypted handoff package is the only place where project-internal notes and
server paths should live. Credentials must never be committed, even inside the
encrypted package unless there is no safer transfer mechanism.

This snapshot deliberately excludes:

- SSH private keys and SSH login guides containing credentials
- sudo passwords, API tokens, cookies, and model-registry tokens
- other users' files, logs, commands, and process details
- large model checkpoints, rollout tensors, and raw videos

The package uses AES-256-GCM with a key derived from the passphrase by Scrypt.
AES-GCM authenticates the ciphertext: a wrong password or modified package fails
before extraction.

