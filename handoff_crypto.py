#!/usr/bin/env python3
"""Pack and unpack the encrypted SmolVLA-QGF handoff container.

The password is read with getpass and is never accepted as a command-line
argument. The encrypted payload is a gzip-compressed tar archive protected by
AES-256-GCM. Scrypt derives the encryption key from the passphrase.
"""

from __future__ import annotations

import argparse
import getpass
import io
import os
import shutil
import struct
import tarfile
from pathlib import Path

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt


MAGIC = b"QGFPACK1"
VERSION = 1
SALT_SIZE = 16
NONCE_SIZE = 12
HEADER = struct.Struct(">8sB16s12s")


def _derive_key(password: str, salt: bytes) -> bytes:
    if not password:
        raise ValueError("Password must not be empty.")
    kdf = Scrypt(salt=salt, length=32, n=2**15, r=8, p=1)
    return kdf.derive(password.encode("utf-8"))


def _tar_directory(source: Path) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz", format=tarfile.PAX_FORMAT) as archive:
        for path in sorted(source.rglob("*")):
            archive.add(path, arcname=path.relative_to(source), recursive=False)
    return buffer.getvalue()


def _safe_extract(archive: tarfile.TarFile, output: Path) -> None:
    root = output.resolve()
    for member in archive.getmembers():
        destination = (output / member.name).resolve()
        if destination != root and root not in destination.parents:
            raise RuntimeError(f"Unsafe archive path: {member.name}")
        if member.issym() or member.islnk():
            raise RuntimeError(f"Links are not allowed in the handoff: {member.name}")
    archive.extractall(output, filter="data")


def pack(source: Path, output: Path, password: str | None = None) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {source}")
    if password is None:
        password = getpass.getpass("Package password: ")
        confirmation = getpass.getpass("Confirm password: ")
        if password != confirmation:
            raise ValueError("Passwords do not match.")

    salt = os.urandom(SALT_SIZE)
    nonce = os.urandom(NONCE_SIZE)
    key = _derive_key(password, salt)
    header = HEADER.pack(MAGIC, VERSION, salt, nonce)
    ciphertext = AESGCM(key).encrypt(nonce, _tar_directory(source), header)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(header + ciphertext)
    temporary.replace(output)
    print(f"Encrypted package written to {output}")


def unpack(package: Path, output: Path, password: str | None = None) -> None:
    payload = package.read_bytes()
    if len(payload) <= HEADER.size:
        raise ValueError("Package is truncated.")
    magic, version, salt, nonce = HEADER.unpack(payload[: HEADER.size])
    if magic != MAGIC or version != VERSION:
        raise ValueError("Not a supported QGF handoff package.")

    if password is None:
        password = getpass.getpass("Package password: ")
    key = _derive_key(password, salt)
    try:
        plaintext = AESGCM(key).decrypt(nonce, payload[HEADER.size :], payload[: HEADER.size])
    except InvalidTag as error:
        raise SystemExit("Wrong password or damaged package; nothing was extracted.") from error

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    temporary = output.with_name(output.name + ".extracting")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        with tarfile.open(fileobj=io.BytesIO(plaintext), mode="r:gz") as archive:
            _safe_extract(archive, temporary)
        if output.exists():
            output.rmdir()
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(f"Handoff extracted to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    pack_parser = subparsers.add_parser("pack", help="Encrypt a directory.")
    pack_parser.add_argument("source", type=Path)
    pack_parser.add_argument("package", type=Path)

    unpack_parser = subparsers.add_parser("unpack", help="Decrypt a package.")
    unpack_parser.add_argument("package", type=Path)
    unpack_parser.add_argument("--output", type=Path, default=Path("handoff"))

    args = parser.parse_args()
    if args.command == "pack":
        pack(args.source, args.package)
    else:
        unpack(args.package, args.output)


if __name__ == "__main__":
    main()
