"""Shared source identity and atomic, non-overwriting JSON publication."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

from .store import canonical


def snapshot_fingerprint(snapshot) -> str:
    """Hash finite source content, independent of row insertion order/revision."""
    source = {"tables": {name: sorted(rows, key=lambda row: canonical(row["id"]))
                         for name, rows in snapshot.tables.items()},
              "context": snapshot.context}
    return hashlib.sha256(canonical(source).encode("utf-8")).hexdigest()


def execution_binding(engine, snapshot) -> dict:
    """Capture declared code versions in one read; source snapshot is immutable."""
    versions = engine.registry_versions
    return {"snapshot_revision": snapshot.revision,
            "snapshot_sha256": snapshot_fingerprint(snapshot),
            "registry_versions": versions,
            "registry_fingerprint": hashlib.sha256(canonical(versions).encode("utf-8")).hexdigest()}


def atomic_write_new_json(path: Path, value) -> None:
    """Expose only a complete file; never replace an existing manifest.

    File data is fsynced before linking the temporary file to its final name.
    This is atomic visibility on the same filesystem, not distributed durability.
    """
    canonical(value)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp",
                                             dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
