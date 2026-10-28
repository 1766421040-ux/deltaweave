"""Create an explicit, deterministic-metadata release ZIP; no model execution."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import stat
import zipfile

import fetch_release as bundle

ROOT = Path(__file__).resolve().parents[1]
FIXED_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def package_release(root: Path, archive_path: Path):
    root, archive_path = Path(root), Path(archive_path)
    manifest_path = root / "artifacts" / "release.json"
    if archive_path.exists() or archive_path.is_symlink() or manifest_path.exists() or manifest_path.is_symlink():
        raise FileExistsError("Archive and release manifest must both be new")
    if archive_path.name != bundle.ASSET_NAME:
        raise ValueError("Archive filename must match the registered release asset URL")
    artifact_manifest = json.loads((root / "artifacts" / "manifest.json").read_text(), object_pairs_hook=bundle.unique_object)
    data_manifest = json.loads((root / "data" / "learning" / "manifest.json").read_text(), object_pairs_hook=bundle.unique_object)
    if artifact_manifest.get("status") != "exported_local_artifacts" or set(artifact_manifest.get("runs", {})) != {"qa", "behavior"}:
        raise ValueError("Both routes must have completed artifact export")
    if set(data_manifest.get("files", {})) != set(bundle.DATA_NAMES):
        raise ValueError("Learning data manifest does not match the registered five files")
    files = {}
    for name in bundle.ALLOWED_MEMBERS:
        path = root
        for part in Path(name).parts:
            path = path / part
            if path.is_symlink():
                raise ValueError(f"Release source contains a symlink: {name}")
        if not path.is_file():
            raise FileNotFoundError(f"Required release file is missing: {name}")
        files[name] = {"sha256": bundle.file_sha256(path), "bytes": path.stat().st_size}
        if not bundle._positive_size(files[name]["bytes"], bundle.MAX_MEMBER_BYTES):
            raise ValueError("Release source exceeds member size limit")
    for route in ("qa", "behavior"):
        expected = artifact_manifest["runs"][route]["adapter_files"]
        if set(expected) != set(bundle.ADAPTER_NAMES):
            raise ValueError("Adapter export manifest file set mismatch")
        for name in bundle.ADAPTER_NAMES:
            if files[f"artifacts/{route}/{name}"] != expected[name]:
                raise ValueError("Adapter bytes no longer match exported evidence")
    for name in bundle.DATA_NAMES:
        if files[f"data/learning/{name}"]["sha256"] != data_manifest["files"][name]["sha256"]:
            raise ValueError("Learning bytes no longer match data manifest")
    if sum(entry["bytes"] for entry in files.values()) > bundle.MAX_UNCOMPRESSED_BYTES:
        raise ValueError("Release exceeds total uncompressed size limit")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as output:
        for name in bundle.ALLOWED_MEMBERS:
            data = (root / name).read_bytes()
            if hashlib.sha256(data).hexdigest() != files[name]["sha256"] or len(data) != files[name]["bytes"]:
                raise ValueError("Release source changed while packaging")
            info = zipfile.ZipInfo(name, date_time=FIXED_TIMESTAMP)
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            info.internal_attr = 0
            info.extra = b""
            info.comment = b""
            output.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    manifest = {"schema_version": 1, "release_tag": "v0.1.0", "publication_status": "pending",
                "archive": {"filename": bundle.ASSET_NAME, "url": bundle.EXPECTED_URL,
                            "sha256": bundle.file_sha256(archive_path), "bytes": archive_path.stat().st_size},
                "files": files,
                "packaging": {"member_order": "lexicographic", "timestamp": list(FIXED_TIMESTAMP),
                              "compression": "deflate", "compression_level": 9,
                              "permissions": "regular files 0644", "comment": "",
                              "determinism_scope": "Fixed names, metadata and input bytes; compressed bytes may depend on zlib version."}}
    bundle.verify_archive(archive_path, manifest)
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / bundle.ASSET_NAME)
    parser.add_argument("--split-existing", action="store_true", help="Split an already verified ZIP into <=4 MiB assets; do not change its bytes")
    args = parser.parse_args()
    if args.split_existing:
        manifest_path = args.root / "artifacts" / "release.json"
        manifest = split_release(args.output, bundle.read_manifest(manifest_path))
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        manifest = package_release(args.root, args.output)
    print(json.dumps({"publication_status": manifest["publication_status"], "archive": manifest["archive"],
                      "files": len(manifest["files"])}, indent=2))


def split_release(archive_path, manifest, part_bytes=bundle.MAX_PART_BYTES):
    """Transport-only split; the original archive and all member hashes stay fixed."""
    bundle.verify_archive(archive_path, manifest)
    if not bundle._positive_size(part_bytes, bundle.MAX_PART_BYTES):
        raise ValueError("Invalid release part size")
    if (manifest["archive"]["bytes"] + part_bytes - 1) // part_bytes > bundle.MAX_PARTS:
        raise ValueError("Too many release parts")
    changed = deepcopy(manifest)
    parts = []
    with Path(archive_path).open("rb") as source:
        for index in range(1, bundle.MAX_PARTS + 1):
            block = source.read(part_bytes)
            if not block:
                break
            suffix = f".part{index:02d}"
            path = Path(archive_path).with_name(bundle.ASSET_NAME + suffix)
            spec = {"filename": path.name, "url": bundle.EXPECTED_URL + suffix,
                    "bytes": len(block), "sha256": hashlib.sha256(block).hexdigest()}
            if path.is_symlink() or (path.exists() and (not path.is_file() or path.stat().st_size != len(block) or bundle.file_sha256(path) != spec["sha256"])):
                raise FileExistsError("Conflicting release part; refusing replacement")
            if not path.exists():
                with path.open("xb") as output:
                    output.write(block)
            parts.append(spec)
    changed["archive"].update({"url": None, "parts": parts})
    bundle.validate_manifest(changed)
    return changed


if __name__ == "__main__":
    main()
