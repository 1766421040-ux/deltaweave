"""Restore the explicit public release bundle with stdlib-only verification.

The manifest must come from the trusted repository. Hashes bind bytes to that
manifest; they are not a replacement for authenticating the manifest itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
import urllib.parse
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ASSET_NAME = "deltaweave-artifacts-v0.1.0.zip"
EXPECTED_URL = "https://github.com/1766421040-ux/deltaweave/releases/download/v0.1.0/" + ASSET_NAME
ADAPTER_NAMES = ("adapter_config.json", "adapter_model.safetensors", "training_manifest.json")
DATA_NAMES = ("train_qa.jsonl", "train_behavior.jsonl", "dev.jsonl", "test.jsonl", "future.jsonl")
ALLOWED_MEMBERS = tuple(sorted(
    [f"artifacts/{route}/{name}" for route in ("qa", "behavior") for name in ADAPTER_NAMES]
    + ["artifacts/manifest.json", "data/learning/manifest.json"]
    + [f"data/learning/{name}" for name in DATA_NAMES]))
MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
MAX_MEMBER_BYTES = 64 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 128 * 1024 * 1024
CHUNK = 256 * 1024


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate manifest JSON key")
        result[key] = value
    return result


def read_manifest(path: Path):
    # The manifest is small, unlike the binary payload. Bound it before parsing.
    if path.stat().st_size > 1024 * 1024:
        raise ValueError("Release manifest is unexpectedly large")
    result = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    validate_manifest(result)
    return result


def _positive_size(value, maximum):
    return isinstance(value, int) and not isinstance(value, bool) and 0 < value <= maximum


def validate_manifest(manifest):
    if manifest.get("schema_version") != 1 or manifest.get("release_tag") != "v0.1.0":
        raise ValueError("Unsupported release manifest version/tag")
    if manifest.get("publication_status") not in {"pending", "published", "verified"}:
        raise ValueError("Unknown publication status")
    archive = manifest.get("archive", {})
    if archive.get("filename") != ASSET_NAME or archive.get("url") != EXPECTED_URL:
        raise ValueError("Unexpected release asset name or URL")
    if not _positive_size(archive.get("bytes"), MAX_ARCHIVE_BYTES):
        raise ValueError("Invalid/oversize archive length")
    if not isinstance(archive.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", archive["sha256"]):
        raise ValueError("Invalid archive SHA256")
    entries = manifest.get("files", {})
    if not isinstance(entries, dict) or set(entries) != set(ALLOWED_MEMBERS):
        raise ValueError("Manifest must contain exactly the explicit 13-file allowlist")
    total = 0
    for name, entry in entries.items():
        if not isinstance(entry, dict) or set(entry) != {"sha256", "bytes"}:
            raise ValueError("Unexpected file-manifest fields")
        if not _positive_size(entry.get("bytes"), MAX_MEMBER_BYTES):
            raise ValueError(f"Invalid/oversize uncompressed file length: {name}")
        if not isinstance(entry.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
            raise ValueError(f"Invalid file SHA256: {name}")
        total += entry["bytes"]
    if total > MAX_UNCOMPRESSED_BYTES:
        raise ValueError("Total uncompressed size exceeds the release limit")


def _member_hash(stream, expected_size, sink=None):
    digest, seen = hashlib.sha256(), 0
    while True:
        block = stream.read(min(CHUNK, expected_size - seen + 1))
        if not block:
            break
        seen += len(block)
        if seen > expected_size:
            raise ValueError("Decompression exceeds the declared member size")
        digest.update(block)
        if sink is not None:
            sink.write(block)
    if seen != expected_size:
        raise ValueError("Decompressed member length mismatch")
    return digest.hexdigest()


def verify_archive(archive_path: Path, manifest):
    """Check archive, metadata and every decompressed byte without extraction."""
    validate_manifest(manifest)
    archive_path = Path(archive_path)
    if archive_path.is_symlink() or not archive_path.is_file():
        raise ValueError("Archive must be an existing regular file, not a symlink")
    archive = manifest["archive"]
    if archive_path.stat().st_size != archive["bytes"] or file_sha256(archive_path) != archive["sha256"]:
        raise ValueError("Archive SHA256/size mismatch")
    with zipfile.ZipFile(archive_path) as bundle:
        infos = bundle.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate ZIP member name")
        if set(names) != set(ALLOWED_MEMBERS) or len(names) != len(ALLOWED_MEMBERS):
            raise ValueError("ZIP members differ from the exact release allowlist")
        for info in infos:
            name = info.filename
            parsed = PurePosixPath(name)
            if (name != info.orig_filename or "\\" in name or "\x00" in name or parsed.is_absolute()
                    or any(part in {".", "..", ""} for part in parsed.parts) or str(parsed) != name):
                raise ValueError("Unsafe ZIP member path")
            kind = stat.S_IFMT(info.external_attr >> 16)
            if info.is_dir() or kind not in {0, stat.S_IFREG}:
                raise ValueError("ZIP directory/symlink/special-file member is forbidden")
            if info.flag_bits & 1 or info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
                raise ValueError("Encrypted or unsupported ZIP member")
            expected = manifest["files"][name]
            if info.file_size != expected["bytes"] or info.compress_size > archive["bytes"]:
                raise ValueError("ZIP declared member length mismatch")
            with bundle.open(info) as stream:
                if _member_hash(stream, expected["bytes"]) != expected["sha256"]:
                    raise ValueError(f"ZIP member SHA256 mismatch: {name}")
    return {"members": len(ALLOWED_MEMBERS), "uncompressed_bytes": sum(x["bytes"] for x in manifest["files"].values())}


def _check_destination(destination: Path, manifest):
    existing, missing = [], []
    for name in ALLOWED_MEMBERS:
        parts = PurePosixPath(name).parts
        cursor = destination
        for index, part in enumerate(parts):
            cursor = cursor / part
            if cursor.is_symlink():
                raise ValueError(f"Destination contains a symlink: {name}")
            if cursor.exists():
                if index < len(parts) - 1:
                    if not cursor.is_dir():
                        raise ValueError(f"Destination parent is not a directory: {name}")
                elif not cursor.is_file():
                    raise ValueError(f"Destination file is not regular: {name}")
        path = destination / name
        expected = manifest["files"][name]
        if path.exists():
            if path.stat().st_size != expected["bytes"] or file_sha256(path) != expected["sha256"]:
                raise FileExistsError(f"Conflicting destination file; refusing overwrite: {name}")
            existing.append(name)
        else:
            missing.append(name)
    return existing, missing


def _open_parent_fd(root_fd, relative_parts):
    """Walk/create directories using no-follow descriptors (macOS/Linux)."""
    current = os.dup(root_fd)
    try:
        for part in relative_parts:
            try:
                os.mkdir(part, mode=0o755, dir_fd=current)
            except FileExistsError:
                pass
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=current)
            os.close(current)
            current = child
        return current
    except BaseException:
        os.close(current)
        raise


def restore_archive(archive_path: Path, manifest, destination: Path):
    verification = verify_archive(archive_path, manifest)
    destination = Path(destination)
    if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
        raise ValueError("Destination must be a directory, not a symlink/file")
    destination = destination.resolve()
    existing, missing = _check_destination(destination, manifest)
    if not missing:
        return {"status": "verified", "restored": 0, "skipped_matching": len(existing), **verification}
    # No destination mutation precedes full archive and conflict verification.
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".deltaweave-restore-", dir=destination) as temporary:
        stage = Path(temporary)
        with zipfile.ZipFile(archive_path) as bundle:
            for index, name in enumerate(missing):
                expected = manifest["files"][name]
                with bundle.open(name) as stream, (stage / str(index)).open("xb") as sink:
                    if _member_hash(stream, expected["bytes"], sink) != expected["sha256"]:
                        raise ValueError("Archive changed during staging")
                os.chmod(stage / str(index), 0o644)
        # Recheck all conflicts after staging, before installing the first file.
        _check_destination(destination, manifest)
        if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
            raise RuntimeError("Safe restoration requires macOS/Linux no-follow directory support")
        root_fd = os.open(destination, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            for index, name in enumerate(missing):
                parts = PurePosixPath(name).parts
                parent_fd = _open_parent_fd(root_fd, parts[:-1])
                try:
                    # A hard link on the same filesystem is atomic and never replaces
                    # an existing path. O_NOFOLLOW above protects parent traversal.
                    os.link(stage / str(index), parts[-1], dst_dir_fd=parent_fd, follow_symlinks=False)
                finally:
                    os.close(parent_fd)
        finally:
            os.close(root_fd)
    return {"status": "verified", "restored": len(missing), "skipped_matching": len(existing), **verification}


def _https_url(url):
    parsed = urllib.parse.urlsplit(url)
    allowed = {"github.com", "release-assets.githubusercontent.com", "objects.githubusercontent.com"}
    if (parsed.scheme != "https" or parsed.hostname not in allowed or parsed.username is not None
            or parsed.password is not None or parsed.port not in {None, 443}):
        raise ValueError("Release downloads require HTTPS on GitHub's asset hosts")


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _https_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download_archive(manifest, cache_dir: Path):
    validate_manifest(manifest)
    spec = manifest["archive"]
    _https_url(spec["url"])
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    final = cache_dir / (spec["sha256"] + ".zip")
    if final.exists() or final.is_symlink():
        if final.is_symlink() or not final.is_file() or final.stat().st_size != spec["bytes"] or file_sha256(final) != spec["sha256"]:
            raise ValueError("Conflicting cached archive; refusing replacement")
        return final
    opener = urllib.request.build_opener(SafeRedirect())
    request = urllib.request.Request(spec["url"], headers={"User-Agent": "DeltaWeave-release-fetch/1", "Accept-Encoding": "identity"})
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix="download-", suffix=".partial", dir=cache_dir, delete=False) as sink:
            temporary = Path(sink.name)
            with opener.open(request, timeout=60) as response:
                _https_url(response.geturl())
                declared = response.headers.get("Content-Length")
                if declared is not None and int(declared) != spec["bytes"]:
                    raise ValueError("HTTP Content-Length differs from release manifest")
                sha = _member_hash(response, spec["bytes"], sink)
            sink.flush()
            os.fsync(sink.fileno())
        if sha != spec["sha256"]:
            raise ValueError("Downloaded archive SHA256 mismatch")
        os.link(temporary, final)  # Refuse any competing cache entry.
        return final
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "artifacts" / "release.json")
    parser.add_argument("--destination", type=Path, default=ROOT)
    parser.add_argument("--archive", type=Path, help="Offline ZIP; the identical checks still apply")
    args = parser.parse_args()
    manifest = read_manifest(args.manifest)
    archive = args.archive if args.archive is not None else download_archive(manifest, ROOT / ".cache" / "releases")
    result = restore_archive(archive, manifest, args.destination)
    result["declared_publication_status"] = manifest["publication_status"]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
