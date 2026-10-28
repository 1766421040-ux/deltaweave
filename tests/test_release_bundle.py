"""Release transport fixtures contain no trained weights or measured results."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fetch_release as fetch
import package_release as package


def fixture(root):
    for name in fetch.ALLOWED_MEMBERS:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("SYNTHETIC TRANSPORT FIXTURE ONLY " + name + "\n").encode())
    adapters = {"status": "exported_local_artifacts", "runs": {}}
    for route in ("qa", "behavior"):
        adapters["runs"][route] = {"adapter_files": {
            name: {"sha256": fetch.file_sha256(root / "artifacts" / route / name),
                   "bytes": (root / "artifacts" / route / name).stat().st_size}
            for name in fetch.ADAPTER_NAMES}}
    (root / "artifacts/manifest.json").write_text(json.dumps(adapters))
    data = {"files": {name: {"sha256": fetch.file_sha256(root / "data/learning" / name)} for name in fetch.DATA_NAMES}}
    (root / "data/learning/manifest.json").write_text(json.dumps(data))
    archive = root / "dist" / fetch.ASSET_NAME
    manifest = package.package_release(root, archive)
    return archive, manifest


def mutated_zip(source, destination, manifest, transform):
    with zipfile.ZipFile(source) as old:
        entries = [(deepcopy(info), old.read(info)) for info in old.infolist()]
    entries = transform(entries)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(destination, "w") as output:
            for info, content in entries:
                output.writestr(info, content)
    changed = deepcopy(manifest)
    changed["archive"]["sha256"] = fetch.file_sha256(destination)
    changed["archive"]["bytes"] = destination.stat().st_size
    return changed


class ReleaseBundleTests(unittest.TestCase):
    def test_deterministic_packaging_exact_members_and_immutable_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            a, ma = fixture(root / "a")
            b, mb = fixture(root / "b")
            self.assertEqual(a.read_bytes(), b.read_bytes())
            self.assertEqual(ma, mb)
            self.assertEqual(ma["publication_status"], "pending")
            self.assertEqual(ma["archive"]["url"], fetch.EXPECTED_URL)
            with zipfile.ZipFile(a) as archive:
                self.assertEqual(archive.namelist(), list(fetch.ALLOWED_MEMBERS))
                self.assertTrue(all(i.date_time == package.FIXED_TIMESTAMP for i in archive.infolist()))
                self.assertNotIn("artifacts/release.json", archive.namelist())
            with self.assertRaises(FileExistsError):
                package.package_release(root / "a", a)

    def test_restore_then_skip_matching_without_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive, manifest = fixture(root / "source")
            target = root / "restored"
            result = fetch.restore_archive(archive, manifest, target)
            self.assertEqual(result["restored"], 13)
            before = {}
            for name, expected in manifest["files"].items():
                self.assertEqual(fetch.file_sha256(target / name), expected["sha256"])
                before[name] = (target / name).stat().st_mtime_ns
            again = fetch.restore_archive(archive, manifest, target)
            self.assertEqual(again["restored"], 0)
            self.assertEqual(again["skipped_matching"], 13)
            self.assertEqual(before, {name: (target / name).stat().st_mtime_ns for name in before})

    def test_conflicting_file_rejected_before_other_extraction(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive, manifest = fixture(root / "source")
            target = root / "destination"
            conflicting = target / "data/learning/train_qa.jsonl"
            conflicting.parent.mkdir(parents=True)
            conflicting.write_text("KEEP EXISTING USER CONTENT")
            with self.assertRaises(FileExistsError):
                fetch.restore_archive(archive, manifest, target)
            self.assertFalse((target / "artifacts").exists())
            self.assertEqual(conflicting.read_text(), "KEEP EXISTING USER CONTENT")

    def test_bad_archive_hash_does_not_create_destination(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive, manifest = fixture(root / "source")
            archive.write_bytes(archive.read_bytes() + b"tamper")
            with self.assertRaisesRegex(ValueError, "SHA256/size"):
                fetch.restore_archive(archive, manifest, root / "destination")
            self.assertFalse((root / "destination").exists())

    def test_unsafe_zip_members_rejected_even_with_matching_archive_hash(self):
        def traversal(entries):
            entries[0][0].filename = "../outside"
            return entries
        def symlink(entries):
            entries[0][0].external_attr = (stat.S_IFLNK | 0o777) << 16
            return entries
        def duplicate(entries):
            return entries + [entries[0]]
        def extra(entries):
            return entries + [(zipfile.ZipInfo("unexpected.txt"), b"extra")]
        def changed_content(entries):
            return entries[:-1] + [(entries[-1][0], entries[-1][1] + b"wrong")]
        def wrong_hash_same_size(entries):
            return entries[:-1] + [(entries[-1][0], b"!" + entries[-1][1][1:])]
        for transform in (traversal, symlink, duplicate, extra, changed_content, wrong_hash_same_size):
            with self.subTest(attack=transform.__name__), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                archive, manifest = fixture(root / "source")
                attack = root / "attack.zip"
                updated = mutated_zip(archive, attack, manifest, transform)
                with self.assertRaises((ValueError, zipfile.BadZipFile)):
                    fetch.restore_archive(attack, updated, root / "destination")
                self.assertFalse((root / "destination").exists())

    def test_manifest_oversize_and_decompression_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive, manifest = fixture(root / "source")
            wrong = deepcopy(manifest)
            name = fetch.ALLOWED_MEMBERS[0]
            wrong["files"][name]["bytes"] = fetch.MAX_MEMBER_BYTES + 1
            with self.assertRaisesRegex(ValueError, "oversize"):
                fetch.restore_archive(archive, wrong, root / "destination")
            wrong = deepcopy(manifest)
            wrong["files"][name]["bytes"] = 1
            with self.assertRaisesRegex(ValueError, "length"):
                fetch.restore_archive(archive, wrong, root / "destination")
            with self.assertRaisesRegex(ValueError, "Decompression"):
                fetch._member_hash(io.BytesIO(b"12345"), 3)

    def test_destination_symlink_cannot_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive, manifest = fixture(root / "source")
            destination, outside = root / "destination", root / "outside"
            destination.mkdir()
            outside.mkdir()
            (destination / "artifacts").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                fetch.restore_archive(archive, manifest, destination)
            self.assertEqual(list(outside.iterdir()), [])

    def test_https_host_restrictions_and_verified_download_cache(self):
        for url in ("http://github.com/asset", "https://example.com/asset", "https://user@github.com/asset"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                fetch._https_url(url)
        fetch._https_url("https://release-assets.githubusercontent.com/file?signature=example")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive, manifest = fixture(root / "source")
            payload = archive.read_bytes()
            class Response(io.BytesIO):
                headers = {"Content-Length": str(len(payload))}
                def geturl(self):
                    return fetch.EXPECTED_URL
            class Opener:
                calls = 0
                def open(self, request, timeout):
                    self.calls += 1
                    return Response(payload)
            opener = Opener()
            with patch.object(fetch.urllib.request, "build_opener", return_value=opener):
                cached = fetch.download_archive(manifest, root / "cache")
                second = fetch.download_archive(manifest, root / "cache")
            self.assertEqual(opener.calls, 1)
            self.assertEqual(cached, second)
            self.assertEqual(hashlib.sha256(cached.read_bytes()).hexdigest(), manifest["archive"]["sha256"])


if __name__ == "__main__":
    unittest.main()
