from __future__ import annotations

import hashlib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts import verify_source_artifacts


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class OptionalSourceVerificationTests(unittest.TestCase):
    def write_manifest(self, root: Path, files: dict[str, bytes]) -> Path:
        manifest = root / "optional_source.sha256"
        manifest.write_text(
            "".join(f"{digest(content)}  {name}\n" for name, content in files.items()),
            encoding="utf-8",
        )
        return manifest

    def test_all_optional_files_absent_can_be_skipped(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.write_manifest(
                root,
                {"optional/source.xlsx": b"xlsx", "optional/source.tsv": b"tsv"},
            )
            self.assertFalse(
                verify_source_artifacts.checksum_manifest_is_available(manifest)
            )

    def test_complete_optional_source_is_verified(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            files = {"optional/source.xlsx": b"xlsx", "optional/source.tsv": b"tsv"}
            manifest = self.write_manifest(root, files)
            for name, content in files.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            self.assertTrue(
                verify_source_artifacts.checksum_manifest_is_available(manifest)
            )
            self.assertEqual(
                verify_source_artifacts.verify_checksum_manifest(manifest), len(files)
            )

    def test_partially_restored_optional_source_fails(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            files = {"optional/source.xlsx": b"xlsx", "optional/source.tsv": b"tsv"}
            manifest = self.write_manifest(root, files)
            path = root / "optional/source.xlsx"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(files["optional/source.xlsx"])
            with self.assertRaisesRegex(RuntimeError, "partially available"):
                verify_source_artifacts.checksum_manifest_is_available(manifest)


if __name__ == "__main__":
    unittest.main()
