from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.verify_gdc_download import load_manifest, verify_download


ROOT = Path(__file__).resolve().parents[1]
VERIFIER = ROOT / "scripts/verify_gdc_download.py"
FILE_IDS = (
    "0019c951-16c5-48d0-85c8-58d96b12d330",
    "0022cd20-f64f-4773-b9ff-a3de0b71b259",
)


class GdcDownloadVerificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.manifest = self.root / "manifest.tsv"
        self.download_root = self.root / "downloads"
        self.payloads = (b"first locked GDC fixture\n", b"second fixture\n")
        self.names = ("first.star_counts.tsv", "second.star_counts.tsv")
        self.write_fixture()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def manifest_rows(self) -> list[dict[str, str]]:
        return [
            {
                "id": file_id,
                "filename": name,
                "md5": hashlib.md5(payload, usedforsecurity=False).hexdigest(),
                "size": str(len(payload)),
                "state": "released",
            }
            for file_id, name, payload in zip(
                FILE_IDS, self.names, self.payloads, strict=True
            )
        ]

    def write_manifest(self, rows: list[dict[str, str]]) -> None:
        with self.manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("id", "filename", "md5", "size", "state"),
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)

    def write_fixture(self) -> None:
        self.write_manifest(self.manifest_rows())
        for file_id, name, payload in zip(
            FILE_IDS, self.names, self.payloads, strict=True
        ):
            target = self.download_root / file_id / name
            target.parent.mkdir(parents=True)
            target.write_bytes(payload)

    def write_download_project(self) -> Path:
        project = self.root / "project"
        scripts = project / "scripts"
        manifest_dir = project / "data/manifests"
        download_root = project / "data/raw/gdc/star_counts"
        scripts.mkdir(parents=True)
        manifest_dir.mkdir(parents=True)
        shutil.copy2(ROOT / "scripts/download_gdc_star_counts.sh", scripts)
        shutil.copy2(VERIFIER, scripts)
        shutil.copy2(self.manifest, manifest_dir / self.manifest.name)
        (manifest_dir / self.manifest.name).rename(
            manifest_dir / "gdc_manifest_tcga_brca_star_counts.tsv"
        )
        for file_id, name, payload in zip(
            FILE_IDS, self.names, self.payloads, strict=True
        ):
            target = download_root / file_id / name
            target.parent.mkdir(parents=True)
            target.write_bytes(payload)
        return project

    def test_complete_download_passes_and_writes_atomic_reports(self) -> None:
        report = self.root / "verification.tsv"
        summary_path = self.root / "summary.json"
        rows, summary = verify_download(
            self.manifest,
            self.download_root,
            report_path=report,
            summary_path=summary_path,
        )
        written_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertTrue(summary["all_files_verified"])
        self.assertEqual(summary["verified_file_count"], 2)
        self.assertEqual(summary["failed_file_count"], 0)
        self.assertEqual(summary["unexpected_file_count"], 0)
        self.assertEqual(summary["partial_file_count"], 0)
        self.assertEqual([row["status"] for row in rows], ["OK", "OK"])
        self.assertEqual(written_summary["platform"], summary["platform"])
        self.assertFalse(list(self.root.glob(".*.part")))

    def test_corrupt_and_missing_files_fail(self) -> None:
        (self.download_root / FILE_IDS[0] / self.names[0]).write_bytes(b"corrupt")
        (self.download_root / FILE_IDS[1] / self.names[1]).unlink()
        rows, summary = verify_download(self.manifest, self.download_root)
        self.assertFalse(summary["all_files_verified"])
        self.assertEqual(summary["failed_file_count"], 2)
        self.assertEqual([row["status"] for row in rows], ["FAILED", "MISSING"])

    def test_unexpected_and_partial_files_fail_strict_validation(self) -> None:
        (self.download_root / "unexpected.txt").write_text(
            "unexpected", encoding="utf-8"
        )
        partial = self.download_root / FILE_IDS[0] / f"{self.names[0]}.part"
        partial.write_bytes(b"partial")
        _, summary = verify_download(self.manifest, self.download_root)
        self.assertFalse(summary["all_files_verified"])
        self.assertEqual(summary["unexpected_files"], ["unexpected.txt"])
        self.assertEqual(
            summary["partial_files"],
            [f"{FILE_IDS[0]}/{self.names[0]}.part"],
        )

    def test_manifest_rejects_path_traversal_before_file_access(self) -> None:
        rows = self.manifest_rows()
        rows[0]["filename"] = "../outside.tsv"
        self.write_manifest(rows)
        with self.assertRaisesRegex(ValueError, "Path separators"):
            load_manifest(self.manifest)

    def test_manifest_rejects_duplicate_ids_and_invalid_fields(self) -> None:
        rows = self.manifest_rows()
        rows[1]["id"] = rows[0]["id"]
        self.write_manifest(rows)
        with self.assertRaisesRegex(ValueError, "Duplicate GDC file id"):
            load_manifest(self.manifest)

        rows = self.manifest_rows()
        rows[0]["md5"] = "not-an-md5"
        self.write_manifest(rows)
        with self.assertRaisesRegex(ValueError, "Invalid MD5"):
            load_manifest(self.manifest)

    def test_cli_uses_no_external_checksum_stat_or_json_commands(self) -> None:
        environment = os.environ.copy()
        environment["PATH"] = str(self.root / "empty-path")
        completed = subprocess.run(
            [
                sys.executable,
                str(VERIFIER),
                "verify",
                "--manifest",
                str(self.manifest),
                "--download-root",
                str(self.download_root),
                "--no-write-reports",
                "--quiet",
            ],
            text=True,
            capture_output=True,
            env=environment,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout, "")
        self.assertEqual(completed.stderr, "")

    def test_downloader_skip_path_runs_on_platform_shell(self) -> None:
        project = self.write_download_project()
        environment = os.environ.copy()
        environment["PYTHON"] = sys.executable
        environment["GDC_DOWNLOAD_WORKERS"] = "2"
        completed = subprocess.run(
            ["bash", "scripts/download_gdc_star_counts.sh"],
            cwd=project,
            text=True,
            capture_output=True,
            env=environment,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn('"all_files_verified": true', completed.stdout)
        log = (project / "logs/gdc_star_counts_download.log").read_text(
            encoding="utf-8"
        )
        self.assertEqual(log.count("SKIP\t"), 2)

    def test_downloader_rejects_invalid_worker_count(self) -> None:
        project = self.write_download_project()
        environment = os.environ.copy()
        environment["PYTHON"] = sys.executable
        environment["GDC_DOWNLOAD_WORKERS"] = "0"
        completed = subprocess.run(
            ["bash", "scripts/download_gdc_star_counts.sh"],
            cwd=project,
            text=True,
            capture_output=True,
            env=environment,
            check=False,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("positive integer", completed.stderr)


if __name__ == "__main__":
    unittest.main()
