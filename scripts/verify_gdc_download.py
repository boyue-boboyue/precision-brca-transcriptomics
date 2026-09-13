#!/usr/bin/env python3
"""Cross-platform validation for the locked GDC download manifest and files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import re
import sys
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data/manifests/gdc_manifest_tcga_brca_star_counts.tsv"
DEFAULT_DOWNLOAD_ROOT = ROOT / "data/raw/gdc/star_counts"
DEFAULT_REPORT = ROOT / "data/metadata/gdc_download_verification.tsv"
DEFAULT_SUMMARY = ROOT / "data/metadata/gdc_download_verification_summary.json"
MANIFEST_COLUMNS = ("id", "filename", "md5", "size", "state")
REPORT_COLUMNS = (
    "file_id",
    "file_name",
    "expected_size",
    "actual_size",
    "expected_md5",
    "actual_md5",
    "status",
)
MD5_PATTERN = re.compile(r"[0-9a-fA-F]{32}\Z")


@dataclass(frozen=True)
class ManifestRecord:
    file_id: str
    file_name: str
    expected_md5: str
    expected_size: int
    state: str

    @property
    def relative_path(self) -> Path:
        return Path(self.file_id) / self.file_name


def new_md5() -> Any:
    """Create an MD5 object on regular and FIPS-enabled Python builds."""
    try:
        return hashlib.md5(usedforsecurity=False)
    except TypeError:  # pragma: no cover - compatibility with older Python builds
        return hashlib.md5()


def file_md5(path: Path) -> str:
    digest = new_md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_component(value: str, *, field: str, line_number: int) -> str:
    if not value or value in {".", ".."}:
        raise ValueError(f"Empty or unsafe {field} at manifest line {line_number}")
    if "/" in value or "\\" in value or Path(value).name != value:
        raise ValueError(
            f"Path separators are not allowed in {field} at manifest line "
            f"{line_number}: {value!r}"
        )
    return value


def load_manifest(path: Path) -> list[ManifestRecord]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"GDC manifest is missing or empty: {path}")

    records: list[ManifestRecord] = []
    seen_ids: set[str] = set()
    seen_targets: set[Path] = set()
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != MANIFEST_COLUMNS:
            raise ValueError(
                f"Unexpected GDC manifest columns in {path}: {reader.fieldnames}; "
                f"expected {list(MANIFEST_COLUMNS)}"
            )
        for line_number, row in enumerate(reader, start=2):
            if None in row or any(row[column] is None for column in MANIFEST_COLUMNS):
                raise ValueError(f"Malformed GDC manifest row at line {line_number}")
            file_id = safe_component(
                row["id"], field="file id", line_number=line_number
            ).lower()
            try:
                parsed_id = uuid.UUID(file_id)
            except ValueError as error:
                raise ValueError(
                    f"Invalid GDC file UUID at manifest line {line_number}: {file_id!r}"
                ) from error
            if str(parsed_id) != file_id:
                raise ValueError(
                    f"Non-canonical GDC file UUID at manifest line {line_number}: "
                    f"{file_id!r}"
                )
            file_name = safe_component(
                row["filename"], field="file name", line_number=line_number
            )
            expected_md5 = row["md5"].lower()
            if not MD5_PATTERN.fullmatch(expected_md5):
                raise ValueError(
                    f"Invalid MD5 at manifest line {line_number}: {row['md5']!r}"
                )
            try:
                expected_size = int(row["size"])
            except ValueError as error:
                raise ValueError(
                    f"Invalid byte size at manifest line {line_number}: {row['size']!r}"
                ) from error
            if expected_size <= 0:
                raise ValueError(
                    f"Byte size must be positive at manifest line {line_number}"
                )
            state = row["state"]
            if state != "released":
                raise ValueError(
                    f"GDC file is not released at manifest line {line_number}: "
                    f"{state!r}"
                )

            record = ManifestRecord(
                file_id=file_id,
                file_name=file_name,
                expected_md5=expected_md5,
                expected_size=expected_size,
                state=state,
            )
            if file_id in seen_ids:
                raise ValueError(f"Duplicate GDC file id at manifest line {line_number}")
            if record.relative_path in seen_targets:
                raise ValueError(
                    f"Duplicate GDC target path at manifest line {line_number}"
                )
            seen_ids.add(file_id)
            seen_targets.add(record.relative_path)
            records.append(record)

    if not records:
        raise ValueError(f"GDC manifest contains no file records: {path}")
    return records


def check_file(path: Path, expected_size: int, expected_md5: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "expected_size": expected_size,
        "actual_size": None,
        "expected_md5": expected_md5.lower(),
        "actual_md5": "",
        "status": "MISSING",
    }
    if not path.is_file():
        return result
    try:
        result["actual_size"] = path.stat().st_size
        result["actual_md5"] = file_md5(path)
    except OSError as error:
        result["status"] = "UNREADABLE"
        result["error"] = str(error)
        return result
    result["status"] = (
        "OK"
        if result["actual_size"] == expected_size
        and result["actual_md5"] == expected_md5.lower()
        else "FAILED"
    )
    return result


def find_unexpected_files(
    download_root: Path, expected_paths: set[Path]
) -> tuple[list[str], list[str]]:
    unexpected: list[str] = []
    partial: list[str] = []
    if not download_root.exists():
        return unexpected, partial
    for path in download_root.rglob("*"):
        if not path.is_file() and not path.is_symlink():
            continue
        relative = path.relative_to(download_root)
        relative_text = relative.as_posix()
        if path.name.endswith(".part"):
            partial.append(relative_text)
        elif relative not in expected_paths:
            unexpected.append(relative_text)
    return sorted(unexpected), sorted(partial)


def verify_download(
    manifest_path: Path,
    download_root: Path,
    *,
    report_path: Path | None = None,
    summary_path: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = load_manifest(manifest_path)
    rows: list[dict[str, Any]] = []
    expected_bytes = 0
    actual_bytes = 0
    verified_count = 0

    for record in records:
        result = check_file(
            download_root / record.relative_path,
            record.expected_size,
            record.expected_md5,
        )
        expected_bytes += record.expected_size
        if result["actual_size"] is not None:
            actual_bytes += int(result["actual_size"])
        verified_count += int(result["status"] == "OK")
        rows.append(
            {
                "file_id": record.file_id,
                "file_name": record.file_name,
                "expected_size": record.expected_size,
                "actual_size": (
                    "" if result["actual_size"] is None else result["actual_size"]
                ),
                "expected_md5": record.expected_md5,
                "actual_md5": result["actual_md5"],
                "status": result["status"],
            }
        )

    expected_paths = {record.relative_path for record in records}
    unexpected_files, partial_files = find_unexpected_files(
        download_root, expected_paths
    )
    failed_count = len(records) - verified_count
    all_files_verified = (
        failed_count == 0
        and len(records) == verified_count
        and expected_bytes == actual_bytes
        and not unexpected_files
        and not partial_files
    )
    summary: dict[str, Any] = {
        "verified_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "platform": platform.system(),
        "python_version": platform.python_version(),
        "manifest_sha256": file_sha256(manifest_path),
        "expected_file_count": len(records),
        "verified_file_count": verified_count,
        "failed_file_count": failed_count,
        "unexpected_file_count": len(unexpected_files),
        "partial_file_count": len(partial_files),
        "expected_bytes": expected_bytes,
        "actual_bytes": actual_bytes,
        "expected_gib": expected_bytes / 1073741824,
        "actual_gib": actual_bytes / 1073741824,
        "unexpected_files": unexpected_files,
        "partial_files": partial_files,
        "all_files_verified": all_files_verified,
    }

    if report_path is not None:
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(
            stream, fieldnames=REPORT_COLUMNS, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
        atomic_write_text(report_path, stream.getvalue())
    if summary_path is not None:
        atomic_write_text(
            summary_path,
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        )
    return rows, summary


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".part", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def add_common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--download-root", type=Path, default=DEFAULT_DOWNLOAD_ROOT)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest_parser = subparsers.add_parser(
        "manifest", help="validate manifest schema and safe target paths"
    )
    manifest_parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    manifest_parser.add_argument("--quiet", action="store_true")

    file_parser = subparsers.add_parser(
        "file", help="validate one downloaded or partial file"
    )
    file_parser.add_argument("--path", type=Path, required=True)
    file_parser.add_argument("--expected-size", type=int, required=True)
    file_parser.add_argument("--expected-md5", required=True)
    file_parser.add_argument("--quiet", action="store_true")

    verify_parser = subparsers.add_parser(
        "verify", help="validate every locked download and reject extra/partial files"
    )
    add_common_paths(verify_parser)
    verify_parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    verify_parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    verify_parser.add_argument("--no-write-reports", action="store_true")
    verify_parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "manifest":
            records = load_manifest(args.manifest)
            result = {
                "status": "PASS",
                "manifest": str(args.manifest),
                "file_count": len(records),
                "expected_bytes": sum(record.expected_size for record in records),
            }
            if not args.quiet:
                print(json.dumps(result, indent=2))
            return 0

        if args.command == "file":
            if args.expected_size <= 0:
                raise ValueError("expected-size must be positive")
            if not MD5_PATTERN.fullmatch(args.expected_md5):
                raise ValueError("expected-md5 must contain exactly 32 hexadecimal digits")
            result = check_file(args.path, args.expected_size, args.expected_md5)
            if not args.quiet or result["status"] != "OK":
                output = sys.stdout if result["status"] == "OK" else sys.stderr
                print(json.dumps(result, indent=2), file=output)
            return 0 if result["status"] == "OK" else 1

        report_path = None if args.no_write_reports else args.report
        summary_path = None if args.no_write_reports else args.summary
        _, summary = verify_download(
            args.manifest,
            args.download_root,
            report_path=report_path,
            summary_path=summary_path,
        )
        if not args.quiet or not summary["all_files_verified"]:
            output = sys.stdout if summary["all_files_verified"] else sys.stderr
            print(json.dumps(summary, indent=2), file=output)
        return 0 if summary["all_files_verified"] else 1
    except (OSError, ValueError) as error:
        print(f"GDC validation error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
