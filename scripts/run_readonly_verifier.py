#!/usr/bin/env python3
"""Run a verifier while preserving already-versioned verification reports."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a verifier command is required after --")
    return args


def main() -> None:
    args = parse_args()
    existing = {
        path: path.read_bytes()
        for base in (ROOT / "data", ROOT / "outputs")
        for path in base.rglob("verification_report.json")
    }
    try:
        completed = subprocess.run(args.command, cwd=ROOT, check=False)
    finally:
        for path, content in existing.items():
            if not path.is_file() or path.read_bytes() != content:
                path.write_bytes(content)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
