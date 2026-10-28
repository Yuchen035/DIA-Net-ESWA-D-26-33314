#!/usr/bin/env python3
"""Verify the distributed file manifest without importing project code."""
import hashlib
import json
from pathlib import Path
import sys


def main():
    root = Path(__file__).resolve().parent.parent
    manifest = root / "SHA256SUMS.txt"
    failures = []
    expected = set()
    if not manifest.is_file():
        print(json.dumps({"ok": False, "error": "SHA256SUMS.txt is missing"}))
        return 1
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split("  ", 1)
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or len(digest) != 64:
            failures.append({"path": relative, "reason": "invalid manifest entry"})
            continue
        expected.add(relative)
        if not path.is_file():
            failures.append({"path": relative, "reason": "missing file"})
            continue
        computed = hashlib.sha256(path.read_bytes()).hexdigest()
        if computed != digest:
            failures.append({"path": relative, "reason": "SHA-256 mismatch"})
    actual = {str(path.relative_to(root)).replace("\\", "/")
              for path in root.rglob("*") if path.is_file()
              and path.name != "SHA256SUMS.txt"
              and "__pycache__" not in path.parts and path.suffix != ".pyc"}
    extras = sorted(actual - expected)
    print(json.dumps({"ok": not failures and not extras,
                      "manifest_entries": len(expected),
                      "failures": failures, "unlisted_files": extras,
                      "scope": "distribution integrity only; no model execution"}, indent=2))
    return 0 if not failures and not extras else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        sys.exit(1)
