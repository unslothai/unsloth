# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Copy Studio's logs into the evidence with the backend's own redactor applied.

Run with Studio's managed interpreter, which ships ``studio.backend``: the
evidence zip is attached to an issue, and the shapes Studio masks in its log
viewer (``utils.log_redaction``) must be masked here too. A port of those
rules drifts; the module itself does not.

    python -X utf8 -I redact_logs.py <source dir> <destination dir> [--backend <dir>]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def backend_dir(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    import studio  # the installed package, beside the interpreter this runs under

    return Path(studio.__file__).resolve().parent / "backend"


def load_redactor(backend: Path):
    sys.path.insert(0, str(backend))
    from utils.log_redaction import redact_log_text

    return redact_log_text


def copy_redacted(source: Path, destination: Path, redact) -> list[str]:
    destination.mkdir(parents = True, exist_ok = True)
    written: list[str] = []
    for log in sorted(p for p in source.rglob("*") if p.is_file()):
        text = log.read_text(encoding = "utf-8", errors = "replace")
        (destination / log.name).write_text(redact(text), encoding = "utf-8")
        written.append(log.name)
    return written


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("destination")
    ap.add_argument("--backend", default = None, help = "studio/backend directory (default: the installed one)")
    args = ap.parse_args()
    redact = load_redactor(backend_dir(args.backend))
    for name in copy_redacted(Path(args.source), Path(args.destination), redact):
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
