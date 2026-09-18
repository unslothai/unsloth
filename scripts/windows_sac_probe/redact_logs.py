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


# Studio does not prune its own log directory: debug_log_sources.py records a
# measured installation carrying 11794 llama-server logs, and the log viewer
# caps each family at ten rather than showing them all. Copying the lot into the
# evidence makes redaction and compression take impractically long, produces a
# zip too large to attach, and ships years of unrelated diagnostics to a public
# issue. Bound it the same two ways the viewer does: the probe window first,
# then a per-family cap.
FAMILY_CAP = 10


def family_of(log: Path) -> str:
    # "backend-backend-1788737374062-2-s01.log" and
    # "server-20260906-194759-pid1820.log" both collapse to their leading word,
    # which is what the viewer groups on.
    return log.parent.name + "/" + log.name.split("-", 1)[0].lower()


def select(source: Path, since: float | None) -> list[Path]:
    logs = sorted((p for p in source.rglob("*") if p.is_file()), key = lambda p: p.stat().st_mtime)
    if since is not None:
        # No fallback to historical logs when the window is empty. It looked
        # like a kindness (an empty studio-logs/ is unhelpful) but the zip is
        # attached to a public issue, and redaction masks tokens, not prompts,
        # file paths or anything else private in a log from last month. The
        # empty-window case is also exactly the interesting one: Studio blocked
        # before its logger created a file. The probe's own Studio stdout is
        # captured separately under raw-logs/, so nothing about this run is
        # lost by declining to reach backwards.
        logs = [p for p in logs if p.stat().st_mtime >= since]
    kept: dict[str, list[Path]] = {}
    for log in logs:
        kept.setdefault(family_of(log), []).append(log)
    out: list[Path] = []
    for family in sorted(kept):
        out.extend(kept[family][-FAMILY_CAP:])
    return sorted(out)


def copy_redacted(
    source: Path,
    destination: Path,
    redact,
    since: float | None = None,
) -> list[str]:
    destination.mkdir(parents = True, exist_ok = True)
    written: list[str] = []
    for log in select(source, since):
        text = log.read_text(encoding = "utf-8", errors = "replace")
        # Relative path, not just the leaf: two log families can hold the same
        # file name in different subdirectories, and flattening let the second
        # silently overwrite the first.
        target = destination / log.relative_to(source)
        target.parent.mkdir(parents = True, exist_ok = True)
        target.write_text(redact(text), encoding = "utf-8")
        written.append(str(log.relative_to(source)))
    return written


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("destination")
    ap.add_argument(
        "--backend", default = None, help = "studio/backend directory (default: the installed one)"
    )
    ap.add_argument(
        "--since",
        default = None,
        help = "ISO 8601 start of the probe window; older logs are left out of the evidence",
    )
    args = ap.parse_args()
    since = None
    if args.since:
        from datetime import datetime
        since = datetime.fromisoformat(args.since).timestamp()
    redact = load_redactor(backend_dir(args.backend))
    for name in copy_redacted(Path(args.source), Path(args.destination), redact, since):
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
