# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Copy Studio's logs into the evidence with the backend's own redactor applied."""

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


FAMILY_CAP = 10


def family_of(log: Path) -> str:
    return log.parent.name + "/" + log.name.split("-", 1)[0].lower()


def select(source: Path, since: float | None) -> list[Path]:
    logs = sorted((p for p in source.rglob("*") if p.is_file()), key = lambda p: p.stat().st_mtime)
    if since is not None:
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
