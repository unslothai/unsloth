# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Read or diff the packed-weight fingerprint of a pre-quantized transformer checkpoint.

  python scripts/prequant_fingerprint.py --fingerprint Wan2.2-TI2V-5B-NVFP4.pt
  python scripts/prequant_fingerprint.py --diff build_a.pt build_b.pt

``--diff`` exits 3 on any difference. The metadata is read through the loader's own restricted
``weights_only`` path, so pointing this at a file that is not one of ours cannot execute anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

BACKEND = Path(__file__).resolve().parent.parent / "studio" / "backend"


def _fingerprint(path: str) -> dict:
    """The fingerprint block of the artifact at ``path``. Raises when it carries none."""
    from core.inference.diffusion_prequant import read_prequant_metadata

    block = (read_prequant_metadata(path) or {}).get("fingerprint") or {}
    if not block.get("modules"):
        raise ValueError(
            f"{path} carries no fingerprint block (built before it existed); rebuild it with "
            "scripts/build_prequant_checkpoint.py"
        )
    return block


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--fingerprint", default = None, help = "checkpoint to print the block of")
    p.add_argument(
        "--diff", nargs = 2, default = None, metavar = ("A", "B"), help = "two checkpoints to diff"
    )
    args = p.parse_args(argv)
    if bool(args.fingerprint) == bool(args.diff):
        print("error: pass exactly one of --fingerprint <ckpt> or --diff <a> <b>", flush = True)
        return 2

    sys.path.insert(0, str(BACKEND))
    # Imported here rather than at module scope: the backend only joins sys.path above.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "build_prequant_checkpoint",
        Path(__file__).resolve().parent / "build_prequant_checkpoint.py",
    )
    build: Any = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = build
    spec.loader.exec_module(build)

    if args.fingerprint:
        try:
            block = _fingerprint(args.fingerprint)
        except Exception as exc:  # noqa: BLE001 -- a tool reports, it does not traceback
            print(f"error: {exc}", flush = True)
            return 2
        print(json.dumps(block, indent = 2, sort_keys = True), flush = True)
        return 0

    a, b = args.diff
    try:
        first, second = _fingerprint(a), _fingerprint(b)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", flush = True)
        return 2
    diffs = build.fingerprint_mismatches(first, second)
    if not diffs:
        print(f"identical: {first.get('count')} quantized weights match", flush = True)
        return 0
    print(f"{len(diffs)} quantized weights differ:", flush = True)
    for key in diffs:
        print(f"  {build.describe_key(key)}", flush = True)
    return 3


if __name__ == "__main__":
    sys.exit(main())
