# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Run the physical Apple Silicon agent-workspace certification and emit JSON evidence."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument(
        "--model",
        default = os.environ.get(
            "UNSLOTH_MLX_AGENT_RUNTIME_MODEL", "unsloth/gemma-3-270m-it"
        ),
    )
    parser.add_argument("--workdir", type = Path, required = True)
    parser.add_argument("--evidence", type = Path)
    parser.add_argument("--timeout-seconds", type = float, default = 300.0)
    parser.add_argument("--max-seq-length", type = int, default = 8192)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError(
            "The real MLX agent-workspace certification requires Apple Silicon."
        )

    repo_root = Path(__file__).resolve().parents[2]
    backend_root = repo_root / "studio" / "backend"
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))
    if str(Path(__file__).resolve().parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent))

    workdir = args.workdir.expanduser().resolve()
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents = True)
    studio_home = workdir / "studio-home"
    studio_home.mkdir()

    # These must be final before any Studio storage or hardware module imports.
    os.environ["UNSLOTH_STUDIO_HOME"] = str(studio_home)
    os.environ["UNSLOTH_IS_PRESENT"] = "1"
    os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
    os.environ.pop("UNSLOTH_ALLOW_CPU", None)
    os.environ.pop("UNSLOTH_STUDIO_DISABLE_DEVICE_PROBE", None)

    from agent_workspace_mlx_runtime_harness import run_certification

    evidence = run_certification(
        workdir,
        model_name = str(args.model),
        timeout_seconds = args.timeout_seconds,
        max_seq_length = args.max_seq_length,
        hf_token = (os.environ.get("HF_TOKEN") or None),
    )
    evidence_path = (
        args.evidence.expanduser().resolve()
        if args.evidence is not None
        else workdir / "evidence.json"
    )
    evidence_path.parent.mkdir(parents = True, exist_ok = True)
    evidence_path.write_text(
        json.dumps(evidence, indent = 2, ensure_ascii = False) + "\n",
        encoding = "utf-8",
    )
    print("\n=== real MLX agent-workspace evidence ===", flush = True)
    print(json.dumps(evidence, indent = 2, ensure_ascii = False), flush = True)
    print(f"[mlx-agent-runtime] wrote {evidence_path}", flush = True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
