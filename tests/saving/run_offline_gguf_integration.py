#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Download real Gemma weights and run offline integration tests for #7481.

Sets ``UNSLOTH_INTEGRATION_IMPORT=1`` for the pytest subprocess so the
real-cache suite is not silently skipped. Requires a host that can import
unsloth (typically GPU).

Example:
  python tests/saving/run_offline_gguf_integration.py
  python tests/saving/run_offline_gguf_integration.py --download-only
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = "unsloth/gemma-3-270m-it-bnb-4bit"
CACHE_ROOT = Path(
    os.environ.get("HF_HOME") or os.path.join(tempfile.gettempdir(), "hf_offline_test_cache")
)


def download():
    from huggingface_hub import snapshot_download

    os.environ.setdefault("HF_HOME", str(CACHE_ROOT))
    path = snapshot_download(REPO, cache_dir = str(CACHE_ROOT / "hub"))
    print("cached at", path)


def run_tests():
    os.environ.setdefault("HF_HOME", str(CACHE_ROOT))
    # Real-cache suite is gated on this; without it every integration test skips
    # and the runner reports success after only the fake-cache unit file ran.
    env = os.environ.copy()
    env["UNSLOTH_INTEGRATION_IMPORT"] = "1"
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/saving/test_offline_gguf_vlm_tokenizer_7481.py",
        "tests/saving/test_offline_gguf_real_cache_integration.py",
        "-q",
    ]
    raise SystemExit(subprocess.call(cmd, cwd = str(Path(__file__).resolve().parents[2]), env = env))


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--download-only", action = "store_true")
    args = parser.parse_args()
    download()
    if not args.download_only:
        run_tests()


if __name__ == "__main__":
    main()
