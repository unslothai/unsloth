#!/usr/bin/env python3
"""Download real Gemma weights and run offline integration tests for #7481.

Example:
  python tests/saving/run_offline_gguf_integration.py
  python tests/saving/run_offline_gguf_integration.py --download-only
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = "unsloth/gemma-3-270m-it-bnb-4bit"
CACHE_ROOT = Path(os.environ.get("HF_HOME", "/tmp/hf_offline_test_cache"))


def download():
    from huggingface_hub import snapshot_download

    os.environ.setdefault("HF_HOME", str(CACHE_ROOT))
    path = snapshot_download(REPO, cache_dir = str(CACHE_ROOT / "hub"))
    print("cached at", path)


def run_tests():
    os.environ.setdefault("HF_HOME", str(CACHE_ROOT))
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/saving/test_offline_gguf_vlm_tokenizer_7481.py",
        "tests/saving/test_offline_gguf_real_cache_integration.py",
        "-q",
    ]
    raise SystemExit(subprocess.call(cmd, cwd = str(Path(__file__).resolve().parents[2])))


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
