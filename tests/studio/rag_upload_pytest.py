# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Keep test homes and caches inside the simulation directory."""

import os
from pathlib import Path
import sys

root = Path(os.environ.get("RAG_SIM_ART_DIR", "./temp/rag-upload")).resolve()
root.mkdir(parents = True, exist_ok = True)
test_home = root / "home"
test_home.mkdir(exist_ok = True)
Path.home = classmethod(lambda cls: test_home)
expanduser = os.path.expanduser
os.path.expanduser = lambda path: (
    str(test_home) + path[1:]
    if isinstance(path, str) and (path == "~" or path.startswith(("~/", "~\\")))
    else expanduser(path)
)
os.environ.update(
    {
        "UNSLOTH_STUDIO_HOME": str(test_home / "studio"),
        "XDG_CACHE_HOME": str(root / "cache"),
        "HF_HOME": str(root / "cache/huggingface"),
        "HYPOTHESIS_STORAGE_DIRECTORY": str(root / "hypothesis"),
    }
)
import pytest

raise SystemExit(pytest.main(sys.argv[1:]))
