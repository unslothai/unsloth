# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""unslothai/unsloth#4381: an installed xformers whose CUDA extension cannot load
(PyPI's cu128 Windows wheel beside a cu130 torch) left only "Xformers = None" in
the banner. Unsloth must say why and how to fix it, and stay quiet when it loads."""

import importlib.util
import os
import subprocess
import sys

import pytest
from real_accelerator import has_real_cuda  # tests/_shared, on sys.path via tests/conftest.py

pytestmark = pytest.mark.skipif(
    not has_real_cuda() or importlib.util.find_spec("xformers") is None,
    reason = "needs a CUDA GPU and an installed xformers",
)

# Raise xformers' own exception, so the message format Unsloth parses is the installed package's.
_BREAK_XFORMERS = """
import xformers._cpp_lib as c
def _register_extensions():
    raise c.xFormersInvalidLibException(c._build_metadata) from OSError("libcudart.so.12: cannot open shared object file")
c._register_extensions = _register_extensions
"""

_IMPORT_UNSLOTH = """
import unsloth.models._utils as u
print("XFORMERS_IS_NONE", u.xformers is None)
"""


def _run(code):
    env = {k: v for k, v in os.environ.items() if k != "UNSLOTH_ENABLE_LOGGING"}
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output = True,
        text = True,
        env = env,
        timeout = 600,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    return result.stdout


def test_broken_xformers_is_reported():
    out = _run(_BREAK_XFORMERS + _IMPORT_UNSLOTH)
    assert "XFORMERS_IS_NONE True" in out
    assert "xformers is installed but cannot load with torch" in out
    assert "(it was built for " in out
    assert "unsloth install-kernels xformers" in out


def test_version_guard_keeps_its_own_fix():
    # Blackwell + xformers <= 0.0.32.post2 is refused on purpose; install-kernels maps torch 2.8 back to that same
    # release, so pointing there would leave xformers off.
    old_blackwell_xformers = (
        "import torch, xformers\n"
        "xformers.__version__ = '0.0.32.post2'\n"
        "torch.cuda.get_device_capability = lambda *args, **kwargs: (12, 0)\n"
    )
    out = _run(old_blackwell_xformers + _IMPORT_UNSLOTH)
    assert "XFORMERS_IS_NONE True" in out
    assert "has a broken FA3 dispatch" in out
    assert "unsloth install-kernels xformers" not in out


def test_working_xformers_prints_nothing():
    import xformers._cpp_lib as c

    if c._cpp_library_load_exception is not None:
        pytest.skip("installed xformers does not load here")
    out = _run(_IMPORT_UNSLOTH)
    assert "XFORMERS_IS_NONE False" in out
    assert "xformers is installed but cannot load" not in out
