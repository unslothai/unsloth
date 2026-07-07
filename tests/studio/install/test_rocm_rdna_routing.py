# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""RDNA 2/3/4 routing, validated on CPU-only CI with no AMD hardware.

tests/_zoo_rocm_spoof.py presents torch as each Radeon gfx arch, then we assert
unsloth_zoo routes it: device_type -> "hip", llama.cpp target -> ("rocm", gfx),
and the per-family ROCm bundle suffix. The torch-facing checks run in a
subprocess so the spoof never leaks into sibling tests and DEVICE_TYPE (cached
at import) resolves from a clean process.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("unsloth_zoo")

_TESTS_DIR = Path(__file__).resolve().parents[2]  # tests/

# gfx -> (expected llama.cpp target, expected ROCm bundle family).
_ARCHES = {
    "gfx1030": (("rocm", "gfx1030"), "gfx103X"),  # RDNA2
    "gfx1031": (("rocm", "gfx1031"), "gfx103X"),
    "gfx1032": (("rocm", "gfx1032"), "gfx103X"),
    "gfx1034": (("rocm", "gfx1034"), "gfx103X"),
    "gfx1100": (("rocm", "gfx1100"), "gfx110X"),  # RDNA3
    "gfx1101": (("rocm", "gfx1101"), "gfx110X"),
    "gfx1102": (("rocm", "gfx1102"), "gfx110X"),
    "gfx1150": (("rocm", "gfx1150"), "gfx1150"),  # RDNA3.5 APU (self-family)
    "gfx1151": (("rocm", "gfx1151"), "gfx1151"),
    "gfx1200": (("rocm", "gfx1200"), "gfx120X"),  # RDNA4
    "gfx1201": (("rocm", "gfx1201"), "gfx120X"),
}

# Child: spoof each arch, then record device_type once (fresh import) and the
# live llama.cpp target per arch. Emits one JSON line the parent parses.
_CHILD = """
import json, sys
sys.path.insert(0, {tests!r})
import _zoo_rocm_spoof as spoof
arches = {arches!r}
spoof.apply(arches[0])
from unsloth_zoo.device_type import get_device_type, is_hip
device_type = [get_device_type(), is_hip()]
from unsloth_zoo import llama_cpp as lc
targets = {{}}
for gfx in arches:
    spoof.apply(gfx)
    targets[gfx] = list(lc._detect_gpu_target())
print("RESULT " + json.dumps({{"device_type": device_type, "targets": targets}}))
"""


@pytest.fixture(scope = "module")
def routed():
    code = _CHILD.format(tests = str(_TESTS_DIR), arches = list(_ARCHES))
    proc = subprocess.run([sys.executable, "-c", code], capture_output = True, text = True)
    line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT ")), None)
    assert line, f"child produced no result.\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    return json.loads(line[len("RESULT ") :])


@pytest.mark.parametrize("gfx", list(_ARCHES))
def test_detect_gpu_target(routed, gfx):
    # RDNA card is routed to its ROCm gfx target (drives the llama.cpp bundle).
    assert tuple(routed["targets"][gfx]) == _ARCHES[gfx][0]


def test_device_type_is_hip(routed):
    # An RDNA card must resolve the compute device_type to "hip".
    assert routed["device_type"] == ["hip", True]


@pytest.mark.parametrize("gfx", list(_ARCHES))
def test_rocm_gfx_family(gfx):
    # Pure mapping (no torch): each gfx picks the right per-family ROCm bundle.
    from unsloth_zoo import llama_cpp as lc
    assert lc._rocm_gfx_family(gfx) == _ARCHES[gfx][1]
