# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("datasets")

BACKEND = Path(__file__).resolve().parent.parent

PROBE = """
import sys
from utils.security import hub_data_files
{before}
import datasets.data_files as data_files
{after}
secret = sys.argv[1]
try:
    data_files.resolve_pattern(secret, base_path = "hf://datasets/someone/repo@main")
    print("HUB_ABSOLUTE read")
except FileNotFoundError:
    print("HUB_ABSOLUTE refused")
# A local dataset may still name its own files by absolute path.
print("LOCAL", data_files.resolve_pattern(secret, base_path = sys.argv[2]) == [secret])
inner = getattr(data_files.resolve_pattern, "__wrapped__", None)
print("DOUBLE", getattr(inner, "_unsloth_hub_card_guard", False))
print("FINDERS", sum(isinstance(f, hub_data_files._Finder) for f in sys.meta_path))
"""


def _probe(tmp_path, before, after):
    secret = tmp_path / "private.csv"
    secret.write_text("secret\n1\n")
    code = textwrap.dedent(PROBE.format(before = before, after = after))
    out = subprocess.run(
        [sys.executable, "-c", code, str(secret), str(tmp_path)],
        cwd = BACKEND,
        capture_output = True,
        text = True,
        timeout = 300,
        env = {"PYTHONPATH": str(BACKEND), "HF_HUB_OFFLINE": "1", "PATH": "/usr/bin:/bin"},
    )
    assert out.returncode == 0, out.stderr
    return out.stdout.split("\n")


# Installed before datasets loads (Studio's startup), after it, and twice: one wrapper, one finder.
@pytest.mark.parametrize(
    "before,after",
    [
        ("hub_data_files.install()", ""),
        ("", "hub_data_files.install()"),
        ("hub_data_files.install(); hub_data_files.install()", "hub_data_files.install()"),
    ],
)
def test_a_hub_card_cannot_name_a_host_file(tmp_path, before, after):
    lines = _probe(tmp_path, before, after)

    assert "HUB_ABSOLUTE refused" in lines
    assert "LOCAL True" in lines
    assert "DOUBLE False" in lines
    assert any(line in ("FINDERS 0", "FINDERS 1") for line in lines)


def test_without_the_guard_datasets_reads_the_host_file(tmp_path):
    # The behaviour the guard exists for, so a datasets release that fixes it shows up here.
    lines = _probe(tmp_path, "", "")

    assert "HUB_ABSOLUTE read" in lines
