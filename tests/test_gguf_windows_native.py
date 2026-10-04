# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native-Windows proof for #7897. Skipped everywhere else.

The Linux suite simulates Windows by injecting ntpath; this asserts the same
properties with the real os.path on real NTFS, so the simulation cannot quietly
diverge from the platform it models. windows-latest runners have a second drive
letter, which makes the cross-drive case real rather than notional.
"""

from __future__ import annotations

import ast
import glob
import os
import sys
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason = "native Windows path semantics")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SAVE_PY = _REPO_ROOT / "unsloth" / "save.py"


def _load_helper():
    src = _SAVE_PY.read_text(encoding = "utf-8")
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == "_model_basename":
            ns: dict = {"os": os}
            exec(compile(ast.Module([node], []), str(_SAVE_PY), "exec"), ns)
            return ns["_model_basename"]
    raise AssertionError("unsloth/save.py defines no _model_basename")


# Ground truth: the platform behaviours the Linux simulation asserts.
def test_join_really_discards_the_prefix_for_a_drive_bearing_component():
    """The mechanism behind #7897, on the real platform."""
    assert os.path.join(r"C:\exp\_gguf", r"D:\M\X.Q5_K_M.gguf") == r"D:\M\X.Q5_K_M.gguf"
    assert os.path.join(r"C:\exp", r"\\server\share\X.gguf") == r"\\server\share\X.gguf"
    # A rooted-but-driveless component keeps the drive, drops the directory.
    assert os.path.join(r"C:\exp", r"\Models\X.gguf") == r"C:\Models\X.gguf"


def test_glob_really_hides_dot_leading_gguf():
    """Why the final listing moved off glob.glob."""
    with tempfile.TemporaryDirectory() as d:
        Path(d, ".BF16.gguf").write_bytes(b"GGUF")
        Path(d, "ok.gguf").write_bytes(b"GGUF")
        assert [os.path.basename(p) for p in glob.glob(os.path.join(d, "*.gguf"))] == ["ok.gguf"]
        assert sorted(p.name for p in Path(d).glob("*.gguf")) == [".BF16.gguf", "ok.gguf"]


def test_posix_basename_would_not_have_fixed_it():
    """os.path.basename is correct here but wrong on the Linux CI that tests it."""
    import posixpath

    assert os.path.basename(r"D:\M\MyModel") == "MyModel"  # ntpath: fine
    assert posixpath.basename(r"D:\M\MyModel") == r"D:\M\MyModel"  # posix: broken


# The fix, on real paths.
@pytest.mark.parametrize(
    "base, expected",
    [
        (r"D:\Models\Merged Models\MyModel", "MyModel"),
        ("D:\\Models\\MyModel\\", "MyModel"),
        (r"\\?\D:\Models\MyModel", "MyModel"),
        (r"\\server\share\Models\MyModel", "MyModel"),
        (r"C:\Users\Ada\OneDrive - X\Llama 3.1 8B", "Llama 3.1 8B"),
        ("D:\\", "model"),
        ("D:", "model"),
        ("unsloth/Qwen3-8B", "Qwen3-8B"),
    ],
)
def test_basename_on_native_windows(base, expected):
    assert _load_helper()(base) == expected


def test_output_stays_on_the_export_drive(tmp_path):
    """Cross-drive: base model on D:, export on the runner's temp drive."""
    helper = _load_helper()
    gguf_dir = str(tmp_path / "run_gguf")
    os.makedirs(gguf_dir, exist_ok = True)

    stem = helper(r"D:\Models\Merged Models\MyModel")
    out = os.path.join(gguf_dir, f"{stem}.Q5_K_M.gguf")

    assert os.path.dirname(out) == gguf_dir
    assert os.path.splitdrive(out)[0].upper() == os.path.splitdrive(gguf_dir)[0].upper()

    Path(out).write_bytes(b"GGUF")
    assert [p.name for p in Path(gguf_dir).glob("*.gguf")] == ["MyModel.Q5_K_M.gguf"]


@pytest.mark.skipif(not os.path.isdir("D:\\"), reason = "runner has no D: drive")
def test_real_second_drive_end_to_end():
    """A real base-model directory on D:, a real export dir on C:."""
    helper = _load_helper()
    base_dir = r"D:\Models\Merged Models\MyModel"
    os.makedirs(base_dir, exist_ok = True)

    with tempfile.TemporaryDirectory() as export_root:
        gguf_dir = os.path.join(export_root, "_tmp_model_ab12_gguf")
        os.makedirs(gguf_dir, exist_ok = True)

        out = os.path.join(gguf_dir, f"{helper(base_dir)}.Q5_K_M.gguf")
        Path(out).write_bytes(b"GGUF")

        # Nothing may be written beside the base model.
        strays = [
            os.path.join(root, f)
            for root, _d, files in os.walk(r"D:\Models")
            for f in files
            if f.lower().endswith(".gguf")
        ]
        assert strays == [], f"GGUF leaked next to the base model: {strays}"
        assert os.path.isfile(out)
