"""`import unsloth` must not force HF_HUB_ENABLE_HF_TRANSFER=1: huggingface_hub < 1.0 then refuses
every download when hf_transfer is missing, and Studio's explicit "0" (Xet fallback) was overwritten."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

_PROBE = """
import sys
if {hide}:
    sys.modules["hf_transfer"] = None
import unsloth.dataprep.synthetic, os
print("HF_TRANSFER=" + repr(os.environ.get("HF_HUB_ENABLE_HF_TRANSFER")))
"""


def _flag_after_import(
    tmp_path,
    *,
    hide,
    preset = None,
    stub = False,
    offline = False,
):
    env = dict(os.environ)
    for key in (
        "HF_HUB_ENABLE_HF_TRANSFER",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
    ):
        env.pop(key, None)
    if preset is not None:
        env["HF_HUB_ENABLE_HF_TRANSFER"] = preset
    if offline:
        env["HF_HUB_OFFLINE"] = "1"
    env["UNSLOTH_ALLOW_CPU"] = "1"
    paths = [str(REPO_ROOT)]
    if stub:
        (tmp_path / "hf_transfer").mkdir()
        (tmp_path / "hf_transfer" / "__init__.py").write_text("")
        paths.append(str(tmp_path))
    env["PYTHONPATH"] = os.pathsep.join(paths + [env.get("PYTHONPATH", "")])
    out = subprocess.run(
        [sys.executable, "-c", _PROBE.format(hide = hide)],
        env = env,
        capture_output = True,
        text = True,
        timeout = 600,
    )
    lines = [l for l in out.stdout.splitlines() if l.startswith("HF_TRANSFER=")]
    if not lines:
        pytest.skip(f"unsloth did not import here: {out.stderr[-500:]}")
    return lines[-1].split("=", 1)[1]


def test_flag_not_forced_when_hf_transfer_missing(tmp_path):
    assert _flag_after_import(tmp_path, hide = True) == "None"


def test_explicit_zero_survives_import(tmp_path):
    assert _flag_after_import(tmp_path, hide = False, preset = "0", stub = True) == "'0'"


def test_flag_enabled_when_hf_transfer_present(tmp_path):
    assert _flag_after_import(tmp_path, hide = False, stub = True) == "'1'"


def test_flag_not_set_offline(tmp_path):
    assert _flag_after_import(tmp_path, hide = False, stub = True, offline = True) == "None"
