"""Regression tests for docker/unsloth_pip_shim.py.

The shim sits ahead of the real pip/uv on PATH inside the Unsloth Docker
notebook environment so a notebook `!pip install ...` / `!uv pip install ...`
cell cannot clobber the baked, ABI-matched cu128 torch/vLLM/transformers stack.
These tests drive main() with UNSLOTH_NB_SHIM=1 and capture the command it would
os.execv, so we can assert what actually reaches the real tool. They cover:

  * -e/--editable paired with its target (a protected editable drops the flag
    too, so pip is never left a dangling `-e`);
  * -P/--upgrade-package values filtered through the protected set (uv cannot be
    told to refresh a baked package);
  * direct wheel URL / local wheel path basenames parsed for protected
    distribution names before URL passthrough.

No GPU or network is required.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SHIM_PATH = REPO_ROOT / "docker" / "unsloth_pip_shim.py"

TORCH_WHEEL_URL = (
    "https://download.pytorch.org/whl/cu128/"
    "torch-2.11.0%2Bcu128-cp312-cp312-linux_x86_64.whl"
)


class _Exec(Exception):
    """Raised by the patched os.execv so main() stops at the exec point and the
    intended command is captured instead of replacing the test process."""

    def __init__(self, path, argv):
        self.path = path
        self.argv = list(argv)


@pytest.fixture()
def shim(tmp_path, monkeypatch):
    """Load a fresh copy of the shim with the transformers marker pointed at a
    temp file and os.execv patched to capture (not perform) the exec."""
    marker = tmp_path / "requested_transformers"
    monkeypatch.setenv("UNSLOTH_NB_TF_MARKER", str(marker))
    monkeypatch.setenv("UNSLOTH_NB_SHIM", "1")

    assert SHIM_PATH.is_file(), f"missing shim: {SHIM_PATH}"
    spec = importlib.util.spec_from_file_location("unsloth_pip_shim_under_test", SHIM_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def _fake_execv(path, argv):
        raise _Exec(path, argv)

    monkeypatch.setattr(mod.os, "execv", _fake_execv)
    mod._marker_path = marker  # convenience for assertions
    return mod


def _run(shim, tool, args):
    """Invoke the shim as `tool install <args>` and return (execd_tail, marker).

    execd_tail is the argument list after the `install` verb that reached the
    real tool, or None when the shim no-op'd (nothing left to install). marker is
    the recorded transformers version, or None.
    """
    if tool == "uv":
        argv = ["uv", "pip", "install", *args]
    else:
        argv = ["pip", "install", *args]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(shim.sys, "argv", argv)
        try:
            shim.main()
            execd = None
        except _Exec as exc:
            # main() builds [REAL[tool]] + head + keep_args; head ends with the
            # `install` verb, so everything after it is what we asserted on.
            i = exc.argv.index("install")
            execd = exc.argv[i + 1 :]
    marker = shim._marker_path.read_text() if shim._marker_path.exists() else None
    return execd, marker


# --------------------------------------------------------------------------
# Item 3541142907 -- pair -e/--editable with its target.
# --------------------------------------------------------------------------
def test_editable_protected_target_drops_flag_and_value(shim):
    # `pip install -e git+...unsloth...#egg=unsloth peft` must NOT become
    # `pip install -e peft` (which pip rejects); it must install just peft.
    execd, _ = _run(
        shim,
        "pip",
        ["-e", "git+https://github.com/unslothai/unsloth.git#egg=unsloth", "peft"],
    )
    assert execd == ["peft"], execd
    assert "-e" not in execd


def test_editable_only_protected_target_noops(shim):
    execd, _ = _run(
        shim, "pip", ["-e", "git+https://github.com/unslothai/unsloth.git#egg=unsloth"]
    )
    assert execd is None  # nothing left to install -> no-op, no dangling -e


def test_editable_unprotected_target_is_kept(shim):
    execd, _ = _run(shim, "pip", ["-e", "./localpkg"])
    assert execd == ["-e", "./localpkg"], execd


def test_editable_long_form_inline_protected(shim):
    execd, _ = _run(
        shim,
        "pip",
        ["--editable=git+https://github.com/unslothai/unsloth.git#egg=unsloth", "peft"],
    )
    assert execd == ["peft"], execd


def test_editable_long_form_inline_unprotected_kept(shim):
    execd, _ = _run(shim, "pip", ["--editable=./localpkg"])
    assert execd == ["--editable=./localpkg"], execd


# --------------------------------------------------------------------------
# Item 3541142906 -- filter uv -P/--upgrade-package values.
# --------------------------------------------------------------------------
def test_upgrade_package_protected_short_flag_dropped(shim):
    # `uv pip install -P torch peft` must not let uv refresh baked torch.
    execd, _ = _run(shim, "uv", ["-P", "torch", "peft"])
    assert execd == ["peft"], execd
    assert "torch" not in execd and "-P" not in execd


def test_upgrade_package_protected_long_inline_dropped(shim):
    execd, marker = _run(shim, "uv", ["--upgrade-package=transformers", "peft"])
    assert execd == ["peft"], execd
    assert "--upgrade-package=transformers" not in execd


def test_upgrade_package_transformers_pin_recorded(shim):
    # A pinned transformers upgrade selector still feeds the sidecar marker.
    execd, marker = _run(shim, "uv", ["-P", "transformers==4.55.0", "peft"])
    assert execd == ["peft"], execd
    assert marker == "4.55.0"


def test_upgrade_package_unprotected_kept(shim):
    execd, _ = _run(shim, "uv", ["-P", "requests", "requests"])
    assert execd == ["-P", "requests", "requests"], execd


def test_upgrade_package_only_protected_noops(shim):
    execd, _ = _run(shim, "uv", ["-P", "torch"])
    assert execd is None  # -P is not itself a target


# --------------------------------------------------------------------------
# Item 3541142908 -- parse protected wheel basenames before URL passthrough.
# --------------------------------------------------------------------------
def test_direct_torch_wheel_url_dropped(shim):
    execd, _ = _run(shim, "pip", [TORCH_WHEEL_URL])
    assert execd is None  # torch wheel URL recognised + dropped -> no-op


def test_local_torch_wheel_path_dropped(shim):
    execd, _ = _run(
        shim, "pip", ["/tmp/wheels/torch-2.11.0+cu128-cp312-cp312-linux_x86_64.whl"]
    )
    assert execd is None


def test_normalised_wheel_name_dropped(shim):
    # unsloth_zoo-*.whl normalises to unsloth-zoo, which is protected.
    execd, _ = _run(shim, "pip", ["https://example.com/unsloth_zoo-1.0-py3-none-any.whl"])
    assert execd is None


def test_unprotected_wheel_url_kept(shim):
    url = "https://example.com/wheels/numpy-2.1.0-cp312-cp312-linux_x86_64.whl"
    execd, _ = _run(shim, "pip", [url])
    assert execd == [url], execd


def test_protected_wheel_in_requirements_file_dropped(shim, tmp_path):
    req = tmp_path / "reqs.txt"
    req.write_text(
        TORCH_WHEEL_URL + "\n" + "snac==1.2.0\n",
        encoding = "utf-8",
    )
    execd, _ = _run(shim, "pip", ["-r", str(req)])
    # The filtered requirements copy still installs snac; torch's wheel line is
    # stripped. execd is `-r <filtered.txt>`.
    assert execd is not None and execd[0] == "-r"
    filtered = Path(execd[1]).read_text(encoding = "utf-8")
    assert "snac==1.2.0" in filtered
    assert "torch" not in filtered


# --------------------------------------------------------------------------
# Guardrails: the ordinary happy paths still work unchanged.
# --------------------------------------------------------------------------
def test_plain_package_passes_through(shim):
    execd, _ = _run(shim, "pip", ["omegaconf==2.3.1"])
    assert execd == ["omegaconf==2.3.1"], execd


def test_bare_transformers_recorded_and_dropped(shim):
    execd, marker = _run(shim, "pip", ["transformers==4.55.0"])
    assert execd is None
    assert marker == "4.55.0"


def test_index_url_value_flag_kept_verbatim(shim):
    execd, _ = _run(shim, "pip", ["--extra-index-url", "https://example.com/simple", "snac"])
    assert execd == ["--extra-index-url", "https://example.com/simple", "snac"], execd
