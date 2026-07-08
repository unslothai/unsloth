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
    "https://download.pytorch.org/whl/cu128/torch-2.11.0%2Bcu128-cp312-cp312-linux_x86_64.whl"
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
    execd, _ = _run(shim, "pip", ["-e", "git+https://github.com/unslothai/unsloth.git#egg=unsloth"])
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
    execd, _ = _run(shim, "pip", ["/tmp/wheels/torch-2.11.0+cu128-cp312-cp312-linux_x86_64.whl"])
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


# --------------------------------------------------------------------------
# Item 3541404842 -- filter editable entries INSIDE a requirements file.
# --------------------------------------------------------------------------
def test_editable_protected_in_requirements_file_dropped(shim, tmp_path):
    req = tmp_path / "reqs.txt"
    req.write_text(
        "-e git+https://github.com/unslothai/unsloth.git#egg=unsloth\nsnac==1.2.0\n",
        encoding = "utf-8",
    )
    execd, _ = _run(shim, "pip", ["-r", str(req)])
    assert execd is not None and execd[0] == "-r", execd
    filtered = Path(execd[1]).read_text(encoding = "utf-8")
    assert "snac==1.2.0" in filtered
    assert "unsloth" not in filtered  # protected editable line stripped


def test_editable_attached_protected_in_requirements_file_dropped(shim, tmp_path):
    req = tmp_path / "reqs.txt"
    req.write_text(
        "-egit+https://github.com/unslothai/unsloth.git#egg=unsloth\nsnac==1.2.0\n",
        encoding = "utf-8",
    )
    execd, _ = _run(shim, "pip", ["-r", str(req)])
    assert execd is not None and execd[0] == "-r", execd
    filtered = Path(execd[1]).read_text(encoding = "utf-8")
    assert "snac==1.2.0" in filtered
    assert "unsloth" not in filtered


def test_editable_unprotected_in_requirements_file_kept(shim, tmp_path):
    # An unprotected editable survives even when the file is otherwise rewritten
    # (torch dropped); only protected editables are stripped.
    req = tmp_path / "reqs.txt"
    req.write_text(
        "-e ./localpkg\ntorch==2.11.0\nsnac==1.2.0\n",
        encoding = "utf-8",
    )
    execd, _ = _run(shim, "pip", ["-r", str(req)])
    assert execd is not None and execd[0] == "-r", execd
    filtered = Path(execd[1]).read_text(encoding = "utf-8")
    assert "./localpkg" in filtered
    assert "snac==1.2.0" in filtered
    assert "torch" not in filtered


# --------------------------------------------------------------------------
# Item 3541404849 -- a nested -c constraint pin is not recorded as a request.
# --------------------------------------------------------------------------
def test_nested_constraint_transformers_pin_not_recorded(shim, tmp_path):
    constraints = tmp_path / "constraints.txt"
    constraints.write_text("transformers==4.55.0\n", encoding = "utf-8")
    req = tmp_path / "reqs.txt"
    req.write_text("-c constraints.txt\nsnac==1.2.0\n", encoding = "utf-8")
    execd, marker = _run(shim, "pip", ["-r", str(req)])
    assert execd is not None and execd[0] == "-r", execd
    # A constraint pin is not an install request -> no sidecar marker written.
    assert marker is None, marker


def test_nested_requirement_transformers_pin_recorded(shim, tmp_path):
    # Contrast: a nested -r requirement DOES carry install requests, so its
    # transformers pin is still recorded for the sidecar.
    nested = tmp_path / "nested.txt"
    nested.write_text("transformers==4.55.0\n", encoding = "utf-8")
    req = tmp_path / "reqs.txt"
    req.write_text("-r nested.txt\nsnac==1.2.0\n", encoding = "utf-8")
    execd, marker = _run(shim, "pip", ["-r", str(req)])
    assert execd is not None and execd[0] == "-r", execd
    assert marker == "4.55.0", marker


# --------------------------------------------------------------------------
# Item 3541404845 -- handle pip's attached short options (-rfile / -cfile / etc).
# --------------------------------------------------------------------------
def test_attached_short_requirement_file_filtered(shim, tmp_path):
    # `pip install -rreqs.txt` (attached) must filter the file AND count as a
    # target -- before the fix it fell through as an opaque option and no-op'd.
    req = tmp_path / "reqs.txt"
    req.write_text("torch==2.11.0\nsnac==1.2.0\n", encoding = "utf-8")
    execd, _ = _run(shim, "pip", ["-r" + str(req)])
    assert execd is not None and execd[0] == "-r", execd
    filtered = Path(execd[1]).read_text(encoding = "utf-8")
    assert "snac==1.2.0" in filtered
    assert "torch" not in filtered


def test_attached_short_constraint_file_filtered(shim, tmp_path):
    constraints = tmp_path / "constraints.txt"
    constraints.write_text("torch==2.11.0\n", encoding = "utf-8")
    execd, _ = _run(shim, "pip", ["-c" + str(constraints), "peft"])
    assert execd is not None and execd[0] == "-c", execd
    assert "peft" in execd
    filtered = Path(execd[1]).read_text(encoding = "utf-8")
    assert "torch" not in filtered


def test_attached_short_editable_protected_dropped(shim):
    execd, _ = _run(
        shim,
        "pip",
        ["-egit+https://github.com/unslothai/unsloth.git#egg=unsloth", "peft"],
    )
    assert execd == ["peft"], execd


def test_attached_short_upgrade_package_protected_dropped(shim):
    execd, _ = _run(shim, "uv", ["-Ptorch", "peft"])
    assert execd == ["peft"], execd
    assert "torch" not in execd and "-P" not in execd


# --------------------------------------------------------------------------
# Item 3541773143 -- a bare wheel filename (no ./ or / prefix) is still a pip
# target from the CWD, so its protected distribution must be parsed too.
# --------------------------------------------------------------------------
def test_bare_torch_wheel_filename_dropped(shim):
    # `pip install torch-2.11.0-...whl` from the CWD must not reinstall torch.
    execd, _ = _run(shim, "pip", ["torch-2.11.0+cu128-cp312-cp312-linux_x86_64.whl"])
    assert execd is None, execd


def test_bare_wheel_in_subdir_dropped(shim):
    execd, _ = _run(shim, "pip", ["dist/torch-2.11.0-cp312-cp312-linux_x86_64.whl"])
    assert execd is None, execd


def test_bare_unprotected_wheel_filename_kept(shim):
    execd, _ = _run(shim, "pip", ["numpy-2.1.0-cp312-cp312-linux_x86_64.whl"])
    assert execd == ["numpy-2.1.0-cp312-cp312-linux_x86_64.whl"], execd


# --------------------------------------------------------------------------
# Item 3541773157 -- a protected VCS URL WITHOUT an #egg= fragment (the egg-less
# form this repo recommends) must be dropped via its repo basename.
# --------------------------------------------------------------------------
def test_vcs_url_without_egg_protected_dropped(shim):
    # git+https://github.com/huggingface/transformers.git -> transformers.
    execd, _ = _run(shim, "pip", ["git+https://github.com/huggingface/transformers.git", "peft"])
    assert execd == ["peft"], execd


def test_vcs_url_without_egg_with_ref_dropped(shim):
    execd, _ = _run(shim, "pip", ["git+https://github.com/unslothai/unsloth-zoo.git@main", "peft"])
    assert execd == ["peft"], execd


def test_vcs_url_without_egg_unprotected_kept(shim):
    url = "git+https://github.com/someone/coolpkg.git"
    execd, _ = _run(shim, "pip", [url])
    assert execd == [url], execd


# --------------------------------------------------------------------------
# Item 3541773153 -- refuse remote (URL) requirement / constraint files in shim
# mode; their protected pins cannot be inspected before the real tool installs.
# --------------------------------------------------------------------------
def test_remote_requirement_url_only_noops(shim):
    execd, _ = _run(shim, "pip", ["-r", "https://example.com/reqs.txt"])
    assert execd is None, execd  # dropped, and no dangling -r left behind


def test_remote_requirement_url_with_other_target_kept(shim):
    execd, _ = _run(shim, "pip", ["-r", "https://example.com/reqs.txt", "peft"])
    assert execd == ["peft"], execd


def test_remote_requirement_inline_form_dropped(shim):
    execd, _ = _run(shim, "pip", ["--requirement=https://example.com/reqs.txt", "peft"])
    assert execd == ["peft"], execd


def test_remote_requirement_attached_form_dropped(shim):
    execd, _ = _run(shim, "pip", ["-rhttps://example.com/reqs.txt", "peft"])
    assert execd == ["peft"], execd


def test_remote_constraint_url_dropped_target_kept(shim):
    execd, _ = _run(shim, "pip", ["-c", "https://example.com/constraints.txt", "peft"])
    assert execd == ["peft"], execd


def test_nested_remote_include_dropped(shim, tmp_path):
    # A local reqs file that pulls a remote include must have that include
    # stripped, not passed through for the real pip to fetch unfiltered.
    req = tmp_path / "reqs.txt"
    req.write_text("-r https://example.com/evil.txt\nsnac==1.2.0\n", encoding = "utf-8")
    execd, _ = _run(shim, "pip", ["-r", str(req)])
    assert execd is not None and execd[0] == "-r", execd
    filtered = Path(execd[1]).read_text(encoding = "utf-8")
    assert "snac==1.2.0" in filtered
    assert "example.com" not in filtered and "://" not in filtered


# --------------------------------------------------------------------------
# Item 3541773164 -- resolver-wide reinstall / ignore-installed flags are
# stripped so they cannot rebuild already-satisfied baked deps.
# --------------------------------------------------------------------------
def test_force_reinstall_flag_stripped(shim):
    execd, _ = _run(shim, "pip", ["--force-reinstall", "peft"])
    assert execd == ["peft"], execd


def test_ignore_installed_short_flag_stripped(shim):
    execd, _ = _run(shim, "pip", ["-I", "peft"])
    assert execd == ["peft"], execd


def test_uv_reinstall_flag_stripped(shim):
    execd, _ = _run(shim, "uv", ["--reinstall", "peft"])
    assert execd == ["peft"], execd


# --------------------------------------------------------------------------
# Item 3541773168 -- uv's --reinstall-package selector is filtered through _KEEP
# exactly like -P/--upgrade-package (both forms, no dangling flag).
# --------------------------------------------------------------------------
def test_reinstall_package_protected_separated_dropped(shim):
    execd, _ = _run(shim, "uv", ["--reinstall-package", "torch", "peft"])
    assert execd == ["peft"], execd
    assert "torch" not in execd and "--reinstall-package" not in execd


def test_reinstall_package_protected_inline_dropped(shim):
    execd, _ = _run(shim, "uv", ["--reinstall-package=torch", "peft"])
    assert execd == ["peft"], execd


def test_reinstall_package_unprotected_kept(shim):
    execd, _ = _run(shim, "uv", ["--reinstall-package", "requests", "requests"])
    assert execd == ["--reinstall-package", "requests", "requests"], execd


def test_reinstall_package_transformers_pin_recorded(shim):
    execd, marker = _run(shim, "uv", ["--reinstall-package", "transformers==4.55.0", "peft"])
    assert execd == ["peft"], execd
    assert marker == "4.55.0", marker


# --------------------------------------------------------------------------
# Item 3542096750 -- parse protected source archives (sdist / zip) too.
# --------------------------------------------------------------------------
def test_sdist_url_protected_dropped(shim):
    url = "https://files.pythonhosted.org/packages/aa/unsloth-2026.7.1.tar.gz"
    execd, _ = _run(shim, "pip", [url, "peft"])
    assert execd == ["peft"], execd


def test_sdist_bare_protected_dropped(shim):
    execd, _ = _run(shim, "pip", ["torch-2.11.0.tar.gz"])
    assert execd is None, execd


def test_sdist_zip_protected_dropped(shim):
    execd, _ = _run(shim, "pip", ["./transformers-4.55.0.zip", "peft"])
    assert execd == ["peft"], execd


def test_sdist_hyphenated_name_protected_dropped(shim):
    # flashinfer-python is protected; the name must survive the hyphen split.
    execd, _ = _run(shim, "pip", ["flashinfer-python-0.5.0.tar.gz"])
    assert execd is None, execd


def test_sdist_unprotected_kept(shim):
    execd, _ = _run(shim, "pip", ["numpy-2.1.0.tar.gz"])
    assert execd == ["numpy-2.1.0.tar.gz"], execd


# --------------------------------------------------------------------------
# Item 3542096760 -- uv's PLURAL --requirements / --constraints go through the
# same filter as the pip-style singular names.
# --------------------------------------------------------------------------
def test_uv_plural_requirements_filtered(shim, tmp_path):
    req = tmp_path / "reqs.txt"
    req.write_text("torch==2.11.0\nsnac==1.2.0\n", encoding = "utf-8")
    execd, _ = _run(shim, "uv", ["--requirements", str(req)])
    assert execd is not None and execd[0] == "--requirements", execd
    filtered = Path(execd[1]).read_text(encoding = "utf-8")
    assert "snac==1.2.0" in filtered
    assert "torch" not in filtered


def test_uv_plural_constraints_filtered(shim, tmp_path):
    constraints = tmp_path / "constraints.txt"
    constraints.write_text("torch==2.11.0\n", encoding = "utf-8")
    execd, _ = _run(shim, "uv", ["--constraints", str(constraints), "peft"])
    assert execd is not None and execd[0] == "--constraints", execd
    assert "peft" in execd
    filtered = Path(execd[1]).read_text(encoding = "utf-8")
    assert "torch" not in filtered


# --------------------------------------------------------------------------
# Item 3542096764 -- neutralise --upgrade-strategy eager so a kept target cannot
# eagerly rebuild already-satisfied baked deps.
# --------------------------------------------------------------------------
def test_upgrade_strategy_eager_dropped(shim):
    execd, _ = _run(shim, "pip", ["-U", "--upgrade-strategy", "eager", "peft"])
    assert execd == ["-U", "peft"], execd


def test_upgrade_strategy_eager_inline_dropped(shim):
    execd, _ = _run(shim, "pip", ["--upgrade-strategy=eager", "peft"])
    assert execd == ["peft"], execd


def test_upgrade_strategy_only_if_needed_also_dropped(shim):
    # only-if-needed is pip's default, so dropping it is a harmless no-op that
    # keeps the kept target installing normally.
    execd, _ = _run(shim, "pip", ["--upgrade-strategy", "only-if-needed", "peft"])
    assert execd == ["peft"], execd
