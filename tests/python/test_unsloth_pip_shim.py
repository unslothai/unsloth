# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

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
            # main() builds [REAL[tool]] + head + keep_args + the protected
            # constraints pair; head ends with `install`, so everything after it
            # is what we assert on. The trailing `--constraint <...>` pair is
            # injected on EVERY install; strip it here so each test asserts on its
            # own args (dedicated tests below cover the pair).
            i = exc.argv.index("install")
            execd = exc.argv[i + 1 :]
            if (
                len(execd) >= 2
                and execd[-2] == "--constraint"
                and os.path.basename(execd[-1]).startswith("unsloth-nb-protected-")
            ):
                execd = execd[:-2]
    marker = shim._marker_path.read_text() if shim._marker_path.exists() else None
    return execd, marker


# --------------------------------------------------------------------------
# Item 3541142907 -- pair -e/--editable with its target (the attached short
# `-e<target>` form from item 3541404845 is folded in here). A protected
# editable such as `pip install -e git+...unsloth...#egg=unsloth peft` must
# NOT become `pip install -e peft` (which pip rejects): the flag drops WITH
# its value, and an unprotected editable is forwarded verbatim.
# --------------------------------------------------------------------------
UNSLOTH_VCS = "git+https://github.com/unslothai/unsloth.git#egg=unsloth"

# Sentinel expectation: the whole command line is forwarded verbatim (execd == args).
KEPT = object()


@pytest.mark.parametrize(
    "args, expected",
    [
        pytest.param(["-e", UNSLOTH_VCS, "peft"], ["peft"], id = "sep-protected"),
        # nothing left to install -> no-op, no dangling -e
        pytest.param(["-e", UNSLOTH_VCS], None, id = "sep-only-protected-noop"),
        pytest.param(["-e", "./localpkg"], KEPT, id = "sep-unprotected-kept"),
        pytest.param(["--editable=" + UNSLOTH_VCS, "peft"], ["peft"], id = "inline-protected"),
        pytest.param(["--editable=./localpkg"], KEPT, id = "inline-unprotected-kept"),
        pytest.param(["-e" + UNSLOTH_VCS, "peft"], ["peft"], id = "attached-protected"),
    ],
)
def test_editable_forms(shim, args, expected):
    execd, _ = _run(shim, "pip", args)
    assert execd == (args if expected is KEPT else expected), execd


# --------------------------------------------------------------------------
# Item 3541142906 -- filter uv -P/--upgrade-package values. `uv pip install
# -P torch peft` must not let uv refresh baked torch; a pinned transformers
# upgrade selector still feeds the sidecar marker.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "args, expected, expected_marker",
    [
        pytest.param(["-P", "torch", "peft"], ["peft"], None, id = "protected-dropped"),
        pytest.param(["--upgrade-package=transformers", "peft"], ["peft"], None, id = "inline"),
        pytest.param(["-P", "transformers==4.55.0", "peft"], ["peft"], "4.55.0", id = "tf-pin"),
        pytest.param(["-P", "requests", "requests"], KEPT, None, id = "unprotected-kept"),
        # -P is not itself a target
        pytest.param(["-P", "torch"], None, None, id = "only-protected-noop"),
    ],
)
def test_upgrade_package_forms(shim, args, expected, expected_marker):
    execd, marker = _run(shim, "uv", args)
    assert execd == (args if expected is KEPT else expected), execd
    assert marker == expected_marker, marker


# --------------------------------------------------------------------------
# Item 3541142908 -- parse protected wheel basenames before URL passthrough
# (a recognised protected wheel URL/path is dropped -> no-op).
# --------------------------------------------------------------------------
NUMPY_WHEEL_URL = "https://example.com/wheels/numpy-2.1.0-cp312-cp312-linux_x86_64.whl"


@pytest.mark.parametrize(
    "args, expected",
    [
        pytest.param([TORCH_WHEEL_URL], None, id = "direct-url"),
        pytest.param(
            ["/tmp/torch-2.11.0+cu128-cp312-cp312-linux_x86_64.whl"], None, id = "local-path"
        ),
        # unsloth_zoo-*.whl normalises to unsloth-zoo, which is protected.
        pytest.param(
            ["https://example.com/unsloth_zoo-1.0-py3-none-any.whl"], None, id = "normalised"
        ),
        pytest.param([NUMPY_WHEEL_URL], KEPT, id = "unprotected-kept"),
    ],
)
def test_wheel_url_and_path_forms(shim, args, expected):
    execd, _ = _run(shim, "pip", args)
    assert execd == (args if expected is KEPT else expected), execd


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
# Item 3541404845 -- handle pip's attached short options (-rfile / -cfile /
# etc). The attached `-e<target>` case lives in test_editable_forms above.
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


def test_attached_short_upgrade_package_protected_dropped(shim):
    execd, _ = _run(shim, "uv", ["-Ptorch", "peft"])
    assert execd == ["peft"], execd
    assert "torch" not in execd and "-P" not in execd


# --------------------------------------------------------------------------
# Item 3541773143 -- a bare wheel filename (no ./ or / prefix) is still a pip
# target from the CWD, so its protected distribution must be parsed too
# (`pip install torch-2.11.0-...whl` must not reinstall torch).
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "args, expected",
    [
        pytest.param(["torch-2.11.0+cu128-cp312-cp312-linux_x86_64.whl"], None, id = "bare-torch"),
        pytest.param(["dist/torch-2.11.0-cp312-cp312-linux_x86_64.whl"], None, id = "subdir-torch"),
        pytest.param(["numpy-2.1.0-cp312-cp312-linux_x86_64.whl"], KEPT, id = "unprotected-kept"),
    ],
)
def test_bare_wheel_filename_forms(shim, args, expected):
    execd, _ = _run(shim, "pip", args)
    assert execd == (args if expected is KEPT else expected), execd


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
R_URL = "https://example.com/reqs.txt"


@pytest.mark.parametrize(
    "args, expected",
    [
        # dropped, and no dangling -r left behind
        pytest.param(["-r", R_URL], None, id = "sep-r-only-noop"),
        pytest.param(["-r", R_URL, "peft"], ["peft"], id = "sep-r-target-kept"),
        pytest.param(["--requirement=" + R_URL, "peft"], ["peft"], id = "inline-r"),
        pytest.param(["-r" + R_URL, "peft"], ["peft"], id = "attached-r"),
        pytest.param(["-c", "https://example.com/constraints.txt", "peft"], ["peft"], id = "sep-c"),
    ],
)
def test_remote_requirement_and_constraint_urls_refused(shim, args, expected):
    execd, _ = _run(shim, "pip", args)
    assert execd == expected, execd


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
@pytest.mark.parametrize(
    "args, expected, expected_marker",
    [
        pytest.param(["--reinstall-package", "torch", "peft"], ["peft"], None, id = "sep-protected"),
        pytest.param(["--reinstall-package=torch", "peft"], ["peft"], None, id = "inline-protected"),
        pytest.param(["--reinstall-package", "requests", "requests"], KEPT, None, id = "unprotected"),
        pytest.param(
            ["--reinstall-package", "transformers==4.55.0", "peft"], ["peft"], "4.55.0", id = "tf-pin"
        ),
    ],
)
def test_reinstall_package_forms(shim, args, expected, expected_marker):
    execd, marker = _run(shim, "uv", args)
    assert execd == (args if expected is KEPT else expected), execd
    assert marker == expected_marker, marker


# --------------------------------------------------------------------------
# Item 3542096750 -- parse protected source archives (sdist / zip) too.
# --------------------------------------------------------------------------
SDIST_URL = "https://files.pythonhosted.org/packages/aa/unsloth-2026.7.1.tar.gz"


@pytest.mark.parametrize(
    "args, expected",
    [
        pytest.param([SDIST_URL, "peft"], ["peft"], id = "url-protected"),
        pytest.param(["torch-2.11.0.tar.gz"], None, id = "bare-protected"),
        pytest.param(["./transformers-4.55.0.zip", "peft"], ["peft"], id = "zip-protected"),
        # flashinfer-python is protected; the name must survive the hyphen split.
        pytest.param(["flashinfer-python-0.5.0.tar.gz"], None, id = "hyphenated-name"),
        pytest.param(["numpy-2.1.0.tar.gz"], KEPT, id = "unprotected-kept"),
    ],
)
def test_source_archive_forms(shim, args, expected):
    execd, _ = _run(shim, "pip", args)
    assert execd == (args if expected is KEPT else expected), execd


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
@pytest.mark.parametrize(
    "args, expected",
    [
        pytest.param(["-U", "--upgrade-strategy", "eager", "peft"], ["-U", "peft"], id = "eager"),
        pytest.param(["--upgrade-strategy=eager", "peft"], ["peft"], id = "inline-eager"),
        # only-if-needed is pip's default, so dropping it is a harmless no-op that
        # keeps the kept target installing normally.
        pytest.param(
            ["--upgrade-strategy", "only-if-needed", "peft"], ["peft"], id = "only-if-needed"
        ),
    ],
)
def test_upgrade_strategy_forms(shim, args, expected):
    execd, _ = _run(shim, "pip", args)
    assert execd == expected, execd


# --------------------------------------------------------------------------
# Resolver-level protection: every forwarded install carries a constraints
# file pinning the installed protected packages, so a kept target's
# DEPENDENCY on an incompatible torch/transformers/etc. fails loudly instead
# of replacing the baked wheel.
# --------------------------------------------------------------------------
def _raw_execd(shim, tool, args):
    """Like _run but WITHOUT stripping the injected constraint pair."""
    argv = ["uv", "pip", "install", *args] if tool == "uv" else ["pip", "install", *args]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(shim.sys, "argv", argv)
        try:
            shim.main()
            return None
        except _Exec as exc:
            return exc.argv[exc.argv.index("install") + 1 :]


def test_forwarded_install_carries_protected_constraints(shim):
    execd = _raw_execd(shim, "pip", ["peft"])
    assert execd is not None and execd[-2] == "--constraint", execd
    pins = Path(execd[-1]).read_text(encoding = "utf-8").strip().splitlines()
    assert pins, "constraints file must pin the installed protected packages"
    assert all("==" in pin for pin in pins), pins
    names = {pin.split("==", 1)[0].lower().replace("_", "-") for pin in pins}
    protected = {"transformers"} | shim._KEEP | {"nvidia-"}
    assert all(
        n in shim._KEEP or n == "transformers" or n.startswith("nvidia-") for n in names
    ), names


def test_noop_install_gets_no_constraints(shim):
    # A cell whose only target is protected still no-ops (no exec at all).
    execd = _raw_execd(shim, "pip", ["torch"])
    assert execd is None


# --------------------------------------------------------------------------
# pip expands ${UPPERCASE} in requirements files AFTER the shim classifies the
# literal text; classification must expand the same way or `${PKG}==...` with
# PKG=torch walks straight past _KEEP.
# --------------------------------------------------------------------------
def test_env_expanded_protected_requirement_dropped(shim, tmp_path, monkeypatch):
    monkeypatch.setenv("PKG", "torch")
    req = tmp_path / "reqs.txt"
    req.write_text("${PKG}==2.11.0\nsnac==1.2.0\n", encoding = "utf-8")
    execd, _ = _run(shim, "pip", ["-r", str(req)])
    assert execd is not None and execd[0] == "-r", execd
    filtered = Path(execd[1]).read_text(encoding = "utf-8")
    assert "snac==1.2.0" in filtered
    assert "${PKG}" not in filtered and "torch" not in filtered


def test_env_expanded_transformers_pin_recorded(shim, tmp_path, monkeypatch):
    monkeypatch.setenv("TF_PKG", "transformers")
    req = tmp_path / "reqs.txt"
    req.write_text("${TF_PKG}==4.56.2\nsnac==1.2.0\n", encoding = "utf-8")
    _, marker = _run(shim, "pip", ["-r", str(req)])
    assert marker == "4.56.2"


def test_unset_env_reference_left_verbatim(shim, tmp_path, monkeypatch):
    monkeypatch.delenv("NOT_SET_ANYWHERE", raising = False)
    req = tmp_path / "reqs.txt"
    req.write_text("${NOT_SET_ANYWHERE}==1.0\nsnac==1.2.0\n", encoding = "utf-8")
    execd, _ = _run(shim, "pip", ["-r", str(req)])
    # Nothing protected detected -> the original file is forwarded unchanged
    # (pip forwards unset references verbatim too).
    assert execd == ["-r", str(req)], execd


# --------------------------------------------------------------------------
# Filtered-copy write failures fail CLOSED: the original file pins protected
# packages, so forwarding it would hand pip exactly what must be filtered.
# --------------------------------------------------------------------------
def test_filter_write_failure_refuses_original_file(shim, tmp_path, monkeypatch):
    req = tmp_path / "reqs.txt"
    req.write_text("torch==2.11.0\nsnac==1.2.0\n", encoding = "utf-8")

    def denied(*args, **kwargs):
        raise OSError(30, "Read-only file system")

    monkeypatch.setattr(shim.tempfile, "mkstemp", denied)
    with pytest.raises(SystemExit, match = "refusing to forward"):
        shim._filter_requirements_file(str(req))


def test_filter_write_failure_clean_file_passes_through(shim, tmp_path, monkeypatch):
    # A file with nothing protected never needs the temp copy, so a broken
    # TMPDIR must not block it.
    req = tmp_path / "reqs.txt"
    req.write_text("snac==1.2.0\n", encoding = "utf-8")

    def denied(*args, **kwargs):
        raise OSError(30, "Read-only file system")

    monkeypatch.setattr(shim.tempfile, "mkstemp", denied)
    path, recorded, dropped = shim._filter_requirements_file(str(req))
    assert path == str(req) and recorded is None and dropped == []


# --------------------------------------------------------------------------
# Item 3567875029 -- uv's --exact performs an exact SYNC (removes packages
# outside the kept target's closure), so it is stripped like the other
# resolver-wide destructive switches.
# --------------------------------------------------------------------------
def test_uv_exact_flag_stripped(shim):
    execd, _ = _run(shim, "uv", ["--exact", "peft"])
    assert execd == ["peft"], execd


# --------------------------------------------------------------------------
# Item 3567875023 -- a local project directory naming a protected package
# (pip install ./transformers, pip install -e ./unsloth) is filtered like the
# wheel/sdist/VCS forms: a same-version dev build slips past the constraints
# file, so the name must come from the project metadata.
# --------------------------------------------------------------------------
def _make_local_project(tmp_path, dirname, project_name):
    proj = tmp_path / dirname
    proj.mkdir()
    (proj / "pyproject.toml").write_text(f'[project]\nname = "{project_name}"\nversion = "1.0"\n')
    return str(proj)


def test_local_dir_protected_by_metadata_dropped(shim, tmp_path):
    # Directory name is innocuous; pyproject names a protected package.
    path = _make_local_project(tmp_path, "my-checkout", "transformers")
    execd, _ = _run(shim, "pip", [path, "peft"])
    assert execd == ["peft"], execd


def test_local_dir_protected_editable_dropped(shim, tmp_path):
    path = _make_local_project(tmp_path, "unsloth", "unsloth")
    execd, _ = _run(shim, "pip", ["-e", path, "peft"])
    assert execd == ["peft"], execd
    assert "-e" not in execd


def test_local_dir_basename_fallback_setup_py(shim, tmp_path):
    # No parseable name in metadata: setup.py + protected basename still drops.
    proj = tmp_path / "torch"
    proj.mkdir()
    (proj / "setup.py").write_text("from setuptools import setup\nsetup()\n")
    execd, _ = _run(shim, "pip", [str(proj), "peft"])
    assert execd == ["peft"], execd


def test_local_dir_unprotected_kept(shim, tmp_path):
    path = _make_local_project(tmp_path, "my-torch-utils", "my-torch-utils")
    execd, _ = _run(shim, "pip", [path])
    assert execd == [path], execd


def test_local_dir_without_metadata_passes_through(shim, tmp_path):
    plain = tmp_path / "datadir"
    plain.mkdir()
    execd, _ = _run(shim, "pip", [str(plain)])
    assert execd == [str(plain)], execd


# --------------------------------------------------------------------------
# Item 3592835033 -- every uv/pip value-taking flag must be in _VALUE_FLAGS.
# `uv pip install --torch-backend cu128 torch` used to drop the protected
# torch but keep the SEPARATED flag pair, exec'ing uv with no install target
# at all (uv hard-errors) instead of no-oping like the attached `=` form; and
# `--extra torch peft` misread the extra NAME "torch" as a protected target,
# leaving a dangling `--extra` that swallowed peft.


@pytest.mark.parametrize(
    "tool, flag, value",
    [
        pytest.param("uv", "--torch-backend", "cu128", id = "uv-torch-backend"),
        pytest.param("uv", "--resolution", "lowest", id = "uv-resolution"),
        pytest.param("uv", "--default-index", "https://mirror/simple", id = "uv-default-index"),
        pytest.param("uv", "--exclude-newer", "2026-01-01", id = "uv-exclude-newer"),
        pytest.param("uv", "-b", "build-constraints.txt", id = "uv-build-constraints-short"),
        pytest.param("pip", "--proxy", "http://proxy:3128", id = "pip-proxy"),
        pytest.param("pip", "--retries", "3", id = "pip-retries"),
        pytest.param("pip", "--trusted-host", "mirror.internal", id = "pip-trusted-host"),
    ],
)
def test_value_flag_protected_only_noops(shim, tool, flag, value):
    # The value must not be mistaken for an install target: with only a
    # protected target the cell is a clean no-op, never a broken exec.
    execd, _ = _run(shim, tool, [flag, value, "torch"])
    assert execd is None, execd


@pytest.mark.parametrize(
    "tool, flag, value",
    [
        pytest.param("uv", "--torch-backend", "cu128", id = "uv-torch-backend"),
        pytest.param("uv", "--resolution", "lowest", id = "uv-resolution"),
        pytest.param("pip", "--proxy", "http://proxy:3128", id = "pip-proxy"),
    ],
)
def test_value_flag_pair_forwarded_with_kept_target(shim, tool, flag, value):
    execd, _ = _run(shim, tool, [flag, value, "torch", "peft"])
    assert execd == [flag, value, "peft"], execd


def test_extra_value_is_not_a_protected_target(shim):
    # `--extra torch` names an EXTRA, not the torch package: the pair stays and
    # peft is not swallowed by a dangling --extra.
    execd, _ = _run(shim, "uv", ["--extra", "torch", "peft"])
    assert execd == ["--extra", "torch", "peft"], execd


def _value_flags_from_help(cmd):
    import re
    import subprocess

    out = subprocess.run(cmd, capture_output = True, text = True).stdout
    flags = set()
    for m in re.finditer(r"^\s+(-\w)?,?\s*(--[\w-]+)[= ]<", out, re.M):
        if m.group(1):
            flags.add(m.group(1))
        flags.add(m.group(2))
    for m in re.finditer(r"^\s+(-\w) <", out, re.M):
        flags.add(m.group(1))
    return flags


# The help-derived drift guards are OPT-IN: repo CI runs whatever pip/uv are
# current that week, so a hard assert here turns every upstream flag addition
# into an unrelated red PR. The authoritative check runs at image BUILD time
# against the exact baked tools (`unsloth_pip_shim.py
# --unsloth-selfcheck-value-flags` in the Dockerfile verify step); set
# UNSLOTH_SHIM_FLAG_DRIFT_CHECK=1 to run these locally.
_DRIFT_OPT_IN = os.environ.get("UNSLOTH_SHIM_FLAG_DRIFT_CHECK") == "1"


@pytest.mark.skipif(not _DRIFT_OPT_IN, reason = "opt-in: UNSLOTH_SHIM_FLAG_DRIFT_CHECK=1")
def test_pip_help_value_flags_all_classified(shim):
    # Drift guard: every value-taking flag `pip install --help` documents must
    # be classified as value-taking by the shim, or its VALUE is misread as an
    # install target (see --torch-backend above).
    known = shim._VALUE_FLAGS | shim._DROP_VALUE_FLAGS
    missing = _value_flags_from_help([sys.executable, "-m", "pip", "install", "--help"]) - known
    assert not missing, f"value flags missing from _VALUE_FLAGS: {sorted(missing)}"


@pytest.mark.skipif(
    not _DRIFT_OPT_IN or not __import__("shutil").which("uv"),
    reason = "opt-in: UNSLOTH_SHIM_FLAG_DRIFT_CHECK=1 (and uv installed)",
)
def test_uv_help_value_flags_all_classified(shim):
    known = shim._VALUE_FLAGS | shim._DROP_VALUE_FLAGS
    missing = _value_flags_from_help(["uv", "pip", "install", "--help"]) - known
    assert not missing, f"value flags missing from _VALUE_FLAGS: {sorted(missing)}"


# --------------------------------------------------------------------------
# Item 3592947879 -- a VCS @ref may itself contain a slash (@feature/foo);
# the ref must be stripped from the PATH before the last-segment split, or
# `git+https://github.com/unslothai/unsloth.git@feature/foo` canonicalizes as
# "foo" and a protected repo installed from a branch dodges _KEEP.


@pytest.mark.parametrize(
    "url",
    [
        pytest.param(
            "git+https://github.com/unslothai/unsloth.git@feature/foo", id = "https-slash-ref"
        ),
        pytest.param(
            "git+ssh://git@github.com/unslothai/unsloth.git@feature/foo",
            id = "ssh-userinfo-and-slash-ref",
        ),
        pytest.param("git+https://github.com/unslothai/unsloth.git@v2026.7", id = "plain-tag-ref"),
        pytest.param("git+https://github.com/unslothai/unsloth.git", id = "no-ref"),
    ],
)
def test_vcs_slash_ref_still_protected(shim, url):
    execd, _ = _run(shim, "pip", [url, "peft"])
    assert execd == ["peft"], execd


def test_vcs_slash_ref_unprotected_kept(shim):
    url = "git+https://github.com/someorg/sometool.git@feature/foo"
    execd, _ = _run(shim, "pip", [url])
    assert execd == [url], execd
