# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""Regression tests for docker/unsloth_pip_shim.py.

Drives main() with UNSLOTH_NB_SHIM=1 and captures the os.execv command, so the
assertions are on what actually reaches the real pip/uv.
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
    """Raised by the patched os.execv so main() stops here."""

    def __init__(self, path, argv):
        self.path = path
        self.argv = list(argv)


class _BakedImage:
    """Stands in for _installed_names() on an image where every bake succeeded.

    Only `in` is asked of the return value, so answering the prefix rule here keeps
    nvidia-* wheels present too, which a plain set of _KEEP cannot express.
    """

    def __init__(self, mod):
        self._mod = mod

    def __contains__(self, name):
        # transformers is baked too; it is out of _KEEP only because the sidecar
        # replaces its VERSION rather than the distribution
        if name == "transformers":
            return True
        return name in self._mod._KEEP or name.startswith(self._mod._KEEP_PREFIX)


@pytest.fixture()
def shim(tmp_path, monkeypatch):
    """Fresh shim copy, marker in tmp_path, os.execv patched to capture the exec."""
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
    # the shim now skips a protected package only when it is really installed, so pin
    # the fully baked image here: otherwise these assertions read the CI venv, which
    # has no torchcodec, and pass or fail on the runner rather than on the shim
    monkeypatch.setattr(mod, "_installed_names", lambda: _BakedImage(mod))
    mod._marker_path = marker
    return mod


def _run(shim, tool, args):
    """(args after `install` that reached the real tool or None, recorded transformers
    version or None). The injected trailing `--constraint <file>` pair is stripped."""
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


UNSLOTH_VCS = "git+https://github.com/unslothai/unsloth.git#egg=unsloth"

KEPT = object()


@pytest.mark.parametrize(
    "args, expected",
    [
        pytest.param(["-e", UNSLOTH_VCS, "snac"], ["snac"], id = "sep-protected"),
        # a protected editable drops the flag WITH its value: never a dangling `-e`
        pytest.param(["-e", UNSLOTH_VCS], None, id = "sep-only-protected-noop"),
        pytest.param(["-e", "./localpkg"], KEPT, id = "sep-unprotected-kept"),
        pytest.param(["--editable=" + UNSLOTH_VCS, "snac"], ["snac"], id = "inline-protected"),
        pytest.param(["--editable=./localpkg"], KEPT, id = "inline-unprotected-kept"),
        pytest.param(["-e" + UNSLOTH_VCS, "snac"], ["snac"], id = "attached-protected"),
    ],
)
def test_editable_forms(shim, args, expected):
    execd, _ = _run(shim, "pip", args)
    assert execd == (args if expected is KEPT else expected), execd


@pytest.mark.parametrize(
    "args, expected, expected_marker",
    [
        pytest.param(["-P", "torch", "snac"], ["snac"], None, id = "protected-dropped"),
        pytest.param(["--upgrade-package=transformers", "snac"], ["snac"], None, id = "inline"),
        pytest.param(["-P", "transformers==4.55.0", "snac"], ["snac"], "4.55.0", id = "tf-pin"),
        pytest.param(["-P", "requests", "requests"], KEPT, None, id = "unprotected-kept"),
        pytest.param(["-P", "torch"], None, None, id = "only-protected-noop"),
    ],
)
def test_upgrade_package_forms(shim, args, expected, expected_marker):
    execd, marker = _run(shim, "uv", args)
    assert execd == (args if expected is KEPT else expected), execd
    assert marker == expected_marker, marker


NUMPY_WHEEL_URL = "https://example.com/wheels/numpy-2.1.0-cp312-cp312-linux_x86_64.whl"


@pytest.mark.parametrize(
    "args, expected",
    [
        pytest.param([TORCH_WHEEL_URL], None, id = "direct-url"),
        pytest.param(
            ["/tmp/torch-2.11.0+cu128-cp312-cp312-linux_x86_64.whl"], None, id = "local-path"
        ),
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
    assert execd is not None and execd[0] == "-r"
    filtered = Path(execd[1]).read_text(encoding = "utf-8")
    assert "snac==1.2.0" in filtered
    assert "torch" not in filtered


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
    assert "unsloth" not in filtered


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


def test_nested_constraint_transformers_pin_not_recorded(shim, tmp_path):
    constraints = tmp_path / "constraints.txt"
    constraints.write_text("transformers==4.55.0\n", encoding = "utf-8")
    req = tmp_path / "reqs.txt"
    req.write_text("-c constraints.txt\nsnac==1.2.0\n", encoding = "utf-8")
    execd, marker = _run(shim, "pip", ["-r", str(req)])
    assert execd is not None and execd[0] == "-r", execd
    assert marker is None, marker


def test_nested_requirement_transformers_pin_recorded(shim, tmp_path):
    nested = tmp_path / "nested.txt"
    nested.write_text("transformers==4.55.0\n", encoding = "utf-8")
    req = tmp_path / "reqs.txt"
    req.write_text("-r nested.txt\nsnac==1.2.0\n", encoding = "utf-8")
    execd, marker = _run(shim, "pip", ["-r", str(req)])
    assert execd is not None and execd[0] == "-r", execd
    assert marker == "4.55.0", marker


def test_attached_short_requirement_file_filtered(shim, tmp_path):
    # attached `-rreqs.txt` must filter the file AND count as a target, or the cell no-ops
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
    execd, _ = _run(shim, "pip", ["-c" + str(constraints), "snac"])
    assert execd is not None and execd[0] == "-c", execd
    assert "snac" in execd
    filtered = Path(execd[1]).read_text(encoding = "utf-8")
    assert "torch" not in filtered


def test_attached_short_upgrade_package_protected_dropped(shim):
    execd, _ = _run(shim, "uv", ["-Ptorch", "snac"])
    assert execd == ["snac"], execd
    assert "torch" not in execd and "-P" not in execd


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


def test_vcs_url_without_egg_protected_dropped(shim):
    execd, _ = _run(shim, "pip", ["git+https://github.com/huggingface/transformers.git", "snac"])
    assert execd == ["snac"], execd


def test_vcs_url_without_egg_with_ref_dropped(shim):
    execd, _ = _run(shim, "pip", ["git+https://github.com/unslothai/unsloth-zoo.git@main", "snac"])
    assert execd == ["snac"], execd


def test_vcs_url_without_egg_unprotected_kept(shim):
    url = "git+https://github.com/someone/coolpkg.git"
    execd, _ = _run(shim, "pip", [url])
    assert execd == [url], execd


# remote requirement/constraint files are refused: their pins cannot be inspected first
R_URL = "https://example.com/reqs.txt"


@pytest.mark.parametrize(
    "args, expected",
    [
        pytest.param(["-r", R_URL], None, id = "sep-r-only-noop"),
        pytest.param(["-r", R_URL, "snac"], ["snac"], id = "sep-r-target-kept"),
        pytest.param(["--requirement=" + R_URL, "snac"], ["snac"], id = "inline-r"),
        pytest.param(["-r" + R_URL, "snac"], ["snac"], id = "attached-r"),
        pytest.param(["-c", "https://example.com/constraints.txt", "snac"], ["snac"], id = "sep-c"),
    ],
)
def test_remote_requirement_and_constraint_urls_refused(shim, args, expected):
    execd, _ = _run(shim, "pip", args)
    assert execd == expected, execd


def test_nested_remote_include_dropped(shim, tmp_path):
    req = tmp_path / "reqs.txt"
    req.write_text("-r https://example.com/evil.txt\nsnac==1.2.0\n", encoding = "utf-8")
    execd, _ = _run(shim, "pip", ["-r", str(req)])
    assert execd is not None and execd[0] == "-r", execd
    filtered = Path(execd[1]).read_text(encoding = "utf-8")
    assert "snac==1.2.0" in filtered
    assert "example.com" not in filtered and "://" not in filtered


# resolver-wide reinstall / ignore-installed flags cannot rebuild satisfied baked deps
def test_force_reinstall_flag_stripped(shim):
    execd, _ = _run(shim, "pip", ["--force-reinstall", "snac"])
    assert execd == ["snac"], execd


def test_ignore_installed_short_flag_stripped(shim):
    execd, _ = _run(shim, "pip", ["-I", "snac"])
    assert execd == ["snac"], execd


def test_uv_reinstall_flag_stripped(shim):
    execd, _ = _run(shim, "uv", ["--reinstall", "snac"])
    assert execd == ["snac"], execd


@pytest.mark.parametrize(
    "args, expected, expected_marker",
    [
        pytest.param(["--reinstall-package", "torch", "snac"], ["snac"], None, id = "sep-protected"),
        pytest.param(["--reinstall-package=torch", "snac"], ["snac"], None, id = "inline-protected"),
        pytest.param(["--reinstall-package", "requests", "requests"], KEPT, None, id = "unprotected"),
        pytest.param(
            ["--reinstall-package", "transformers==4.55.0", "snac"], ["snac"], "4.55.0", id = "tf-pin"
        ),
    ],
)
def test_reinstall_package_forms(shim, args, expected, expected_marker):
    execd, marker = _run(shim, "uv", args)
    assert execd == (args if expected is KEPT else expected), execd
    assert marker == expected_marker, marker


SDIST_URL = "https://files.pythonhosted.org/packages/aa/unsloth-2026.7.1.tar.gz"


@pytest.mark.parametrize(
    "args, expected",
    [
        pytest.param([SDIST_URL, "snac"], ["snac"], id = "url-protected"),
        pytest.param(["torch-2.11.0.tar.gz"], None, id = "bare-protected"),
        pytest.param(["./transformers-4.55.0.zip", "snac"], ["snac"], id = "zip-protected"),
        # a hyphenated protected name must survive the sdist hyphen split
        pytest.param(["flashinfer-python-0.5.0.tar.gz"], None, id = "hyphenated-name"),
        pytest.param(["numpy-2.1.0.tar.gz"], KEPT, id = "unprotected-kept"),
    ],
)
def test_source_archive_forms(shim, args, expected):
    execd, _ = _run(shim, "pip", args)
    assert execd == (args if expected is KEPT else expected), execd


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
    execd, _ = _run(shim, "uv", ["--constraints", str(constraints), "snac"])
    assert execd is not None and execd[0] == "--constraints", execd
    assert "snac" in execd
    filtered = Path(execd[1]).read_text(encoding = "utf-8")
    assert "torch" not in filtered


# --upgrade-strategy eager would let a kept target rebuild satisfied baked deps
@pytest.mark.parametrize(
    "args, expected",
    [
        pytest.param(["-U", "--upgrade-strategy", "eager", "snac"], ["-U", "snac"], id = "eager"),
        pytest.param(["--upgrade-strategy=eager", "snac"], ["snac"], id = "inline-eager"),
        pytest.param(
            ["--upgrade-strategy", "only-if-needed", "snac"], ["snac"], id = "only-if-needed"
        ),
    ],
)
def test_upgrade_strategy_forms(shim, args, expected):
    execd, _ = _run(shim, "pip", args)
    assert execd == expected, execd


# every forwarded install carries a constraints file, so an incompatible dependency
# fails loudly instead of replacing the wheel
def _raw_execd(shim, tool, args):
    argv = ["uv", "pip", "install", *args] if tool == "uv" else ["pip", "install", *args]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(shim.sys, "argv", argv)
        try:
            shim.main()
            return None
        except _Exec as exc:
            return exc.argv[exc.argv.index("install") + 1 :]


class _FakeDist:
    def __init__(self, name, version):
        self.metadata = {"Name": name}
        self.version = version


def _fake_distributions(monkeypatch, *pairs):
    """Pin what _protected_constraints_file sees as INSTALLED, else the assertions
    depend on the ambient venv. The shim imports it inside the function, so patch
    it at its source."""
    monkeypatch.setattr(
        "importlib.metadata.distributions",
        lambda: [_FakeDist(n, v) for n, v in pairs],
    )


def test_forwarded_install_carries_protected_constraints(shim, monkeypatch):
    _fake_distributions(monkeypatch, ("transformers", "5.14.1"), ("trl", "0.24.0"))
    execd = _raw_execd(shim, "pip", ["snac"])
    assert execd is not None, "an unprotected target must still be forwarded"
    assert len(execd) >= 2 and execd[-2] == "--constraint", execd
    pins = Path(execd[-1]).read_text(encoding = "utf-8").strip().splitlines()
    assert pins, "constraints file must pin the installed protected packages"
    assert all("==" in pin for pin in pins), pins
    names = {pin.split("==", 1)[0].lower().replace("_", "-") for pin in pins}
    protected = {"transformers"} | shim._KEEP | {"nvidia-"}
    assert all(
        n in shim._KEEP or n == "transformers" or n.startswith("nvidia-") for n in names
    ), names


def test_forwarded_install_without_protected_packages_has_no_constraints(shim, monkeypatch):
    _fake_distributions(monkeypatch, ("snac", "1.2.1"))
    execd = _raw_execd(shim, "pip", ["snac"])
    assert execd is not None, "the install must still be forwarded"
    assert "--constraint" not in execd, execd


def test_noop_install_gets_no_constraints(shim):
    execd = _raw_execd(shim, "pip", ["torch"])
    assert execd is None


# pip expands ${UPPERCASE} in requirements files after the shim classifies the literal
# text, so classification must expand the same way or `${PKG}==` walks past _KEEP
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
    assert execd == ["-r", str(req)], execd


# a filtered-copy write failure must fail CLOSED: forwarding the original hands pip
# exactly what must be filtered
def test_filter_write_failure_refuses_original_file(shim, tmp_path, monkeypatch):
    req = tmp_path / "reqs.txt"
    req.write_text("torch==2.11.0\nsnac==1.2.0\n", encoding = "utf-8")

    def denied(*args, **kwargs):
        raise OSError(30, "Read-only file system")

    monkeypatch.setattr(shim.tempfile, "mkstemp", denied)
    with pytest.raises(SystemExit, match = "refusing to forward"):
        shim._filter_requirements_file(str(req))


# transformers is already out of the real arguments by the time the marker is written,
# so a swallowed failure reported a plain success while the model cells went on
# importing the baked version. It must warn instead: aborting would turn an unwritable
# path the user cannot act on into a hard notebook failure, but silence is worse.
def test_marker_write_failure_warns_and_does_not_claim_success(shim, monkeypatch, capsys):
    def denied(*args, **kwargs):
        raise OSError(30, "Read-only file system")

    monkeypatch.setattr(shim.os, "makedirs", denied)
    monkeypatch.setattr(shim.sys, "argv", ["pip", "install", "transformers==4.99.0"])
    shim.main()
    err = capsys.readouterr()
    assert "WARNING" in err.err and "will NOT" in err.err
    assert "will activate its sidecar" not in err.out


# dirname() is "" for a bare relative MARKER and makedirs("") raises FileNotFoundError,
# an OSError: without the guard, failing closed would turn that config into a hard abort
def test_bare_relative_marker_still_records(shim, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(shim, "MARKER", "requested_transformers")
    monkeypatch.setattr(shim.sys, "argv", ["pip", "install", "transformers==4.99.0"])
    shim.main()
    assert (tmp_path / "requested_transformers").read_text() == "4.99.0"


def test_filter_write_failure_clean_file_passes_through(shim, tmp_path, monkeypatch):
    req = tmp_path / "reqs.txt"
    req.write_text("snac==1.2.0\n", encoding = "utf-8")

    def denied(*args, **kwargs):
        raise OSError(30, "Read-only file system")

    monkeypatch.setattr(shim.tempfile, "mkstemp", denied)
    path, recorded, dropped = shim._filter_requirements_file(str(req))
    assert path == str(req) and recorded is None and dropped == []


# the constraints file is the ONLY thing holding the baked stack when a forwarded
# package pulls an incompatible transitive pin, and `_extras_only_target` forwards a
# protected `name[extras]` because of it, so its write must fail CLOSED too instead of
# reading like "nothing to protect"
def test_protected_constraints_write_failure_refuses_install(shim, monkeypatch):
    _fake_distributions(monkeypatch, ("torch", "2.11.0"))

    def denied(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(shim.tempfile, "mkstemp", denied)
    with pytest.raises(SystemExit, match = "refusing to install"):
        _raw_execd(shim, "pip", ["snac"])


def test_protected_metadata_enumeration_failure_refuses_install(shim, monkeypatch):
    def denied():
        raise OSError(5, "Input/output error")

    monkeypatch.setattr("importlib.metadata.distributions", denied)
    with pytest.raises(SystemExit, match = "refusing to install"):
        _raw_execd(shim, "pip", ["snac"])


# an interrupted install leaves a `.dist-info` with no METADATA behind, and that one
# unreadable dist must not take the pins for every other package down with it
def test_one_unreadable_dist_does_not_drop_the_other_pins(shim, monkeypatch):
    class _BrokenDist:
        @property
        def metadata(self):
            raise KeyError("Name")

        @property
        def version(self):
            raise KeyError("Version")

    monkeypatch.setattr(
        "importlib.metadata.distributions",
        lambda: [_BrokenDist(), _FakeDist("torch", "2.11.0"), _FakeDist("snac", "1.2.1")],
    )
    execd = _raw_execd(shim, "pip", ["snac"])
    assert execd is not None and execd[-2] == "--constraint", execd
    pins = Path(execd[-1]).read_text(encoding = "utf-8").split()
    assert pins == ["torch==2.11.0"], pins


# uv --exact is an exact SYNC: it removes packages outside the kept target's closure
def test_uv_exact_flag_stripped(shim):
    execd, _ = _run(shim, "uv", ["--exact", "snac"])
    assert execd == ["snac"], execd


# a local project dir naming a protected package needs its name from the project
# metadata: a same-version dev build slips past the constraints file
def _make_local_project(tmp_path, dirname, project_name):
    proj = tmp_path / dirname
    proj.mkdir()
    (proj / "pyproject.toml").write_text(f'[project]\nname = "{project_name}"\nversion = "1.0"\n')
    return str(proj)


def test_local_dir_protected_by_metadata_dropped(shim, tmp_path):
    path = _make_local_project(tmp_path, "my-checkout", "transformers")
    execd, _ = _run(shim, "pip", [path, "snac"])
    assert execd == ["snac"], execd


def test_local_dir_protected_editable_dropped(shim, tmp_path):
    path = _make_local_project(tmp_path, "unsloth", "unsloth")
    execd, _ = _run(shim, "pip", ["-e", path, "snac"])
    assert execd == ["snac"], execd
    assert "-e" not in execd


def test_local_dir_basename_fallback_setup_py(shim, tmp_path):
    proj = tmp_path / "torch"
    proj.mkdir()
    (proj / "setup.py").write_text("from setuptools import setup\nsetup()\n")
    execd, _ = _run(shim, "pip", [str(proj), "snac"])
    assert execd == ["snac"], execd


def test_local_dir_unprotected_kept(shim, tmp_path):
    path = _make_local_project(tmp_path, "my-torch-utils", "my-torch-utils")
    execd, _ = _run(shim, "pip", [path])
    assert execd == [path], execd


def test_local_dir_without_metadata_passes_through(shim, tmp_path):
    plain = tmp_path / "datadir"
    plain.mkdir()
    execd, _ = _run(shim, "pip", [str(plain)])
    assert execd == [str(plain)], execd


# every uv/pip value-taking flag must be in _VALUE_FLAGS, or its VALUE is misread as
# an install target (`--extra torch snac` swallowed snac behind a dangling --extra)
@pytest.mark.parametrize(
    "tool, flag, value",
    [
        pytest.param("uv", "--torch-backend", "cu128", id = "uv-torch-backend"),
        pytest.param("uv", "--resolution", "lowest", id = "uv-resolution"),
        pytest.param("uv", "--default-index", "https://mirror/simple", id = "uv-default-index"),
        pytest.param("uv", "--exclude-newer", "2026-01-01", id = "uv-exclude-newer"),
        pytest.param("uv", "-b", "build-constraints.txt", id = "uv-build-constraints-short"),
        pytest.param("uv", "--prerelease-package", "snac", id = "uv-prerelease-package"),
        pytest.param("pip", "--proxy", "http://proxy:3128", id = "pip-proxy"),
        pytest.param("pip", "--retries", "3", id = "pip-retries"),
        pytest.param("pip", "--trusted-host", "mirror.internal", id = "pip-trusted-host"),
    ],
)
def test_value_flag_protected_only_noops(shim, tool, flag, value):
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
    execd, _ = _run(shim, tool, [flag, value, "torch", "snac"])
    assert execd == [flag, value, "snac"], execd


def test_extra_value_is_not_a_protected_target(shim):
    execd, _ = _run(shim, "uv", ["--extra", "torch", "snac"])
    assert execd == ["--extra", "torch", "snac"], execd


def test_uv_per_package_value_flags_classified(shim):
    # the whole family, so the next sibling uv adds fails here, not in the image build
    known = shim._VALUE_FLAGS | shim._DROP_VALUE_FLAGS
    family = {
        "--config-settings-package",
        "--exclude-newer-package",
        "--no-build-isolation-package",
        "--no-editable-package",
        "--no-sources-package",
        "--prerelease-package",
        "--refresh-package",
        "--reinstall-package",
        "--upgrade-package",
    }
    assert family <= known, sorted(family - known)


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


# opt-in: repo CI runs whatever pip/uv are current, so a hard assert would redden every
# upstream flag addition. The image build's --unsloth-selfcheck-value-flags is
# authoritative, being run against the baked tools.
_DRIFT_OPT_IN = os.environ.get("UNSLOTH_SHIM_FLAG_DRIFT_CHECK") == "1"


@pytest.mark.skipif(not _DRIFT_OPT_IN, reason = "opt-in: UNSLOTH_SHIM_FLAG_DRIFT_CHECK=1")
def test_pip_help_value_flags_all_classified(shim):
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


# a VCS @ref may contain a slash (@feature/foo); strip it before the last-segment
# split, else the ref's basename dodges _KEEP
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
    execd, _ = _run(shim, "pip", [url, "snac"])
    assert execd == ["snac"], execd


def test_vcs_slash_ref_unprotected_kept(shim):
    url = "git+https://github.com/someorg/sometool.git@feature/foo"
    execd, _ = _run(shim, "pip", [url])
    assert execd == [url], execd


# the constraints pair must go BEFORE `--`: both pip and uv parse everything after
# the terminator as a requirement and reject "Invalid requirement: '--constraint'"
def _execd_full(shim, tool, args):
    argv = ["uv", "pip", "install", *args] if tool == "uv" else ["pip", "install", *args]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(shim.sys, "argv", argv)
        with pytest.raises(_Exec) as exc:
            shim.main()
    tail = exc.value.argv[exc.value.argv.index("install") + 1 :]
    return tail


@pytest.mark.parametrize("tool", ["pip", "uv"])
def test_constraints_precede_the_end_of_options_marker(shim, tool):
    execd = _execd_full(shim, tool, ["--", "snac"])
    assert "--constraint" in execd, execd
    assert execd.index("--constraint") < execd.index("--"), (
        f"the constraints pair lands after `--`, so the real tool parses "
        f"--constraint as a requirement and the cell fails: {execd}"
    )
    assert execd[execd.index("--") :] == ["--", "snac"], execd


@pytest.mark.parametrize("tool", ["pip", "uv"])
def test_end_of_options_marker_still_protects_the_baked_stack(shim, tool):
    execd = _execd_full(shim, tool, ["--", "torch", "snac"])
    assert "torch" not in execd, execd
    assert execd[execd.index("--") :] == ["--", "snac"], execd


@pytest.mark.parametrize("tool", ["pip", "uv"])
def test_without_a_terminator_the_pair_is_still_appended_last(shim, tool):
    execd = _execd_full(shim, tool, ["snac"])
    assert execd[0] == "snac", execd
    assert execd[-2] == "--constraint", execd


# PEP 503: a name compares equal under any run of `-`, `_` or `.`, so `unsloth.zoo`
# IS `unsloth-zoo`. Every _canon early return must normalise, not just collapse "_".
@pytest.mark.parametrize(
    "token",
    [
        "unsloth.zoo @ git+https://github.com/unslothai/unsloth_zoo",
        "git+https://github.com/unslothai/unsloth_zoo#egg=unsloth.zoo",
        "https://example.invalid/unsloth.zoo-1.0-py3-none-any.whl",
        "unsloth.zoo-1.0.tar.gz",
        "git+https://github.com/unslothai/unsloth.zoo",
        "unsloth.zoo==2026.9.1",
        "unsloth__zoo==2026.9.1",
        "UNSLOTH.ZOO==2026.9.1",
    ],
)
def test_dotted_spellings_of_a_protected_package_are_still_protected(shim, token):
    assert shim._canon(token) == "unsloth-zoo", token
    assert shim._canon(token) in shim._KEEP, token


@pytest.mark.parametrize(
    "token, expected",
    [
        ("nvidia.cublas-cu12==1.0", "nvidia-cublas-cu12"),
        ("huggingface.hub==1.0", "huggingface-hub"),
    ],
)
def test_prefix_and_plain_matches_normalize_too(shim, token, expected):
    name = shim._canon(token)
    assert name == expected, token
    assert name in shim._KEEP or name.startswith(shim._KEEP_PREFIX), token


def test_normalization_does_not_merge_distinct_distributions(shim):
    for token in ("torch-directml==0.2", "torch_tensorrt==2.0", "torchsde==0.2"):
        assert shim._canon(token) not in shim._KEEP, token


# --group and --requirements-from-script ARE the install target, with no package on
# the command line; as ordinary option/value pairs the shim no-op'd while printing "ok"
@pytest.mark.parametrize(
    "args",
    [
        ["--group", "test"],
        ["--group=test"],
        ["--group", "sub/pyproject.toml:dev"],
        ["--requirements-from-script", "demo.py"],
        ["--requirements-from-script=demo.py"],
    ],
)
def test_dependency_group_flags_are_install_targets(shim, args):
    execd = _execd_full(shim, "pip", args)
    for tok in args:
        assert tok in execd, (args, execd)


@pytest.mark.parametrize("args", [["--no-deps"], ["--quiet"], ["torch"]])
def test_flag_only_or_fully_protected_cells_still_no_op(shim, args):
    """The guard must keep no-op'ing where there really is nothing to install."""
    argv = ["pip", "install", *args]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(shim.sys, "argv", argv)
        shim.main()


# --- hashed lock files: a requirement continues across physical lines --------------
# `pip-compile --generate-hashes` / `uv pip compile --generate-hashes` emit
#     torch==2.11.0 \
#         --hash=sha256:... \
#         --hash=sha256:...
# Filtering physical lines dropped only the first row and published the orphaned
# `--hash` rows; uv then refuses the whole file with
# "Unexpected '-', expected '-c', '-e', '-r' or the start of a requirement".
HASHED_LOCK = (
    "colorama==0.4.6 \\\n"
    "    --hash=sha256:08695f5cb7ed6e0531a20572697297273c47b8cae5a63ffc6d6ed5c201be6e44 \\\n"
    "    --hash=sha256:4f1d9991f5acc0ca119f9d443620b77f9d6b33703e51011c16baf57afb285fc6\n"
    "    # via -r in.txt\n"
    "torch==2.11.0 \\\n"
    "    --hash=sha256:1111111111111111111111111111111111111111111111111111111111111111 \\\n"
    "    --hash=sha256:2222222222222222222222222222222222222222222222222222222222222222\n"
    "idna==3.18 \\\n"
    "    --hash=sha256:7f952cbe720b688055e3f87de14f5c3e5fdaa8bc3928985c4077ca689de849a2\n"
)


def test_hashed_requirement_drops_its_continuation_lines(shim, tmp_path):
    req = tmp_path / "locked.txt"
    req.write_text(HASHED_LOCK, encoding = "utf-8")
    execd, _ = _run(shim, "uv", ["-r", str(req)])
    assert execd is not None and execd[0] == "-r", execd
    filtered = Path(execd[1]).read_text(encoding = "utf-8")
    # no ORPHANED option row: every `--hash` must continue the line above it
    prev = ""
    for line in filtered.splitlines():
        if line.strip().startswith("--hash"):
            assert prev.rstrip().endswith("\\"), filtered
        prev = line
    assert "torch" not in filtered, filtered
    assert "1111111111111111" not in filtered and "2222222222222222" not in filtered, filtered
    # the untouched requirements keep BOTH their pin and every hash row
    assert "colorama==0.4.6 \\" in filtered, filtered
    assert "08695f5cb7ed6e0531a20572697297273c47b8cae5a63ffc6d6ed5c201be6e44" in filtered
    assert "4f1d9991f5acc0ca119f9d443620b77f9d6b33703e51011c16baf57afb285fc6" in filtered
    assert "idna==3.18 \\" in filtered, filtered
    assert "7f952cbe720b688055e3f87de14f5c3e5fdaa8bc3928985c4077ca689de849a2" in filtered


def test_hashed_transformers_still_records_its_version(shim, tmp_path):
    req = tmp_path / "locked.txt"
    req.write_text(
        "transformers==4.55.0 \\\n"
        "    --hash=sha256:3333333333333333333333333333333333333333333333333333333333333333\n"
        "snac==1.2.0\n",
        encoding = "utf-8",
    )
    execd, marker = _run(shim, "pip", ["-r", str(req)])
    assert marker == "4.55.0", marker
    filtered = Path(execd[1]).read_text(encoding = "utf-8")
    assert "snac==1.2.0" in filtered
    assert "transformers" not in filtered and "3333333333333333" not in filtered, filtered


def test_continued_protected_requirement_leaves_no_orphan_specifier(shim, tmp_path):
    req = tmp_path / "reqs.txt"
    req.write_text("torch \\\n    ==2.11.0\nsnac==1.2.0\n", encoding = "utf-8")
    execd, _ = _run(shim, "pip", ["-r", str(req)])
    filtered = Path(execd[1]).read_text(encoding = "utf-8")
    assert "==2.11.0" not in filtered, filtered
    assert "snac==1.2.0" in filtered, filtered


# --- extras of a baked package ----------------------------------------------------
# `pip install "datasets[audio]"` against an installed datasets ADDS the extra's
# dependencies, it does not replace datasets, so dropping the token lost every one of
# them and still printed "ok". The injected --constraint pins the baked version.
@pytest.mark.parametrize(
    "arg, expected",
    [
        pytest.param("datasets[audio]", "datasets[audio]", id = "bare"),
        pytest.param("unsloth[studio]", "unsloth[studio]", id = "unsloth-studio"),
        # the pin is what would replace the bake, so only the pin is dropped
        pytest.param("datasets[audio]==4.3.0", "datasets[audio]", id = "pinned"),
        pytest.param("datasets[audio, vision]", "datasets[audio, vision]", id = "multi"),
        pytest.param(
            'datasets[audio]; python_version >= "3.10"',
            'datasets[audio] ; python_version >= "3.10"',
            id = "marker",
        ),
    ],
)
def test_extras_of_baked_package_are_forwarded(shim, arg, expected):
    execd, _ = _run(shim, "pip", [arg])
    assert execd == [expected], execd


@pytest.mark.parametrize(
    "arg",
    [
        # a direct reference REPLACES the distribution, extras or not
        "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git",
        "torch[opt] @ https://example.com/torch-2.11.0-py3-none-any.whl",
    ],
)
def test_extras_with_direct_reference_still_dropped(shim, arg):
    execd, _ = _run(shim, "pip", [arg])
    assert execd is None, execd


def test_extras_of_baked_package_in_requirements_file(shim, tmp_path):
    req = tmp_path / "reqs.txt"
    req.write_text("datasets[audio]==4.3.0\ntorch==2.11.0\nsnac==1.2.0\n", encoding = "utf-8")
    execd, _ = _run(shim, "pip", ["-r", str(req)])
    filtered = Path(execd[1]).read_text(encoding = "utf-8")
    assert "datasets[audio]" in filtered, filtered
    assert "datasets[audio]==4.3.0" not in filtered, filtered
    assert "torch" not in filtered, filtered
    assert "snac==1.2.0" in filtered, filtered


# transformers is the one protected name outside _KEEP, because the sidecar replaces its
# VERSION rather than the distribution. `pip install "transformers[deepspeed]"` is a
# documented HF install line, so the extras must be forwarded here exactly as they are
# for every _KEEP package; dropping the whole token recorded the pin and then printed
# "nothing to install ... ok" while deepspeed never arrived.
@pytest.mark.parametrize(
    "arg, expected, expected_marker",
    [
        pytest.param(
            "transformers[deepspeed]==5.5.0", "transformers[deepspeed]", "5.5.0", id = "pinned"
        ),
        pytest.param("transformers[torch]", "transformers[torch]", None, id = "bare"),
        pytest.param(
            "transformers[torch, sentencepiece]",
            "transformers[torch, sentencepiece]",
            None,
            id = "multi",
        ),
    ],
)
def test_transformers_extras_are_forwarded_and_the_pin_is_still_recorded(
    shim, arg, expected, expected_marker
):
    execd, marker = _run(shim, "pip", [arg])
    assert execd == [expected], execd
    assert marker == expected_marker, marker


def test_transformers_extras_direct_reference_still_dropped(shim):
    # a direct reference REPLACES transformers, which is what the sidecar is for
    execd, marker = _run(
        shim, "pip", ["transformers[torch] @ git+https://github.com/huggingface/transformers"]
    )
    assert execd is None, execd
    assert marker is None, marker


def test_transformers_extras_in_requirements_file(shim, tmp_path):
    req = tmp_path / "reqs.txt"
    req.write_text("transformers[deepspeed]==5.5.0\nsnac==1.2.0\n", encoding = "utf-8")
    execd, marker = _run(shim, "pip", ["-r", str(req)])
    filtered = Path(execd[1]).read_text(encoding = "utf-8")
    assert "transformers[deepspeed]\n" in filtered, filtered
    assert "5.5.0" not in filtered, filtered
    assert "snac==1.2.0" in filtered, filtered
    assert marker == "5.5.0", marker


def test_transformers_extras_are_dropped_when_transformers_is_not_installed(shim, monkeypatch):
    """No baked transformers means no version to hold in place: forwarding the extras
    would install whatever transformers pip resolves, so keep the sidecar route."""
    monkeypatch.setattr(shim, "_installed_names", lambda: set())
    execd, marker = _run(shim, "pip", ["transformers[deepspeed]==5.5.0"])
    assert execd is None, execd
    assert marker == "5.5.0", marker
