# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Name a torchvision whose compiled ops do not match torch.

`torchvision_compatibility_check` compared version metadata, which cannot see
an ABI break. Found by running `Gemma4_(E2B)_GRPO`: its T4 branch installs
vllm==0.9.2 beside Colab's torch, and `import unsloth` then died with
`RuntimeError: operator torchvision::nms does not exist`, raised from
`transformers/image_utils.py` and naming nothing. The vLLM half of the same
breakage was already handled.
"""

import ast
import builtins
import pathlib
import sys
from unittest import mock

import pytest

from unsloth import import_fixes


_NMS = RuntimeError("operator torchvision::nms does not exist")


def test_the_nms_break_is_recognised():
    assert import_fixes._is_broken_torchvision_error(_NMS)


@pytest.mark.parametrize(
    "message",
    [
        "/usr/lib/torchvision/_C.so: undefined symbol: _ZN3c10",
        "libc10.so: cannot open shared object file: No such file or directory",
        "No module named 'torchvision.io.video'",
        "No module named 'torchvision.io._video'",
    ],
)
def test_the_other_shapes_of_the_same_break_are_recognised(message):
    """A half-overwritten install and an ABI mismatch reach us differently."""
    assert import_fixes._is_broken_torchvision_error(ImportError(message))


@pytest.mark.parametrize(
    "message",
    [
        # A CPU-only or driverless box: torchvision cannot load, and that is
        "libcuda.so.1: cannot open shared object file: No such file or directory",
        "libnvrtc.so: cannot open shared object file: No such file or directory",
        "/lib/libjpeg.so: undefined symbol: jpeg_resync_to_restart",
    ],
)
def test_an_unrelated_loader_failure_is_not_claimed(message):
    """The probe imports torchvision where nothing used to, so it must not turn
    a failure it did not cause into a hard error on `import unsloth`."""
    assert not import_fixes._is_broken_torchvision_error(ImportError(message))


def test_an_unrelated_error_is_not_claimed():
    assert not import_fixes._is_broken_torchvision_error(ValueError("something else"))
    assert not import_fixes._is_broken_torchvision_error(None)


def test_a_chained_cause_is_followed():
    """torchvision surfaces the loader error as __cause__ of its own."""
    outer = ImportError("cannot import name 'ops' from 'torchvision'")
    outer.__cause__ = _NMS
    assert import_fixes._is_broken_torchvision_error(outer)


def _probe_with_import_raising(
    error,
    required = (0, 26),
    torch_version_raw = "2.11.0",
    torchvision_version_raw = "0.26.0",
):
    """Run the probe with `import torchvision` raising `error`."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torchvision" or name.startswith("torchvision."):
            raise error
        return real_import(name, *args, **kwargs)

    with mock.patch.dict(sys.modules):
        for name in [n for n in sys.modules if n.startswith("torchvision")]:
            sys.modules.pop(name, None)
        with mock.patch.object(builtins, "__import__", fake_import):
            import_fixes._probe_torchvision_binary(
                torch_version_raw, torchvision_version_raw, required
            )


def test_a_broken_binary_raises_something_actionable():
    with pytest.raises(ImportError) as excinfo:
        _probe_with_import_raising(_NMS)
    text = str(excinfo.value)
    # The cause, the fix, and the escape hatch, in the one message.
    assert "torchvision==0.26.0" in text and "torch==2.11.0" in text
    assert "force-reinstall --no-deps --no-cache-dir" in text
    assert "UNSLOTH_SKIP_TORCHVISION_CHECK=1" in text
    assert excinfo.value.__cause__ is _NMS


def test_the_repair_command_cannot_replace_torch():
    """Every torchvision wheel requires an exact `torch==X.Y.Z`, so an unpinned
    upgrade resolves the newest torchvision and drags a new torch in with it."""
    command = import_fixes._torchvision_repair_command((0, 26))
    assert "--no-deps" in command, "torch must not be a candidate for replacement"
    assert "--upgrade" not in command, "the newest release is not what repairs a binary"
    assert "torchvision==0.26.*" in command


def test_the_repair_command_names_the_companion_release():
    """The gate passes on a lower bound (torch 2.4 accepts torchvision >= 0.19),
    so an installed 0.20 reaches the probe; 0.19 is what repairs that box."""
    assert "torchvision==0.19.*" in import_fixes._torchvision_repair_command((0, 19))
    # No table entry: still pinned to nothing rather than to the wrong thing.
    assert "torchvision" in import_fixes._torchvision_repair_command(None)


def test_the_probe_is_told_which_release_the_table_wanted():
    """Otherwise the message cannot name the companion version."""
    source = ast.unparse(_check_function())
    assert (
        "_probe_torchvision_binary(torch_version_raw, torchvision_version_raw, required)" in source
    )


def test_an_unrelated_import_error_is_left_alone():
    """The probe must not turn every torchvision import failure into ours."""
    _probe_with_import_raising(ImportError("No module named 'some_optional_dep'"))


def test_a_healthy_torchvision_is_silent():
    pytest.importorskip("torchvision")
    import_fixes._probe_torchvision_binary("2.11.0", "0.26.0")


_SOURCE = (pathlib.Path(import_fixes.__file__)).read_text(encoding = "utf-8")


def _check_function():
    tree = ast.parse(_SOURCE)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "torchvision_compatibility_check":
            return node
    pytest.fail("torchvision_compatibility_check has moved or been renamed")


def test_the_probe_runs_on_the_path_the_table_calls_compatible():
    """Which is the only blind spot: a mismatch the table sees already raises."""
    assert "_probe_torchvision_binary" in ast.unparse(_check_function())


def test_the_skip_variable_still_skips_everything():
    """It guards the whole function, so it must come before the probe."""
    source = ast.unparse(_check_function())
    assert source.index("UNSLOTH_SKIP_TORCHVISION_CHECK") < source.index(
        "_probe_torchvision_binary"
    )
    with mock.patch.dict("os.environ", {"UNSLOTH_SKIP_TORCHVISION_CHECK": "1"}):
        with mock.patch.object(import_fixes, "_probe_torchvision_binary") as probe:
            import_fixes.torchvision_compatibility_check()
    probe.assert_not_called()




def test_the_repair_names_the_wheel_for_this_torch_patch():
    """`0.22.*` on a torch 2.7.0 host resolves torchvision 0.22.1, which requires
    torch 2.7.1, and `--no-deps` then keeps the 2.7.0 that does not match it. The
    advertised repair would rebuild the mismatch it is meant to fix."""
    from unsloth.import_fixes import _torchvision_repair_command

    assert '"torchvision==0.22.0"' in _torchvision_repair_command((0, 22, 0))
    assert '"torchvision==0.22.1"' in _torchvision_repair_command((0, 22, 1))
    assert ".*" not in _torchvision_repair_command((0, 24, 1))


def test_a_minor_only_pair_still_gets_a_command():
    """The table and the forward-compat formula both answer with two numbers when
    the torch version carries no patch. Nothing to derive, so the range stands."""
    from unsloth.import_fixes import _torchvision_repair_command

    assert '"torchvision==0.22.*"' in _torchvision_repair_command((0, 22))
    assert '"torchvision"' in _torchvision_repair_command(None)


def test_the_pairing_this_relies_on_is_what_pypi_publishes():
    """The whole fix rests on torchvision's patch tracking torch's. Asserted
    against the real metadata rather than against the table, and skipped rather
    than failed when the network is unavailable."""
    import json
    import urllib.error
    import urllib.request

    import pytest

    expected = {"0.22.0": "torch==2.7.0", "0.22.1": "torch==2.7.1"}
    for torchvision_version, torch_requirement in expected.items():
        try:
            with urllib.request.urlopen(
                f"https://pypi.org/pypi/torchvision/{torchvision_version}/json", timeout = 20
            ) as response:
                metadata = json.load(response)
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            pytest.skip(f"pypi.org unreachable: {error}")
        requirements = metadata["info"].get("requires_dist") or []
        assert (
            torch_requirement in requirements
        ), f"torchvision {torchvision_version} no longer requires {torch_requirement}"


def test_the_repair_command_keeps_the_backend_torch_was_built_for():
    """PyPI carries one torchvision build per release and it is the CUDA one:
    `torchvision-0.22.0-cp310-manylinux_2_28_x86_64.whl` links libcudart.so.12,
    libc10_cuda.so and libtorch_cuda.so, while `0.22.0+rocm6.3` links
    libamdhip64.so.6, libc10_hip.so and libtorch_hip.so. `--no-deps` keeps the
    installed torch, so on a ROCm, XPU or CPU host an unqualified pin swaps the
    working wheel for the CUDA one and reproduces the exact `operator
    torchvision::nms does not exist` this command is handed out to clear
    (reproduced end to end on torch 2.7.1+cpu with torchvision 0.22.1+cpu)."""

    def advice(torch_raw):
        """The message a user on `torch_raw` is actually shown."""
        with pytest.raises(ImportError) as excinfo:
            _probe_with_import_raising(
                _NMS,
                required = (0, 22, 0),
                torch_version_raw = torch_raw,
                torchvision_version_raw = "0.22.0",
            )
        return str(excinfo.value)

    # CUDA families included:
    # CUDA families included: PyPI ships exactly one of them, so `cu118only*` (pyproject.toml:176) is as mismatched
    for tag in ("rocm6.3", "rocm6.2.4", "xpu", "cpu", "cu118", "cu126", "cu128"):
        command = advice(f"2.7.0+{tag}")
        assert f"--index-url https://download.pytorch.org/whl/{tag}" in command, command
        assert "torchvision==0.22.0" in command, command

    # No local tag, so PyPI's own build is the one that pairs with it.
    assert "--index-url" not in advice("2.7.0")
    assert "force-reinstall" in advice("2.7.0")


def test_a_build_no_public_index_carries_is_not_sent_to_pip():
    """A vendor or source build has no index that pairs with it, and a nightly's
    companion version is synthesised from the release numbers alone, so any
    pinned reinstall installs a wheel that cannot load against the installed
    torch. This repo ships such builds itself: the `rocm72-torch291` extra
    installs `torch 2.9.1+rocm7.2.0.lw.git7e1940d4` beside a repo.radeon.com
    torchvision 0.24.0, and the table would otherwise advertise PyPI's 0.24.1."""

    def advice(torch_raw, required):
        with pytest.raises(ImportError) as excinfo:
            _probe_with_import_raising(
                _NMS,
                required = required,
                torch_version_raw = torch_raw,
                torchvision_version_raw = "0.24.0",
            )
        return str(excinfo.value)

    for raw, required in (
        ("2.9.1+rocm7.2.0.lw.git7e1940d4", (0, 24, 1)),  # Radeon Linux extra Radeon Windows extra built from source
        ("2.9.1+rocmsdk20260116", (0, 24, 1)),
        ("2.7.0+git1a2b3c", (0, 22, 0)),
        ("2.12.0.dev20260801+cpu", (0, 27, 0)),
        ("2.11.0a1+cu128", (0, 26, 0)),
        ("2.11.0b2+cu128", (0, 26, 0)),
        ("2.7.0rc1", (0, 22, 0)),
    ):
        text = advice(raw, required)
        assert "pip install" not in text, text
        assert f"torch=={raw}" in text, text


def test_a_conda_torch_is_not_sent_to_pypis_torchvision(tmp_path):
    """conda records the backend in the build string and leaves the version
    plain, so a conda CPU or ROCm torch reaches the tag check looking exactly
    like a PyPI one. `--no-deps` then keeps that torch beside PyPI's CUDA-only
    torchvision, which is the mismatch the command is handed out to clear."""
    conda_meta = tmp_path / "conda-meta"
    conda_meta.mkdir()
    (conda_meta / "pytorch-2.5.1-py3.12_cuda12.4_cudnn9_0.json").write_text("{}")
    # Same version, unrelated package: it must not answer for torch.
    (conda_meta / "pytorch-lightning-2.5.1-pyhd8ed1ab_0.json").write_text("{}")

    def advice(torch_raw):
        with pytest.raises(ImportError) as excinfo:
            _probe_with_import_raising(
                _NMS,
                required = (0, 20, 1),
                torch_version_raw = torch_raw,
                torchvision_version_raw = "0.20.1",
            )
        return str(excinfo.value)

    with mock.patch.object(sys, "prefix", str(tmp_path)):
        conda = advice("2.5.1")
        assert "pip install" not in conda, conda
        assert "torch==2.5.1" in conda, conda
        # A different version in the same prefix is pip's, and still gets pip's command:
        assert "pip install" in advice("2.6.0")

    # Without the ledger nothing changes:
    assert "pip install" in advice("2.5.1")
