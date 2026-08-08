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
        # not what this probe is for.
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


def _probe_with_import_raising(error, required = (0, 26)):
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
            import_fixes._probe_torchvision_binary("2.11.0", "0.26.0", required)


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
