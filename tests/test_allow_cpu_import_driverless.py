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
"""`UNSLOTH_ALLOW_CPU=1` has to survive the import on a driverless host.

A CUDA-built torch with no usable device -- a driverless container, a CI runner,
a laptop with the runtime and no card -- is exactly what that variable exists
for. `get_device_type()` deliberately keeps `DEVICE_TYPE` at `"cuda"` there, so
every `if DEVICE_TYPE == "cuda": torch.cuda.get_device_capability()` at module
scope runs with nothing to query and raises `RuntimeError: No CUDA GPUs are
available` out of `_lazy_init()`.

The import is process-global and one-shot, so every case here runs in a fresh
interpreter with `CUDA_VISIBLE_DEVICES=""`.
"""

import os
import pathlib
import re
import subprocess
import sys
import textwrap

import pytest
import torch

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_UNSLOTH_DIR = _ROOT / "unsloth"

_NO_DEVICE = "No CUDA GPUs are available"


def _run(code, **env):
    """Fresh interpreter, this checkout on the path, every CUDA device hidden."""
    path = [str(_ROOT)]
    if os.environ.get("PYTHONPATH"):
        path.append(os.environ["PYTHONPATH"])
    # A runner (or a conftest) that exports UNSLOTH_ALLOW_CPU must not decide the
    # cases for us: each one says for itself whether the child gets it.
    clean = {k: v for k, v in os.environ.items() if k != "UNSLOTH_ALLOW_CPU"}
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output = True,
        text = True,
        env = dict(
            clean,
            PYTHONPATH = os.pathsep.join(path),
            CUDA_VISIBLE_DEVICES = "",
            **env,
        ),
        timeout = 900,
    )


def _needs_cuda_build():
    """Only a CUDA-built torch reaches the branch under test. On a ROCm or a
    CPU-only build `DEVICE_TYPE` is not `"cuda"` and there is nothing to prove --
    that is the Windows-on-ARM shape, not this one."""
    if getattr(torch.version, "hip", None):
        pytest.skip("ROCm build: DEVICE_TYPE is not the cuda branch this covers")
    if not getattr(torch.version, "cuda", None):
        pytest.skip("torch is not built against CUDA, so no device can go missing")


def _import_attempt():
    _needs_cuda_build()
    return _run(
        """
        import unsloth
        from unsloth import FastLanguageModel, FastModel
        import unsloth.models._utils as _utils
        import unsloth._gpu_init as _gpu_init
        print("DEVICE_TYPE", _gpu_init.DEVICE_TYPE)
        print("SUPPORTS_BFLOAT16", _gpu_init.SUPPORTS_BFLOAT16, _utils.SUPPORTS_BFLOAT16)
        print("HAS_FLASH_ATTENTION", _utils.HAS_FLASH_ATTENTION)
        print("IMPORT_OK")
        """,
        UNSLOTH_ALLOW_CPU = "1",
    )


_FRAME = re.compile(r'^\s*File "([^"]+)", line \d+', re.MULTILINE)

_TORCH_DIR = str(pathlib.Path(torch.__file__).parent)


def _culprit(text):
    """The deepest traceback frame outside torch: the line that did the asking.

    Every frame above it merely imported the module that asked, so matching on
    "any frame under this directory" would blame unsloth for an unsloth_zoo
    probe -- `unsloth/models/_utils.py` is on the import path either way.
    """
    outside = [f for f in _FRAME.findall(text) if not f.startswith(_TORCH_DIR + os.sep)]
    return outside[-1] if outside else None


def _under(path, directory):
    return path is not None and path.startswith(str(directory) + os.sep)


def test_the_devices_really_are_hidden_and_the_variable_is_what_opens_the_import():
    """Without the variable the import must still refuse, and refuse with the
    documented message. If this ever passes, the host has a visible GPU and every
    other case in this file is vacuous."""
    _needs_cuda_build()
    out = _run("import unsloth")
    assert out.returncode != 0, "a device is visible; this file proves nothing here"
    assert "You need a GPU" in out.stderr, out.stderr[-3000:]


def test_no_unsloth_module_probes_a_device_that_is_not_there():
    """The claim this repo can make on its own. Scoped to unsloth's own files so a
    stale `unsloth_zoo` on the path cannot redden it -- see the next case."""
    out = _import_attempt()
    if out.returncode == 0:
        return
    if _NO_DEVICE not in out.stderr:
        return
    culprit = _culprit(out.stderr)
    assert not _under(culprit, _UNSLOTH_DIR), (
        f"an import-time site in unsloth asked a missing device what it can do: "
        f"{culprit}\n" + out.stderr[-2000:]
    )


def test_the_import_succeeds_on_a_driverless_host():
    """End to end. The chain runs through unsloth_zoo as well
    (`compiler.py`, `loss_utils.py`), so an unsloth_zoo without those guards
    leaves this unprovable rather than failed."""
    out = _import_attempt()
    if out.returncode != 0 and _NO_DEVICE in out.stderr:
        import unsloth_zoo
        zoo = pathlib.Path(unsloth_zoo.__file__).parent
        if _under(_culprit(out.stderr), zoo):
            pytest.skip(
                "unsloth_zoo on this path still probes a missing device at import "
                "time; needs the matching unsloth-zoo guards"
            )
    assert out.returncode == 0, out.stderr[-3000:]
    assert "IMPORT_OK" in out.stdout, out.stdout


def test_a_host_with_no_device_claims_no_capability():
    """Where a capability cannot be read, the conservative answer is the one that
    only costs float32. Claiming bfloat16 here fails at the first cast instead."""
    out = _import_attempt()
    if out.returncode != 0:
        pytest.skip("the import does not complete here; covered by the cases above")
    assert "DEVICE_TYPE cuda" in out.stdout, out.stdout
    assert "SUPPORTS_BFLOAT16 False False" in out.stdout, out.stdout
    assert "HAS_FLASH_ATTENTION False" in out.stdout, out.stdout
