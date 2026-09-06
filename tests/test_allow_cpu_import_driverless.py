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

A CPU-only wheel reaches the same branches, because `get_device_type()` answers
`"cuda"` for the variable before it looks at the torch build. It raises
`AssertionError: Torch not compiled with CUDA enabled` from the same
`_lazy_init()` instead, so both spellings count as "asked a device that is not
there what it can do". That is the build CI's `Repo tests (CPU)` job installs.

The import is process-global and one-shot, so every case here runs in a fresh
interpreter with `CUDA_VISIBLE_DEVICES=""`.
"""

import importlib.util
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

# Both spellings of "you asked a device that is not there what it can do". A CUDA-built torch with the devices hidden
# raises the first out of _lazy_init(); a CPU-only wheel raises the second from the same place. CI's `Repo tests (CPU)`
# job installs the CPU wheel, so the second shape is the one it would see.
_NO_DEVICE = (
    "No CUDA GPUs are available",
    "Torch not compiled with CUDA enabled",
)


def _asked_a_missing_device(text):
    return any(message in text for message in _NO_DEVICE)


def _run(
    code,
    extra_path = (),
    **env,
):
    """Fresh interpreter, this checkout on the path, every CUDA device hidden."""
    path = [str(entry) for entry in extra_path]
    path.append(str(_ROOT))
    if os.environ.get("PYTHONPATH"):
        path.append(os.environ["PYTHONPATH"])
    # A runner (or a conftest) that exports UNSLOTH_ALLOW_CPU must not decide the cases for us: each one says for itself
    # whether the child gets it.
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


def _needs_the_cuda_branch():
    """The guards under test sit behind `DEVICE_TYPE == "cuda"`, and a CPU-only
    wheel gets there too.

    `get_device_type()` answers `"cuda"` for `UNSLOTH_ALLOW_CPU=1` before it
    looks at `torch.cuda.is_available()` or at the torch build
    (`unsloth/device_type.py`), so the CPU wheel CI installs in `Repo tests
    (CPU)` reaches both new branches -- it just raises
    "Torch not compiled with CUDA enabled" rather than "No CUDA GPUs are
    available" when they are missing, which `_NO_DEVICE` now covers. Skipping on
    that build left the only job that discovers this file reporting four skips.

    MLX is the real exception: there `DEVICE_TYPE` is `"mlx"` and none of this
    runs. ROCm keeps its own skip because `is_available()` answers from a
    different runtime there."""
    if importlib.util.find_spec("mlx") is not None:
        pytest.skip("MLX runtime: DEVICE_TYPE is 'mlx', not the cuda branch this covers")
    if getattr(torch.version, "hip", None):
        pytest.skip("ROCm build: DEVICE_TYPE is not the cuda branch this covers")


_IMPORT_ATTEMPT_CODE = """
    import unsloth
    from unsloth import FastLanguageModel, FastModel
    import unsloth.models._utils as _utils
    import unsloth._gpu_init as _gpu_init
    print("DEVICE_TYPE", _gpu_init.DEVICE_TYPE)
    print("SUPPORTS_BFLOAT16", _gpu_init.SUPPORTS_BFLOAT16, _utils.SUPPORTS_BFLOAT16)
    print("HAS_FLASH_ATTENTION", _utils.HAS_FLASH_ATTENTION)
    print("IMPORT_OK")
    """


@pytest.fixture(scope = "module")
def _import_attempt_result():
    """One child interpreter, shared by every case that reads the same attempt.

    Three tests below ask different questions of the *same* driverless import:
    whether unsloth's own files probed a device, whether the import finished, and
    what it claimed about capability. They ran it three times, and `import unsloth`
    is ~14s of startup, so two thirds of this file's 55s was the same subprocess
    over again. The child is read-only from the tests' point of view -- they only
    inspect its returncode, stdout and stderr -- so one run answers all three.

    Module-scoped rather than session-scoped: nothing outside this file wants it,
    and a session fixture would keep the CompletedProcess alive for the whole run.
    """
    _needs_the_cuda_branch()
    return _run(_IMPORT_ATTEMPT_CODE, UNSLOTH_ALLOW_CPU = "1")


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
    _needs_the_cuda_branch()
    out = _run("import unsloth")
    assert out.returncode != 0, "a device is visible; this file proves nothing here"
    assert "You need a GPU" in out.stderr, out.stderr[-3000:]


def test_no_unsloth_module_probes_a_device_that_is_not_there(_import_attempt_result):
    """The claim this repo can make on its own. Scoped to unsloth's own files so a
    stale `unsloth_zoo` on the path cannot redden it -- see the next case."""
    out = _import_attempt_result
    if out.returncode == 0:
        return
    if not _asked_a_missing_device(out.stderr):
        return
    culprit = _culprit(out.stderr)
    assert not _under(culprit, _UNSLOTH_DIR), (
        f"an import-time site in unsloth asked a missing device what it can do: "
        f"{culprit}\n" + out.stderr[-2000:]
    )


def test_the_import_succeeds_on_a_driverless_host(_import_attempt_result):
    """End to end. The chain runs through unsloth_zoo as well
    (`compiler.py`, `loss_utils.py`), so an unsloth_zoo without those guards
    leaves this unprovable rather than failed."""
    out = _import_attempt_result
    if out.returncode != 0 and _asked_a_missing_device(out.stderr):
        import unsloth_zoo
        zoo = pathlib.Path(unsloth_zoo.__file__).parent
        if _under(_culprit(out.stderr), zoo):
            pytest.skip(
                "unsloth_zoo on this path still probes a missing device at import "
                "time; needs the matching unsloth-zoo guards"
            )
    assert out.returncode == 0, out.stderr[-3000:]
    assert "IMPORT_OK" in out.stdout, out.stdout


_NO_LIBCUDA_PROBE = """
import os
import subprocess

_log = open(os.environ["PROBE_LOG"], "a")


def _record(what):
    _log.write(what + "\\n")
    _log.flush()


# Root is the default in a container, and it is the only euid that reaches the
# ldconfig arm. Claim it, and stub out the two calls that would touch the host so
# the test observes the attempt without performing it.
os.geteuid = lambda: 0
os.system = lambda command: (_record("system: " + command), 0)[1]

_check_output = subprocess.check_output


def _check(command, *args, **kwargs):
    _record("check_output: " + repr(command))
    return _check_output(command, *args, **kwargs)


subprocess.check_output = _check

# A driverless host has no libcuda for ldconfig to find, which is what triton's
# probe reports by raising.
import triton.backends.nvidia.driver as _driver


def _no_libcuda(*args, **kwargs):
    raise RuntimeError("probe: libcuda.so.1 not found by ldconfig")


_driver.libcuda_dirs = _no_libcuda
"""


def test_a_driverless_import_does_not_try_to_repair_cuda_linkage(tmp_path):
    """`UNSLOTH_ALLOW_CPU=1` says there is no device, so there is no linkage to
    repair.

    The `except` arm around `libcuda_dirs()` predates this branch and was only
    reachable with a device present. Left unguarded it now fires on every
    driverless import: as root it ldconfigs the container's linker cache through
    an unguarded `ls` subprocess, and otherwise it warns that CUDA is broken on a
    host the caller already said has no card.
    """
    _needs_the_cuda_branch()
    try:
        found = importlib.util.find_spec("triton.backends.nvidia.driver")
    except Exception:
        found = None
    if found is None:
        pytest.skip("no triton nvidia backend here, so libcuda_dirs is never called")
    probe = tmp_path / "sitecustomize.py"
    probe.write_text(_NO_LIBCUDA_PROBE, encoding = "utf-8")
    log = tmp_path / "calls.log"
    log.write_text("", encoding = "utf-8")
    out = _run(
        "import unsloth\nprint('IMPORT_OK')",
        extra_path = (tmp_path,),
        UNSLOTH_ALLOW_CPU = "1",
        PROBE_LOG = str(log),
    )
    if out.returncode != 0 and _asked_a_missing_device(out.stderr):
        pytest.skip("the import does not complete here; covered by the cases above")
    assert out.returncode == 0, out.stderr[-3000:]
    # Scoped to the two calls the repair arm makes.
    calls = log.read_text(encoding = "utf-8")
    assert "ldconfig" not in calls, f"a driverless import ran ldconfig:\n{calls}"
    assert (
        "check_output: ['ls'" not in calls
    ), f"a driverless import shelled out to ls to hunt for a CUDA install:\n{calls}"
    assert "CUDA is not linked properly" not in out.stderr, out.stderr[-3000:]


def test_a_host_with_no_device_claims_no_capability(_import_attempt_result):
    """Where a capability cannot be read, the conservative answer is the one that
    only costs float32. Claiming bfloat16 here fails at the first cast instead."""
    out = _import_attempt_result
    if out.returncode != 0:
        pytest.skip("the import does not complete here; covered by the cases above")
    assert "DEVICE_TYPE cuda" in out.stdout, out.stdout
    assert "SUPPORTS_BFLOAT16 False False" in out.stdout, out.stdout
    assert "HAS_FLASH_ATTENTION False" in out.stdout, out.stdout
