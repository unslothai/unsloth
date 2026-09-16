# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A backend-neutral GPU memory cap for the training worker (unsloth#8178).

The only lever on how much GPU memory a training run may take was
`UNSLOTH_ROCM_MEM_FRACTION`, read in one place behind `_hw.IS_ROCM`, so NVIDIA had
no cap at all. `set_per_process_memory_fraction` lives on `torch.cuda` for both
vendors, so one variable can serve both: `UNSLOTH_GPU_MEM_FRACTION`.

Two properties this file exists to hold:

1. Nothing changes for anyone who sets nothing. With no override the policy
   answers 1.0 on every non-ROCm backend, which is what torch does with no cap,
   and the ROCm arm is delegated unchanged to `_rocm_memory_fraction`.
2. The ROCm name still wins on a ROCm host, so a machine that already exports it
   keeps exactly the cap it had.

The wiring in section 1h of `run_training_process` cannot be called directly (it
sits inside a function that spawns a trainer), so the block is sliced out of the
source and executed against fake `_hw`, `torch` and `logger` objects. That is a
real execution of the shipped lines, not a source-substring check.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.training.worker import (
    _DISCRETE_MEM_FRACTION,
    _GPU_MEM_FRACTION_ENV,
    _MEM_FRACTION_ENV,
    _gpu_memory_fraction,
    _mem_fraction_env_names,
    _mem_fraction_env_value,
    _rocm_memory_fraction,
)

GIB = 1024**3

_WORKER_PY = Path(__file__).resolve().parents[1] / "core" / "training" / "worker.py"

# Non-ROCm backends the worker can meet. "cuda" is the one wired; the rest are here so a
# later backend cannot quietly inherit a cap it was never measured for.
_OTHER_BACKENDS = ["cuda", "xpu", "mps", "cpu"]
_PLATFORMS = ["linux", "win32", "darwin"]


# ── The default: nobody who sets nothing is affected ─────────────────────────


@pytest.mark.parametrize("backend", _OTHER_BACKENDS)
@pytest.mark.parametrize("platform", _PLATFORMS)
@pytest.mark.parametrize("total", [0, 8 * GIB, 24 * GIB, 128 * GIB])
@pytest.mark.parametrize("is_unified", [False, True])
def test_every_non_rocm_backend_is_uncapped_without_an_override(
    backend: str, platform: str, total: int, is_unified: bool
) -> None:
    assert _gpu_memory_fraction(total, is_unified, platform, backend) == 1.0


@pytest.mark.parametrize("platform", _PLATFORMS)
@pytest.mark.parametrize("total", [0, 8 * GIB, 24 * GIB, 80 * GIB, 128 * GIB, 512 * GIB])
@pytest.mark.parametrize("is_unified", [False, True])
@pytest.mark.parametrize("denominator", [None, 220 * GIB])
def test_the_rocm_arm_is_byte_identical_to_the_helper_it_delegates_to(
    platform: str, total: int, is_unified: bool, denominator: int | None
) -> None:
    """The whole compatibility claim in one assertion, over the corners the ROCm
    policy actually branches on."""
    assert _gpu_memory_fraction(
        total, is_unified, platform, "rocm", None, denominator
    ) == _rocm_memory_fraction(total, is_unified, platform, None, denominator)


def test_a_known_rocm_answer_is_unchanged():
    """Spot values, so a refactor of both sides at once still fails here."""
    assert _gpu_memory_fraction(24 * GIB, False, "linux", "rocm") == _DISCRETE_MEM_FRACTION
    assert _gpu_memory_fraction(128 * GIB, True, "win32", "rocm") == 1.0
    assert _gpu_memory_fraction(24 * GIB, True, "linux", "rocm") == pytest.approx(0.80)


# ── The override ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("backend", _OTHER_BACKENDS + ["rocm"])
@pytest.mark.parametrize("raw, expected", [("0.5", 0.5), ("0.95", 0.95), ("1.0", 1.0), (" 0.25 ", 0.25)])
def test_a_usable_override_wins_on_every_backend(backend: str, raw: str, expected: float) -> None:
    assert _gpu_memory_fraction(128 * GIB, False, "linux", backend, raw) == pytest.approx(expected)


@pytest.mark.parametrize("backend", _OTHER_BACKENDS)
@pytest.mark.parametrize("bad", ["", "  ", "abc", "0", "0.0", "-0.5", "1.5", "nan", "inf"])
def test_an_unusable_override_leaves_a_non_rocm_backend_uncapped(backend: str, bad: str) -> None:
    assert _gpu_memory_fraction(128 * GIB, False, "linux", backend, bad) == 1.0


@pytest.mark.parametrize("bad", ["", "  ", "abc", "0", "0.0", "-0.5", "1.5", "nan", "inf"])
def test_an_unusable_override_leaves_rocm_on_its_computed_cap(bad: str) -> None:
    assert _gpu_memory_fraction(128 * GIB, True, "linux", "rocm", bad) == _rocm_memory_fraction(
        128 * GIB, True, "linux", None
    )


# ── Which variable is read ───────────────────────────────────────────────────


def test_rocm_reads_its_own_name_first():
    assert _mem_fraction_env_names("rocm") == (_MEM_FRACTION_ENV, _GPU_MEM_FRACTION_ENV)


@pytest.mark.parametrize("backend", _OTHER_BACKENDS)
def test_other_backends_read_only_the_neutral_name(backend: str) -> None:
    assert _mem_fraction_env_names(backend) == (_GPU_MEM_FRACTION_ENV,)


def test_the_rocm_name_still_wins_on_a_rocm_host():
    """An existing ROCm setup keeps the cap it had, whatever else is exported."""
    environ = {_MEM_FRACTION_ENV: "0.7", _GPU_MEM_FRACTION_ENV: "0.3"}
    assert _mem_fraction_env_value("rocm", environ) == ("0.7", _MEM_FRACTION_ENV)
    assert _gpu_memory_fraction(128 * GIB, True, "linux", "rocm", "0.7") == 0.7


def test_the_neutral_name_serves_a_rocm_host_that_only_sets_it():
    environ = {_GPU_MEM_FRACTION_ENV: "0.6"}
    assert _mem_fraction_env_value("rocm", environ) == ("0.6", _GPU_MEM_FRACTION_ENV)


def test_an_unusable_rocm_value_falls_through_to_the_neutral_one():
    environ = {_MEM_FRACTION_ENV: "abc", _GPU_MEM_FRACTION_ENV: "0.5"}
    assert _mem_fraction_env_value("rocm", environ) == ("0.5", _GPU_MEM_FRACTION_ENV)


def test_when_nothing_parses_the_first_variable_that_was_set_is_named():
    """So the warning names a variable the user really exported."""
    environ = {_MEM_FRACTION_ENV: "abc", _GPU_MEM_FRACTION_ENV: "2.0"}
    assert _mem_fraction_env_value("rocm", environ) == ("abc", _MEM_FRACTION_ENV)


def test_nothing_set_resolves_to_nothing():
    assert _mem_fraction_env_value("rocm", {}) == (None, None)
    assert _mem_fraction_env_value("cuda", {}) == (None, None)


def test_a_cuda_host_ignores_the_rocm_only_name():
    """The ROCm variable describes a ROCm-specific reserve policy; honouring it on
    NVIDIA would cap a machine whose owner meant something else entirely."""
    environ = {_MEM_FRACTION_ENV: "0.5"}
    assert _mem_fraction_env_value("cuda", environ) == (None, None)


def test_an_empty_string_is_treated_as_unset():
    assert _mem_fraction_env_value("cuda", {_GPU_MEM_FRACTION_ENV: ""}) == ("", _GPU_MEM_FRACTION_ENV)
    assert _gpu_memory_fraction(0, False, "linux", "cuda", "") == 1.0


# ── Section 1h, executed ─────────────────────────────────────────────────────


def _section_1h_source() -> str:
    source = _WORKER_PY.read_text(encoding = "utf-8")
    start = source.index("    # ── 1h. Explicit GPU memory cap ──")
    end = source.index("    # ── 2. Now import ML libraries")
    return textwrap.dedent(source[start:end])


def _make_logger():
    """A logger that keeps the rendered lines, so the assertions read what a user sees."""
    recorded = SimpleNamespace(info = [], warning = [], debug = [])

    def make(level):
        def log(message, *args):
            getattr(recorded, level).append(message % args if args else message)

        return log

    logger = SimpleNamespace(info = make("info"), warning = make("warning"), debug = make("debug"))
    return logger, recorded


class _FakeCuda:
    def __init__(self, available = True, total = 24 * GIB, name = "NVIDIA GeForce RTX 4090"):
        self._available = available
        self._total = total
        self._name = name
        self.fraction = None

    def is_available(self):
        return self._available

    def set_per_process_memory_fraction(self, fraction):
        self.fraction = fraction

    def get_device_properties(self, index):
        return SimpleNamespace(total_memory = self._total, name = self._name)


def _run_section_1h(*, is_rocm, environ, cuda = None, platform = "linux"):
    """Execute the shipped block with fakes standing in for the process globals."""
    import sys as _real_sys

    cuda = cuda if cuda is not None else _FakeCuda()
    logger, recorded = _make_logger()
    fake_torch = SimpleNamespace(cuda = cuda)

    from core.training import worker as worker_module

    def resolve_env(backend, environ_ = None):
        # The shipped line calls this with one argument, so the fake environment has to
        # arrive here rather than through a patched os module.
        return worker_module._mem_fraction_env_value(backend, environ if environ_ is None else environ_)

    namespace = {
        "_hw": SimpleNamespace(IS_ROCM = is_rocm),
        "os": SimpleNamespace(environ = environ),
        "sys": SimpleNamespace(platform = platform, modules = dict(_real_sys.modules, torch = fake_torch)),
        "logger": logger,
        "_mem_fraction_env_value": resolve_env,
        "_parse_mem_fraction_env": worker_module._parse_mem_fraction_env,
        "_gpu_memory_fraction": worker_module._gpu_memory_fraction,
        "torch": fake_torch,
    }
    # `import torch` inside the block must reach the fake, not the real wheel.
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            return fake_torch
        return real_import(name, *args, **kwargs)

    builtins_map = __builtins__ if isinstance(__builtins__, dict) else vars(__builtins__)
    namespace["__builtins__"] = dict(builtins_map, __import__ = fake_import)

    exec(compile(_section_1h_source(), str(_WORKER_PY), "exec"), namespace)
    return cuda, recorded


def test_the_block_does_nothing_when_no_variable_is_set():
    cuda, log = _run_section_1h(is_rocm = False, environ = {})
    assert cuda.fraction is None
    assert log.info == [] and log.warning == []


def test_the_block_does_nothing_on_a_rocm_host():
    """Section 1g already served it; running both would cap twice and log twice."""
    cuda, log = _run_section_1h(
        is_rocm = True, environ = {_GPU_MEM_FRACTION_ENV: "0.5"},
    )
    assert cuda.fraction is None
    assert log.info == []


def test_the_block_caps_an_nvidia_device():
    cuda, log = _run_section_1h(is_rocm = False, environ = {_GPU_MEM_FRACTION_ENV: "0.75"})
    assert cuda.fraction == pytest.approx(0.75)
    assert len(log.info) == 1
    assert _GPU_MEM_FRACTION_ENV in log.info[0]
    assert "18.0 of 24.0 GiB allowed" in log.info[0]


def test_the_block_ignores_the_rocm_only_variable_on_nvidia():
    cuda, log = _run_section_1h(is_rocm = False, environ = {_MEM_FRACTION_ENV: "0.5"})
    assert cuda.fraction is None
    assert log.info == [] and log.warning == []


@pytest.mark.parametrize("bad", ["abc", "0", "1.5", "-1", "nan"])
def test_an_unusable_value_warns_and_leaves_the_device_uncapped(bad: str):
    cuda, log = _run_section_1h(is_rocm = False, environ = {_GPU_MEM_FRACTION_ENV: bad})
    assert cuda.fraction is None
    assert len(log.warning) == 1
    assert _GPU_MEM_FRACTION_ENV in log.warning[0]
    assert "uncapped" in log.warning[0]


def test_a_host_with_no_cuda_device_is_a_debug_line_not_a_crash():
    cuda, log = _run_section_1h(
        is_rocm = False,
        environ = {_GPU_MEM_FRACTION_ENV: "0.5"},
        cuda = _FakeCuda(available = False),
    )
    assert cuda.fraction is None
    assert log.info == [] and log.warning == []
    assert len(log.debug) == 1


def test_a_wheel_that_reports_no_total_still_caps():
    cuda, log = _run_section_1h(
        is_rocm = False,
        environ = {_GPU_MEM_FRACTION_ENV: "0.5"},
        cuda = _FakeCuda(total = 0),
    )
    assert cuda.fraction == pytest.approx(0.5)
    assert "device total unreported" in log.info[0]


class _ExplodingCuda(_FakeCuda):
    def set_per_process_memory_fraction(self, fraction):
        raise RuntimeError("no CUDA-capable device is detected")


def test_a_failure_inside_the_block_never_takes_the_run_down():
    cuda, log = _run_section_1h(
        is_rocm = False,
        environ = {_GPU_MEM_FRACTION_ENV: "0.5"},
        cuda = _ExplodingCuda(),
    )
    assert log.debug and "Could not set GPU memory fraction" in log.debug[0]


@pytest.mark.parametrize("platform", _PLATFORMS)
def test_the_cap_is_the_same_on_every_platform(platform: str):
    """The neutral cap is a user preference, not a driver workaround, so unlike the
    ROCm reserve it does not vary by platform."""
    cuda, _ = _run_section_1h(
        is_rocm = False, environ = {_GPU_MEM_FRACTION_ENV: "0.6"}, platform = platform,
    )
    assert cuda.fraction == pytest.approx(0.6)


def test_the_block_is_still_gated_on_the_rocm_flag():
    assert "if not _hw.IS_ROCM:" in _section_1h_source()


def test_no_settings_route_reads_the_new_variable():
    """The Settings control is a deliberate follow-up. If one is added, this test is
    the reminder that the env var and the control need one policy path, not two."""
    backend_root = Path(__file__).resolve().parents[1]
    offenders = []
    for path in (backend_root / "routes").rglob("*.py"):
        if _GPU_MEM_FRACTION_ENV in path.read_text(encoding = "utf-8"):
            offenders.append(str(path.relative_to(backend_root)))
    assert offenders == [], offenders
