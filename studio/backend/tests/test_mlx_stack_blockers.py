# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The MLX gate has to say what it is unhappy about.

`mlx_unavailable` is a single verdict covering three packages and four runtime
imports, and the greyed-out Train row could only answer it with "run `unsloth
studio update`". That is a dead end for the usual cause: an update that ran, and
a resolver backtrack that left one package missing or too old for the pinned
transformers. These cover the blocker list that message is built from.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import utils.mlx_repair as mr  # noqa: E402


def _fake_versions(monkeypatch, installed: dict[str, str]):
    """Report `installed` as the distributions present, and nothing else."""
    from importlib.metadata import PackageNotFoundError

    def version(name: str) -> str:
        if name in installed:
            return installed[name]
        raise PackageNotFoundError(name)

    monkeypatch.setattr("importlib.metadata.version", version)


def _healthy(**overrides: str) -> dict[str, str]:
    """The floors mlx_repair requires, so "healthy" here follows a floor bump.

    Hardcoding a healthy version made raising the mlx-lm floor to 0.31.2 fail six
    tests that are about mlx-vlm, import errors and line length, none of which had
    anything to say about mlx-lm.
    """
    return {**mr._MLX_MIN_VERSIONS, **overrides}


def test_a_healthy_stack_reports_no_blockers(monkeypatch):
    _fake_versions(monkeypatch, _healthy())
    monkeypatch.setattr(mr, "_mlx_runtime_import_blocker", lambda: None)
    assert mr.mlx_stack_blockers() == []
    assert mr.mlx_stack_available() is True


def test_a_missing_package_is_named_with_the_version_it_needs(monkeypatch):
    _fake_versions(monkeypatch, {k: v for k, v in _healthy().items() if k != "mlx-vlm"})
    monkeypatch.setattr(mr, "_mlx_runtime_import_blocker", lambda: None)
    blockers = mr.mlx_stack_blockers()
    assert any("mlx-vlm is not installed" in blocker for blocker in blockers)
    assert any("0.4.4" in blocker for blocker in blockers)
    assert mr.mlx_stack_available() is False


def test_a_backtracked_package_names_the_version_it_found(monkeypatch):
    # The reported shape: present, importable, and too old for VLM Train/Export.
    _fake_versions(monkeypatch, _healthy(**{"mlx-vlm": "0.1.0"}))
    monkeypatch.setattr(mr, "_mlx_runtime_import_blocker", lambda: None)
    blockers = mr.mlx_stack_blockers()
    assert blockers == ["mlx-vlm 0.1.0 is older than 0.4.4"]


def test_every_bad_package_is_listed_not_just_the_first(monkeypatch):
    _fake_versions(monkeypatch, {"mlx": "0.1.0"})
    monkeypatch.setattr(mr, "_mlx_runtime_import_blocker", lambda: None)
    blockers = mr.mlx_stack_blockers()
    assert len(blockers) == 3, blockers


def test_an_import_that_raises_is_reported_with_its_error(monkeypatch):
    # Versions satisfied but the module will not load, which is what a mlx-vlm
    # built against a different transformers looks like from here.
    _fake_versions(monkeypatch, _healthy())

    def explode(module: str):
        raise ImportError("cannot import name 'AutoProcessor' from 'transformers'")

    monkeypatch.setattr(mr.importlib, "import_module", explode)
    blockers = mr.mlx_stack_blockers()
    assert len(blockers) == 1
    assert "does not import" in blockers[0]
    assert "AutoProcessor" in blockers[0]
    assert mr.mlx_stack_available() is False


def test_versions_are_checked_before_imports(monkeypatch):
    """A too-old package must be named without loading it into this process."""
    _fake_versions(monkeypatch, _healthy(**{"mlx-vlm": "0.1.0"}))

    def never(module: str):
        raise AssertionError("imported a package the version check already rejected")

    monkeypatch.setattr(mr.importlib, "import_module", never)
    assert mr.mlx_stack_blockers() == ["mlx-vlm 0.1.0 is older than 0.4.4"]


def test_the_detail_line_never_raises_and_stays_short(monkeypatch):
    from utils.hardware import hardware as hw

    monkeypatch.setattr(hw, "_MLX_BLOCKERS_MEASURED", None)

    def explode() -> list[str]:
        raise RuntimeError("no")

    monkeypatch.setattr(mr, "mlx_stack_blockers", explode)
    assert hw._mlx_stack_detail() is None

    monkeypatch.setattr(mr, "mlx_stack_blockers", lambda: [])
    assert hw._mlx_stack_detail() is None

    monkeypatch.setattr(mr, "mlx_stack_blockers", lambda: ["a", "b", "c", "d"])
    detail = hw._mlx_stack_detail()
    assert detail == "a; b; c"


@pytest.mark.parametrize("reason", ["intel_mac", "no_gpu", "detection_failed", None])
def test_only_the_mlx_verdict_carries_a_detail(monkeypatch, reason):
    """Nothing else has anything specific to add, so nothing else may claim to."""
    from utils.hardware import hardware as hw

    monkeypatch.setattr(hw, "CHAT_ONLY_REASON", reason)
    monkeypatch.setattr(hw, "CHAT_ONLY_DETAIL", None)
    assert hw.CHAT_ONLY_DETAIL is None


# The detail only means anything beside the reason it explains, so it travels with it
# through every place the verdict is saved, restored, discarded or read.
def test_a_failed_forced_redetect_restores_the_detail(monkeypatch):
    """detect_hardware() puts back the verdict a raising pass clobbered, detail included.

    Without it the restored verdict is still mlx_unavailable but has lost the blocker,
    so the row goes back to the generic message this change exists to replace.
    """
    from utils.hardware import hardware as hw

    monkeypatch.setattr(hw, "DEVICE", hw.DeviceType.CPU)
    monkeypatch.setattr(hw, "CHAT_ONLY", True)
    monkeypatch.setattr(hw, "CHAT_ONLY_REASON", "mlx_unavailable")
    monkeypatch.setattr(hw, "CHAT_ONLY_DETAIL", "mlx-vlm 0.1.0 is older than 0.4.4")
    hw.DETECTION_COMPLETE.set()

    def explode():
        # A pass clears the verdict before it probes; this one dies in between.
        hw.CHAT_ONLY_REASON = None
        hw.CHAT_ONLY_DETAIL = None
        raise RuntimeError("probe died")

    monkeypatch.setattr(hw, "_detect_hardware_locked", explode)
    with pytest.raises(RuntimeError):
        hw.detect_hardware()

    assert hw.CHAT_ONLY_REASON == "mlx_unavailable"
    assert hw.CHAT_ONLY_DETAIL == "mlx-vlm 0.1.0 is older than 0.4.4"


def test_a_discarded_verdict_takes_the_detail_with_it(monkeypatch):
    from utils.hardware import hardware as hw

    monkeypatch.setattr(hw, "CHAT_ONLY_REASON", "mlx_unavailable")
    monkeypatch.setattr(hw, "CHAT_ONLY_DETAIL", "mlx-lm is not installed (needs >=0.31.2)")
    hw._discard_detection_locked()
    assert hw.CHAT_ONLY_REASON is None
    assert hw.CHAT_ONLY_DETAIL is None


def test_health_reads_the_detail_inside_the_guarded_snapshot(monkeypatch):
    """A forced re-detect starting mid-read must not pair one pass's reason with another's
    detail, which is what reading the global after the snapshot allowed."""
    import main

    monkeypatch.setattr(main._hw_module, "DEVICE", main._hw_module.DeviceType.CPU)
    monkeypatch.setattr(main._hw_module, "CHAT_ONLY", True)
    monkeypatch.setattr(main._hw_module, "CHAT_ONLY_REASON", "mlx_unavailable")
    monkeypatch.setattr(main._hw_module, "CHAT_ONLY_DETAIL", "mlx-vlm 0.1.0 is older than 0.4.4")
    main._hw_module.DETECTION_COMPLETE.set()

    snapshot = main._hardware_snapshot()
    assert snapshot is not None
    assert len(snapshot) == 3, "the detail has to come out of the same guarded read"
    assert snapshot[1] == "mlx_unavailable"
    assert snapshot[2] == "mlx-vlm 0.1.0 is older than 0.4.4"

    # A later pass clearing the globals cannot change what this snapshot reports.
    monkeypatch.setattr(main._hw_module, "CHAT_ONLY_DETAIL", None)
    assert snapshot[2] == "mlx-vlm 0.1.0 is older than 0.4.4"


# The gate and the detail ask the same question, so it is asked once. On the host that
# needs the detail the mlx imports are the ones that hang, and this module already treats
# them as able to park indefinitely; asking twice there is what a second call costs.
def test_the_gate_measures_the_stack_once(monkeypatch):
    from utils.hardware import hardware as hw

    monkeypatch.setattr(hw, "_MLX_BLOCKERS_MEASURED", None)
    calls: list[int] = []

    def counted() -> list[str]:
        calls.append(1)
        return ["mlx-vlm 0.1.0 is older than 0.4.4"]

    monkeypatch.setattr(mr, "mlx_stack_blockers", counted)
    assert hw._has_usable_mlx_stack() is False
    assert hw._mlx_stack_detail() == "mlx-vlm 0.1.0 is older than 0.4.4"
    assert len(calls) == 1, f"the stack was probed {len(calls)} times for one verdict"


def test_a_measurement_is_used_once_and_not_kept(monkeypatch):
    """A list left over from an earlier pass describes a stack since re-measured."""
    from utils.hardware import hardware as hw

    monkeypatch.setattr(hw, "_MLX_BLOCKERS_MEASURED", None)
    monkeypatch.setattr(mr, "mlx_stack_blockers", lambda: ["mlx is not installed"])
    hw._has_usable_mlx_stack()
    assert hw._mlx_stack_detail() == "mlx is not installed"
    assert hw._MLX_BLOCKERS_MEASURED is None

    # Nothing measured, so the detail measures for itself rather than reusing the above.
    monkeypatch.setattr(mr, "mlx_stack_blockers", lambda: ["mlx-lm is not installed"])
    assert hw._mlx_stack_detail() == "mlx-lm is not installed"


def test_a_healthy_gate_still_reads_as_usable(monkeypatch):
    from utils.hardware import hardware as hw

    monkeypatch.setattr(hw, "_MLX_BLOCKERS_MEASURED", None)
    monkeypatch.setattr(mr, "mlx_stack_blockers", lambda: [])
    assert hw._has_usable_mlx_stack() is True


def test_an_unreadable_gate_falls_back_to_the_bare_import(monkeypatch):
    """mlx_repair should always import; a host where it cannot is not forced chat-only."""
    from utils.hardware import hardware as hw

    monkeypatch.setattr(hw, "_MLX_BLOCKERS_MEASURED", ["stale"])

    def explode() -> list[str]:
        raise RuntimeError("mlx_repair is unimportable")

    monkeypatch.setattr(mr, "mlx_stack_blockers", explode)
    monkeypatch.setattr(hw, "_has_mlx", lambda: True)
    assert hw._has_usable_mlx_stack() is True
    # And it published nothing, rather than leaving the stale list to be read as this
    # pass's answer.
    assert hw._MLX_BLOCKERS_MEASURED is None


# A blocker line goes into /api/health and into the Train row's native tooltip. Neither
# renders a paragraph, and a dyld failure lists every path it tried.
def test_a_long_import_error_is_folded_to_one_bounded_line(monkeypatch):
    _fake_versions(monkeypatch, _healthy())

    def explode(module: str):
        raise ImportError(
            "dlopen failed:\n  tried: '/opt/one/lib.so' (no such file)\n"
            "  tried: '/opt/two/lib.so' (mach-o, but wrong architecture)\n" + "x" * 400
        )

    monkeypatch.setattr(mr.importlib, "import_module", explode)
    blocker = mr.mlx_stack_blockers()[0]
    assert "\n" not in blocker
    assert len(blocker) < 200, f"{len(blocker)} chars reaches the tooltip: {blocker}"
    assert blocker.endswith("...)"), blocker
    # Still says which module and which error, which is the whole point of the line.
    assert blocker.startswith("mlx.core does not import (ImportError:")


def test_a_malformed_installed_version_is_bounded_too(monkeypatch):
    """Version metadata is read from disk, and an interrupted install can leave junk."""
    junk = "1.0\n" + "y" * 500
    _fake_versions(monkeypatch, _healthy(mlx = junk))
    monkeypatch.setattr(mr, "_mlx_runtime_import_blocker", lambda: None)
    blocker = mr.mlx_stack_blockers()[0]
    assert "\n" not in blocker
    assert len(blocker) < 200, f"{len(blocker)} chars reaches the tooltip"
    assert blocker.startswith("mlx 1.0 y")


# A repair that installed and then failed its own validation has still changed the
# environment, so the verdict beside it was measured against a stack that no longer exists:
# it can name a package the install has since put there.
def _fake_hardware(monkeypatch, calls: list[str]):
    """Stand the real hardware module's re-detection down, keeping the module identity."""
    from contextlib import nullcontext

    from utils.hardware import hardware as hw

    monkeypatch.setattr(hw, "detect_hardware", lambda: calls.append("detect"))
    monkeypatch.setattr(hw, "owning_detection_epoch", lambda epoch: nullcontext())
    monkeypatch.setattr(hw, "current_detection_epoch", lambda: None)
    return hw


def test_a_repair_that_failed_validation_still_remeasures(monkeypatch):
    called: list[str] = []
    _fake_hardware(monkeypatch, called)
    monkeypatch.setattr(mr, "attempt_mlx_repair", lambda: False)
    monkeypatch.setattr(mr, "_environment_mutated", True)
    mr._run_repair_and_redetect()
    assert called == ["detect"], "the stale detail was left describing a replaced stack"


def test_a_repair_that_never_ran_does_not_remeasure(monkeypatch):
    """Nothing changed, so re-running the mlx imports would cost latency for nothing."""
    called: list[str] = []
    _fake_hardware(monkeypatch, called)
    monkeypatch.setattr(mr, "attempt_mlx_repair", lambda: False)
    monkeypatch.setattr(mr, "_environment_mutated", False)
    mr._run_repair_and_redetect()
    assert called == []


def test_a_successful_repair_still_remeasures(monkeypatch):
    called: list[str] = []
    _fake_hardware(monkeypatch, called)
    monkeypatch.setattr(mr, "attempt_mlx_repair", lambda: True)
    monkeypatch.setattr(mr, "_environment_mutated", True)
    mr._run_repair_and_redetect()
    assert called == ["detect"]


# uv passes every package with --reinstall-package, so it removes and replaces them as it
# goes: a timeout or a non-zero exit part way through has already changed the stack.
@pytest.mark.parametrize(
    "outcome",
    ["timeout", "nonzero"],
)
def test_an_install_that_died_part_way_still_counts_as_mutating(monkeypatch, outcome):
    import subprocess as sp

    monkeypatch.setattr(mr, "_environment_mutated", False)
    monkeypatch.setattr(mr, "_uv_install_cmd", lambda *a, **k: ["uv", "pip", "install"])
    monkeypatch.setattr(mr, "_transformers_constraint_args", lambda: ([], None))
    monkeypatch.setattr(mr, "_mlx_install_env", dict)

    def run(*a, **k):
        if outcome == "timeout":
            raise sp.TimeoutExpired(cmd = "uv", timeout = 1)
        return sp.CompletedProcess(args = "uv", returncode = 1, stdout = "boom")

    monkeypatch.setattr(mr.subprocess, "run", run)
    assert mr.attempt_mlx_repair() is False
    assert mr._environment_mutated is True, (
        "a half-applied reinstall leaves the pre-repair detail describing a stack that "
        "is no longer on disk"
    )


def test_a_venv_uv_refuses_is_not_marked_mutated(monkeypatch):
    """uv gave up before resolving an interpreter, so it installed nothing."""
    import subprocess as sp

    monkeypatch.setattr(mr, "_environment_mutated", False)
    monkeypatch.setattr(mr, "_uv_install_cmd", lambda *a, **k: ["uv", "pip", "install"])
    monkeypatch.setattr(mr, "_transformers_constraint_args", lambda: ([], None))
    monkeypatch.setattr(mr, "_mlx_install_env", dict)
    monkeypatch.setattr(
        mr.subprocess,
        "run",
        lambda *a, **k: sp.CompletedProcess(
            args = "uv",
            returncode = 2,
            stdout = mr._UNRESOLVED_PYTHON_MARKER + " at /x",
        ),
    )
    assert mr.attempt_mlx_repair() is False
    assert mr._environment_mutated is False


def test_uv_missing_never_marks_the_environment(monkeypatch):
    monkeypatch.setattr(mr, "_environment_mutated", False)
    monkeypatch.setattr(mr, "_uv_install_cmd", lambda *a, **k: None)
    monkeypatch.setattr(mr, "_transformers_constraint_args", lambda: ([], None))
    assert mr.attempt_mlx_repair() is False
    assert mr._environment_mutated is False


# ── Concurrent first-import race (#9120) ──────────────────────────────────────


class _FakeImportWithRace:
    """import_module that raises the partial-import ImportError N times for one
    module (the shape another startup thread mid-`import transformers` produces),
    then succeeds."""

    def __init__(self, racing_module: str, failures: int):
        self._racing_module = racing_module
        self._failures_left = failures
        self.calls: list[str] = []

    def __call__(self, module: str):
        self.calls.append(module)
        if module == self._racing_module and self._failures_left > 0:
            self._failures_left -= 1
            raise ImportError(
                "cannot import name 'AutoTokenizer' from 'transformers' "
                "(/venv/site-packages/transformers/__init__.py)"
            )
        return None


def test_concurrent_partial_import_is_retried_not_reported(monkeypatch):
    # The #9120 shape: a healthy stack whose first mlx_lm import races another
    # startup thread's first transformers import. One retry must clear it —
    # reporting it greyed out Train/Export for the whole process lifetime.
    fake = _FakeImportWithRace("mlx_lm", failures = 1)
    monkeypatch.setattr(mr.importlib, "import_module", fake)
    sleeps: list[float] = []
    monkeypatch.setattr(mr.time, "sleep", sleeps.append)

    assert mr._mlx_runtime_import_blocker() is None
    assert fake.calls.count("mlx_lm") == 2, "the race must be retried, not reported"


def test_broken_install_import_is_reported_on_the_first_attempt(monkeypatch):
    # A missing module is a real install problem, never the race: no retries,
    # no added latency before the chat-only verdict.
    fake = _FakeImportWithRace("mlx_lm", failures = 99)
    fake_bad = fake.__call__

    def strict(module: str):
        fake.calls.append(module)
        if module == "mlx_lm":
            raise ImportError("No module named 'mlx_lm'")
        return None

    monkeypatch.setattr(mr.importlib, "import_module", strict)
    monkeypatch.setattr(mr.time, "sleep", lambda _s: pytest.fail("no retry expected"))

    blocker = mr._mlx_runtime_import_blocker()
    assert blocker is not None and "No module named 'mlx_lm'" in blocker
    assert fake.calls.count("mlx_lm") == 1


def test_persistent_partial_import_reports_after_exhausting_retries(monkeypatch):
    # If the signature persists past the retries it is not a race after all:
    # the verdict must still name it instead of looping or passing.
    fake = _FakeImportWithRace("mlx_lm", failures = 99)
    monkeypatch.setattr(mr.importlib, "import_module", fake)
    sleeps: list[float] = []
    monkeypatch.setattr(mr.time, "sleep", sleeps.append)

    blocker = mr._mlx_runtime_import_blocker()
    assert blocker is not None and "cannot import name" in blocker
    assert fake.calls.count("mlx_lm") == 3
    assert len(sleeps) == 2, "backoff between attempts, none after the last"
