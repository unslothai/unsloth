# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A --no-torch install declined the training stack, so the self-heal must not put it back.

`install.sh --no-torch` is GGUF-only by request. The runtime self-heal used to gate on
Apple Silicon, the kill switch and stack availability, but never on the install mode, so
it reinstalled mlx/mlx-lm/mlx-vlm about 20 seconds after first launch and turned Train and
Export on for someone who had opted out.

The half that is easy to get wrong is the unknown case. `recorded_no_torch()` answers
Optional[bool], and None means "nothing recorded it", which is what every install predating
the manifest key looks like. Reading None as False keeps today's behaviour, which is right;
reading it as True would silently disable the repair for every older install on the
planet. These tests pin both halves, plus every way the lookup can fail.
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import utils.mlx_repair as mr  # noqa: E402


@pytest.fixture(autouse = True)
def _isolated(monkeypatch):
    """Same reset as test_mlx_repair.py: the attempt latch is process-global, and a worker
    that outlives its test runs the real repair against the next one."""
    monkeypatch.setattr(mr, "_attempted", False)
    monkeypatch.setattr(mr, "_environment_mutated", False)
    monkeypatch.delenv(mr.DISABLE_ENV_VAR, raising = False)
    yield
    for thread in threading.enumerate():
        if thread.name == "mlx-autorepair":
            thread.join(timeout = 5)
            assert not thread.is_alive()


def _install_manifest_returning(monkeypatch, value):
    """Stand in for studio.install_manifest, which _installed_without_torch imports lazily.

    A callable raising is spelled by passing an exception instance, since that is the other
    thing the real module can do here (an unreadable manifest, a partially installed tree).
    """
    import types

    module = types.ModuleType("studio.install_manifest")

    def recorded_no_torch():
        if isinstance(value, BaseException):
            raise value
        return value

    module.recorded_no_torch = recorded_no_torch
    package = types.ModuleType("studio")
    package.install_manifest = module
    monkeypatch.setitem(sys.modules, "studio", package)
    monkeypatch.setitem(sys.modules, "studio.install_manifest", module)


# --- _installed_without_torch: only a literal True opts out ------------------------


def test_true_means_no_torch(monkeypatch):
    _install_manifest_returning(monkeypatch, True)
    assert mr._installed_without_torch() is True


def test_false_means_a_normal_install(monkeypatch):
    _install_manifest_returning(monkeypatch, False)
    assert mr._installed_without_torch() is False


def test_unknown_keeps_todays_behaviour(monkeypatch):
    """None is an install made before anything recorded the mode. It must read as False, or
    this change would disable the self-heal for every venv installed before it landed."""
    _install_manifest_returning(monkeypatch, None)
    assert mr._installed_without_torch() is False


@pytest.mark.parametrize("value", ["1", "true", 1, ["no-torch"], object()])
def test_a_truthy_non_bool_is_not_an_opt_out(monkeypatch, value):
    """recorded_no_torch normalises a hand-edited manifest itself, so anything that reaches
    here as a non-bool is a contract violation, not consent. `is True` says so."""
    _install_manifest_returning(monkeypatch, value)
    assert mr._installed_without_torch() is False


@pytest.mark.parametrize(
    "error",
    [ImportError("no studio package"), OSError("unreadable manifest"), ValueError("bad json")],
)
def test_a_failed_lookup_fails_open(monkeypatch, error):
    """Fail open, not closed: an unreadable manifest must not quietly cost an Apple Silicon
    user their training stack."""
    _install_manifest_returning(monkeypatch, error)
    assert mr._installed_without_torch() is False


def test_a_missing_studio_package_fails_open(monkeypatch):
    monkeypatch.setitem(sys.modules, "studio", None)  # import raises
    assert mr._installed_without_torch() is False


def _real_install_manifest(monkeypatch, venv_root: Path):
    """The shipped studio/install_manifest.py, reading a real manifest file.

    Loaded by path and registered as `studio.install_manifest`, because that is the name
    _installed_without_torch imports and the repo root is not on sys.path in the backend
    test job. Everything below it is the real code: read_manifest, the marker fallback and
    the truthy-string tolerance all run for real."""
    import importlib.util
    import types

    source = Path(__file__).resolve().parents[2] / "install_manifest.py"
    spec = importlib.util.spec_from_file_location("studio.install_manifest", source)
    module = importlib.util.module_from_spec(spec)
    package = types.ModuleType("studio")
    package.install_manifest = module
    monkeypatch.setitem(sys.modules, "studio", package)
    monkeypatch.setitem(sys.modules, "studio.install_manifest", module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "venv_root", lambda: venv_root)
    return module


@pytest.mark.parametrize(
    ("manifest", "expected"),
    [
        ({"no_torch": True}, True),
        ({"no_torch": False}, False),
        ({"no_torch": "1"}, True),  # tolerated hand edit
        ({"no_torch": "no"}, False),
        ({"steps_total": 12}, False),  # an install predating the key
        (None, False),  # no manifest at all
    ],
)
def test_against_a_real_manifest_on_disk(monkeypatch, tmp_path, manifest, expected):
    """The same question asked through the shipped manifest reader rather than a stub, so a
    change to its parsing cannot pass this file while breaking the gate."""
    module = _real_install_manifest(monkeypatch, tmp_path)
    if manifest is not None:
        (tmp_path / module.MANIFEST_NAME).write_text(json.dumps(manifest), encoding = "utf-8")
    assert mr._installed_without_torch() is expected


def test_a_corrupt_manifest_does_not_disable_the_repair(monkeypatch, tmp_path):
    module = _real_install_manifest(monkeypatch, tmp_path)
    (tmp_path / module.MANIFEST_NAME).write_text("{not json", encoding = "utf-8")
    assert mr._installed_without_torch() is False


# --- start_mlx_autorepair_if_needed: the gate itself -------------------------------


def _hardware(monkeypatch, *, blames_mlx: bool):
    """start_mlx_autorepair_if_needed imports utils.hardware.hardware inside the function, so
    patch the module rather than a name on mlx_repair."""
    import utils.hardware.hardware as hw

    monkeypatch.setattr(hw, "verdict_blames_the_mlx_stack", lambda: blames_mlx)
    monkeypatch.setattr(hw, "current_detection_epoch", lambda: 1)
    monkeypatch.setattr(hw, "detect_hardware", lambda: None)
    monkeypatch.setattr(hw, "settle_the_no_torch_verdict", lambda epoch: False)
    overturns = []
    monkeypatch.setattr(
        hw, "overturn_the_mlx_verdict", lambda epoch: bool(overturns.append(epoch)) or True
    )
    return overturns


def _apple_silicon_without_mlx(monkeypatch):
    monkeypatch.setattr(mr, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(mr, "mlx_stack_available", lambda: False)


def test_no_torch_install_starts_no_repair(monkeypatch):
    _apple_silicon_without_mlx(monkeypatch)
    _hardware(monkeypatch, blames_mlx = False)
    monkeypatch.setattr(mr, "_installed_without_torch", lambda: True)
    attempts = []
    monkeypatch.setattr(mr, "attempt_mlx_repair", lambda **_kw: attempts.append(1) or True)

    assert mr.start_mlx_autorepair_if_needed() is False
    assert attempts == [], "a --no-torch install must not be given the training stack back"


def test_no_torch_install_starts_no_repair_even_when_the_verdict_blames_mlx(monkeypatch):
    """The verdict blaming MLX is what carries the opt-out past the first gate, so the
    second one has to hold. Without it, a --no-torch venv whose health check says
    'mlx_unavailable' would still be handed a reinstall, which is the reported bug."""
    _apple_silicon_without_mlx(monkeypatch)
    _hardware(monkeypatch, blames_mlx = True)
    monkeypatch.setattr(mr, "_installed_without_torch", lambda: True)
    attempts = []
    monkeypatch.setattr(mr, "attempt_mlx_repair", lambda **_kw: attempts.append(1) or True)

    assert mr.start_mlx_autorepair_if_needed() is False
    assert attempts == []
    assert mr._attempted is False, "the one-shot latch must not be spent by a declined repair"


def test_a_normal_install_still_repairs(monkeypatch):
    """The regression check: the opt-out must not cost the users it is not about."""
    _apple_silicon_without_mlx(monkeypatch)
    _hardware(monkeypatch, blames_mlx = False)
    monkeypatch.setattr(mr, "_installed_without_torch", lambda: False)
    attempts = []
    monkeypatch.setattr(mr, "attempt_mlx_repair", lambda **_kw: attempts.append(1) or True)

    assert mr.start_mlx_autorepair_if_needed() is True
    for thread in threading.enumerate():
        if thread.name == "mlx-autorepair":
            thread.join(timeout = 5)
    assert attempts == [1]


def test_an_older_install_with_no_recorded_mode_still_repairs(monkeypatch):
    """End to end through the real _installed_without_torch, on the manifest an install
    predating the key leaves: unknown, so the repair runs as it always did."""
    _apple_silicon_without_mlx(monkeypatch)
    _hardware(monkeypatch, blames_mlx = False)
    _install_manifest_returning(monkeypatch, None)
    attempts = []
    monkeypatch.setattr(mr, "attempt_mlx_repair", lambda **_kw: attempts.append(1) or True)

    assert mr.start_mlx_autorepair_if_needed() is True
    for thread in threading.enumerate():
        if thread.name == "mlx-autorepair":
            thread.join(timeout = 5)
    assert attempts == [1]


def test_a_no_torch_install_still_overturns_a_verdict_that_blames_mlx(monkeypatch):
    """Opting out declines a reinstall, not a correct verdict. That is why the opt-out was
    spelled into the kill switch's condition rather than given an early return of its own:
    a venv that does have a usable stack must still be allowed to say so, and a --no-torch
    install can have one, for instance from a `pip install mlx-lm` the user ran themselves."""
    monkeypatch.setattr(mr, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(mr, "mlx_stack_available", lambda: True)
    monkeypatch.setattr(mr, "_installed_without_torch", lambda: True)
    overturns = _hardware(monkeypatch, blames_mlx = True)
    attempts = []
    monkeypatch.setattr(mr, "attempt_mlx_repair", lambda **_kw: attempts.append(1) or True)

    assert mr.start_mlx_autorepair_if_needed() is False
    assert overturns == [1], "the chat-only verdict was left standing against a usable stack"
    assert attempts == [], "an adequate stack needs no reinstall"


def test_a_no_torch_install_settles_its_verdict_once_the_probe_measures_unusable(monkeypatch):
    """With mlx on disk the verdict boots as mlx_unavailable so the sidebar keeps polling for
    the overturn; once the probe measures the stack unusable, nothing is coming, so it settles
    as the opt-out it is and the poll stops on the next read."""
    import utils.hardware.hardware as hw

    _apple_silicon_without_mlx(monkeypatch)
    _hardware(monkeypatch, blames_mlx = True)
    monkeypatch.setattr(mr, "_installed_without_torch", lambda: True)
    settled = []
    monkeypatch.setattr(
        hw, "settle_the_no_torch_verdict", lambda epoch: settled.append(epoch) or True
    )

    assert mr.start_mlx_autorepair_if_needed() is False
    assert settled == [1], "settled with the epoch read before the measurement"


def test_the_kill_switch_does_not_settle_the_verdict_as_no_torch(monkeypatch):
    import utils.hardware.hardware as hw

    _apple_silicon_without_mlx(monkeypatch)
    _hardware(monkeypatch, blames_mlx = True)
    monkeypatch.setattr(mr, "_installed_without_torch", lambda: False)
    monkeypatch.setenv(mr.DISABLE_ENV_VAR, "1")
    settled = []
    monkeypatch.setattr(
        hw, "settle_the_no_torch_verdict", lambda epoch: settled.append(epoch) or True
    )

    assert mr.start_mlx_autorepair_if_needed() is False
    assert settled == [], "the kill switch is a normal install; its verdict is not an opt-out"
