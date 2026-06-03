# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Tests for core/training/training.py:_cleanup_cancelled_checkpoints."""

import os
import sys
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))


@pytest.fixture
def outputs_setup(tmp_path, monkeypatch):
    """Point outputs_root() at a temp dir so cleanup is allowed to run on it.

    The training module binds ``outputs_root`` at import time
    (``from utils.paths import outputs_root``), so we have to patch
    the symbol on the importer module, not on storage_roots.
    """
    from core.training import training as training_mod

    monkeypatch.setattr(training_mod, "outputs_root", lambda: tmp_path)
    return tmp_path


def _mk_dir(parent: Path, name: str) -> Path:
    p = parent / name
    p.mkdir()
    (p / "marker.txt").write_text(name)
    return p


def test_completed_checkpoints_are_preserved(outputs_setup):
    """The big regression: prior to this fix, every completed
    checkpoint-N/ was rmtree'd on Cancel, destroying resume points."""
    from core.training.training import _cleanup_cancelled_checkpoints

    out = outputs_setup / "run-1"
    out.mkdir()
    ckpts = [_mk_dir(out, f"checkpoint-{n}") for n in (200, 400, 600)]
    tmp = _mk_dir(out, "tmp-checkpoint-800")

    _cleanup_cancelled_checkpoints(out)

    for c in ckpts:
        assert c.exists(), f"completed {c.name} was destroyed"
        assert (c / "marker.txt").exists()
    assert not tmp.exists(), "in-flight tmp-checkpoint-800 should be removed"


def test_in_flight_tmp_checkpoints_removed(outputs_setup):
    from core.training.training import _cleanup_cancelled_checkpoints

    out = outputs_setup / "run-2"
    out.mkdir()
    _mk_dir(out, "tmp-checkpoint-100")
    _mk_dir(out, "tmp-checkpoint-200")
    _mk_dir(out, "checkpoint-50")  # completed, kept

    _cleanup_cancelled_checkpoints(out)

    assert not (out / "tmp-checkpoint-100").exists()
    assert not (out / "tmp-checkpoint-200").exists()
    assert (out / "checkpoint-50").exists()


def test_non_checkpoint_dirs_left_alone(outputs_setup):
    from core.training.training import _cleanup_cancelled_checkpoints

    out = outputs_setup / "run-3"
    out.mkdir()
    _mk_dir(out, "logs")
    _mk_dir(out, "tensorboard")
    _mk_dir(out, "checkpoint-final")  # non-int suffix, kept
    _mk_dir(out, "checkpoint-best")
    _mk_dir(out, "tmp-checkpoint-99")

    _cleanup_cancelled_checkpoints(out)

    for n in ("logs", "tensorboard", "checkpoint-final", "checkpoint-best"):
        assert (out / n).exists(), f"{n} should be preserved"
    assert not (out / "tmp-checkpoint-99").exists()


def test_output_dir_outside_outputs_root_is_refused(tmp_path, monkeypatch):
    """Containment check: even if a bug passed an output_dir outside
    outputs_root, the cleanup must refuse to touch it."""
    from core.training import training as training_mod
    from core.training.training import _cleanup_cancelled_checkpoints

    inside = tmp_path / "inside"
    inside.mkdir()
    monkeypatch.setattr(training_mod, "outputs_root", lambda: inside)

    outside = tmp_path / "outside"
    outside.mkdir()
    _mk_dir(outside, "tmp-checkpoint-1")

    _cleanup_cancelled_checkpoints(outside)

    assert (
        outside / "tmp-checkpoint-1"
    ).exists(), "must not rmtree under a path outside outputs_root"


def test_symlinked_output_dir_skipped(outputs_setup):
    """A symlinked output_dir is skipped so the realpath check can't be
    leveraged to delete content via a symlink trick."""
    from core.training.training import _cleanup_cancelled_checkpoints

    real = outputs_setup / "real-run"
    real.mkdir()
    _mk_dir(real, "tmp-checkpoint-1")

    link = outputs_setup / "link-run"
    try:
        link.symlink_to(real, target_is_directory = True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported on this filesystem / platform")

    _cleanup_cancelled_checkpoints(link)

    assert (real / "tmp-checkpoint-1").exists(), "symlinked output_dir must be skipped"


def test_missing_output_dir_is_noop(outputs_setup):
    from core.training.training import _cleanup_cancelled_checkpoints

    _cleanup_cancelled_checkpoints(outputs_setup / "does-not-exist")
    # Should not raise; nothing to assert beyond non-failure.


def test_symlinked_child_skipped(outputs_setup):
    """A symlinked tmp-checkpoint-* child must not be deleted, so the
    realpath bypass cannot redirect rmtree to arbitrary content."""
    from core.training.training import _cleanup_cancelled_checkpoints

    out = outputs_setup / "run-symchild"
    out.mkdir()
    target = outputs_setup / "external"
    target.mkdir()
    (target / "important.txt").write_text("keep me")

    link = out / "tmp-checkpoint-99"
    try:
        link.symlink_to(target, target_is_directory = True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported on this filesystem / platform")

    _cleanup_cancelled_checkpoints(out)

    assert (
        target / "important.txt"
    ).exists(), "symlink target outside outputs_root must not be rmtree'd"


def test_non_numeric_tmp_checkpoint_suffix_preserved(outputs_setup):
    """HF Trainer's partials are tmp-checkpoint-<step>. A user-named
    tmp-checkpoint-final / tmp-checkpoint-backup / tmp-checkpoint-notes
    must NOT be deleted by the cancel cleanup."""
    from core.training.training import _cleanup_cancelled_checkpoints

    out = outputs_setup / "run-non-numeric"
    out.mkdir()
    numeric = _mk_dir(out, "tmp-checkpoint-100")
    user_final = _mk_dir(out, "tmp-checkpoint-final")
    user_backup = _mk_dir(out, "tmp-checkpoint-backup")
    user_notes = _mk_dir(out, "tmp-checkpoint-user-notes")

    _cleanup_cancelled_checkpoints(out)

    assert not numeric.exists(), "in-flight tmp-checkpoint-100 should be removed"
    assert user_final.exists(), "user dir tmp-checkpoint-final must be preserved"
    assert user_backup.exists(), "user dir tmp-checkpoint-backup must be preserved"
    assert user_notes.exists(), "user dir tmp-checkpoint-user-notes must be preserved"
