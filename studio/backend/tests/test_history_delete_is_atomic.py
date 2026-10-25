# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Artifacts are removed only after the row is gone; a same-parent rename stages the directory first."""

import shutil
from pathlib import Path

import pytest

from routes import training_history


@pytest.fixture
def outputs(tmp_path, monkeypatch):
    root = tmp_path / "outputs"
    root.mkdir()
    monkeypatch.setattr(training_history, "outputs_root", lambda: root)
    monkeypatch.setattr(
        training_history, "_active_training_output_dir", lambda: None, raising = False
    )
    monkeypatch.setattr(
        training_history, "_output_dir_shared", lambda *a, **k: False, raising = False
    )

    import contextlib

    from core.training import lifecycle

    monkeypatch.setattr(lifecycle, "training_lifecycle_guard", contextlib.nullcontext)
    return root


def _run_dir(outputs, name = "run-1"):
    d = outputs / name
    d.mkdir()
    (d / "adapter_model.safetensors").write_bytes(b"\x00" * 128)
    (d / "adapter_config.json").write_text("{}")
    return d


def test_a_failed_row_delete_leaves_the_artifacts_intact(outputs):
    """The regression: artifacts must not be destroyed before the row is gone."""
    run_dir = _run_dir(outputs)
    before = sorted(p.name for p in run_dir.iterdir())

    outcome, original, staged = training_history._delete_run_output_dir_guarded(
        "run-1", str(run_dir)
    )
    assert outcome == "deleted"
    assert original is not None and staged is not None

    training_history._restore_staged_output_dir(original, staged)

    assert run_dir.is_dir(), (
        "a failed database delete destroyed the artifacts; the row would survive "
        "pointing at a directory that no longer exists, which looks exactly like the "
        "deliberate keep-history outcome"
    )
    assert sorted(p.name for p in run_dir.iterdir()) == before
    assert not staged.exists()


def test_staging_hides_the_directory_before_the_row_is_committed(outputs):
    """Staging is a rename, so the run is logically gone the moment it succeeds."""
    run_dir = _run_dir(outputs)

    outcome, original, staged = training_history._delete_run_output_dir_guarded(
        "run-1", str(run_dir)
    )

    assert outcome == "deleted"
    assert not run_dir.exists()
    assert staged.is_dir()
    assert staged.parent == run_dir.parent, "the rename must stay on the same filesystem"
    assert staged.name.startswith(f".{run_dir.name}.deleting-")


def test_a_committed_delete_purges_the_staged_copy(outputs):
    run_dir = _run_dir(outputs)
    _, original, staged = training_history._delete_run_output_dir_guarded("run-1", str(run_dir))

    assert training_history._purge_staged_output_dir("run-1", original, staged) is True

    assert not staged.exists()
    assert not run_dir.exists()
    assert list(outputs.iterdir()) == []


def test_a_purge_that_fails_puts_the_directory_back_under_its_own_name(outputs, monkeypatch):
    """The staged name is hidden and randomized, so a leaked tombstone is unrecoverable."""
    run_dir = _run_dir(outputs)
    _, original, staged = training_history._delete_run_output_dir_guarded("run-1", str(run_dir))

    def refuse(path):
        raise OSError(16, "Device or resource busy")

    monkeypatch.setattr(training_history.shutil, "rmtree", refuse)

    assert training_history._purge_staged_output_dir("run-1", original, staged) is False
    assert run_dir.is_dir()
    assert not staged.exists()
    assert [p.name for p in outputs.iterdir()] == ["run-1"]


def test_an_active_run_is_refused_before_anything_moves(outputs, monkeypatch):
    run_dir = _run_dir(outputs)
    monkeypatch.setattr(
        training_history, "_active_training_output_dir", lambda: str(run_dir), raising = False
    )

    outcome, original, staged = training_history._delete_run_output_dir_guarded(
        "run-1", str(run_dir)
    )

    assert outcome == "active"
    assert (original, staged) == (None, None)
    assert run_dir.is_dir()


def test_a_shared_output_dir_is_refused_before_anything_moves(outputs, monkeypatch):
    run_dir = _run_dir(outputs)
    monkeypatch.setattr(training_history, "_output_dir_shared", lambda *a, **k: True, raising = False)

    outcome, original, staged = training_history._delete_run_output_dir_guarded(
        "run-1", str(run_dir)
    )

    assert outcome == "shared"
    assert (original, staged) == (None, None)
    assert run_dir.is_dir()


def test_a_missing_shared_output_dir_is_already_deleted(outputs, monkeypatch):
    missing = outputs / "gone"
    monkeypatch.setattr(training_history, "_output_dir_shared", lambda *a, **k: True)

    outcome, original, staged = training_history._delete_run_output_dir_guarded(
        "run-1", str(missing)
    )

    assert outcome == "deleted"
    assert (original, staged) == (None, None)


def test_the_outputs_root_itself_is_never_staged(outputs):
    outcome, original, staged = training_history._delete_run_output_dir_guarded(
        "run-1", str(outputs)
    )

    assert outcome == "failed"
    assert (original, staged) == (None, None)
    assert outputs.is_dir()


def test_an_already_absent_directory_is_a_clean_success(outputs):
    """Nothing to move, nothing to purge -- and a retry after a crash lands here."""
    outcome, original, staged = training_history._delete_run_output_dir_guarded(
        "run-1", str(outputs / "gone")
    )

    assert outcome == "deleted"
    assert (original, staged) == (None, None)


def test_a_path_outside_the_outputs_root_is_refused(outputs, tmp_path):
    stranger = tmp_path / "elsewhere"
    stranger.mkdir()
    (stranger / "keep.txt").write_text("do not touch")

    outcome, original, staged = training_history._delete_run_output_dir_guarded(
        "run-1", str(stranger)
    )

    assert outcome == "failed"
    assert (original, staged) == (None, None)
    assert (stranger / "keep.txt").read_text() == "do not touch"
