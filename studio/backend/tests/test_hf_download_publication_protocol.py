# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Exhaustive schedules use the production decisions, with transfers as explicit fake steps."""

import hashlib
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from .test_hf_download_pinned_revision import _BACKEND, _FakeHub, _NEW, _OLD, _REPO


_RACE_CHILD = r"""
import hashlib, os, sys, threading
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from huggingface_hub import snapshot_download, hf_hub_download
from hub.workers import hf_download as w
from hub.utils.download_manifest import ExpectedFile
repo_id, old, new = 'fake/repo', 'a' * 40, 'b' * 40
cache = Path(os.environ['HF_HUB_CACHE'])
weights = 'model.safetensors'
kwargs = dict(repo_id=repo_id, token=False, allow_patterns=[weights])
old_path = Path(snapshot_download(revision=old, **kwargs))
ref = old_path.parent.parent / 'refs' / 'main'
ref.parent.mkdir(parents=True, exist_ok=True)
ref.write_text(old)
old_ready, new_ready, old_done = (threading.Event() for _ in range(3))
results, errors = {}, []
def old_transfer(**kw):
    old_ready.set()
    assert new_ready.wait(10)
    return snapshot_download(**kw)
def new_transfer(**kw):
    new_ready.set()
    assert old_done.wait(10)
    return snapshot_download(**kw)
def run(name, transfer, commit, size):
    try:
        results[name] = w._pinned_snapshot_download(
            transfer, commit, expected_files=[ExpectedFile(weights, size)], **kwargs)
    except BaseException as exc:
        errors.append(repr(exc))
    finally:
        if name == 'old':
            old_done.set()
a = threading.Thread(target=run, args=('old', old_transfer, old, old_path.joinpath(weights).stat().st_size))
b = threading.Thread(target=run, args=('new', new_transfer, new, int(sys.argv[2])))
a.start()
assert old_ready.wait(10)
b.start()
a.join(20)
b.join(20)
assert not a.is_alive() and not b.is_alive()
assert not errors, errors
assert set(results) == {'old', 'new'}, results
assert Path(results['new']).joinpath(weights).is_file()
loaded = hf_hub_download(repo_id, weights, cache_dir=str(cache), local_files_only=True)
assert new in loaded, (ref.read_text(), loaded)
assert hashlib.sha256(Path(loaded).read_bytes()).hexdigest() == sys.argv[3]
print('Both workers succeeded; offline hf_hub_download resolved verified B.')
"""


def test_cached_a_finishes_before_later_b_does_not_disown_b(tmp_path):
    """A's no-op completion must not revoke the later-started B's publication."""
    hub = _FakeHub()
    try:
        weights = hub.commits[_NEW]["model.safetensors"]
        env = dict(os.environ)
        env.pop("HF_TOKEN", None)
        env.update(
            HF_ENDPOINT = hub.endpoint,
            HF_HOME = str(tmp_path / "hf"),
            HF_HUB_CACHE = str(tmp_path / "hf" / "hub"),
            HF_HUB_DISABLE_XET = "1",
            HF_HUB_ENABLE_HF_TRANSFER = "0",
            HF_HUB_DISABLE_TELEMETRY = "1",
            HF_HUB_DISABLE_IMPLICIT_TOKEN = "1",
            HF_HUB_OFFLINE = "0",
            CUDA_VISIBLE_DEVICES = "",
        )
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                _RACE_CHILD,
                str(_BACKEND),
                str(len(weights)),
                hashlib.sha256(weights).hexdigest(),
            ],
            env = env,
            text = True,
            capture_output = True,
            timeout = 60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    finally:
        hub.close()


def _schedules(workers, canonical_starts):
    """All linear extensions of start < transfer-done < verify-result < finish.

    For three workers, remove only label symmetry: first starter is worker 0,
    second worker 1, third worker 2. All target/outcome assignments still run.
    Every labelled schedule maps to exactly one of these via worker renaming.
    """

    def visit(phases, history):
        if all(phase == 4 for phase in phases):
            yield history
            return
        first_unstarted = next((i for i, phase in enumerate(phases) if not phase), None)
        for worker, phase in enumerate(phases):
            if phase == 4 or (canonical_starts and phase == 0 and worker != first_unstarted):
                continue
            updated = list(phases)
            updated[worker] += 1
            yield from visit(updated, (*history, (worker, phase)))

    return tuple(visit([0] * workers, ()))


@pytest.mark.parametrize("workers,orderings", [(2, 70), (3, 5775)])
@pytest.mark.parametrize("previous", [None, _OLD, _NEW])
def test_every_publication_interleaving(workers, orderings, previous):
    from hub.workers import hf_download as w

    schedules = _schedules(workers, canonical_starts = workers == 3)
    assert len(schedules) == orderings
    checked = 0
    # The fake transfer records completion; verification independently returns
    # success, failure, or cancellation. Only the final step may change a ref.
    for assignments in itertools.product(
        itertools.product((_OLD, _NEW), ("ok", "fail", "cancel")), repeat = workers
    ):
        for schedule in schedules:
            state = w._MainRefState(published_commit = previous)
            started, succeeded, sequences, transferred, verified, finished = (
                [],
                set(),
                {},
                set(),
                {},
                set(),
            )
            ref = previous
            for worker, phase in schedule:
                commit, outcome = assignments[worker]
                if phase == 0:
                    started.append(worker)
                    state, sequences[worker] = w._start_main_ref(state, commit)
                elif phase == 1:
                    transferred.add(worker)  # fake transfer: no ref side effects
                elif phase == 2:
                    assert worker in transferred
                    verified[worker] = outcome == "ok"
                else:
                    finished.add(worker)
                    finished.add(worker)
                    ok = verified[worker]
                    before = ref
                    state = w._finish_main_ref(state, sequences[worker], ok)
                    if ok:
                        succeeded.add(worker)
                        ref = state.published_commit
                    else:
                        assert ref == before  # failure/cancel cannot hide a peer
                # Independent oracle: scan chronological starts backwards for a
                # terminal verified success, without using the reducer's sequence.
                winner = next((i for i in reversed(started) if i in succeeded), None)
                expected = previous if winner is None else assignments[winner][0]
                assert ref == expected, (assignments, schedule, worker, phase)
                assert state.published_commit == expected
                assert dict(state.attempts) == {
                    sequences[i]: assignments[i][0] for i in started if i not in finished
                }
            assert not state.attempts
            checked += 1
    assert checked == orderings * 6**workers
    print(f"{workers} workers, initial={previous}: {checked:,} schedules checked")


@pytest.mark.parametrize(
    "outcome", ["fail", "cancel", "verify_fail", "verify_cancel", "wrong_commit"]
)
@pytest.mark.parametrize("previous", [None, _OLD])
def test_failed_attempt_never_exposes_landed_partial(monkeypatch, tmp_path, outcome, previous):
    from huggingface_hub import constants
    from hub.workers import hf_download as w
    from hub.utils import download_manifest as dm

    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    repo = tmp_path / "models--fake--repo"
    ref = repo / "refs" / "main"
    if previous is not None:
        ref.parent.mkdir(parents = True)
        ref.write_text(previous)
    calls = []

    def transfer(**kwargs):
        calls.append(kwargs["revision"])
        assert w._read_ref(ref) == previous
        target = repo / "snapshots" / (_OLD if outcome == "wrong_commit" else _NEW)
        target.mkdir(parents = True)
        (target / "weights").write_bytes(b"partial")
        if outcome == "fail":
            raise OSError("transfer failed after bytes landed")
        if outcome == "cancel":
            raise KeyboardInterrupt
        return str(target)

    if outcome == "verify_cancel":

        def cancel(*args):
            raise KeyboardInterrupt

        monkeypatch.setattr(dm, "verify_against_disk", cancel)
    with pytest.raises(KeyboardInterrupt if "cancel" in outcome else OSError):
        w._pinned_snapshot_download(
            transfer,
            _NEW,
            repo_id = _REPO,
            expected_files = [dm.ExpectedFile("weights", 100)],
        )
    assert calls == [_NEW]  # failure never retries an older branch snapshot
    assert w._read_ref(ref) == previous
    state_file = tmp_path / ".locks" / repo.name / "studio-main.json"
    state = w._load_main_ref_state(state_file, previous)
    assert not state.attempts
    assert state.published_sequence == 0


def test_coordination_failure_does_not_transfer_unpinned(monkeypatch, tmp_path):
    from huggingface_hub import constants
    from hub.workers import hf_download as w

    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    state_file = tmp_path / ".locks" / "models--fake--repo" / "studio-main.json"
    state_file.parent.mkdir(parents = True)
    state_file.write_text("{broken journal")

    def transfer(**kwargs):
        pytest.fail("coordination failure must not run an unpinned transfer")

    with pytest.raises(json.JSONDecodeError):
        w._pinned_snapshot_download(transfer, _NEW, repo_id = _REPO)


def test_temporary_ref_files_stay_outside_refs(monkeypatch, tmp_path):
    from hub.workers import hf_download as w

    ref = tmp_path / "repo" / "refs" / "main"
    replace = os.replace

    def checked_replace(source, target):
        assert Path(source).parent == ref.parent.parent
        assert list(ref.parent.iterdir()) == []
        replace(source, target)

    monkeypatch.setattr(w.os, "replace", checked_replace)
    w._write_ref(ref, _OLD)
    assert list(ref.parent.iterdir()) == [ref]
    assert ref.read_text() == _OLD


@pytest.mark.parametrize("failure", ["state", "ref", "cancel_after_replace"])
def test_failed_publication_does_not_disown_older_success(monkeypatch, tmp_path, failure):
    from huggingface_hub import constants
    from hub.workers import hf_download as w

    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    repo = tmp_path / "models--fake--repo"
    ref = repo / "refs" / "main"
    ref.parent.mkdir(parents = True)
    baseline = "c" * 40
    ref.write_text(baseline)
    original_write = w._write_ref
    injected = False

    def write(path, value):
        nonlocal injected
        is_publication = (
            path.name == "studio-main.json" and json.loads(value)["published_sequence"] == 2
            if failure == "state"
            else path == ref and value == _NEW
        )
        if is_publication and not injected:
            injected = True
            if failure == "cancel_after_replace":
                original_write(path, value)
                raise KeyboardInterrupt
            raise OSError("publication write failed")
        original_write(path, value)

    monkeypatch.setattr(w, "_write_ref", write)

    def new_transfer(**kwargs):
        target = repo / "snapshots" / _NEW
        target.mkdir(parents = True)
        return str(target)

    def old_transfer(**kwargs):
        with pytest.raises(KeyboardInterrupt if failure == "cancel_after_replace" else OSError):
            w._pinned_snapshot_download(new_transfer, _NEW, repo_id = _REPO)
        assert injected
        assert ref.read_text() == baseline
        target = repo / "snapshots" / _OLD
        target.mkdir(parents = True)
        return str(target)

    w._pinned_snapshot_download(old_transfer, _OLD, repo_id = _REPO)
    assert ref.read_text() == _OLD
    state_file = tmp_path / ".locks" / repo.name / "studio-main.json"
    state = w._load_main_ref_state(state_file, baseline)
    assert state.published_sequence == 1
    assert not state.attempts
