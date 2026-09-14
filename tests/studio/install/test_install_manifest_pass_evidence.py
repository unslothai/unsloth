# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The evidence the idempotent dependency pass records and reads back.

Every skip in install_python_stack.py is gated on three things: the previous run
recorded that it did this exact work, the inputs are byte-identical, and a cheap
on-disk check of the OUTPUT still passes. This file covers the pieces of that in
install_manifest.py -- the additive manifest keys, the digests, the constraint
check and the sidecar predicate -- because a false "already done" here is an
install nobody can tell apart from a finished one.
"""

from __future__ import annotations

import ast
import contextlib
import errno
import importlib.util
import json
import os
import pathlib
import stat
import subprocess
import sys
import sysconfig
import tempfile
import textwrap
import time
import unittest.mock

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "studio" / "install_manifest.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("studio_install_manifest_evidence", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


im = _load_module()


# -- digests -------------------------------------------------------------------


def test_pass_inputs_are_a_superset_of_the_tracked_requirements() -> None:
    """The pass reads more files than verify_install fingerprints, and they stay two lists:
    verify_install compares the whole `requirement_files` dict, so widening THAT one reports
    every install in the field as `studio_install_requirements_changed`.
    """
    assert im.PASS_INPUT_FILES[: len(im.TRACKED_REQUIREMENT_FILES)] == (
        im.TRACKED_REQUIREMENT_FILES
    )
    for extra in (
        "diffusers-pin.txt",
        "triton-kernels.txt",
        "single-env/constraints.txt",
    ):
        assert extra in im.PASS_INPUT_FILES
        assert extra not in im.TRACKED_REQUIREMENT_FILES


def test_the_shipped_requirements_tree_supplies_every_pass_input() -> None:
    """A name nothing on disk answers is evidence that can never match."""
    digests = im.pass_input_digests(REPO_ROOT / "studio" / "backend" / "requirements")
    assert set(digests) == set(im.PASS_INPUT_FILES)


def test_an_absent_or_unreadable_input_has_no_digest(tmp_path: pathlib.Path) -> None:
    """None never equals a recorded digest, so the step that reads it runs."""
    assert im.digest_file(tmp_path / "nope.txt") is None
    assert im.digest_file(tmp_path) is None
    target = tmp_path / "there.txt"
    target.write_bytes(b"payload")
    assert im.digest_file(target) == im.digest_file(str(target))
    assert (tmp_path / "nope.txt").name not in im.pass_input_digests(tmp_path)


def test_a_one_byte_edit_changes_the_digest(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "studio.txt"
    target.write_text("structlog\n", encoding = "utf-8")
    before = im.digest_file(target)
    target.write_text("structlog \n", encoding = "utf-8")
    assert im.digest_file(target) != before


# -- additive manifest keys ----------------------------------------------------


def _payload(root: pathlib.Path) -> dict:
    return json.loads((root / im.MANIFEST_NAME).read_text(encoding = "utf-8"))


def test_extra_keys_are_written_and_schema_stays_1(tmp_path: pathlib.Path) -> None:
    im.write_manifest(
        root = tmp_path,
        req_root = tmp_path,
        package_name = "pytest",
        extra = {"pass_inputs": {"studio.txt": "abc"}, "pip_check_ok": True},
    )
    payload = _payload(tmp_path)
    assert payload["schema"] == im.MANIFEST_SCHEMA == 1
    assert payload["pass_inputs"] == {"studio.txt": "abc"}
    assert payload["pip_check_ok"] is True


def test_a_none_extra_is_left_out_entirely(tmp_path: pathlib.Path) -> None:
    """Absent means unknown for these keys too, exactly as for no_torch."""
    im.write_manifest(
        root = tmp_path, req_root = tmp_path, package_name = "pytest", extra = {"mlx_health": None}
    )
    assert "mlx_health" not in _payload(tmp_path)


def test_extras_cannot_shadow_a_field_verify_install_reads(tmp_path: pathlib.Path) -> None:
    """`extra` is evidence, never authority: package_version and requirement_files
    are what decides whether this install is complete."""
    im.write_manifest(
        root = tmp_path,
        req_root = tmp_path,
        package_name = "pytest",
        extra = {"package_version": "9.9.9", "requirement_files": {"studio.txt": "no"}, "schema": 7},
    )
    payload = _payload(tmp_path)
    assert payload["schema"] == 1
    assert payload["requirement_files"] == im.requirement_digests(tmp_path)
    assert payload["package_version"] != "9.9.9"


def test_update_manifest_merges_without_touching_the_rest(tmp_path: pathlib.Path) -> None:
    im.write_manifest(root = tmp_path, req_root = tmp_path, package_name = "pytest")
    before = _payload(tmp_path)
    assert im.update_manifest(root = tmp_path, mlx_health = {"ok": True}) is True
    after = _payload(tmp_path)
    assert after["mlx_health"] == {"ok": True}
    assert after["completed_at_ms"] == before["completed_at_ms"]
    assert after["requirement_files"] == before["requirement_files"]


def test_update_manifest_never_creates_one(tmp_path: pathlib.Path) -> None:
    """The manifest's presence means the install finished; a post-manifest probe
    must not be able to claim that on its own."""
    assert im.update_manifest(root = tmp_path, mlx_health = {"ok": True}) is False
    assert not (tmp_path / im.MANIFEST_NAME).exists()


def test_update_manifest_merges_into_the_manifest_it_replaces(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    """The caller spends minutes gathering this evidence (the MLX probe waits up to 180 s).
    If a second updater removed the manifest and finished a new pass in that time, merging
    into a copy read before the probe would put the old pass's fields back -- including
    no_torch and the torch flavour, which a later update acts on."""
    im.write_manifest(root = tmp_path, req_root = tmp_path, package_name = "pytest", no_torch = True)
    assert _payload(tmp_path)["no_torch"] is True
    real_lock = im._manifest_lock
    done: list[int] = []

    @contextlib.contextmanager
    def _lock_with_a_peer_ahead_of_us(root = None):
        # The second updater got there first: it removed the manifest, ran its pass and wrote
        # a new one. Everything it did is complete before this call takes the lock.
        if not done:
            done.append(1)
            assert im.remove_manifest(root = tmp_path) is True
            im.write_manifest(
                root = tmp_path, req_root = tmp_path, package_name = "pytest", no_torch = False
            )
        with real_lock(root) as locked:
            yield locked

    monkeypatch.setattr(im, "_manifest_lock", _lock_with_a_peer_ahead_of_us)
    assert im.update_manifest(root = tmp_path, mlx_health = {"ok": True}) is True
    after = _payload(tmp_path)
    assert after["mlx_health"] == {"ok": True}
    assert after["no_torch"] is False, "the newer pass's fields were overwritten"


def test_the_manifest_lock_is_exclusive_across_processes(tmp_path: pathlib.Path) -> None:
    """The writers serialise against another PROCESS, not just another thread: setup.sh,
    setup.ps1, the installer and the CLI are separate processes on one venv."""
    order = tmp_path / "order.txt"
    child_code = "\n".join(
        [
            f"import sys, time, pathlib",
            f"sys.path.insert(0, {str(pathlib.Path(im.__file__).resolve().parent)!r})",
            f"import install_manifest as im",
            f"with im._manifest_lock(pathlib.Path({str(tmp_path)!r})):",
            f"    pathlib.Path({str(tmp_path / 'held')!r}).write_text('1', encoding='utf-8')",
            f"    time.sleep(1.5)",
            f"    fh = open({str(order)!r}, 'a', encoding='utf-8')",
            f"    fh.write('child-released\\n')",
            f"    fh.close()",
        ]
    )
    child = subprocess.Popen([sys.executable, "-c", child_code])
    try:
        held = tmp_path / "held"
        deadline = time.time() + 20
        while not held.exists() and time.time() < deadline:
            time.sleep(0.02)
        assert held.exists(), "the child never took the lock"
        with im._manifest_lock(tmp_path):
            with order.open("a", encoding = "utf-8") as fh:
                fh.write("parent-acquired\n")
    finally:
        child.wait(timeout = 30)
    # Ordering, not duration: the parent's acquire cannot land inside the child's hold.
    assert order.read_text(encoding = "utf-8").split() == ["child-released", "parent-acquired"]


def test_the_advisory_write_declines_rather_than_publish_unserialised(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    """A peer holding the lock past the timeout is mid-pass. write_manifest and
    remove_manifest must still go ahead there (failing an install is worse), but this one
    merges evidence a probe gathered minutes ago: publishing beside that peer risks putting
    its removed completion marker back over a half-built venv, and losing the evidence costs
    one probe."""
    monkeypatch.setattr(im, "LOCK_WAIT_SECONDS", 0.3)
    im.write_manifest(root = tmp_path, req_root = tmp_path, package_name = "pytest")
    before = _payload(tmp_path)
    holder = "\n".join(
        [
            "import sys, time, pathlib",
            f"sys.path.insert(0, {str(pathlib.Path(im.__file__).resolve().parent)!r})",
            "import install_manifest as im",
            f"with im._manifest_lock(pathlib.Path({str(tmp_path)!r})):",
            f"    pathlib.Path({str(tmp_path / 'held')!r}).write_text('1', encoding='utf-8')",
            "    time.sleep(30)",
        ]
    )
    child = subprocess.Popen([sys.executable, "-c", holder])
    try:
        deadline = time.time() + 20
        while not (tmp_path / "held").exists() and time.time() < deadline:
            time.sleep(0.02)
        assert (tmp_path / "held").exists(), "the child never took the lock"
        assert im.update_manifest(root = tmp_path, mlx_health = {"ok": True}) is False
        assert _payload(tmp_path) == before
        # The two that must never fail an install still do their work.
        assert im.write_manifest(root = tmp_path, req_root = tmp_path, package_name = "pytest")
        assert im.remove_manifest(root = tmp_path) is True
    finally:
        child.kill()
        child.wait(timeout = 30)


@pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason = "needs O_NOFOLLOW")
def test_a_symlink_on_the_lock_name_is_not_followed(tmp_path: pathlib.Path) -> None:
    """Followed, it would open or create a file somewhere else entirely, and a replacement
    of the target behind it would let two holders think they had the same lock."""
    elsewhere = tmp_path / "elsewhere.txt"
    os.symlink(elsewhere, tmp_path / im.LOCK_NAME)
    with im._manifest_lock(tmp_path) as locked:
        assert locked is False
    assert not elsewhere.exists(), "the symlink's target was created"
    # ...and the writers still work, unserialised, as they did before the lock existed.
    assert im.write_manifest(root = tmp_path, req_root = tmp_path, package_name = "pytest")


def test_a_filesystem_without_locking_is_not_waited_out(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    """Some NFS and SMB mounts answer immediately that they do not implement locking. Only
    contention is worth waiting out; retrying that answer would cost the whole deadline on
    every manifest write."""
    fcntl = pytest.importorskip("fcntl")
    monkeypatch.setattr(im, "LOCK_WAIT_SECONDS", 5.0)

    def _unsupported(*_args, **_kwargs):
        raise OSError(errno.ENOTSUP, "locking not supported")

    monkeypatch.setattr(fcntl, "flock", _unsupported)
    started = time.monotonic()
    with im._manifest_lock(tmp_path) as locked:
        assert locked is False
    assert time.monotonic() - started < 1.0


def test_a_stuck_peer_does_not_wedge_the_lock(tmp_path: pathlib.Path, monkeypatch) -> None:
    """A process suspended or stopped while holding the lock must not stop every later
    update: after the wait the writer goes ahead unserialised, which is what shipped before
    the lock existed. Windows does this on its own (msvcrt's LK_LOCK gives up); POSIX flock
    waits forever unless asked not to."""
    monkeypatch.setattr(im, "LOCK_WAIT_SECONDS", 0.3)
    holder = "\n".join(
        [
            "import sys, time, pathlib",
            f"sys.path.insert(0, {str(pathlib.Path(im.__file__).resolve().parent)!r})",
            "import install_manifest as im",
            f"with im._manifest_lock(pathlib.Path({str(tmp_path)!r})):",
            f"    pathlib.Path({str(tmp_path / 'held')!r}).write_text('1', encoding='utf-8')",
            "    time.sleep(30)",
        ]
    )
    child = subprocess.Popen([sys.executable, "-c", holder])
    try:
        held = tmp_path / "held"
        deadline = time.time() + 20
        while not held.exists() and time.time() < deadline:
            time.sleep(0.02)
        assert held.exists(), "the child never took the lock"
        started = time.monotonic()
        with im._manifest_lock(tmp_path):
            pass
        waited = time.monotonic() - started
        assert waited < 10, f"waited {waited:.1f}s on a peer that never lets go"
        # ...and the writers still answer while that peer holds it.
        assert im.write_manifest(root = tmp_path, req_root = tmp_path, package_name = "pytest")
    finally:
        child.kill()
        child.wait(timeout = 30)


def test_every_manifest_writer_takes_the_lock() -> None:
    """A writer outside it reintroduces the race the lock exists for, and nothing in the
    payloads themselves would show it."""
    source = pathlib.Path(im.__file__).read_text(encoding = "utf-8")
    tree = ast.parse(source)
    writers = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name in ("write_manifest", "update_manifest", "remove_manifest")
    }
    assert set(writers) == {"write_manifest", "update_manifest", "remove_manifest"}
    for name, node in writers.items():
        body = ast.unparse(node)
        assert "_manifest_lock(" in body, f"{name} replaces the manifest outside the lock"
        assert "os.replace" not in body or "_manifest_lock(" in body


def test_a_root_that_cannot_hold_a_lock_still_writes(tmp_path: pathlib.Path, monkeypatch) -> None:
    """This module has to run inside a half-built venv and on filesystems that cannot lock.
    Unserialised is the behaviour that shipped before; failing the install is not."""

    def _no_open(*_args, **_kwargs):
        raise OSError("no lock file here")

    monkeypatch.setattr(pathlib.Path, "open", _no_open)
    with im._manifest_lock(tmp_path):
        pass
    monkeypatch.undo()
    assert im.write_manifest(root = tmp_path, req_root = tmp_path, package_name = "pytest")
    assert im.update_manifest(root = tmp_path, mlx_health = {"ok": True}) is True
    assert im.remove_manifest(root = tmp_path) is True


def test_remove_manifest_keeps_the_live_one_when_the_parked_name_cannot_be_cleared(
    tmp_path: pathlib.Path,
) -> None:
    """setup.ps1 reads True here as permission to replace pip, torch and triton, and the
    dependency pass refuses to run behind a parked copy it cannot clear. Dropping the live
    manifest first would put that refusal after the mutations, on a venv that can no longer
    verify, and every later update would stop at the same place."""
    im.write_manifest(root = tmp_path, req_root = tmp_path, package_name = "pytest")
    live = tmp_path / im.MANIFEST_NAME
    # A directory on the reserved name refuses both the rename and the unlink, on every OS.
    blocked = tmp_path / im.PREVIOUS_MANIFEST_NAME
    blocked.mkdir()
    (blocked / "keep.txt").write_text("x", encoding = "utf-8")

    assert im.remove_manifest(root = tmp_path) is False
    assert live.exists(), "the venv must still verify when the pass cannot be entered"

    # Cleared, the same call parks as usual.
    (blocked / "keep.txt").unlink()
    blocked.rmdir()
    assert im.remove_manifest(root = tmp_path) is True
    assert not live.exists() and (tmp_path / im.PREVIOUS_MANIFEST_NAME).exists()


def test_remove_manifest_refuses_an_unclearable_parked_copy_with_no_live_manifest(
    tmp_path: pathlib.Path,
) -> None:
    """An interrupted run already took the live manifest. Nothing is parked by this call, so
    a surviving copy is that dead run's, and answering True would send setup.ps1 into its
    pip/torch mutations ahead of the refusal the pass makes on exactly that path."""
    blocked = tmp_path / im.PREVIOUS_MANIFEST_NAME
    blocked.mkdir()
    (blocked / "keep.txt").write_text("x", encoding = "utf-8")
    assert im.remove_manifest(root = tmp_path) is False

    (blocked / "keep.txt").unlink()
    blocked.rmdir()
    assert im.remove_manifest(root = tmp_path) is True


def test_a_dead_runs_parked_copy_does_not_outlive_the_next_invalidation(
    tmp_path: pathlib.Path,
) -> None:
    """It would otherwise be read as this pass's evidence."""
    (tmp_path / im.PREVIOUS_MANIFEST_NAME).write_text("{}", encoding = "utf-8")
    assert im.remove_manifest(root = tmp_path) is True
    assert not (tmp_path / im.PREVIOUS_MANIFEST_NAME).exists()


def test_remove_manifest_parks_over_a_stale_copy_it_can_clear(tmp_path: pathlib.Path) -> None:
    """The ordinary case the fallback must not punish: a leftover file from a run that died
    is replaced, not treated as an obstruction."""
    im.write_manifest(root = tmp_path, req_root = tmp_path, package_name = "pytest")
    (tmp_path / im.PREVIOUS_MANIFEST_NAME).write_text("{}", encoding = "utf-8")
    assert im.remove_manifest(root = tmp_path) is True
    assert not (tmp_path / im.MANIFEST_NAME).exists()
    assert im.read_previous_manifest(root = tmp_path)["schema"] == im.MANIFEST_SCHEMA


@pytest.mark.skipif(os.name == "nt", reason = "POSIX file modes")
def test_the_manifest_keeps_the_mode_it_had(tmp_path: pathlib.Path) -> None:
    """It used to be written through Path.write_text, so it carried the umask default and
    anything else on the machine could read it. mkstemp creates at 0600, and silently
    narrowing a file other tooling may read is a change nobody asked for."""
    previous = os.umask(0o022)
    try:
        im.write_manifest(root = tmp_path, req_root = tmp_path, package_name = "pytest")
        live = tmp_path / im.MANIFEST_NAME
        assert stat.S_IMODE(live.stat().st_mode) == 0o644
        # ...and a mode the user tightened stays tightened.
        os.chmod(live, 0o600)
        assert im.update_manifest(root = tmp_path, mlx_health = {"ok": True}) is True
        assert stat.S_IMODE(live.stat().st_mode) == 0o600
        im.write_manifest(root = tmp_path, req_root = tmp_path, package_name = "pytest")
        assert stat.S_IMODE(live.stat().st_mode) == 0o600
    finally:
        os.umask(previous)


def test_two_writers_do_not_share_a_temp_file(tmp_path: pathlib.Path) -> None:
    """A temp name shared between writers is a second writer's file as much as this one's:
    one could remove or overwrite the other's copy between the write and the replace, and
    publish the wrong payload into the manifest that says the install finished."""
    seen: list[str] = []
    real_mkstemp = tempfile.mkstemp

    def _record(*args, **kwargs):
        descriptor, name = real_mkstemp(*args, **kwargs)
        seen.append(pathlib.Path(name).name)
        return descriptor, name

    with unittest.mock.patch.object(tempfile, "mkstemp", _record):
        im.write_manifest(root = tmp_path, req_root = tmp_path, package_name = "pytest")
        assert im.update_manifest(root = tmp_path, mlx_health = {"ok": True}) is True
    assert len(seen) == 2 and seen[0] != seen[1], seen
    assert all(name.startswith(im.MANIFEST_NAME + ".") for name in seen)
    # Neither copy outlives its writer.
    assert not list(tmp_path.glob("*.tmp"))
    assert _payload(tmp_path)["mlx_health"] == {"ok": True}


def test_update_manifest_does_not_recreate_one_removed_while_it_worked(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    """A peer running an older build of this module removes the manifest without taking the
    lock. Writing it back would put a completion marker over the venv that peer is part-way
    through rebuilding."""
    im.write_manifest(root = tmp_path, req_root = tmp_path, package_name = "pytest")
    live = tmp_path / im.MANIFEST_NAME
    real_read = im.read_manifest

    def _read_then_the_peer_removes_it(root = None):
        data = real_read(root)
        live.unlink(missing_ok = True)
        return data

    monkeypatch.setattr(im, "read_manifest", _read_then_the_peer_removes_it)
    assert im.update_manifest(root = tmp_path, mlx_health = {"ok": True}) is False
    monkeypatch.undo()
    assert not live.exists()
    assert not list(tmp_path.glob("*.tmp")), "the temp copy outlived the call"


def test_update_manifest_with_nothing_to_say_is_a_no_op(tmp_path: pathlib.Path) -> None:
    im.write_manifest(root = tmp_path, req_root = tmp_path, package_name = "pytest")
    assert im.update_manifest(root = tmp_path) is False
    assert im.update_manifest(root = tmp_path, mlx_health = None) is False


def test_update_manifest_survives_a_corrupt_manifest(tmp_path: pathlib.Path) -> None:
    (tmp_path / im.MANIFEST_NAME).write_text("{not json", encoding = "utf-8")
    assert im.update_manifest(root = tmp_path, pip_check_ok = True) is False
    assert (tmp_path / im.MANIFEST_NAME).read_text(encoding = "utf-8") == "{not json"


def test_update_manifest_cannot_shadow_a_field_verify_install_reads(tmp_path: pathlib.Path) -> None:
    """The same guard write_manifest applies to `extra`, on the merge path. This one merges
    into a manifest that already means "the install finished", so a caller able to rewrite
    `package_version` or `prefix` leaves a file that validates and describes nobody's install.
    """
    im.write_manifest(root = tmp_path, req_root = tmp_path, package_name = "pytest")
    before = _payload(tmp_path)
    assert (
        im.update_manifest(
            root = tmp_path,
            schema = 7,
            package_version = "9.9.9",
            requirement_files = {"studio.txt": "no"},
            prefix = "/somewhere/else",
            steps_total = 999,
            mlx_health = {"ok": True},
        )
        is True
    )
    after = _payload(tmp_path)
    for key in ("schema", "package_version", "requirement_files", "prefix", "steps_total"):
        assert after[key] == before[key], key
    # The additive key it was actually called for still lands.
    assert after["mlx_health"] == {"ok": True}


def test_update_manifest_with_only_protected_keys_writes_nothing(tmp_path: pathlib.Path) -> None:
    """Nothing left to merge is "record nothing", the same answer as no arguments at
    all -- and, in particular, not a rewrite of the file with the keys dropped."""
    im.write_manifest(root = tmp_path, req_root = tmp_path, package_name = "pytest")
    raw = (tmp_path / im.MANIFEST_NAME).read_bytes()
    assert im.update_manifest(root = tmp_path, package_version = "9.9.9") is False
    assert (tmp_path / im.MANIFEST_NAME).read_bytes() == raw


def test_both_writers_refuse_the_same_keys() -> None:
    """One constant, because two copies of this list is how the two writers would drift. Every
    key write_manifest sets from its own arguments is in it, the optional three included:
    absent means "unknown", and only a build that knew the answer may write one.
    """
    for key in (
        "schema",
        "completed_at_ms",
        "package",
        "package_version",
        "python",
        "platform",
        "prefix",
        "steps_total",
        "requirement_files",
        "no_torch",
        "expected_torch_tag",
        "expected_torch_tag_pinned",
    ):
        assert key in im.PROTECTED_MANIFEST_KEYS, key
    source = MODULE_PATH.read_text(encoding = "utf-8")
    assert source.count("PROTECTED_MANIFEST_KEYS: Tuple[str, ...] = (") == 1
    # Both writers, and no third spelling of the rule.
    assert source.count("key not in PROTECTED_MANIFEST_KEYS") == 1
    assert source.count("or key in PROTECTED_MANIFEST_KEYS") == 1


def test_an_optional_field_cannot_be_invented_by_evidence(tmp_path: pathlib.Path) -> None:
    """no_torch absent means "unknown", and a GGUF-only venv is what the wrong answer
    costs: the next update reinstalls torch into it."""
    im.write_manifest(root = tmp_path, req_root = tmp_path, package_name = "pytest")
    assert im.update_manifest(root = tmp_path, no_torch = True) is False
    assert "no_torch" not in _payload(tmp_path)
    im.write_manifest(
        root = tmp_path, req_root = tmp_path, package_name = "pytest", extra = {"no_torch": True}
    )
    assert "no_torch" not in _payload(tmp_path)


# -- constraints ---------------------------------------------------------------


def test_an_absent_distribution_does_not_violate_a_constraint(tmp_path: pathlib.Path) -> None:
    """A constraints file never asks for an install, so nothing to install is
    nothing to repair -- otherwise every pass would run forever."""
    constraints = tmp_path / "constraints.txt"
    constraints.write_text("nonexistent-dist-xyz<2\n", encoding = "utf-8")
    assert im.violated_constraints(constraints, installed = {}) == []


def test_a_resident_version_outside_the_window_is_reported(tmp_path: pathlib.Path) -> None:
    constraints = tmp_path / "constraints.txt"
    constraints.write_text(
        "# a comment\n\npytest>=99\nanyio<1  # inline\n-r other.txt\n", encoding = "utf-8"
    )
    violated = im.violated_constraints(constraints, installed = {"pytest": "8.0.0", "anyio": "0.1"})
    assert violated == ["pytest"]


def test_an_unpinned_or_unreadable_constraints_file_reports_nothing(tmp_path: pathlib.Path) -> None:
    bare = tmp_path / "bare.txt"
    bare.write_text("pytest\n", encoding = "utf-8")
    assert im.violated_constraints(bare, installed = {"pytest": "8.0.0"}) == []
    assert im.violated_constraints(tmp_path / "gone.txt", installed = {}) == []


def test_the_shipped_constraints_file_parses() -> None:
    """Not an assertion about this machine's venv, only that the real file is
    readable by the parser the skip gate relies on."""
    real = REPO_ROOT / "studio" / "backend" / "requirements" / "single-env" / "constraints.txt"
    assert isinstance(im.violated_constraints(real, installed = {}), list)


# -- sidecar predicate ---------------------------------------------------------


def _sidecar(
    root: pathlib.Path,
    name: str,
    version: str,
    *,
    files = ("__init__.py",),
) -> None:
    """A flat `pip --target` tree: package dir, dist-info, RECORD with sizes."""
    package = root / name.replace("-", "_")
    package.mkdir(parents = True, exist_ok = True)
    rows = []
    for relative in files:
        target = package / relative
        target.parent.mkdir(parents = True, exist_ok = True)
        target.write_bytes(b"x" * 32)
        rows.append(f"{package.name}/{relative},sha256=deadbeef,32")
    dist_info = root / f"{name.replace('-', '_')}-{version}.dist-info"
    dist_info.mkdir(parents = True, exist_ok = True)
    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n", encoding = "utf-8"
    )
    (dist_info / "RECORD").write_text("\n".join(rows) + "\n", encoding = "utf-8")


@pytest.fixture
def sidecar(tmp_path: pathlib.Path) -> pathlib.Path:
    root = tmp_path / ".venv_t5_530"
    root.mkdir()
    (root / im.SIDECAR_OWNED_MARKER).write_text("", encoding = "utf-8")
    _sidecar(root, "transformers", "5.3.0")
    _sidecar(root, "huggingface_hub", "1.8.0")
    _sidecar(root, "hf_xet", "1.4.2")
    _sidecar(root, "tiktoken", "0.9.0")
    return root


PINS = ("transformers==5.3.0", "huggingface_hub==1.8.0", "hf_xet==1.4.2", "tiktoken")


def test_a_complete_sidecar_is_current(sidecar: pathlib.Path) -> None:
    assert im.sidecar_is_current(sidecar, PINS) == (True, "")


def test_a_missing_or_empty_directory_is_not_current(tmp_path: pathlib.Path) -> None:
    assert im.sidecar_is_current(tmp_path / "gone", PINS)[0] is False
    empty = tmp_path / "empty"
    empty.mkdir()
    assert im.sidecar_is_current(empty, PINS) == (False, "empty")


def test_an_unowned_directory_is_never_current(sidecar: pathlib.Path) -> None:
    """Rebuilding starts with rm -rf, so "current" has to mean "and ours"."""
    (sidecar / im.SIDECAR_OWNED_MARKER).unlink()
    current, reason = im.sidecar_is_current(sidecar, PINS)
    assert current is False and im.SIDECAR_OWNED_MARKER in reason


def test_a_wrong_version_names_itself(sidecar: pathlib.Path) -> None:
    current, reason = im.sidecar_is_current(sidecar, ("transformers==5.5.0",) + PINS[1:])
    assert current is False
    assert "transformers==5.3.0" in reason and "5.5.0" in reason


def test_an_unpinned_name_only_has_to_be_present(sidecar: pathlib.Path) -> None:
    assert im.sidecar_is_current(sidecar, ("tiktoken",)) == (True, "")
    current, reason = im.sidecar_is_current(sidecar, ("regex",))
    assert current is False and "regex" in reason


def test_metadata_without_its_package_tree_is_not_current(sidecar: pathlib.Path) -> None:
    """The failure mode the version check alone cannot see: an interrupted pip
    leaves METADATA behind and takes the module with it."""
    import shutil

    shutil.rmtree(sidecar / "transformers")
    current, reason = im.sidecar_is_current(sidecar, PINS)
    assert current is False and "directory missing" in reason


def test_a_truncated_recorded_file_forces_a_rebuild(sidecar: pathlib.Path) -> None:
    (sidecar / "transformers" / "__init__.py").write_bytes(b"")
    current, reason = im.sidecar_is_current(sidecar, PINS)
    assert current is False
    assert "0 bytes, expected 32" in reason


@pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
def test_the_file_check_kill_switch_reaches_the_setup_predicate(
    sidecar: pathlib.Path, monkeypatch, value
) -> None:
    """The runtime honours UNSLOTH_SKIP_SIDECAR_FILE_CHECK so a false positive can be
    turned off without a release; a setup-side scan that ignored it would be the one
    path left that still wipes the sidecar. Package and version checks still apply."""
    (sidecar / "transformers" / "__init__.py").write_bytes(b"")
    monkeypatch.setenv(im.SIDECAR_FILE_CHECK_ENV, value)
    assert im.sidecar_is_current(sidecar, PINS) == (True, "")
    current, reason = im.sidecar_is_current(sidecar, ("transformers==5.5.0",) + PINS[1:])
    assert current is False and "5.5.0" in reason


@pytest.mark.parametrize("value", ["0", "false", "", "maybe"])
def test_a_non_true_kill_switch_value_changes_nothing(
    sidecar: pathlib.Path, monkeypatch, value
) -> None:
    (sidecar / "transformers" / "__init__.py").write_bytes(b"")
    monkeypatch.setenv(im.SIDECAR_FILE_CHECK_ENV, value)
    assert im.sidecar_is_current(sidecar, PINS)[0] is False


def test_a_pinned_package_without_a_record_forces_a_rebuild(sidecar: pathlib.Path) -> None:
    """pip and uv write RECORD last: a pinned dist-info without one is an interrupted
    install whose payload the size check cannot see. The optional package is not held
    to it, since its own top-up clears a recordless dist-info."""
    (sidecar / "hf_xet-1.4.2.dist-info" / "RECORD").unlink()
    (sidecar / "hf_xet" / "__init__.py").write_bytes(b"")
    ok, reason = im.sidecar_is_current(sidecar, PINS)
    assert ok is False
    assert "hf_xet: RECORD is missing" in reason
    (sidecar / "hf_xet-1.4.2.dist-info" / "RECORD").write_text(
        "hf_xet/__init__.py,sha256=deadbeef,0\n", encoding = "utf-8"
    )
    assert im.sidecar_is_current(sidecar, PINS) == (True, "")
    (sidecar / "tiktoken-0.9.0.dist-info" / "RECORD").unlink()
    assert im.sidecar_is_current(sidecar, PINS) == (True, "")


def test_a_deleted_recorded_file_forces_a_rebuild(sidecar: pathlib.Path) -> None:
    (sidecar / "transformers" / "__init__.py").unlink()
    current, reason = im.sidecar_is_current(sidecar, PINS)
    assert current is False and "is missing" in reason


def test_a_larger_file_is_a_collision_not_damage(sidecar: pathlib.Path) -> None:
    (sidecar / "transformers" / "__init__.py").write_bytes(b"x" * 64)
    assert im.sidecar_is_current(sidecar, PINS) == (True, "")


def test_an_extension_built_for_another_interpreter_is_rejected(
    tmp_path: pathlib.Path, sidecar: pathlib.Path
) -> None:
    """A sidecar survives a Python upgrade intact and stops importing."""
    tag = "{}{}{}".format(
        sys.version_info.major,
        sys.version_info.minor,
        "t" if sysconfig.get_config_var("Py_GIL_DISABLED") else "",
    )
    foreign = "990" if not tag.startswith("99") else "980"
    _sidecar(
        sidecar,
        "regex",
        "2024.1.1",
        files = ("__init__.py", f"_regex.cpython-{foreign}-x86_64-linux-gnu.so"),
    )
    current, reason = im.sidecar_is_current(sidecar, PINS)
    assert current is False
    assert f"targets cp{foreign}" in reason and f"interpreter is cp{tag}" in reason


def test_a_tag_in_a_directory_name_says_nothing_about_the_binary(sidecar: pathlib.Path) -> None:
    """Wiping several hundred MB over a directory name is the false positive this
    check must not have."""
    _sidecar(sidecar, "regex", "2024.1.1", files = ("__init__.py", "regex.cp312.libs/libhelper.so"))
    assert im.sidecar_is_current(sidecar, PINS) == (True, "")


def test_every_distribution_is_scanned_not_only_the_pinned_ones(sidecar: pathlib.Path) -> None:
    """The whole directory goes on sys.path, so a truncated regex/ shadows the
    base install and breaks `import transformers` just as a truncated
    transformers/ does."""
    _sidecar(sidecar, "regex", "2024.1.1")
    (sidecar / "regex" / "__init__.py").write_bytes(b"")
    current, reason = im.sidecar_is_current(sidecar, PINS)
    assert current is False and "regex" in reason


def _module_dist(
    root: pathlib.Path,
    name: str,
    version: str,
    *,
    modules = ("six.py",),
) -> None:
    """A distribution that installs top-level MODULES, with no package directory."""
    rows = []
    for relative in modules:
        target = root / relative
        target.parent.mkdir(parents = True, exist_ok = True)
        target.write_bytes(b"x" * 32)
        rows.append(f"{relative},sha256=deadbeef,32")
    dist_info = root / f"{name.replace('-', '_')}-{version}.dist-info"
    dist_info.mkdir(parents = True, exist_ok = True)
    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n", encoding = "utf-8"
    )
    rows.append(f"{dist_info.name}/METADATA,,")
    (dist_info / "RECORD").write_text("\n".join(rows) + "\n", encoding = "utf-8")


def test_a_module_only_distribution_is_current(sidecar: pathlib.Path) -> None:
    """six installs six.py and nothing else, so there is no directory named after it.

    Called stale, `sidecar_is_current` deletes and refetches a healthy several-hundred-MB
    tree on every single update -- for a pin that is satisfied.
    """
    _module_dist(sidecar, "six", "1.17.0")
    assert im.sidecar_is_current(sidecar, ("six==1.17.0",)) == (True, "")
    assert im.sidecar_is_current(sidecar, ("six",)) == (True, "")


def test_a_module_only_distribution_still_answers_on_its_version(sidecar: pathlib.Path) -> None:
    """The payload fallback decides whether the files arrived, never which release."""
    _module_dist(sidecar, "six", "1.17.0")
    current, reason = im.sidecar_is_current(sidecar, ("six==1.16.0",))
    assert current is False
    assert "six==1.17.0" in reason and "1.16.0" in reason


def test_a_module_only_distribution_whose_module_is_gone_is_not_current(
    sidecar: pathlib.Path,
) -> None:
    """The case the directory probe was there for, on the path that replaces it: an
    interrupted pip leaves the METADATA and takes the module with it."""
    _module_dist(sidecar, "six", "1.17.0")
    (sidecar / "six.py").unlink()
    current, reason = im.sidecar_is_current(sidecar, ("six==1.17.0",))
    assert current is False and "directory missing" in reason


def test_an_import_name_that_matches_neither_spelling_is_current(sidecar: pathlib.Path) -> None:
    """pillow -> PIL. Neither `pillow` nor `pillow` with dashes swapped is on disk, and
    guessing the mapping is not something an installer can do."""
    _module_dist(sidecar, "pillow", "11.0.0", modules = ("PIL/__init__.py", "PIL/Image.py"))
    assert im.sidecar_is_current(sidecar, ("pillow==11.0.0",)) == (True, "")


def test_a_console_script_alone_does_not_prove_the_payload_arrived(sidecar: pathlib.Path) -> None:
    """pip records ../../bin/hf and uv records bin/hf, and neither is inside the tree the
    training worker puts on sys.path. Believing them fails CLOSED on a real sidecar."""
    _module_dist(sidecar, "toolonly", "1.0", modules = ("bin/toolonly",))
    current, reason = im.sidecar_is_current(sidecar, ("toolonly==1.0",))
    assert current is False and "directory missing" in reason


def test_a_distribution_with_no_record_at_all_is_not_current(sidecar: pathlib.Path) -> None:
    """No directory and no RECORD to fall back to leaves nothing that says the files
    landed, and the answer has to be the one that repairs rather than the one that
    ships a sidecar the worker cannot import."""
    dist_info = sidecar / "ghost-1.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: ghost\nVersion: 1.0\n", encoding = "utf-8"
    )
    current, reason = im.sidecar_is_current(sidecar, ("ghost==1.0",))
    assert current is False and "directory missing" in reason


def test_a_pinned_sidecar_package_with_no_record_is_rebuilt(sidecar: pathlib.Path) -> None:
    """An absent RECORD used to say nothing about damage, since the answer costs a
    several-hundred-MB refetch; but pip and uv write RECORD last, so a pinned package
    without one is an interrupted install whose payload the size check cannot see."""
    (sidecar / "transformers-5.3.0.dist-info" / "RECORD").unlink()
    ok, reason = im.sidecar_is_current(sidecar, PINS)
    assert ok is False
    assert reason == "transformers: RECORD is missing"


def test_the_scan_is_bounded(sidecar: pathlib.Path) -> None:
    """No installer wraps this in a timeout, so a stalled mount must not wedge setup."""
    assert im.sidecar_is_current(sidecar, PINS, budget_seconds = 0.0)[0] is True


# -- the CLI shim both shells call ---------------------------------------------


def _shim(*args: str):
    import subprocess
    return subprocess.run(
        [sys.executable, str(MODULE_PATH), *args],
        capture_output = True,
        text = True,
        encoding = "utf-8",
    )


def test_the_shim_reports_current(sidecar: pathlib.Path) -> None:
    result = _shim("sidecar", str(sidecar), *PINS)
    assert result.returncode == 0
    assert result.stdout.strip() == "sidecar: current"


def test_the_shim_reports_the_reason_and_exits_1(sidecar: pathlib.Path) -> None:
    result = _shim("sidecar", str(sidecar), "transformers==5.5.0")
    assert result.returncode == 1
    assert result.stdout.startswith("sidecar: ")
    assert "5.5.0" in result.stdout


def test_the_shim_marks_its_own_output(sidecar: pathlib.Path) -> None:
    """An install_manifest.py predating the shim has no __main__ block at all, so
    running it exits 0 with no output. Both shells require the marker line before
    believing exit 0, or that silence reads as "current" and no sidecar is ever
    rebuilt again."""
    assert im._SIDECAR_CLI_MARKER == "sidecar:"
    for args in ((), ("sidecar",), ("nonsense",), ("sidecar", str(sidecar))):
        result = _shim(*args)
        if result.returncode == 2:
            assert not result.stdout.strip().startswith(im._SIDECAR_CLI_MARKER)


def test_the_shim_refuses_what_it_does_not_implement() -> None:
    for args in ((), ("nonsense",), ("sidecar",)):
        assert _shim(*args).returncode == 2


def test_an_absent_tiktoken_is_optional_but_a_present_one_is_held_to_its_record(
    sidecar: pathlib.Path,
) -> None:
    """Absence is what is optional: a sidecar without tiktoken is current, and setup's
    top-up adds it. Present, tiktoken's RECORD is held to the disk like every other
    package's, since a file it names that is not there is a tokenizer that fails at
    import. A required package's RECORD is held to the disk too."""
    import shutil

    (sidecar / "tiktoken" / "__init__.py").unlink()
    current, reason = im.sidecar_is_current(sidecar, PINS)
    assert current is False and "tiktoken" in reason
    shutil.rmtree(sidecar / "tiktoken")
    shutil.rmtree(next(sidecar.glob("tiktoken-*.dist-info")))
    assert im.sidecar_is_current(sidecar, PINS) == (True, "")
    (sidecar / "hf_xet" / "__init__.py").unlink()
    current, reason = im.sidecar_is_current(sidecar, PINS)
    assert current is False and "hf_xet" in reason


def test_write_manifest_never_raises_on_a_payload_json_cannot_encode(
    tmp_path: pathlib.Path,
) -> None:
    """`extra` is caller-composed and the docstring promises this never raises. It is the last
    act of a pass that has already installed everything, so a TypeError out of json.dumps
    would end the update with no manifest, which every reader takes for a half-built install.
    """
    assert (
        im.write_manifest(
            root = tmp_path,
            req_root = tmp_path,
            package_name = "pytest",
            extra = {"known_unmet": {"studio.txt"}},
        )
        is None
    )
    assert not (tmp_path / im.MANIFEST_NAME).exists()


def test_an_unencodable_extra_leaves_an_existing_manifest_alone(tmp_path: pathlib.Path) -> None:
    """Refusing is the safe direction: the previous record is still true of this venv."""
    im.write_manifest(
        root = tmp_path, req_root = tmp_path, package_name = "pytest", extra = {"pip_check_ok": True}
    )
    before = _payload(tmp_path)
    assert (
        im.write_manifest(
            root = tmp_path,
            req_root = tmp_path,
            package_name = "pytest",
            extra = {"bad": object()},
        )
        is None
    )
    assert _payload(tmp_path) == before


def test_both_writers_refuse_the_same_unencodable_payload(tmp_path: pathlib.Path) -> None:
    im.write_manifest(root = tmp_path, req_root = tmp_path, package_name = "pytest")
    assert im.update_manifest(root = tmp_path, mlx_health = {1, 2}) is False
    assert (
        im.write_manifest(
            root = tmp_path, req_root = tmp_path, package_name = "pytest", extra = {"x": {1, 2}}
        )
        is None
    )


def test_the_pass_lock_reads_as_contended_only_while_a_peer_holds_it(
    tmp_path: pathlib.Path,
) -> None:
    """A second pass on one venv must see the first one, across processes."""
    child_code = "\n".join(
        [
            "import sys, time, pathlib",
            f"sys.path.insert(0, {str(pathlib.Path(im.__file__).resolve().parent)!r})",
            "import install_manifest as im",
            f"with im.pass_lock(pathlib.Path({str(tmp_path)!r})) as owned:",
            "    assert owned",
            f"    pathlib.Path({str(tmp_path / 'held')!r}).write_text('1', encoding='utf-8')",
            "    time.sleep(2.0)",
        ]
    )
    child = subprocess.Popen([sys.executable, "-c", child_code])
    try:
        held = tmp_path / "held"
        deadline = time.time() + 20
        while not held.exists() and time.time() < deadline:
            time.sleep(0.02)
        assert held.exists(), "the child never took the pass lock"
        with im.pass_lock(tmp_path) as uncontended:
            assert uncontended is False
    finally:
        child.wait(timeout = 30)
    # And the moment it lets go, the next pass is free to trust its evidence again.
    with im.pass_lock(tmp_path) as uncontended:
        assert uncontended is True


def test_the_pass_lock_never_waits_for_the_peer(tmp_path: pathlib.Path) -> None:
    """It is tested, not waited on: a pass runs for minutes, so a waiter would be worse."""
    with im.pass_lock(tmp_path):
        started = time.monotonic()
        code = "\n".join(
            [
                "import sys, pathlib",
                f"sys.path.insert(0, {str(pathlib.Path(im.__file__).resolve().parent)!r})",
                "import install_manifest as im",
                f"with im.pass_lock(pathlib.Path({str(tmp_path)!r})) as owned:",
                "    print(owned)",
            ]
        )
        peer = subprocess.run(
            [sys.executable, "-c", code], capture_output = True, text = True, timeout = 60
        )
    assert peer.stdout.strip() == "False", peer.stderr
    assert time.monotonic() - started < 5.0


def test_a_root_that_cannot_hold_a_pass_lock_reads_as_uncontended(tmp_path: pathlib.Path) -> None:
    """Best effort, as the manifest lock: an unlockable filesystem keeps the fast path."""
    with im.pass_lock(tmp_path / "does" / "not" / "exist") as uncontended:
        assert uncontended is True


def test_a_symlink_on_the_pass_lock_name_is_not_followed(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "elsewhere"
    target.write_text("untouched", encoding = "utf-8")
    (tmp_path / im.PASS_LOCK_NAME).symlink_to(target)
    with im.pass_lock(tmp_path) as uncontended:
        assert uncontended is True
    assert target.read_text(encoding = "utf-8") == "untouched"


def test_remove_manifest_reports_a_refusal_rather_than_raising(tmp_path: pathlib.Path) -> None:
    """An unsearchable venv directory made Path.exists() raise EACCES straight out of
    remove_manifest. It returned False before the parked copy existed, and it must still."""
    if os.name == "nt" or os.geteuid() == 0:
        pytest.skip("needs POSIX permissions and a non-root user")
    root = tmp_path / "venv"
    root.mkdir()
    im.write_manifest(root = root, req_root = tmp_path, package_name = "pytest")
    os.chmod(root, 0o000)
    try:
        assert im.remove_manifest(root) is False
    finally:
        os.chmod(root, 0o755)


def test_the_presence_check_answers_blocked_when_it_cannot_look(tmp_path: pathlib.Path) -> None:
    """Unknown reads as still there: every caller is asking whether a marker blocks the pass."""
    if os.name == "nt" or os.geteuid() == 0:
        pytest.skip("needs POSIX permissions and a non-root user")
    hidden = tmp_path / "hidden"
    hidden.mkdir()
    target = hidden / "unsloth_install_manifest.previous.json"
    target.write_text("{}", encoding = "utf-8")
    os.chmod(hidden, 0o000)
    try:
        assert im.manifest_is_present(target) is True
    finally:
        os.chmod(hidden, 0o755)
    assert im.manifest_is_present(tmp_path / "absent.json") is False


@pytest.mark.parametrize(
    "code, expected",
    [
        (errno.ENOENT, False),
        (errno.ENOTDIR, False),
        (errno.EBADF, False),
        (errno.ELOOP, False),
        (errno.EACCES, True),
        (errno.EPERM, True),
        (errno.ESTALE, True),
        (errno.EIO, True),
        (errno.ETIMEDOUT, True),
        (errno.ENAMETOOLONG, True),
    ],
)
def test_the_presence_check_reads_the_errno_itself(monkeypatch, code, expected) -> None:
    """Not Path.exists(): 3.13 raises EACCES out of it and 3.14 returns False (gh-101357), so
    it means "absent" on one interpreter and "unknown" on the other. Only the four pathlib
    treated as absent before 3.14 are absent here; anything else is a marker still in place."""

    def refuse(self, *args, **kwargs):
        raise OSError(code, os.strerror(code))

    monkeypatch.setattr(pathlib.Path, "stat", refuse)
    assert im.manifest_is_present(pathlib.Path("/whatever")) is expected


def test_a_path_this_interpreter_cannot_encode_holds_no_manifest(monkeypatch) -> None:
    def refuse(self, *args, **kwargs):
        raise ValueError("embedded null character")

    monkeypatch.setattr(pathlib.Path, "stat", refuse)
    assert im.manifest_is_present(pathlib.Path("/whatever")) is False
