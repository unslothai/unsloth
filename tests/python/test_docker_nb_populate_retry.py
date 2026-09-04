# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""First-boot population must not claim a commit it did not finish copying.

A `cp -a` that fails is skipped silently and gets no state entry, so phase 1b never
restores it -- and stamping the commit anyway tells the phase 2 refresh it is already
synced, so the notebook is gone for good on an offline container.

Driven end to end against the real script with the refresh disabled. The blocked path
is a plain FILE where a directory has to be, which is ENOTDIR for root too, so this
cannot silently pass under a root CI container the way a chmod would.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNC = REPO_ROOT / "docker" / "unsloth_sync_notebooks.sh"

TEMPLATE_COMMIT = "a" * 40

behavioural = pytest.mark.skipif(
    any(shutil.which(tool) is None for tool in ("bash", "sha256sum")),
    reason = "needs bash and sha256sum",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _template(tmp_path: Path) -> Path:
    template = tmp_path / "template"
    (template / "sub").mkdir(parents = True)
    (template / "a.ipynb").write_text("A", encoding = "utf-8")
    (template / "sub" / "b.ipynb").write_text("B", encoding = "utf-8")
    (template / ".unsloth_template_commit").write_text(TEMPLATE_COMMIT + "\n", encoding = "utf-8")
    return template


def _run(tmp_path: Path, template: Path, dest: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(SYNC)],
        capture_output = True,
        text = True,
        timeout = 300,
        env = dict(
            os.environ,
            UNSLOTH_NOTEBOOKS_TEMPLATE = str(template),
            UNSLOTH_NOTEBOOKS_DIR = str(dest),
            # phase 2 needs a clone; the offline half must retry on its own
            UNSLOTH_SKIP_NOTEBOOK_REFRESH = "1",
            UNSLOTH_SKIP_NOTEBOOK_VIEW = "1",
            UNSLOTH_KEEP_COLAB_INTRO = "1",
        ),
    )


def _state(dest: Path) -> dict:
    path = dest / ".unsloth_sync_state"
    if not path.is_file():
        return {}
    out = {}
    for line in path.read_text(encoding = "utf-8").splitlines():
        parts = line.split("  ", 1)
        if len(parts) == 2:
            out[parts[1]] = parts[0]
    return out


@behavioural
def test_a_failed_copy_leaves_the_commit_unstamped_and_retries_next_start(tmp_path: Path):
    template = _template(tmp_path)
    dest = tmp_path / "dest"
    dest.mkdir()
    # a regular file, so the copy into it is ENOTDIR for root too
    (dest / "sub").write_text("blocked", encoding = "utf-8")

    first = _run(tmp_path, template, dest)
    assert first.returncode == 0, first.stdout + first.stderr

    assert (dest / "a.ipynb").read_text(encoding = "utf-8") == "A"
    assert "a.ipynb" in _state(dest)
    assert "sub/b.ipynb" not in _state(dest), "a copy that failed must not be recorded"
    assert not (dest / ".unsloth_sync_commit").exists(), (
        "stamping the template commit after an incomplete populate makes the "
        "phase 2 refresh exit early, so the missing notebook never comes back"
    )
    assert (dest / ".unsloth_sync_partial").exists()

    (dest / "sub").unlink()

    second = _run(tmp_path, template, dest)
    assert second.returncode == 0, second.stdout + second.stderr

    assert (dest / "sub" / "b.ipynb").read_text(encoding = "utf-8") == "B"
    assert _state(dest)["sub/b.ipynb"] == _sha256(dest / "sub" / "b.ipynb")
    assert _state(dest)["a.ipynb"] == _sha256(dest / "a.ipynb")
    assert (dest / ".unsloth_sync_commit").read_text(encoding = "utf-8").strip() == (TEMPLATE_COMMIT)
    assert not (dest / ".unsloth_sync_partial").exists()


@behavioural
def test_a_clean_populate_stamps_the_commit_and_does_not_re_run(tmp_path: Path):
    template = _template(tmp_path)
    dest = tmp_path / "dest"
    dest.mkdir()

    assert _run(tmp_path, template, dest).returncode == 0
    assert (dest / ".unsloth_sync_commit").read_text(encoding = "utf-8").strip() == (TEMPLATE_COMMIT)
    assert not (dest / ".unsloth_sync_partial").exists()
    before = _state(dest)
    assert set(before) == {"a.ipynb", "sub/b.ipynb"}

    # a re-run of phase 1 would drop this file's managed record, so an identical
    # state is what proves the block was skipped
    (dest / "a.ipynb").write_text("edited", encoding = "utf-8")

    assert _run(tmp_path, template, dest).returncode == 0
    assert _state(dest) == before
    assert (dest / "a.ipynb").read_text(encoding = "utf-8") == "edited"


@behavioural
def test_the_partial_marker_is_not_recorded_as_a_notebook(tmp_path: Path):
    # record_state() walks every file under DEST, so a dotfile missing from its skip
    # list reaches users as a notebook to sync
    template = _template(tmp_path)
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "sub").write_text("blocked", encoding = "utf-8")

    assert _run(tmp_path, template, dest).returncode == 0
    assert (dest / ".unsloth_sync_partial").exists()
    assert ".unsloth_sync_partial" not in _state(dest)

    source = SYNC.read_text(encoding = "utf-8")
    skip = source[source.index("record_state() {") :]
    skip = skip[: skip.index("printf")]
    assert ".unsloth_sync_partial" in skip


@behavioural
def test_the_retry_keeps_records_the_refresh_added(tmp_path: Path):
    """A retry rebuilds the state from the TEMPLATE alone, so anything the refresh did
    in between is thrown away: a template file whose bytes upstream changed now differs
    from the template and hits the "kept existing user file" branch, and a notebook that
    exists only upstream is never visited at all. Both become user-owned for good, while
    the commit marker is stamped anyway so it looks converged."""
    template = _template(tmp_path)
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "sub").write_text("blocked", encoding = "utf-8")

    assert _run(tmp_path, template, dest).returncode == 0
    assert (dest / ".unsloth_sync_partial").exists()

    # what the refresh does between the two boots
    (dest / "a.ipynb").write_text("A-v2-from-upstream", encoding = "utf-8")
    (dest / "remote_only.ipynb").write_text("R", encoding = "utf-8")
    (dest / ".unsloth_sync_state").write_text(
        f"{_sha256(dest / 'a.ipynb')}  a.ipynb\n"
        f"{_sha256(dest / 'remote_only.ipynb')}  remote_only.ipynb\n",
        encoding = "utf-8",
    )

    (dest / "sub").unlink()
    second = _run(tmp_path, template, dest)
    assert second.returncode == 0, second.stdout + second.stderr

    state = _state(dest)
    assert state.get("a.ipynb") == _sha256(
        dest / "a.ipynb"
    ), "the refreshed copy is ours, not the user's; dropping its record freezes it"
    assert state.get("remote_only.ipynb") == _sha256(
        dest / "remote_only.ipynb"
    ), "a notebook that exists only upstream is never walked by the populate loop"
    assert state.get("sub/b.ipynb") == _sha256(dest / "sub" / "b.ipynb")
    assert (dest / "a.ipynb").read_text(
        encoding = "utf-8"
    ) == "A-v2-from-upstream", "the retry must not overwrite the newer copy with the baked template"
    assert len(state) == 3, state


@behavioural
def test_a_first_boot_records_only_what_it_populated(tmp_path: Path):
    """Non-vacuity for the merge above: with no prior state there is nothing to carry
    forward, so a stray file in DEST must not acquire a record."""
    template = _template(tmp_path)
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "not_ours.ipynb").write_text("N", encoding = "utf-8")

    assert _run(tmp_path, template, dest).returncode == 0
    assert set(_state(dest)) == {"a.ipynb", "sub/b.ipynb"}


# Phase 1b restores a notebook the user deleted from the BAKED template, and records
# the template's hash for it. When the refresh had already moved that notebook past
# the image, the restore silently walks it backwards -- and phase 2 exits on
# `remote == last`, so the sync marker has to come off or it stays there until
# upstream happens to commit again.

UPSTREAM_COMMIT = "b" * 40


def _restored_run(tmp_path: Path, template: Path, dest: Path, recorded: str, body: str | None):
    """State says `recorded` for a.ipynb; `body` is what is on disk, None to delete it."""
    (dest / ".unsloth_sync_state").write_text(
        f"{recorded}  a.ipynb\n{_sha256(template / 'sub' / 'b.ipynb')}  sub/b.ipynb\n",
        encoding = "utf-8",
    )
    (dest / ".unsloth_sync_commit").write_text(UPSTREAM_COMMIT + "\n", encoding = "utf-8")
    if body is None:
        (dest / "a.ipynb").unlink()
    else:
        (dest / "a.ipynb").write_text(body, encoding = "utf-8")
    return _run(tmp_path, template, dest)


@behavioural
def test_a_notebook_restored_backwards_drops_the_sync_marker(tmp_path: Path):
    template, dest = _template(tmp_path), tmp_path / "dest"
    assert _run(tmp_path, template, dest).returncode == 0

    # the refresh had taken a.ipynb past the baked "A"; the user then deletes it
    run = _restored_run(tmp_path, template, dest, "c" * 64, None)
    assert run.returncode == 0, run.stdout + run.stderr

    assert (dest / "a.ipynb").read_text(encoding = "utf-8") == "A", "it was restored"
    assert not (dest / ".unsloth_sync_commit").is_file(), (
        "the marker survived a downgrade, so the refresh exits on remote == last and "
        "the notebook stays on the image's older copy"
    )
    assert _state(dest)["a.ipynb"] == _sha256(template / "a.ipynb")
    assert "1 needing a refresh" in run.stdout, run.stdout


@behavioural
def test_a_restore_that_changes_nothing_keeps_the_marker(tmp_path: Path):
    """Non-vacuity and the cost control: an ordinary delete of a notebook that never
    moved past the image must not force a full clone on the next start."""
    template, dest = _template(tmp_path), tmp_path / "dest"
    assert _run(tmp_path, template, dest).returncode == 0

    run = _restored_run(tmp_path, template, dest, _sha256(template / "a.ipynb"), None)
    assert run.returncode == 0, run.stdout + run.stderr

    assert (dest / "a.ipynb").read_text(encoding = "utf-8") == "A"
    assert (dest / ".unsloth_sync_commit").read_text(encoding = "utf-8").strip() == UPSTREAM_COMMIT
    assert "0 needing a refresh" in run.stdout, run.stdout


@behavioural
def test_a_notebook_still_on_disk_is_not_touched(tmp_path: Path):
    """The restore is only for files that are GONE; a user's edit stays and keeps the
    marker, because nothing was walked backwards."""
    template, dest = _template(tmp_path), tmp_path / "dest"
    assert _run(tmp_path, template, dest).returncode == 0

    run = _restored_run(tmp_path, template, dest, "c" * 64, "MY OWN WORK")
    assert run.returncode == 0, run.stdout + run.stderr

    assert (dest / "a.ipynb").read_text(encoding = "utf-8") == "MY OWN WORK"
    assert (dest / ".unsloth_sync_commit").read_text(encoding = "utf-8").strip() == UPSTREAM_COMMIT


# --- a stale state staging file the populate cannot truncate --------------------------
# A run killed between the truncate and the mv leaves $STATE.tmp behind. When that run
# was a different uid (root once, then `--user`) the populate cannot empty it, every
# append fails, and the mv -- which needs write on DEST, not on the file -- publishes
# the FOREIGN file as our state. 0444 on a file this uid owns reproduces that, since
# open-for-write consults the owner bits.
@behavioural
def test_a_stale_unwritable_state_temp_is_not_published_as_our_state(tmp_path: Path):
    template = _template(tmp_path)
    dest = tmp_path / "dest"
    dest.mkdir()
    stale = dest / ".unsloth_sync_state.tmp"
    stale.write_text("deadbeef  stale.ipynb\n", encoding = "utf-8")
    stale.chmod(0o444)

    run = _run(tmp_path, template, dest)
    assert run.returncode == 0, run.stdout + run.stderr

    state = _state(dest)
    assert (
        "stale.ipynb" not in state
    ), "a leftover from an interrupted run was adopted as the state of this one"
    assert state == {
        "a.ipynb": _sha256(dest / "a.ipynb"),
        "sub/b.ipynb": _sha256(dest / "sub" / "b.ipynb"),
    }, state
    assert (dest / ".unsloth_sync_commit").read_text(encoding = "utf-8").strip() == TEMPLATE_COMMIT
    assert not (dest / ".unsloth_sync_partial").exists()


@behavioural
def test_an_unstageable_state_leaves_the_marker_off_instead_of_copying(tmp_path: Path):
    """Fail CLOSED when the staging path cannot be cleared either, exactly as the
    refresh child does: notebooks we cannot record read as user edits on the next run,
    which is `kept`, not `failed`, so the marker would be stamped over them for good.
    A DIRECTORY at the staging path is unlinkable and untruncatable for root too."""
    template = _template(tmp_path)
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / ".unsloth_sync_state.tmp").mkdir()

    run = _run(tmp_path, template, dest)
    assert run.returncode == 0, run.stdout + run.stderr

    assert not (dest / ".unsloth_sync_state").exists(), run.stdout + run.stderr
    assert not (
        dest / ".unsloth_sync_commit"
    ).exists(), "stamping the commit here strands every notebook this run copied"
    assert (dest / ".unsloth_sync_partial").exists()
    assert not (dest / "a.ipynb").exists(), "nothing may be published without a record"
    assert "could not be staged" in run.stdout, run.stdout


# Staging the temp file proves DEST was writable ONCE, not that it stays so. A quota
# or ENOSPC that lands on a post-copy `printf >> "$STATE.tmp"` leaves the notebook on
# disk with no record, and an unrecorded file is read as a user edit and never
# refreshed again -- while the marker is stamped anyway, because only `cp` failures
# were counted. RLIMIT_FSIZE reproduces it without needing a real full filesystem:
# the tiny notebooks copy fine, the growing state file is what hits the ceiling.
def _run_capped(template: Path, dest: Path, max_bytes: int):
    import resource
    import signal

    def _cap():
        # SIGXFSZ would kill the script outright; the shell must SEE the write error
        signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
        resource.setrlimit(resource.RLIMIT_FSIZE, (max_bytes, max_bytes))

    return subprocess.run(
        ["bash", str(SYNC)],
        capture_output = True,
        text = True,
        timeout = 300,
        preexec_fn = _cap,
        env = dict(
            os.environ,
            UNSLOTH_NOTEBOOKS_TEMPLATE = str(template),
            UNSLOTH_NOTEBOOKS_DIR = str(dest),
            UNSLOTH_SKIP_NOTEBOOK_REFRESH = "1",
            UNSLOTH_SKIP_NOTEBOOK_VIEW = "1",
            UNSLOTH_KEEP_COLAB_INTRO = "1",
        ),
    )


def _many_template(tmp_path: Path, count: int) -> Path:
    template = tmp_path / "template"
    template.mkdir(parents = True)
    for i in range(count):
        (template / f"n{i}.ipynb").write_text(f"N{i}", encoding = "utf-8")
    (template / ".unsloth_template_commit").write_text(TEMPLATE_COMMIT + "\n", encoding = "utf-8")
    return template


@behavioural
def test_a_failed_state_append_holds_back_the_commit_marker(tmp_path: Path):
    """A record lost after a successful copy must count as a population failure."""
    template = _many_template(tmp_path, 15)
    dest = tmp_path / "dest"
    dest.mkdir()

    run = _run_capped(template, dest, 512)
    assert run.returncode == 0, run.stdout + run.stderr

    copied = sorted(p.name for p in dest.glob("*.ipynb"))
    recorded = _state(dest)
    assert len(copied) > len(
        recorded
    ), "the cap did not bite; this test proves nothing unless some append failed"
    assert not (
        dest / ".unsloth_sync_commit"
    ).exists(), "notebooks copied but not recorded are user-owned for good once this is stamped"
    assert (dest / ".unsloth_sync_partial").exists(), run.stdout


@behavioural
def test_the_retry_after_a_failed_append_converges(tmp_path: Path):
    """And the retry must actually finish the job, not just defer forever."""
    template = _many_template(tmp_path, 15)
    dest = tmp_path / "dest"
    dest.mkdir()

    _run_capped(template, dest, 512)
    assert not (dest / ".unsloth_sync_commit").exists()

    run = _run(tmp_path, template, dest)
    assert run.returncode == 0, run.stdout + run.stderr
    assert len(_state(dest)) == 15, _state(dest)
    assert (dest / ".unsloth_sync_commit").exists(), run.stdout
    assert not (dest / ".unsloth_sync_partial").exists()


# CLASS GUARD, not another instance test. An unchecked append to a staged state file
# has now been found four separate times in this script: the populate copy loop, the
# populate merge block, the refresh restore loop (RS_TMP) and the refresh publish loop
# (TMPSTATE). Each time the record was lost while the failure counter stayed 0, so a
# truncated state was published AND the commit marker advanced, which strands every
# notebook whose hash was dropped. Rather than wait for the fifth, require every append
# to a staged state file to be accounted for.
_STAGED_STATE_TARGETS = ('>> "$STATE.tmp"', '>> "$TMPSTATE"', '>> "$RS_TMP"')

# an append is accounted for if it handles its own failure, or if the very next lines
# count one
_ACCOUNTED = (
    "|| populate_failed=",
    "|| failed=",
    "|| rs_ok=0",
    "|| unrecorded ",
    "|| drop_unrecordable ",
    "populate_failed=$((populate_failed + 1))",
    "failed=$((failed + 1))",
    "rs_ok=0",
)


def _exempt_lines(lines):
    """record_state() is dead code (no caller anywhere; the companion test below
    keeps it that way), and record_tmpstate() is the shared helper whose whole job is
    to perform the append and report the status its CALLERS must handle."""
    out = set()
    for name in ("record_state() {", "record_tmpstate() {"):
        start = next(i for i, l in enumerate(lines) if l.startswith(name))
        end = next(i for i in range(start, len(lines)) if lines[i] == "}")
        out.update(range(start, end + 1))
    return out


def test_every_staged_state_append_is_checked():
    lines = SYNC.read_text(encoding = "utf-8").splitlines()
    exempt = _exempt_lines(lines)
    unchecked = []
    for i, line in enumerate(lines):
        if i in exempt:
            continue
        if not any(t in line for t in _STAGED_STATE_TARGETS):
            continue
        window = " ".join(lines[i : i + 3])
        if not any(marker in window for marker in _ACCOUNTED):
            unchecked.append((i + 1, line.strip()))
    assert not unchecked, (
        "these appends to a staged state file neither handle their own failure nor "
        "have one counted immediately after, so a lost record would still publish a "
        "truncated state and advance the commit marker:\n"
        + "\n".join(f"  line {n}: {t}" for n, t in unchecked)
    )


def test_every_record_tmpstate_call_handles_a_failed_write():
    """The class that keeps recurring is not the raw append, it is the UNHANDLED
    record failure. Counting it is not enough on its own: the truncated state is
    published either way, so the file must also be rolled back or the write retried.
    Every call site has to say which."""
    lines = SYNC.read_text(encoding = "utf-8").splitlines()
    exempt = _exempt_lines(lines)
    bare = []
    for i, line in enumerate(lines):
        if i in exempt or "record_tmpstate " not in line:
            continue
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if "||" in stripped or stripped.startswith("if record_tmpstate"):
            continue
        bare.append((i + 1, stripped))
    assert not bare, (
        "these record_tmpstate calls ignore a failed write, so the notebook stays on "
        "disk with no record and the next refresh treats it as a user edit:\n"
        + "\n".join(f"  line {n}: {t}" for n, t in bare)
    )


def test_a_failed_record_can_roll_the_notebook_back():
    """drop_unrecordable must only ever remove OUR copy, never a user edit."""
    body = SYNC.read_text(encoding = "utf-8")
    assert "drop_unrecordable() {" in body
    fn = body[body.index("drop_unrecordable() {") :]
    fn = fn[: fn.index("\n}")]
    assert (
        'hash_of "$_d"' in fn and "${LAST[$1]:-}" in fn
    ), "the rollback must be gated on the file still matching the last record"


def test_the_marker_advance_is_gated_on_the_failure_counters():
    """The guard above is only worth anything while the counters still hold the
    marker back."""
    body = SYNC.read_text(encoding = "utf-8")
    assert '[ "$failed" -eq 0 ] && [ "$published" -eq 1 ]' in body
    assert '[ "$populate_failed" -eq 0 ]' in body


def test_record_state_is_still_uncalled():
    """The exemption above depends on it. A call site makes its unchecked append live,
    and it has a second problem waiting: its loop is a PIPELINE, so a failure counter
    set inside would not survive the subshell."""
    body = SYNC.read_text(encoding = "utf-8")
    calls = [
        ln.strip()
        for ln in body.splitlines()
        if "record_state" in ln
        and not ln.startswith("record_state() {")
        and not ln.lstrip().startswith("#")
    ]
    assert not calls, f"record_state gained a caller; the exemption is now unsound: {calls}"
