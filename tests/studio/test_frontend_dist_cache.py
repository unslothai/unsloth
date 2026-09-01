# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The frontend dist cache and BOTH installers' rebuild checks must read the same inputs.

Measured over 13 distinct Linux jobs on main, the frontend build is a median 36s of a
74s install, 49% of it. On Windows it is 96s of a ~257s install (`[72s] building
frontend...` -> `[168s] frontend built`), 37% of it, across five more jobs. The cache
exists to stop paying that eighteen times per commit.

What makes it safe is not the cache action, it is the agreement between three places:

    studio/setup.sh          rebuilds when anything under frontend/ (maxdepth 1, minus
                             bun.lock), frontend/src or frontend/public is NEWER than
                             frontend/dist
    studio/setup.ps1         the same predicate, over the same three groups, against
                             `(Get-Item $DistDir).LastWriteTime`
    the action's cache key   hashes exactly those three path groups

A hit therefore means the build inputs are byte-identical, which is strictly stronger
than the mtime test it rides on. Break the agreement and nothing goes red: the cache
keeps hitting and quietly starts serving a dist built from inputs the key no longer
covers, and every job downstream tests a stale bundle that passes. That is the whole
reason this file exists, and it is why it asserts against setup.sh's AND setup.ps1's own
source rather than a list written down here. A list written here would agree with itself
forever while the scripts moved.

Since the Windows jobs were added the key lives in ONE place,
`.github/actions/frontend-dist-restore`, and `install-unsloth-local` delegates to it.
Twelve workflows with their own copy of a key whose drift is silent would drift twelve
ways.

Five subtler failure modes are pinned too, each of which looks like success:

  * `restore-keys` on this cache. A near-miss download cache still supplies most of the
    wheels; a near-miss dist is a bundle built from different source. Wrong, not partial.
  * A restore with no touch. actions/cache restores through tar, which preserves the
    original mtimes, so the restored dist is older than the checkout that just wrote
    every source file and the installer rebuilds anyway. The cache would cost a
    download, save nothing, and report a hit.
  * A touch that does not update what the READER reads. setup.ps1 reads
    `(Get-Item $DistDir).LastWriteTime`; whether MSYS `touch` on a directory handle
    lands in that field is not something this repo has evidence for either way, so the
    Windows branch writes that property by name and the POSIX branch keeps the `touch`
    that is measured working on main.
  * An empty `hashFiles`. It returns "" when a glob matches nothing, collapsing every
    commit onto one key and serving an arbitrary dist.
  * A hit that was rebuilt anyway. Invisible in every signal except the wall clock, so
    the save action asserts it from the install log and fails the job.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml


REPO = Path(__file__).resolve().parents[2]
ACTIONS = REPO / ".github" / "actions"
RESTORE_ACTION = ACTIONS / "frontend-dist-restore" / "action.yml"
SAVE_ACTION = ACTIONS / "frontend-dist-save" / "action.yml"
INSTALL_ACTION = ACTIONS / "install-unsloth-local" / "action.yml"
WORKFLOWS = REPO / ".github" / "workflows"
SETUP_SH = REPO / "studio" / "setup.sh"
SETUP_PS1 = REPO / "studio" / "setup.ps1"


def _steps(action: Path) -> list[dict]:
    doc = yaml.safe_load(action.read_text(encoding = "utf-8")) or {}
    return [s for s in (doc.get("runs") or {}).get("steps") or [] if isinstance(s, dict)]


def _step(action: Path, fragment: str) -> dict | None:
    for step in _steps(action):
        if fragment.lower() in str(step.get("name", "")).lower():
            return step
    return None


def _code(step: dict) -> str:
    """A step's `run:` body with comment lines removed.

    Every assertion in this file that greps a script body goes through here, and that is
    not tidiness. These steps are heavily commented -- deliberately, since the reasoning
    is the point -- and the comments quote the very strings the assertions look for:
    "touch", "LastWriteTime", "building frontend", "exit 1". So an assertion run against
    the raw body can be satisfied by the explanation of the code instead of the code,
    and a guard that passes because of a comment is exactly the silence this design
    exists to remove. Two mutations survived that way before this helper existed.

    Line comments only. A `#` inside a string is left alone, which is why the split is
    anchored to the start of a line rather than done anywhere in it.
    """
    body = str(step.get("run", ""))
    return "\n".join(l for l in body.splitlines() if not l.lstrip().startswith("#"))


def _restore_step() -> dict:
    step = _step(RESTORE_ACTION, "Restore the built frontend")
    assert step is not None, (
        "frontend-dist-restore no longer restores a built frontend. If the cache was "
        "removed on purpose, delete this file; if it was renamed, retarget it."
    )
    return step


def _key() -> str:
    return str((_restore_step().get("with") or {}).get("key", ""))


def _balanced(text: str, open_at: int) -> str:
    """The substring inside the parentheses that open at ``open_at``.

    A non-greedy ``hashFiles\\((.*?)\\)`` stops at the first ``)``, which is the wrong
    one the moment an argument is itself a call -- and the key's arguments are
    ``format()`` calls now, because the globs carry a checkout prefix.
    """
    depth = 0
    for i in range(open_at, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[open_at + 1 : i]
    raise AssertionError(f"unbalanced parentheses from offset {open_at} in {text!r}")


def _key_patterns() -> tuple[set[str], set[str]]:
    """(hashed, excluded) path groups, read out of the key itself.

    Every argument is `format('{0}<glob>', inputs.path-prefix)`, so the glob is the
    quoted literal with its `{0}` placeholder stripped. Read structurally rather than by
    substring, so dropping a path group -- or prefixing one but not another -- shows up
    as a missing entry rather than as a passing test.

    Negations are separated out rather than treated as more hashed paths.
    `studio/frontend/*` is RECURSIVE in @actions/glob (matchDirectories plus implicit
    descendants), so it currently sweeps in `frontend/tests/**`, which the rebuild check
    never reads; #9380 narrows the key with `!studio/frontend/tests/**` and
    `!studio/frontend/scripts/**`. Splitting them here is what lets this guard describe
    the truth whichever order that PR and this one land in, instead of pinning the
    literal argument list and needing a follow-up edit either way.
    """
    at = _key().find("hashFiles(")
    assert at != -1, f"the dist cache key does not call hashFiles: {_key()!r}"
    inner = _balanced(_key(), at + len("hashFiles"))
    hashed, excluded = set(), set()
    for literal in re.findall(r"'([^']*)'", inner):
        if "studio/" not in literal:
            continue  # the format spec's own placeholder, or a prefix argument
        glob = literal.replace("{0}", "")
        target = excluded if glob.startswith("!") else hashed
        target.add(glob.lstrip("!").removesuffix("/**").rstrip("/*").rstrip("/"))
    return hashed, excluded


def _key_globs() -> set[str]:
    return _key_patterns()[0]


def _staleness_inputs_sh() -> set[str]:
    """The paths setup.sh's rebuild check compares against frontend/dist."""
    text = SETUP_SH.read_text(encoding = "utf-8")
    block = re.search(
        r"Detect whether frontend needs building(.*?)end packaged/Tauri guard", text, re.S
    )
    assert block, "could not find the frontend staleness check in studio/setup.sh"
    body = block.group(1)
    assert "-newer" in body, "the setup.sh block found is not the mtime staleness check"
    found = set()
    for m in re.finditer(r'"\$SCRIPT_DIR/(frontend[^"]*)"', body):
        path = m.group(1)
        if path.endswith("/dist"):
            continue
        found.add("studio/" + path)
    return found


def _staleness_inputs_ps1() -> set[str]:
    """The paths setup.ps1's rebuild check compares against frontend/dist.

    Read out of setup.ps1 the same way the setup.sh reader works, and for the same
    reason. The Windows block does not spell its paths out as string literals -- it
    walks `@("src", "public")` under `$FrontendDir` recursively and then `$FrontendDir`
    itself non-recursively -- so this reconstructs the three groups from that structure.
    Adding a fourth directory to the foreach list, or dropping one, changes what comes
    back here and the key stops covering the check.
    """
    text = SETUP_PS1.read_text(encoding = "utf-8")
    block = re.search(
        r'\$DistDir = Join-Path \$FrontendDir "dist"(.*?)# Provision Node when the frontend build',
        text,
        re.S,
    )
    assert block, "could not find the frontend staleness check in studio/setup.ps1"
    body = block.group(1)
    # The anchor that proves this is the mtime comparison and not some other block.
    assert re.search(r"\(Get-Item \$DistDir\)\.LastWriteTime", body), (
        "the setup.ps1 block found does not read (Get-Item $DistDir).LastWriteTime, so "
        "it is not the staleness check this key is supposed to agree with"
    )

    found = set()
    # The recursive subdirectory sweep:
    sub = re.search(r"foreach \(\$subDir in @\(([^)]*)\)\)", body)
    assert sub, "setup.ps1's staleness check no longer sweeps a list of subdirectories"
    for name in re.findall(r'"([^"]+)"', sub.group(1)):
        found.add(f"studio/frontend/{name}")
    # The non-recursive top-level file sweep over $FrontendDir itself.
    assert re.search(
        r"Get-ChildItem -Path \$FrontendDir -File", body
    ), "setup.ps1's staleness check no longer scans the top-level frontend files"
    found.add("studio/frontend")
    return found


@pytest.mark.parametrize(
    "reader",
    [
        pytest.param(_staleness_inputs_sh, id = "setup.sh"),
        pytest.param(_staleness_inputs_ps1, id = "setup.ps1"),
    ],
)
def test_the_key_covers_every_path_the_rebuild_check_reads(reader) -> None:
    missing = sorted(reader() - _key_globs())
    assert not missing, (
        f"{reader.__name__} says the installer decides to rebuild the frontend by "
        f"looking at {missing}, and the dist cache key does not hash them. A change to "
        f"those files would not change the key, so the cache would hit and serve a dist "
        f"built from different source, and nothing would go red. Key: {_key()!r}"
    )


def test_the_two_installers_agree_on_what_makes_a_dist_stale() -> None:
    """One key serves both, so the two checks reading different paths would be a bug.

    Neither script alone can reveal that. setup.sh could gain a path group, the key
    could gain it too, both Linux tests would pass, and Windows would keep hitting a key
    that no longer describes what setup.ps1 reads.
    """
    assert _staleness_inputs_sh() == _staleness_inputs_ps1(), (
        f"studio/setup.sh checks {sorted(_staleness_inputs_sh())} but studio/setup.ps1 "
        f"checks {sorted(_staleness_inputs_ps1())}. They share one cache key, so the "
        f"key can only be right for both if they read the same inputs. Fix the scripts, "
        f"or split the key by runner.os and say why here."
    )


def test_the_key_does_not_hash_paths_the_rebuild_check_ignores() -> None:
    """Not a style rule: an over-broad key silently destroys the hit rate.

    bun.lock is the deliberate exception. Both scripts must exclude it because the
    install regenerates it and it would self-trigger every run; the cache has no such
    problem, and a lockfile change means different dependencies and so a different
    bundle. It is covered by the `studio/frontend/*` glob, which is why that glob is
    allowed to be broader than the checks' maxdepth-1 scan rather than being narrowed to
    match it.
    """
    extra = sorted(_key_globs() - _staleness_inputs_sh() - _staleness_inputs_ps1())
    assert extra == [], (
        f"the dist cache key hashes {extra}, which neither installer's rebuild check "
        f"reads. Every unrelated edit to those paths would miss the cache for no reason. "
        f"If the extra path genuinely affects the built bundle, say so where the key is "
        f"defined and widen this test deliberately."
    )


def test_no_exclusion_hides_a_path_the_rebuild_check_reads() -> None:
    """Narrowing the key is right; narrowing it past what the installers read is a bug.

    `studio/frontend/*` is recursive in @actions/glob, so the key sweeps in directories
    the rebuild check never looks at (`frontend/tests/**` alone is 456 files here), and
    every unrelated edit to them misses the cache for nothing. #9380 fixes that with `!`
    negations, which is a hit-rate improvement and safe.

    One negation too many is not safe, and it fails in the opposite, silent direction: a
    `!studio/frontend/src/**` would leave the key unchanged across a real source edit, so
    the cache would hit and serve the previous bundle. This is the guard for that, and it
    is why the exclusions are read out of the key separately rather than folded in with
    the hashed paths.
    """
    hashed, excluded = _key_patterns()
    reads = _staleness_inputs_sh() | _staleness_inputs_ps1()
    offenders = sorted(e for e in excluded if any(r == e or r.startswith(e + "/") for r in reads))
    assert not offenders, (
        f"the dist cache key excludes {offenders}, which the installers' rebuild check "
        f"DOES read. An edit there would not change the key, so the cache would hit and "
        f"serve a bundle built from different source. Key: {_key()!r}"
    )
    for e in excluded:
        assert any(e.startswith(h + "/") for h in hashed), (
            f"the key excludes {e!r}, which none of its hashed globs {sorted(hashed)} "
            f"would have matched anyway. A negation that subtracts nothing reads as a "
            f"narrowing that is not in force."
        )


def test_the_key_excludes_the_frontend_subdirs_the_rebuild_check_never_reads() -> None:
    """The other direction, and the one that protects #9380 from being undone.

    Its sibling above forbids an exclusion that hides a path the rebuild check DOES read.
    This forbids the reverse: dropping an exclusion, which puts the key back to hashing
    every file under `frontend/tests` (456 of them) and `frontend/scripts`. Neither is
    read by either installer's staleness check, so an edit to a frontend TEST would evict
    a dist whose bundle is byte-identical.

    That regression is invisible. The cache still works, still hits sometimes, and simply
    hits less -- there is no failure to notice, only a number nobody is watching.

    Derived from the tree rather than a written-down list, so a frontend subdirectory
    added later surfaces here as a decision to make instead of quietly costing hit rate.
    """
    frontend = REPO / "studio" / "frontend"
    if not frontend.is_dir():
        pytest.skip("studio/frontend is absent")
    _, excluded = _key_patterns()
    read = {Path(p).name for p in _staleness_inputs_sh() if Path(p).name != "frontend"}
    # dist is the cache payload itself; node_modules is never committed.
    ignored = {"dist", "node_modules"}
    unread = {
        d.name
        for d in frontend.iterdir()
        if d.is_dir() and d.name not in read and d.name not in ignored
    }
    missing = sorted(d for d in unread if f"studio/frontend/{d}" not in excluded)
    assert not missing, (
        f"studio/frontend/{{{','.join(missing)}}} is hashed into the dist cache key but "
        f"neither installer's rebuild check reads it, so every edit under it evicts a "
        f"dist that would have been byte-identical. `studio/frontend/*` descends into "
        f"subdirectories. Either add `!studio/frontend/<dir>/**` to the key, or say at "
        f"the key why that directory genuinely changes the built bundle."
    )


def test_the_dist_cache_has_no_restore_keys() -> None:
    with_ = _restore_step().get("with") or {}
    assert "restore-keys" not in with_, (
        "the frontend dist cache has restore-keys. A prefix hit would serve a bundle "
        "built from DIFFERENT source, which is wrong rather than partial. The uv "
        "download cache in install-unsloth-local does want them, and that contrast is "
        "the point: a near-miss download still supplies most of the wheels."
    )


# The touch, which is where a hit stops being a hit.
# ---------------------------------------------------------------------------
def _touch_steps() -> list[dict]:
    return [s for s in _steps(RESTORE_ACTION) if "outrank its sources" in str(s.get("name", ""))]


def test_a_restored_dist_is_made_newer_than_the_checkout_on_every_os() -> None:
    """One branch per OS, and BOTH have to exist.

    A single `shell: bash` + `touch` step would look complete and cover Windows by
    accident at best: setup.ps1 reads `(Get-Item $DistDir).LastWriteTime`, and whether
    MSYS `touch` on a directory handle updates that field is not something anyone here
    has evidence for. The five Windows jobs would report a hit and rebuild anyway, which
    costs a download and saves nothing while looking exactly like success.
    """
    steps = _touch_steps()
    assert steps, (
        "nothing makes the restored dist outrank its sources. actions/cache restores "
        "through tar, which preserves the original mtimes, so the freshly checked-out "
        "tree is newer than the restored dist and the installer rebuilds anyway."
    )
    covered = " ".join(str(s.get("if", "")) for s in steps)
    assert "runner.os != 'Windows'" in covered and "runner.os == 'Windows'" in covered, (
        f"the touch does not branch on runner.os, so one platform is unhandled: "
        f"{[s.get('if') for s in steps]}"
    )
    for step in steps:
        assert "steps.restore.outputs.cache-hit == 'true'" in str(step.get("if", "")), (
            f"the touch in {step.get('name')!r} is not gated on a cache hit, so a MISS "
            f"would touch a dist that was never restored and suppress the build that "
            f"has to happen"
        )


def test_the_windows_touch_writes_the_property_setup_ps1_reads() -> None:
    """`touch` and `LastWriteTime` are not interchangeable claims on NTFS.

    setup.ps1 reads `(Get-Item $DistDir).LastWriteTime`. The Windows branch writes that
    same property through the same API, so no inference is needed about MSYS's utime
    path. If someone collapses the two branches back into one bash `touch`, this is the
    test that says why not.
    """
    win = [s for s in _touch_steps() if "runner.os == 'Windows'" in str(s.get("if", ""))]
    assert len(win) == 1, f"expected exactly one Windows touch branch, got {len(win)}"
    step = win[0]
    assert step.get("shell") == "pwsh", (
        f"the Windows touch runs under {step.get('shell')!r}. It has to be pwsh: the "
        f"point is to write the same property setup.ps1 reads, by name."
    )
    body = _code(step)
    assert re.search(r"\(Get-Item[^)]*\)\.LastWriteTime\s*=", body), (
        f"the Windows touch does not assign (Get-Item ...).LastWriteTime, which is the "
        f"exact field studio/setup.ps1:3526-3549 compares against: {body!r}"
    )


def test_the_posix_touch_still_touches() -> None:
    posix = [s for s in _touch_steps() if "runner.os != 'Windows'" in str(s.get("if", ""))]
    assert len(posix) == 1, f"expected exactly one POSIX touch branch, got {len(posix)}"
    step = posix[0]
    assert step.get("shell") == "bash", step.get("shell")
    assert re.search(
        r"^\s*touch\s+\"?\$?\{?DIST", _code(step), re.M
    ), f"the POSIX branch no longer touches the dist directory: {_code(step)!r}"


# What each branch has to do AFTER stamping the directory, expressed as the two things that distinguish "I called the
# API" from "the dist actually ended up newer": re-READ the timestamp, and COMPARE it against the sources with a branch
# that can fail.
_READBACK = {
    # `find ... -newer "$DIST"` is setup.sh's own predicate, re-run.
    "posix": (r'-newer\s+"\$DIST"', r'if\s+\[\s+-n\s+"\$newer"\s+\]'),
    # A second Get-Item read (the first one is the assignment), then the comparison.
    "Windows": (r"\$distTime\s*=\s*\(Get-Item[^)]*\)\.LastWriteTime", r"if\s*\(\$newer\)"),
}


@pytest.mark.parametrize("os_name", sorted(_READBACK))
def test_each_touch_reads_its_work_back(os_name: str) -> None:
    """Touching is not the claim that matters; the dist ENDING UP newer is.

    A `touch` that returns 0 and a `find -newer dist` that comes back empty are
    different statements, and only the second one stops the rebuild. Both branches
    re-evaluate the installer's own predicate and fail loudly, because the alternative is
    discovering it as 96s that nobody attributes to anything.

    Asserted as the specific read-and-compare, not as "the body contains ::error:: and
    exit 1 somewhere". It was written the loose way first, and neutering the entire
    Windows comparison did not turn it red: the earlier "restored no directory" check
    supplied both strings, and `bun.lock` survived in the Get-ChildItem filter. A guard
    satisfied by a different guard standing next to it is not measuring anything.
    """
    marker = "runner.os == 'Windows'" if os_name == "Windows" else "runner.os != 'Windows'"
    step = next(s for s in _touch_steps() if marker in str(s.get("if", "")))
    body = _code(step)
    read, compare = _READBACK[os_name]
    assert re.search(read, body), (
        f"the {os_name} touch branch never re-reads the timestamp it just wrote, so it "
        f"proves only that the call returned: {body!r}"
    )
    at = re.search(compare, body)
    assert at, (
        f"the {os_name} touch branch does not compare the restored dist against its "
        f"sources after stamping it: {body!r}"
    )
    tail = body[at.start() :]
    assert "::error::" in tail and "exit 1" in tail, (
        f"the {os_name} touch branch compares, then cannot fail on the result, so a "
        f"stamp that did not take is silent: {tail!r}"
    )
    assert "bun.lock" in body, (
        f"the {os_name} re-check does not exclude bun.lock, so it is not the predicate "
        f"the installer is about to evaluate (the install regenerates that file, so it "
        f"would self-trigger every run)"
    )


def test_a_degenerate_key_is_refused_before_the_restore_runs() -> None:
    steps = _steps(RESTORE_ACTION)
    guard = next(
        (i for i, s in enumerate(steps) if "hashes nothing" in str(s.get("name", ""))), None
    )
    restore = next(
        (i for i, s in enumerate(steps) if "cache/restore" in str(s.get("uses", ""))), None
    )
    assert guard is not None, (
        'nothing refuses an empty hashFiles result. It returns "" when a glob matches '
        "no file, which collapses every commit onto one key and serves an arbitrary "
        "dist, with the restore succeeding and the build skipped."
    )
    assert "exit 1" in _code(steps[guard]), "the degenerate-key check does not fail the job"
    assert restore is not None and guard < restore, (
        "the degenerate-key check runs AFTER the restore, so an empty key is used to "
        "look something up first -- and on a repo where anything ever saved under that "
        "empty key, the lookup hits"
    )


def test_the_degenerate_key_check_runs_on_a_miss_too() -> None:
    """Gating it on a hit would blind it to the case it is actually for.

    An empty key is what a MOVED FRONTEND produces, and on the first run after such a
    move there is nothing under the empty key yet, so the restore misses and a
    hit-gated check says nothing. The build then runs, the save writes the freshly
    built dist under the empty key, and from the next run onward every commit hits it.
    The damage is done on the miss; the check has to be there for it.
    """
    step = _step(RESTORE_ACTION, "hashes nothing")
    assert step is not None, "the degenerate-key check is gone"
    cond = str(step.get("if", "")).strip()
    assert not cond, (
        f"the degenerate-key check is conditional ({cond!r}). It must run unconditionally: "
        f"an empty key comes from a moved frontend layout, and on the first run after "
        f"that move the restore MISSES -- so a hit-gated check is silent for exactly the "
        f"run that goes on to poison the key."
    )


def test_no_windows_job_reaches_the_posix_install_composite() -> None:
    """install-unsloth-local runs `bash install.sh`; Windows is installed by install.ps1.

    Worth asserting rather than leaving to review, because the failure would not be a
    clean "wrong installer" error. Git Bash exists on windows-latest, so `bash
    install.sh --local --no-torch` starts, and the composite writes UV_CACHE_DIR into
    $GITHUB_ENV for every later step on the way. The five Windows jobs call install.ps1
    from their own `shell: pwsh` step and take the dist cache through
    frontend-dist-restore/-save directly, which is why those two are OS-agnostic and
    this one is not.
    """
    offenders = []
    for name, jid, job in _jobs():
        runs_on = str(job.get("runs-on", ""))
        matrix = str(((job.get("strategy") or {}).get("matrix") or ""))
        windows = "windows" in runs_on.lower() or "windows" in matrix.lower()
        if not windows:
            continue
        for step in job.get("steps") or []:
            if "install-unsloth-local" in str(step.get("uses", "")):
                offenders.append(f"{name}:{jid} (runs-on: {runs_on})")
    assert not offenders, (
        f"these Windows jobs use install-unsloth-local, which runs `bash install.sh`: "
        f"{offenders}. On Windows that is the wrong installer and it does not fail "
        f"cleanly -- Git Bash is present, so it starts. Use install.ps1 with "
        f"frontend-dist-restore/-save around it."
    )


# ---------------------------------------------------------------------------
def test_the_dist_cache_is_saved_on_main_only() -> None:
    step = _step(SAVE_ACTION, "Save the built frontend")
    assert step is not None, "the dist cache is restored but never saved, so it can only ever miss"
    cond = str(step.get("if", ""))
    assert re.search(r"github\.ref\s*==\s*'refs/heads/main'", cond), (
        f"the dist cache is saved off main: {cond!r}. A PR-scoped entry can only be "
        f"restored by re-runs of that same PR while still counting against the shared "
        f"budget, evicting the copy every PR can read."
    )
    assert "always()" not in cond, (
        "the dist save runs under always(), so an install that failed part-way through "
        "the frontend build would store a partial dist under an immutable key and serve "
        "it to every later run. Leaving always() off means a failed install simply "
        "skips this step."
    )


def test_a_cache_hit_that_rebuilt_anyway_fails_the_job() -> None:
    """The one failure mode of this cache that no other signal reveals.

    Green job, `Cache hit for: fe-dist-...` in the log, a healthy hit rate on the cache
    dashboard, and 96s still spent. Without this assertion the only way to notice is for
    someone to time the install by hand, which is how the cost was found the first time.
    """
    step = _step(SAVE_ACTION, "reused and not rebuilt")
    assert step is not None, (
        "nothing checks that a restored dist was actually reused. A hit whose touch did "
        "not take rebuilds the frontend and reports success."
    )
    assert str(step.get("if", "")).strip() == "inputs.cache-hit == 'true'", (
        f"the reuse assertion is not gated on a hit: {step.get('if')!r}. On a MISS the "
        f"installer is supposed to build, so asserting there would fail every cold run."
    )
    body = _code(step)
    assert (
        "building frontend" in body
    ), "the reuse assertion does not look for the rebuild marker both installers emit"
    # Each guarded condition, and the failure that must follow it.
    for condition, what in (
        (r'\[\s+!\s+-f\s+"\$INSTALL_LOG"\s+\]', "a missing install log"),
        (r'\[\s+!\s+-d\s+"\$DIST"\s+\]', "a dist that vanished during the install"),
        (r"grep\s+-qi\s+'building frontend'", "the rebuild marker"),
    ):
        at = re.search(condition, body)
        assert at, f"the reuse assertion no longer checks for {what}: {body!r}"
        block = body[at.start() : at.start() + 600]
        assert "exit 1" in block.split("\nfi")[0], (
            f"the reuse assertion detects {what} and does not fail the job for it. "
            f"'Found nothing to read' must not read the same as 'passed': {block!r}"
        )
    assert "exit 0" not in body, (
        f"the reuse assertion contains an `exit 0`, which is how this check gets "
        f"disarmed while still looking present -- a branch that returns success is "
        f"indistinguishable from one that verified something: {body!r}"
    )


@pytest.mark.parametrize(
    "script,marker",
    [
        (SETUP_SH, "building frontend"),
        (SETUP_PS1, "building frontend"),
        (SETUP_SH, "up to date"),
        (SETUP_PS1, "up to date"),
    ],
)
def test_the_markers_the_reuse_assertion_greps_for_still_exist(script: Path, marker: str) -> None:
    """The assertion above is a grep, and a grep for a string nobody prints is vacuous.

    Renaming either marker in an installer would disarm the reuse check without anything
    going red -- the exact shape of silence this whole file is about. Pinned here so the
    rename fails in pytest, one file away from the edit, rather than in CI six weeks
    later as unexplained minutes.
    """
    assert marker in script.read_text(encoding = "utf-8"), (
        f"{script.name} no longer prints {marker!r}, so the reuse assertion in "
        f"frontend-dist-save greps for a string that never appears. Update both "
        f"together."
    )


# One definition of the key, and where it may be referenced from.
# ---------------------------------------------------------------------------
def test_the_cache_key_has_exactly_one_definition() -> None:
    """Nine call sites with their own copy would drift, and the drift is silent.

    That is the entire argument for the composite pair over pasting four steps into
    five workflows: the key and the two installers' staleness checks have to agree, and
    an agreement maintained in nine places is not maintained.
    """
    definers = []
    for path in sorted(list(ACTIONS.rglob("action.yml")) + list(WORKFLOWS.glob("*.yml"))):
        if re.search(r"key:\s*fe-dist-", path.read_text(encoding = "utf-8")):
            definers.append(str(path.relative_to(REPO)))
    assert definers == [".github/actions/frontend-dist-restore/action.yml"], (
        f"the fe-dist cache key is defined in {definers}. It must have exactly one "
        f"definition: a second copy agrees today and drifts silently, with the cache "
        f"still hitting while it serves a dist built from inputs the key no longer "
        f"covers."
    )


def test_install_unsloth_local_delegates_rather_than_carrying_its_own_copy() -> None:
    uses = [str(s.get("uses", "")) for s in _steps(INSTALL_ACTION)]
    assert "./.github/actions/frontend-dist-restore" in uses, uses
    assert "./.github/actions/frontend-dist-save" in uses, uses
    assert not any("actions/cache" in u and "fe" in u for u in uses)


def _jobs():
    for f in sorted(WORKFLOWS.glob("*.yml")):
        doc = yaml.safe_load(f.read_text(encoding = "utf-8"))
        if isinstance(doc, dict) and isinstance(doc.get("jobs"), dict):
            for jid, job in doc["jobs"].items():
                if isinstance(job, dict):
                    yield f.name, jid, job


def test_no_nested_checkout_job_calls_an_action_that_nests_another_one() -> None:
    """`uses: ./X` resolves as $GITHUB_WORKSPACE/X, in a composite too, and takes no expressions.

    So a job that checks this repo out under `unsloth/` can call
    `./unsloth/.github/actions/install-unsloth-local` and the runner will find it -- and
    then fail inside it, on `uses: ./.github/actions/frontend-dist-restore`, with "Can't
    find 'action.yml'". Three jobs here do check out that way (notebooks-ci
    api-introspect, version-compat-ci zoo-imports-under-spoof and grpo-fake-run); none
    calls this action today, and this keeps it that way with a reason attached instead of
    that error message.

    A nested-checkout job that wants the dist cache calls frontend-dist-restore and
    frontend-dist-save directly and passes their `path-prefix`. Those two nest nothing.
    """
    nesting = {
        p.parent.name
        for p in ACTIONS.rglob("action.yml")
        if re.search(r"uses:\s*\./\.github/actions/", p.read_text(encoding = "utf-8"))
    }
    assert nesting, "no composite action nests another any more; retarget or delete this test"
    offenders = []
    for name, jid, job in _jobs():
        steps = job.get("steps") or []
        if not any(
            "actions/checkout" in str(s.get("uses", "")) and (s.get("with") or {}).get("path")
            for s in steps
        ):
            continue
        for step in steps:
            uses = str(step.get("uses", ""))
            if any(uses.endswith(f"/.github/actions/{n}") for n in sorted(nesting)):
                offenders.append(f"{name}:{jid}: {uses}")
    assert not offenders, (
        f"these jobs check this repo out into a subdirectory and call an action that "
        f"itself references a local action by workspace-relative path, which the runner "
        f"resolves from GITHUB_WORKSPACE and cannot be prefixed (uses: takes no "
        f"expressions): {offenders}. Call frontend-dist-restore/-save directly with "
        f"path-prefix instead."
    )


def test_a_nested_checkout_caller_must_pass_a_prefix_that_can_match() -> None:
    """The prefix is a string glued in front of a glob, so a missing slash matches nothing.

    hashFiles returns "" for that rather than failing, and an empty key collapses every
    commit onto one entry. The action refuses an empty hash at runtime; this catches the
    same mistake in pytest, and catches the opposite one -- a nested checkout that
    forgets the prefix entirely -- which the action cannot distinguish from a moved
    frontend.
    """
    offenders = []
    for name, jid, job in _jobs():
        steps = job.get("steps") or []
        checkout_dirs = [
            str((s.get("with") or {}).get("path")).strip("/")
            for s in steps
            if "actions/checkout" in str(s.get("uses", ""))
            and (s.get("with") or {}).get("path")
            and not str((s.get("with") or {}).get("repository") or "")
            .rstrip("/")
            .endswith(("/unsloth",))
            or (
                "actions/checkout" in str(s.get("uses", ""))
                and (s.get("with") or {}).get("path")
                and not (s.get("with") or {}).get("repository")
            )
        ]
        for step in steps:
            if "frontend-dist-" not in str(step.get("uses", "")):
                continue
            prefix = str((step.get("with") or {}).get("path-prefix", ""))
            if prefix and not prefix.endswith("/"):
                offenders.append(f"{name}:{jid}: path-prefix {prefix!r} has no trailing slash")
            expected = f"{checkout_dirs[0]}/" if checkout_dirs else ""
            if prefix != expected:
                offenders.append(
                    f"{name}:{jid}: path-prefix is {prefix!r} but the repo is checked "
                    f"out at {expected!r}"
                )
    assert not offenders, (
        "these frontend-dist call sites pass a prefix that cannot resolve, so hashFiles "
        "matches nothing and the key is degenerate:\n  " + "\n  ".join(offenders)
    )


def _produces_on_main(name: str) -> bool:
    """Whether ``name``'s triggers can routinely put github.ref on refs/heads/main.

    The save is gated on `refs/heads/main`, so this is what decides whether a save step
    in that workflow can ever fire. `pull_request` gives `refs/pull/N/merge`; `push` to
    main and `schedule` (which runs on the default branch) both give the real thing.

    `workflow_dispatch` is deliberately NOT counted. It can be dispatched from main, so a
    save would technically fire -- but only when a human remembers to press the button,
    and a cache that fills on that schedule is not a cache. Counting it would make the
    rule below vacuous, since almost every workflow here has one.
    """
    doc = yaml.safe_load((WORKFLOWS / name).read_text(encoding = "utf-8"))
    on = doc.get("on", doc.get(True)) or {}
    if isinstance(on, str):
        on = {on: None}
    elif isinstance(on, list):
        on = dict.fromkeys(on)
    if "schedule" in on:
        return True
    push = on.get("push")
    if push is None and "push" not in on:
        return False
    branches = (push or {}).get("branches") if isinstance(push, dict) else None
    return branches is None or "main" in branches


def test_every_restored_dist_is_also_saved_and_wired_to_its_restore() -> None:
    """A restore with no save fills nothing; a save reading the wrong id saves nothing.

    Both halves are silent when wrong: the save takes the key and the hit flag from the
    restore's outputs, so a renamed or missing id yields empty inputs and an entry that
    is never written, with a green job either way.

    The pair is always required. Whether its UPLOAD half is enabled is derived from the
    workflow's triggers rather than allowlisted, so the one consumer-only lane
    (startup-profile-ci, which has no `push` and no `schedule`) passes `save: 'false'`
    and still runs the reuse assertion -- and adding `push:` there without enabling the
    save, or enabling the save without the trigger, both go red.
    """
    offenders = []
    for name, jid, job in _jobs():
        steps = job.get("steps") or []
        restore = next(
            (s for s in steps if "frontend-dist-restore" in str(s.get("uses", ""))), None
        )
        save = next((s for s in steps if "frontend-dist-save" in str(s.get("uses", ""))), None)
        if restore is None and save is None:
            continue
        if restore is None or save is None:
            offenders.append(f"{name}:{jid}: restore={restore is not None} save={save is not None}")
            continue
        ident = restore.get("id")
        if not ident:
            offenders.append(f"{name}:{jid}: the restore step has no id")
            continue
        with_ = save.get("with") or {}
        for field in ("cache-hit", "key"):
            if f"steps.{ident}.outputs.{field}" not in str(with_.get(field, "")):
                offenders.append(f"{name}:{jid}: save does not take {field} from steps.{ident}")
        if steps.index(restore) > steps.index(save):
            offenders.append(f"{name}:{jid}: the save runs before the restore")

        uploads = str(with_.get("save", "true")) != "false"
        if uploads and not _produces_on_main(name):
            offenders.append(
                f"{name}:{jid}: saves the dist cache, but {name} has no `push` to main "
                f"and no `schedule`, so github.ref is never refs/heads/main and the "
                f"upload can only fire on a hand-dispatched run. Pass save: 'false'."
            )
        if not uploads and _produces_on_main(name):
            offenders.append(
                f"{name}:{jid}: passes save: 'false', but {name} DOES run on main, so "
                f"it is a producer and is declining to fill the cache every PR reads."
            )
    assert not offenders, "\n  ".join(["broken frontend-dist wiring:"] + offenders)


def test_at_least_one_producer_actually_fills_the_cache() -> None:
    """A cache every lane consumes and none populates hits exactly never.

    The rule above is a per-job consistency check and would be perfectly happy with
    `save: 'false'` everywhere, as long as no workflow ran on main. This is the check
    that the set is non-empty.
    """
    producers = {
        name
        for name, jid, job in _jobs()
        for s in job.get("steps") or []
        if "frontend-dist-save" in str(s.get("uses", ""))
        and str((s.get("with") or {}).get("save", "true")) != "false"
    }
    assert producers, "no job saves the frontend dist cache, so every restore can only ever miss"


def test_the_restore_comes_before_the_install_and_the_save_after_it() -> None:
    """Either one on the wrong side is inert and still looks right.

    A restore after the install restores a dist nothing will read; a save before it
    stores whatever the previous run left behind.
    """
    offenders = []
    for name, jid, job in _jobs():
        steps = job.get("steps") or []
        idx = {
            role: i
            for i, s in enumerate(steps)
            for role in ("restore", "save")
            if f"frontend-dist-{role}" in str(s.get("uses", ""))
        }
        if "restore" not in idx:
            continue
        # The step that INVOKES the installer, not every step that mentions it.
        # These workflows also parse install.ps1 with the AST and grep logs/install.log, and a bare `install\.(ps1|sh)`
        # match picks those up and reports a false ordering bug.
        installs = [
            i
            for i, s in enumerate(steps)
            if re.search(r"install\.(ps1|sh) --local", str(s.get("run", "")))
        ]
        if not installs:
            offenders.append(f"{name}:{jid}: restores a dist but never runs an installer")
            continue
        if idx["restore"] > min(installs):
            offenders.append(f"{name}:{jid}: the restore runs after the install")
        if "save" in idx and idx["save"] < max(installs):
            offenders.append(f"{name}:{jid}: the save runs before the install")
    assert not offenders, "\n  ".join(["misordered frontend-dist steps:"] + offenders)


# Named, not detected: a lane whose whole point is a cold machine should have to be removed from this list
# ---------------------------------------------------------------------------
COLD_INSTALL_WORKFLOWS = (
    "clean-machine-install-ci.yml",
    "desktop-app-clean-machine-ci.yml",
    "interrupted-install-ci.yml",
    "release-desktop.yml",
)

# Cold at JOB level, inside a workflow whose other jobs legitimately use the cache.
COLD_INSTALL_JOBS = (("studio-windows-inference-smoke.yml", "no-vs-cpu"),)


@pytest.mark.parametrize("name", COLD_INSTALL_WORKFLOWS)
def test_cold_install_lanes_never_adopt_this_action(name: str) -> None:
    """A prebuilt frontend on a lane named for a cold machine proves nothing, and passes.

    These exist to show the installer works where nothing is present. Handing one a
    frontend that was built on another machine last week removes 96s of the thing they
    are testing, and the lane still reports success.
    """
    path = WORKFLOWS / name
    if not path.exists():
        pytest.skip(f"{name} no longer exists")
    text = path.read_text(encoding = "utf-8")
    assert "frontend-dist-" not in text, (
        f"{name} uses the frontend dist cache. A cold-install lane served a prebuilt "
        f"frontend proves nothing and still goes green."
    )
    assert "install-unsloth-local" not in text, (
        f"{name} uses install-unsloth-local, which now restores a prebuilt frontend as "
        f"well as warming uv's cache."
    )


@pytest.mark.parametrize("name,jid", COLD_INSTALL_JOBS)
def test_cold_install_jobs_never_adopt_this_action(name: str, jid: str) -> None:
    """Workflow-level exclusion is not enough when the cold lane is one job of several.

    studio-windows-inference-smoke.yml's `inference-smoke` is a target and its
    `no-vs-cpu` is not, in the same file, so a check that reads whole files would either
    forbid both or permit both.
    """
    doc = yaml.safe_load((WORKFLOWS / name).read_text(encoding = "utf-8"))
    job = (doc.get("jobs") or {}).get(jid)
    assert job is not None, f"{name} no longer has job {jid}; update COLD_INSTALL_JOBS"
    offenders = [
        str(s.get("uses", ""))
        for s in job.get("steps") or []
        if "frontend-dist-" in str(s.get("uses", ""))
        or "install-unsloth-local" in str(s.get("uses", ""))
    ]
    assert not offenders, (
        f"{name}:{jid} is a deliberate cold-install lane and must not be served a "
        f"prebuilt frontend: {offenders}"
    )


def test_the_actions_are_actually_used() -> None:
    """Otherwise every assertion above guards something nothing runs."""
    users = [
        p.name
        for p in WORKFLOWS.glob("*.yml")
        if "frontend-dist-restore" in p.read_text(encoding = "utf-8")
    ]
    assert len(users) >= 5, f"only {len(users)} workflows restore a dist: {users}"


def test_the_guard_is_reading_real_files() -> None:
    """Every assertion above passes vacuously if these files stop being found."""
    for path in (RESTORE_ACTION, SAVE_ACTION, INSTALL_ACTION, SETUP_SH, SETUP_PS1):
        assert path.is_file(), path
    assert len(_staleness_inputs_sh()) >= 3, _staleness_inputs_sh()
    assert len(_staleness_inputs_ps1()) >= 3, _staleness_inputs_ps1()
    assert len(_key_globs()) >= 3, _key_globs()
