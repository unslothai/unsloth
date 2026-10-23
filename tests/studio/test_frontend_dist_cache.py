# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""setup.sh, setup.ps1 and the dist cache key must read the same inputs, or a stale dist is served."""

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
    """Run body with whole-line comments dropped, so greps cannot match the explanations quoting them."""
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
    """Text inside the parens at open_at; a non-greedy match stops early when an argument is a call."""
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
    """Hashed and excluded globs read from the cache key; negations go to the excluded set, not hashed."""
    at = _key().find("hashFiles(")
    assert at != -1, f"the dist cache key does not call hashFiles: {_key()!r}"
    inner = _balanced(_key(), at + len("hashFiles"))
    hashed, excluded = set(), set()
    for literal in re.findall(r"'([^']*)'", inner):
        if "studio/" not in literal:
            continue
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
    """The three path groups setup.ps1's rebuild check reads, rebuilt from its foreach structure."""
    text = SETUP_PS1.read_text(encoding = "utf-8")
    block = re.search(
        # Matched on the stable "# Provision Node" prefix, not the whole sentence.
        r'\$DistDir = Join-Path \$FrontendDir "dist"(.*?)\n# Provision Node ',
        text,
        re.S,
    )
    assert block, "could not find the frontend staleness check in studio/setup.ps1"
    body = block.group(1)
    assert re.search(r"\(Get-Item \$DistDir\)\.LastWriteTime", body), (
        "the setup.ps1 block found does not read (Get-Item $DistDir).LastWriteTime, so "
        "it is not the staleness check this key is supposed to agree with"
    )

    found = set()
    sub = re.search(r"foreach \(\$subDir in @\(([^)]*)\)\)", body)
    assert sub, "setup.ps1's staleness check no longer sweeps a list of subdirectories"
    for name in re.findall(r'"([^"]+)"', sub.group(1)):
        found.add(f"studio/frontend/{name}")
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
    """Over-broad keys silently cost hit rate; bun.lock is hashed on purpose via studio/frontend/*."""
    extra = sorted(_key_globs() - _staleness_inputs_sh() - _staleness_inputs_ps1())
    assert extra == [], (
        f"the dist cache key hashes {extra}, which neither installer's rebuild check "
        f"reads. Every unrelated edit to those paths would miss the cache for no reason. "
        f"If the extra path genuinely affects the built bundle, say so where the key is "
        f"defined and widen this test deliberately."
    )


def test_no_exclusion_hides_a_path_the_rebuild_check_reads() -> None:
    """An exclusion that hides a path a rebuild check reads would serve a stale dist from the cache."""
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
    """Dropping the frontend/tests and frontend/scripts exclusions costs hit rate silently."""
    frontend = REPO / "studio" / "frontend"
    if not frontend.is_dir():
        pytest.skip("studio/frontend is absent")
    _, excluded = _key_patterns()
    read = {Path(p).name for p in _staleness_inputs_sh() if Path(p).name != "frontend"}
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


def _touch_steps() -> list[dict]:
    return [s for s in _steps(RESTORE_ACTION) if "outrank its sources" in str(s.get("name", ""))]


def test_a_restored_dist_is_made_newer_than_the_checkout_on_every_os() -> None:
    """Each OS needs its own touch: setup.ps1 reads LastWriteTime, which a bash touch may not update."""
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
    """The Windows branch writes LastWriteTime by name; touch is not proven to update that field on NTFS."""
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


# After stamping, each branch must re-read the timestamp and compare it with a branch that
# can fail.
_READBACK = {
    "posix": (r'-newer\s+"\$DIST"', r'if\s+\[\s+-n\s+"\$newer"\s+\]'),
    "Windows": (r"\$distTime\s*=\s*\(Get-Item[^)]*\)\.LastWriteTime", r"if\s*\(\$newer\)"),
}


@pytest.mark.parametrize("os_name", sorted(_READBACK))
def test_each_touch_reads_its_work_back(os_name: str) -> None:
    """A touch returning 0 proves nothing; each branch must re-check that the dist ends up newer."""
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
    """The empty-key check runs on a miss too: the first save after a frontend move writes that key."""
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
    """No Windows job may reach install-unsloth-local, which runs bash install.sh, not install.ps1."""
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
    # Each condition paired with its failure: counting `exit 1`s let a warning-plus-exit-0 through.
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
    """The reuse assertion greps for markers the installers print; renaming one would disarm it silently."""
    assert marker in script.read_text(encoding = "utf-8"), (
        f"{script.name} no longer prints {marker!r}, so the reuse assertion in "
        f"frontend-dist-save greps for a string that never appears. Update both "
        f"together."
    )


def test_the_cache_key_has_exactly_one_definition() -> None:
    """The cache key must have exactly one definition; copies in nine call sites would drift silently."""
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
    """A nested-checkout job must not call an action that nests another; uses: ./ paths fail inside it."""
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
    """A nested caller's path-prefix must match something; an empty hashFiles collapses every commit."""
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
    """Whether triggers put github.ref on refs/heads/main; workflow_dispatch is excluded as not routine."""
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
    """Every restore needs a paired save wired to its outputs; both halves fail silently when wrong."""
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
    """At least one workflow must run the save on main, or the cache never hits."""
    producers = {
        name
        for name, jid, job in _jobs()
        for s in job.get("steps") or []
        if "frontend-dist-save" in str(s.get("uses", ""))
        and str((s.get("with") or {}).get("save", "true")) != "false"
    }
    assert producers, "no job saves the frontend dist cache, so every restore can only ever miss"


def test_the_restore_comes_before_the_install_and_the_save_after_it() -> None:
    """A restore after the install, or a save before it, is inert and still looks right."""
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
        # The step that INVOKES the installer: others AST-parse install.ps1 or grep its log.
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


COLD_INSTALL_WORKFLOWS = (
    "clean-machine-install-ci.yml",
    "desktop-app-clean-machine-ci.yml",
    "interrupted-install-ci.yml",
    "release-desktop.yml",
)

COLD_INSTALL_JOBS = (("studio-windows-inference-smoke.yml", "no-vs-cpu"),)


@pytest.mark.parametrize("name", COLD_INSTALL_WORKFLOWS)
def test_cold_install_lanes_never_adopt_this_action(name: str) -> None:
    """Cold-install lanes must never adopt this action; a prebuilt frontend hides the cost they test."""
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
    """Exclusion must be per job, since one workflow can hold both a cold lane and a target job."""
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
