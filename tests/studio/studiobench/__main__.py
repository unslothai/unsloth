# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""studiobench: the only thing a tester runs.

    python -m tests.studio.studiobench --doctor
    python -m tests.studio.studiobench --tier quick --attach http://127.0.0.1:5310
    python -m tests.studio.studiobench --tier standard --branch main

EVERY heavy import is lazy. `--help` and `--doctor` are the first two things an external tester
runs and they must work on a machine with nothing installed -- a machine where `--doctor` exists
precisely to say what is missing. An ImportError at the top of this file would make the tool
unable to report its own missing dependencies, which is the one failure mode a doctor may not have.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Optional

# BUMPED BECAUSE THE INSTRUMENTS CHANGED, not because the tool gained features. This is one of the
# fields the comparability key is computed over, and it is the ONLY field that can separate a run
# taken before this change from one taken after it: the corpus is untouched, so the corpus hash,
# the tier and the rungs all match across the change.
#
# They must not match. `reasoning_toggle` now settles on a quiet DOM, so
# `highlight_spans_while_open` reads 74,250 where it used to read 44,075 on the same bundle, and
# `open_ms` now terminates on a settled mount rather than on the `data-state` flip, which makes it
# a different quantity rather than a better-measured one. A key that certified those two runs as
# comparable would be the exact failure this key exists to prevent, committed by the guard itself.
#
# Any future change to what an instrument MEASURES has to bump this for the same reason.
TOOL_VERSION = "0.2.0"
TIERS = ("fast", "quick", "standard", "full")
TIER_RUNGS = {
    # One rung, and it is 100K on purpose. The 10K rung was measured across six PRs and could
    # not separate any of them from the null control: at that size the UI work disappears
    # underneath the scene's own scripted timings (copy_markdown read 204 ms on all fourteen
    # arms). 100K is the smallest rung that carries real load -- jank index 29 against 0.6,
    # worst frame 1,855 ms against 214 -- so it is the only rung worth an iteration loop.
    "fast": ["100K"],
    "quick": ["1K", "10K"],
    "standard": ["1K", "10K", "100K"],
    "full": ["1K", "10K", "100K", "500K", "1M"],
}
# Wall clock budgets, from the design. The deficit-scheduled pacer is what makes them honest: a
# slow machine gets the same stream duration, so it misses SLOTS rather than overrunning the tier.
TIER_BUDGET_S = {"fast": 5 * 60, "quick": 5 * 60, "standard": 20 * 60, "full": 60 * 60}


def _log(msg: str = "") -> None:
    print(msg, flush = True)


def engines_installed(probe_text: str) -> list:
    """The engines the doctor's probe reported as PRESENT, from its one-line answer.

    The probe prints `chromium, webkit (not installed), firefox (unavailable)`; anything carrying a
    parenthesised note is a name without an executable behind it.
    """
    out = []
    for part in str(probe_text).split(","):
        name = part.strip()
        if name and "(" not in name:
            out.append(name)
    return out


# ── doctor ──────────────────────────────────────────────────────────


def doctor(args) -> int:
    """What is here, what is missing, and what each missing thing costs. Never raises."""
    ok = True
    _log(f"studiobench {TOOL_VERSION}")
    _log(f"  python      {sys.version.split()[0]} on {platform.system()} " f"{platform.machine()}")

    def check(
        label: str,
        fn,
        *,
        fatal: bool = True,
        cost: str = "",
    ) -> bool:
        nonlocal ok
        try:
            detail = fn()
            _log(f"  [ok]   {label}: {detail}")
            return True
        except Exception as exc:  # noqa: BLE001
            mark = "FAIL" if fatal else "warn"
            _log(f"  [{mark}] {label}: {type(exc).__name__}: {exc}")
            if cost:
                _log(f"         -> {cost}")
            if fatal:
                ok = False
            return False

    def _playwright():
        import subprocess
        import playwright  # noqa: F401

        # In a SUBPROCESS. Starting and stopping Playwright's node driver just to read the engine
        # paths leaves asyncio tearing down a cancelled task, and it prints a TargetClosedError
        # traceback to stderr after the check has already passed. A doctor that prints a
        # traceback next to `[ok]` is a doctor nobody believes.
        probe = (
            "from playwright.sync_api import sync_playwright\n"
            "with sync_playwright() as pw:\n"
            "    import pathlib\n"
            "    out = []\n"
            "    for n in ('chromium', 'webkit', 'firefox'):\n"
            "        try:\n"
            "            p = getattr(pw, n).executable_path\n"
            "            out.append(n if pathlib.Path(p).exists() else n + ' (not installed)')\n"
            "        except Exception:\n"
            "            out.append(n + ' (unavailable)')\n"
            "    print(', '.join(out))\n"
        )
        got = subprocess.run(
            [sys.executable, "-c", probe], capture_output = True, text = True, timeout = 120
        )
        if got.returncode != 0:
            raise RuntimeError((got.stderr or "the engine probe failed").strip().splitlines()[-1])
        text = got.stdout.strip()
        # THE PACKAGE IS NOT THE ENGINE. `pip install playwright` and `playwright install` are two
        # steps and the README says so, so the machine with the package and no downloaded binary is
        # the ordinary case rather than an exotic one. Say so in the line the reader sees, with the
        # command that fixes it, rather than reporting a bare `[ok]` they will act on.
        #
        # REPORTED, NOT FATAL, and deliberately so. This doctor has to keep running on a machine
        # with no engine at all, because that is exactly the machine an external tester points it
        # at first; a non-zero exit there would turn the one command that explains the problem
        # into another thing that fails without saying why.
        if not engines_installed(text):
            return (
                f"{text}; no engine is downloaded. Run `playwright install webkit` "
                "(chromium on Windows) before benchmarking"
            )
        return text

    def _psutil():
        import psutil
        return f"{psutil.__version__}; RSS sampling available"

    def _corpus():
        from .fixture.corpus import Corpus, RUNGS, plan_rung

        c = Corpus.load()
        lines = []
        for rung in RUNGS:
            p = plan_rung(c, rung)
            lines.append(f"{rung}={p.total_chars:,}c")
        return (
            f"corpus_hash {c.corpus_hash[:16]}, {len(c.manifest['units'])} units "
            f"({' '.join(lines)})"
        )

    def _registries():
        from .instruments import available, import_errors
        from .scene import action_names

        errs = import_errors()
        note = "" if not errs else f"; {len(errs)} module(s) failed to import: {list(errs)}"
        return f"{len(action_names())} actions, " f"instruments {[n for n, _ in available()]}{note}"

    def _engine():
        from .runtime.browser import default_engine
        name, _, note = default_engine()
        return f"{name} -- {note}"

    check(
        "playwright",
        _playwright,
        cost = "no browser can be driven; install with `pip install playwright` then "
        "`playwright install`",
    )
    check(
        "psutil",
        _psutil,
        fatal = False,
        cost = "RSS is reported as null with a reason instead of a number; everything else runs",
    )
    check("frozen corpus", _corpus, cost = "the corpus cannot be loaded, so no cell can be built")
    check("registries", _registries)
    check("desktop webview proxy", _engine, fatal = False)

    if args.attach:

        def _studio():
            from .runtime.bundle_guard import check_bundle
            from .runtime.lifecycle import wait_for_healthz

            if not wait_for_healthz(args.attach, 10):
                raise RuntimeError(f"{args.attach}/healthz did not answer 200")
            v = check_bundle(args.attach)
            if not v.production:
                raise RuntimeError(v.reason)
            return f"production build, react-dom bundleType {v.bundle_type}"

        check(f"studio at {args.attach}", _studio)

    def _pacer():
        from .pacer import Pacer
        p = Pacer().start()
        try:
            return f"bound to {p.base_url}"
        finally:
            p.stop()

    check("pacer", _pacer)
    _log()
    _log("doctor: PASS" if ok else "doctor: FAIL")
    return 0 if ok else 1


# ── the run ─────────────────────────────────────────────────────────


def _windowed_arms(spec: str, labels: list) -> set:
    """The arms `--windowed-arm` names, checked against the arms this run will actually have.

    A PURE ARGUMENT CHECK, WHICH IS WHY IT RUNS BEFORE ANYTHING IS STARTED. It used to run after
    both Unsloth installs had been launched, the pacer bound and the browser opened -- and the
    `SystemExit` it raises for a typo (`--windowed-arm treatments`) left every one of them running,
    because the cleanup `finally` does not begin until the cell loop much further down. No
    `bundle.close()`, no `pacer.stop()`, no `stop_studio()`, no watchdog cancellation: a mistyped
    flag cost a browser and up to two Unsloth servers, still holding their ports.
    """
    names = {name.strip() for name in (spec or "").split(",") if name.strip()}
    unknown = names - set(labels)
    if unknown:
        raise SystemExit(
            f"--windowed-arm names {sorted(unknown)}, which is not an arm in this run "
            f"({sorted(labels)})"
        )
    return names


def side_home(explicit, out, label: str, *, ab: bool) -> Path:
    """`UNSLOTH_STUDIO_HOME` for one side. THE TWO A/B SIDES NEVER SHARE ONE.

    With `--ab` and `--home` together, both iterations used to select the same directory, so the
    treatment's `install.sh` ran into the base's home -- and into the base's clone, which is
    derived from it -- while the base server was already running out of it. The two arms then
    shared or overwrote each other's binaries, which is the one thing an A/B may not do: whatever
    it reported afterwards was one build measured against itself, wearing two labels.
    """
    if not explicit:
        return Path(out) / f"studio_home_{label}"
    return Path(explicit) / label if ab else Path(explicit)


def side_specs(args, ab_ref) -> list:
    """`(label, ref, attach url, port, password)` per side. One without `--ab`, two with it.

    EACH SIDE CARRIES ITS OWN PASSWORD. Unsloth mints a bootstrap password per home, so two Unsloth instances
    the caller booted separately have two different ones; authenticating both with the single
    `--password` meant the base logged in and the treatment answered 401 every time, which made the
    advertised `--attach` + `--attach-b` A/B unusable unless both servers had been preconfigured
    with the same secret. `--password-b` defaults to `--password`, so one Unsloth, one home or two
    homes already rotated to the bench password all behave as before.
    """
    specs = [("base", args.branch, args.attach, args.port, args.password)]
    if ab_ref:
        specs.append(
            (
                "treatment",
                ab_ref,
                args.attach_b,
                args.port + 1,
                getattr(args, "password_b", "") or args.password,
            )
        )
    return specs


def planned_rungs(args) -> list:
    """The ladder this run will actually walk: `--rungs` when given, else the tier's own.

    NORMALISED AND CHECKED HERE, because a rung label that `RUNGS` does not carry is not merely a
    late crash. `--rungs "1K, 10K"` split to `[\"1K\", \" 10K\"]`, and the first thing `run` does
    with that list is record it: `run_meta` carries the rungs the run PROMISED, `plan_rung` does
    not reach `RUNGS[rung]` until `build_cells` a hundred lines later, and `recorded_ladder` folds
    every `run_meta` in the file. So a resumed run mistyped this way appended a rung nothing can
    ever satisfy to a payload that was complete, and `--report` scored it INCOMPLETE from then on
    -- the same permanent damage the ladder-ratio refusal had to learn to roll back, arriving
    through the argument parser. The install and the browser sit between the two points as well,
    so the check is worth minutes even when the payload is fresh.

    Whitespace around a comma and a lowercase suffix are the two ways to type this that read as
    correct, so both are accepted rather than rejected; anything else is named against the ladder
    it was measured against.
    """
    if not args.rungs:
        return list(TIER_RUNGS[args.tier])
    # Imported here, not at module scope, on the same rule the rest of this file follows: `--help`
    # has to answer on a machine with nothing installed.
    from .fixture.corpus import RUNGS

    rungs = [label.strip().upper() for label in args.rungs.split(",")]
    rungs = [label for label in rungs if label]
    unknown = [label for label in rungs if label not in RUNGS]
    if not rungs or unknown:
        named = ", ".join(repr(label) for label in unknown) or "nothing"
        raise SystemExit(
            f"--rungs {args.rungs!r} names {named}, which is not a rung. "
            f"The ladder is {', '.join(RUNGS)}, comma-separated."
        )
    return rungs


#: What a cell costs BESIDES its film, measured from the steps `CellRunner._run_inner` runs around
#: it: seeding the thread and reading it back, the navigation and thread wait, the 1.5 s idle
#: calibration, the chars-per-token measurement, the drain and the two censuses. Comfortably over
#: what the small rungs cost and about what 1M costs.
CELL_OVERHEAD_S = 60
#: One optional `--surfaces` sweep per arm, before the cells.
SURFACE_SWEEP_S = 120
#: The same generosity the tier budget already carries, applied to the planned work.
WATCHDOG_MARGIN = 3
#: An absolute ceiling on the MEASUREMENT half of the deadline. The point of a watchdog is that a
#: wedged run cannot outlive it, so scaling with the plan must not scale without limit: `--reps
#: 1000` would otherwise arm a deadline measured in months and the watchdog would guarantee
#: nothing at all. 24 hours is past any plan this tool supports and short of forever.
WATCHDOG_MAX_MEASUREMENT_S = 24 * 60 * 60


def planned_work_s(
    tier: str,
    rungs: list,
    reps: int,
    arms: int,
    surfaces: bool = False,
) -> float:
    """The wall clock the CELLS THIS RUN PLANNED will take, from the plan itself.

    The film is a fixed-duration one -- that is the whole design of `scene.schedule` -- so its
    length is known before it runs: `SCENES[tier].duration_ms`, plus the per-cell work around it,
    times one cell per (rung, rep, arm).
    """
    from .scene import schedule as scene_schedule

    scene = scene_schedule.SCENES.get(tier, scene_schedule.QUICK)
    cells = max(1, len(rungs)) * max(1, int(reps)) * max(1, int(arms))
    return cells * (scene.duration_ms / 1000.0 + CELL_OVERHEAD_S) + (
        SURFACE_SWEEP_S * max(1, int(arms)) if surfaces else 0
    )


def watchdog_deadline_s(
    tier: str,
    specs: list,
    *,
    rungs: Optional[list] = None,
    reps: int = 1,
    surfaces: bool = False,
) -> float:
    """The hard-exit deadline for a whole run: the measurement it PLANNED plus the setup it must
    sit through.

    THE MEASUREMENT BUDGET IS THE MEASUREMENT'S. `TIER_BUDGET_S` is the wall clock of the cells --
    the README's table says so, and says the install is not in it -- and three times that is the
    generous margin the watchdog wants around them. Arming it before `install_studio` charged a
    multi-gigabyte clone and build, which this tool itself allows 45 minutes for, against a fast
    tier's 15 minutes; an A/B does that twice, serially, before the first cell. The watchdog then
    fired during setup on a perfectly healthy run, and it fires through `os._exit`, so the `finally`
    that stops the Unsloth instances it started never ran either. Every side this run INSTALLS adds its own
    documented budget; an attached side installs nothing and adds nothing.

    AND THE TIER BUDGET IS NOT THE PLAN. It is one number per tier, and three things the caller
    controls multiply the work underneath it: `--ab` runs every cell TWICE, `--reps N` runs the
    whole ladder N times, and `--rungs` can name a ladder the tier never had. A standard A/B at
    four repetitions -- three rungs, four reps, two arms -- is 24 cells of the 243 second standard
    film, 5,832 seconds of film before a single thread is seeded, against a deadline of
    `20 min * 3 = 3,600` seconds for an attached pair that adds no install budget. The watchdog
    hard-exited a healthy run 40% of the way through it, through `os._exit`, taking the payload's
    remaining rows and the `finally` that stops the Unsloth instances with it. Reps 2 clears it as well once
    seeding is counted.

    So the measurement half is the LARGER of the tier's own budget and the planned work with the
    same 3x margin around it, which leaves every documented single-arm ladder exactly where it was
    -- a fast tier is one 57 second film, nowhere near its 900 seconds -- and grows only when the
    caller asks for more work than the tier describes. `WATCHDOG_MAX_MEASUREMENT_S` caps it, because
    a deadline that scales without limit is not a deadline.
    """
    from .runtime.lifecycle import INSTALL_TIMEOUT_S

    owned = sum(1 for spec in specs if not spec[2])
    arms = max(1, len(specs))
    planned = planned_work_s(
        tier, list(rungs) if rungs else list(TIER_RUNGS[tier]), reps, arms, surfaces
    )
    measurement = max(TIER_BUDGET_S[tier] * WATCHDOG_MARGIN, planned * WATCHDOG_MARGIN)
    return min(measurement, WATCHDOG_MAX_MEASUREMENT_S) + INSTALL_TIMEOUT_S * owned


def completion_exit_code(rows: list, resumed: int = 0) -> int:
    """0 when every cell this run asked for is complete, whether it ran them or found them.

    A RUN WHOSE WORK WAS ALREADY DONE IS A SUCCESS. `--resume` against a finished output skips
    every work item and leaves `rows` empty, and requiring at least one newly executed row then
    reported the finished output as exit 1 -- which makes an idempotent retry fail in automation
    after paying the whole install-and-launch cost. An EMPTY run with nothing resumed is still a
    failure: a payload with no cells passing every check is the same false negative in a costume.
    """
    completed = sum(1 for r in rows if r.get("completed"))
    if not rows and not resumed:
        return 1
    return 0 if completed == len(rows) else 1


def is_null_control(sides: list) -> bool:
    """Is this A/B the same build against itself?

    DETECTED, NOT DECLARED, and detected by BUILD IDENTITY rather than by URL. A self-managed null
    control -- `--branch main --ab main` -- installs the same ref twice and launches the two copies
    on different ports, so their base URLs necessarily differ; keying on the URL classified the one
    calibration run this tool exists to support as an ordinary A/B, skipped
    `noise_floor_from_null_control()`, and printed "no null control ran" underneath a table
    comparing a build with itself. Equal refs on two builds this run installed itself is a null
    control whatever ports they landed on.

    Two ATTACHED Unsloth instances are a different matter: the refs are whatever the caller typed and the
    harness cannot see what is deployed at either URL, so those are only a null control when both
    sides are the same URL.

    WHICH IS WHY THE URL IS ASKED FIRST. That rule was stated here and then not applied: the ref
    comparison ran ahead of it, so `--attach U --attach-b U --branch main --ab fix` -- one server,
    two labels the harness cannot check -- returned False on the unequal labels before the equal
    URL was ever looked at. One Unsloth measured against itself was then rendered as an ordinary
    A/B, free to publish temporal noise as an improvement, and `noise_floor_from_null_control` was
    skipped so nothing downstream had a floor to refuse it with. Two sides on one URL are one
    build whatever they were called. The owned case is untouched: `side_specs` gives the second
    side `port + 1`, so two Unsloth instances this run launched never share a URL and still fall through to
    the commit comparison below.

    AND A REF IS A POINTER, which is the rule `commit_problems` already states for `--resume` and
    this decision did not apply. The two owned sides are cloned into separate repos and fetched one
    after the other, with a whole clone, build and launch between them; `install_studio` alone
    budgets 45 minutes. A push to `main` inside that window leaves `--branch main --ab main` with
    two DIFFERENT builds under one ref name, and calling that a null control does not merely
    mislabel it: `scoring.ab.compare` voids the run and empties `regressions`, and
    `noise_floor_from_null_control` hands the delta back as THIS MACHINE'S NOISE FLOOR for the next
    A/B to be judged against. Measured on a 12% regression between two commits: the regression is
    erased and 12% is published as the floor, which would then hide every real effect under 12% on
    that machine. That is a control that could not have detected the effect it was clearing.

    So the BUILDS are compared, not their names. An empty commit on either side is not a
    difference, the same rule `commit_problems` applies: an attached side has none to declare, and
    neither does a payload written before commits were recorded.
    """
    if len(sides) < 2:
        return False
    base, treatment = sides[0], sides[1]
    if base.get("base_url") == treatment.get("base_url"):
        return True
    if base.get("ref") != treatment.get("ref"):
        return False
    if not (base.get("owns") and treatment.get("owns")):
        return False
    base_commit = str(base.get("commit") or "")
    treatment_commit = str(treatment.get("commit") or "")
    if base_commit and treatment_commit:
        return base_commit == treatment_commit
    return True


def _ab_label(sides: list, is_null: bool) -> str:
    """The title on the A/B table, naming the BUILDS whenever the ref alone would mislead.

    A null control states the commit it compared with itself, so a reader can tell which build the
    machine's noise floor was measured on. And the case this exists for: two installs of one ref
    that resolved to two different commits are NOT a null control, and printing "main -> main" over
    them would read as one. Naming both commits is the only honest title for that table.
    """
    if len(sides) < 2:
        return sides[0]["ref"] if sides else ""
    base_commit = str(sides[0].get("commit") or "")
    treatment_commit = str(sides[1].get("commit") or "")
    if is_null:
        at = f" @ {base_commit[:12]}" if base_commit else ""
        return f"null control: {sides[0]['ref']}{at} vs itself"
    if sides[0]["ref"] == sides[1]["ref"] and base_commit and treatment_commit:
        return f"{sides[0]['ref']} {base_commit[:12]} -> {treatment_commit[:12]}"
    return f"{sides[0]['ref']} -> {sides[1]['ref']}"


def arm_origins(specs: list) -> list:
    """Each side's ORIGIN, resolved exactly as the acquisition loop resolves its base URL.

    An attached side is the URL the caller typed; one this run installs is launched by
    `launch_studio` on the port `side_specs` handed it and `StudioInstall.base_url` is
    `http://127.0.0.1:{port}`. Read from the specs rather than from the sides so the answer is
    available BEFORE anything is cloned, built or launched.

    CANONICALISED, because a typed URL is not an origin. `origin_scoped` gates on
    `window.location.origin`, which lower-cases the scheme and host, drops a port the scheme
    implies and keeps no path -- so `http://studio:80` and `http://studio` are ONE origin to the
    browser and were two to a comparison on the strings. See `browser_origin`, which is what
    `origin_scoped` now gates on too, so the refusal below and the predicate it protects are
    reading the same thing.
    """
    from .runtime.ab import browser_origin
    return [
        (browser_origin(attach) if attach else f"http://127.0.0.1:{port}")
        for _label, _ref, attach, port, _password in specs
    ]


def stream_cost_injection_problem(specs: list, inject_ms) -> str | None:
    """Why `--inject-stream-cost-ms` cannot be honoured against these sides. `None` when it can.

    THE INJECTION IS GATED BY ORIGIN AND NOTHING ELSE. Both arms are driven by one browser
    context and one page, so the init scripts assembled in `run` are the context's, not an arm's:
    `add_init_script` fires on every document. `origin_scoped` is the only discriminator available
    and it discriminates on `window.location.origin`, so two arms served from ONE origin both
    match the treatment's predicate and both burn the injected cost.

    That configuration is not a mistake the caller has to be warned off in general -- one attached
    Unsloth driven twice is a null control `is_null_control` detects on purpose, and
    `test_one_attached_studio_driven_twice_is_a_null_control` pins it. It is only fatal WITH the
    injection, and it is fatal quietly: `evaluate_stream_cost_recovery_gate` reads back
    `(injected_rate - base_rate) * chars`, both rates carry the burn, the difference is zero, and
    the gate fails with "the accumulator is under-attributing" -- a verdict against a metric that
    was working, delivered by the one flag whose entire job is to tell those two apart.

    ONE ORIGIN CAN BE TWO SPELLINGS, which is why `arm_origins` canonicalises rather than
    comparing what was typed. `--attach http://studio --attach-b http://studio:80` is one server
    under two names and a browser reports `http://studio` for both, so the treatment's injection is
    gated on an origin no document has: it burns on NEITHER arm and the difference is zero for the
    other reason. Spelled the other way round the base's predicate is the dead one, the treatment's
    matches every document, and both arms burn. Either way the run reaches the same false verdict,
    and neither is visible in the two URLs the caller typed -- so the refusal names the origin both
    resolve to alongside the spellings.

    Refused rather than isolated. Isolating by arm would mean toggling the burn at every cell
    boundary from the driver, which puts the injection's own timing inside the measured window;
    the cheap and honest answer is to give the two arms two origins.
    """
    if not inject_ms or len(specs) < 2:
        return None
    origins = arm_origins(specs)
    if origins[0] != origins[1]:
        return None
    typed = [spec[2] for spec in specs[:2]]
    spelling = (
        f" ({typed[0]} and {typed[1]} are one origin under two names)"
        if all(typed) and typed[0].rstrip("/") != typed[1].rstrip("/")
        else ""
    )
    return (
        f"--inject-stream-cost-ms needs the two arms on DIFFERENT origins, and both are "
        f"{origins[0]}{spelling}. The injection is installed as a context init script gated on "
        f"window.location.origin, so one origin means both arms burn the cost, the difference "
        f"between them is zero and the recovery gate blames the metric for it. Point --attach and "
        f"--attach-b at two Unsloth instances, or drop --attach and let this run install both."
    )


def stop_owned_sides(
    installs: list,
    stop,
    *,
    keep: bool = False,
) -> None:
    """Stop every Unsloth THIS RUN launched. An attached one belongs to the caller and is left alone.

    `installs` is the `(install, owns)` list the acquisition loop builds, so a side that has not
    been reached yet is simply not in it. `keep` is `--keep-studio`, which asks for exactly this
    leak.
    """
    if keep:
        return
    for side_install, side_owns in installs:
        if side_owns and side_install is not None:
            stop(side_install)


def run(args, ab_ref = None) -> int:
    """Take the output directory FIRST, then run inside it.

    THE LOCK IS THE FIRST THING THE RUN DOES, and it used to be almost the last. It was taken where
    the `Recorder` opens `payload.jsonl`, which is after `prepare_payload` has archived what was
    already in the directory and after every clone, build and launch. So a second invocation into a
    busy `--out` without `--resume` was refused only once it had:

      * RENAMED THE LIVE PAYLOAD of the run it was about to be refused in favour of. `rename` does
        not disturb a writer -- the first run's descriptor names the inode, not the path -- so that
        run went on recording into `payload-<stamp>.jsonl` while `payload.jsonl`, the one name
        `--report`, `--assert-liveness` and the next `--resume` all open, was gone. The rule
        `prepare_payload` states for itself is that a refusal leaves the payload exactly as it
        found it, and this broke it in the one case the guard exists for.
      * CLONED, BUILT AND LAUNCHED A STUDIO ON A MACHINE THAT WAS MEASURING. Contention between two
        runs sharing one `--out` is the whole reason for the guard, so it may not be the price of
        the refusal.

    Held through setup and the cells, and handed to the `Recorder` rather than taken twice. The
    `finally` releases it on every path, including the refusals that leave by returning, so a
    second `run()` in one process -- which is how the tests drive this -- still gets the directory.
    """
    from .runtime.types import OutDirLock, Paths

    # THE ARM NAMES, BEFORE A SINGLE PROCESS EXISTS AND BEFORE A SINGLE DIRECTORY. Nothing below
    # this line can be undone without the cleanup `finally`, and that block is a long way down; see
    # `_windowed_arms`. The labels are READ OFF THE SPECS rather than spelled a second time, so the
    # check and the sides cannot drift into disagreeing about what an arm is called -- and
    # `side_specs` is a pure list build, so asking it this early starts nothing.
    #
    # AHEAD OF THE OUTPUT DIRECTORY as well as ahead of the processes. A mistyped `--windowed-arm`
    # must cost nothing at all, and a directory tree with a lock file in it is not nothing. See
    # test_studiobench_windowed_arm_names.
    specs = side_specs(args, ab_ref)
    arm_labels = [label for label, _ref, _attach, _port, _password in specs]
    windowed = _windowed_arms(getattr(args, "windowed_arm", ""), arm_labels)

    out = Path(args.out or f"studiobench-{args.tier}-{int(time.time())}").resolve()
    paths = Paths.under(out)
    out_lock = OutDirLock.take(paths.out)
    try:
        return _run_holding_out_dir(args, ab_ref, specs, arm_labels, windowed, paths, out_lock)
    finally:
        out_lock.release()


def _run_holding_out_dir(args, ab_ref, specs, arm_labels, windowed, paths, out_lock) -> int:
    from .fixture.corpus import Corpus
    from .instruments import build as build_instruments  # noqa: F401
    from .pacer import Pacer
    from .runtime import browser as browser_mod
    from .runtime.bundle_guard import check_bundle
    from .runtime.lifecycle import (
        authenticate,
        external_checkpoint_id,
        install_studio,
        launch_studio,
        pacer_provider,
        register_provider,
        seed_init_script,
        stop_studio,
        wait_for_healthz,
    )
    from .runtime.seeder import Seeder
    from .runtime.readiness import MODE_FULL, MODE_WINDOWED
    from .runtime.session import CellRunner, build_cells, ensure_probe_image, make_context
    from .runtime import resources

    _log(f"studiobench {TOOL_VERSION}  tier={args.tier}  out={paths.out}")

    corpus = Corpus.load()
    _log(f"  corpus_hash {corpus.corpus_hash}")

    # READ NOW, BEFORE ANYTHING IS STARTED OR MOVED. The probe's source is not needed until the
    # browser is launched, but reading it there means a path typo raises after Unsloth and the pacer
    # are up and before the cleanup `finally` that would stop them is entered, so a detached Unsloth
    # keeps running and holds its port. The cheapest correct fix is to fail while there is nothing
    # to clean up: a missing, unreadable or non-UTF-8 file is a mistake in the invocation, and the
    # right moment to say so is the first second of the run.
    #
    # AHEAD OF `prepare_payload` FOR THE SAME REASON THE ARCHIVE IS AHEAD OF THE RECORDER. Reusing
    # an `--out` without `--resume` archives the payload that is already there, so reading the
    # probe afterwards meant a path typo exited 2 having run no benchmark and having already taken
    # `payload.jsonl` off the path every reader looks at: `--report`, `--assert-liveness` and the
    # next `--resume` all open that name. A refusal that the first millisecond of the run could
    # have issued must not cost the previous run's payload its standard path.
    extra_init = os.environ.get("SBENCH_EXTRA_INIT_SCRIPT")
    extra_init_source = ""
    if extra_init:
        try:
            extra_init_source = Path(extra_init).read_text(encoding = "utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            _log(
                f"  FATAL: SBENCH_EXTRA_INIT_SCRIPT={extra_init} could not be read: "
                f"{type(exc).__name__}: {exc}"
            )
            return 2

    # EVERY A/B ARGUMENT CHECK, AHEAD OF `prepare_payload`, FOR THE REASON STATED DIRECTLY ABOVE.
    # These three refuse the INVOCATION -- nothing they read comes from an Unsloth, a browser or a
    # payload, only `args` and `specs`, both of which exist before this function starts anything.
    # Left below the archive they cost the previous run its payload's standard path: a fresh
    # `--ab X --home H --out DIR` over a finished DIR archived `payload.jsonl` and then exited 2
    # having run no benchmark, so the next `--resume` found nothing to continue and silently
    # re-ran the whole ladder, while `--report` and `--assert-liveness` opened the standard name
    # and found no rows. `rollback_session_rows` states the rule: a refusal has to leave the
    # payload it refused exactly as it found it. That rule covers `invalidate_stale_reports`
    # below for the same reason it covers the archive: it REPLACES `summary.md` and `ab.md`, so
    # a refusal reaching it would take the previous run's reports down with its payload.
    if ab_ref:
        if args.attach and not args.attach_b:
            _log("  --ab with --attach needs --attach-b URL: the second build has to be somewhere.")
            return 2
        # Here rather than beside the init script it guards, because by then two Unsloth instances have been
        # cloned, built and launched to run a validation that cannot say anything. See
        # `stream_cost_injection_problem`.
        injection_problem = stream_cost_injection_problem(
            specs, getattr(args, "inject_stream_cost_ms", None)
        )
        if injection_problem:
            _log(f"  {injection_problem}")
            return 2
        # ONE HOME CANNOT HOLD TWO BUILDS. `install_studio` checks the ref out into a repo
        # directory derived from the home and installs from it, so two arms sharing a home share
        # one checkout: the second install overwrites the first and BOTH arms then serve whichever
        # build was installed last. The A/B compares a build against itself and reports a clean,
        # confident "no difference" -- which is the same failure mode as a copy timing that is
        # really a sleep, and just as invisible in the payload.
        #
        # Measured, before this refused: two runs of the same pair, one at 716 ms and 718 ms for
        # base and treatment, the other at 2,583 ms and 2,614 ms. Nearly equal WITHIN each run and
        # 3.6x apart BETWEEN them, because each run was internally uniform and the two runs were
        # serving different builds. Read as a within-run comparison it says the change does
        # nothing; the difference was sitting between the runs the whole time.
        if not args.attach and args.home:
            _log(
                "  --home cannot be used with --ab: both arms would install into that one "
                "directory, the second install would overwrite the first, and both arms would "
                "then serve the same build. Drop --home and each arm gets its own "
                "studio_home_<label> under --out."
            )
            return 2

    # BEFORE the first install, the first launch and the first recorded row. See `prepare_payload`:
    # a refusal that arrives after two clones and two builds has cost the caller an hour to say
    # something it could have said in a millisecond, and an archive that arrives after the Recorder
    # has opened the file has already appended this run's header to the payload it was moving.
    archived = prepare_payload(
        paths, requested_identity(args, ab_ref, corpus.corpus_hash), resume = bool(args.resume)
    )

    # AND THE REPORTS THE ARCHIVE LEFT BEHIND. See `invalidate_stale_reports`: this is the one
    # point every reuse of an output directory passes through, whichever of the two ways it
    # invalidates what is already sitting there.
    invalidate_stale_reports(paths.out, archived = archived, extra_init = extra_init)

    # Armed AFTER the sides are known, because what it has to cover depends on them: see
    # `watchdog_deadline_s`. Nothing between the `side_specs` call above and here can hang --
    # `_windowed_arms` compares two sets, `Corpus.load` reads a generated fixture, and
    # `prepare_payload` and `invalidate_stale_reports` each touch a handful of files in `--out`.
    watchdog = browser_mod.install_wall_clock_watchdog(
        watchdog_deadline_s(
            args.tier,
            specs,
            rungs = planned_rungs(args),
            reps = args.reps,
            surfaces = bool(args.surfaces),
        ),
        "studiobench",
        _log,
    )
    if ab_ref:
        _log(f"  A/B: base={args.branch} vs treatment={ab_ref}, interleaved in ONE session")

    installs = []
    sides = []
    # EVERY SIDE THIS RUN LAUNCHED IS STOPPED WHEN THE SETUP AROUND IT FAILS. The sides are
    # acquired one after another, so the base is already SERVING while the treatment clones and
    # builds, and the cleanup that stops them both is the `finally` under the cells -- which a
    # failure up here never reaches. The Unsloth instances are the resources that outlive this process:
    # `launch_studio` detaches the server with `setsid -f`, so an abandoned one keeps its port.
    # It is not idle there. Unsloth's own launcher ABORTS rather than binding when it finds one of
    # its own servers on the requested port (`studio/backend/run.py`, `_resolve_port` with
    # `avoid_own_studio`), so the next attempt's server exits and `wait_for_healthz` takes its 200
    # from the STALE process: that run measures the build this one installed while `run_meta`
    # records the ref it was asked for.
    #
    # A RETURN IS A FAILURE HERE TOO. The health check below and the development-build gate leave
    # by returning, with both Unsloth instances up, which is the same abandoned server by a quieter route --
    # so the guard is a `finally` on the whole of setup rather than an `except` on the install.
    setup_complete = False
    try:
        for label, ref, attach, port, password in specs:
            if attach:
                side_url = attach.rstrip("/")
                side_install, owns = None, False
                _log(f"  {label}: attaching to {side_url}")
            else:
                home = side_home(args.home, paths.out, label, ab = bool(ab_ref))
                _log(f"  {label}: installing Unsloth from {ref} into {home} (this takes a while)")
                side_install = install_studio(ref, home)
                launch_studio(side_install, port, paths.out / "logs" / f"studio_{label}.log")
                side_url, owns = side_install.base_url, True
                _log(f"  {label}: Unsloth up at {side_url}")
            installs.append((side_install, owns))
            sides.append(
                {
                    "label": label,
                    "ref": ref,
                    "base_url": side_url,
                    "owns": owns,
                    "password": password,
                    "commit": getattr(side_install, "commit", None) or "",
                }
            )

        # THE BUILD, NOW THAT IT IS KNOWN. `prepare_payload` has already agreed the refs match,
        # and a ref is a pointer: see `commit_problems`. This is the first moment the commit
        # behind it exists, and it is still before the browser, the pacer and every cell, so a
        # resume onto a moved branch costs the install it has already paid and nothing more.
        if args.resume:
            resolved = resolved_commits(sides)
            commit_issues: list = []
            for recorded in recorded_identities(paths.payload_jsonl):
                for problem in commit_problems(recorded, resolved):
                    if problem not in commit_issues:
                        commit_issues.append(problem)
            if commit_issues:
                raise SystemExit(
                    f"refusing to resume {paths.payload_jsonl}: the ref matches but the build "
                    "does not.\n  "
                    + "\n  ".join(commit_issues)
                    + "\nThe cells already in this payload were measured on the commit it "
                    "records, and the rungs it still owes would be measured on the one installed "
                    "now, under one header naming one ref. Resume with the commit the payload "
                    "was recorded at, or re-run into a fresh --out."
                )

        base_url = sides[0]["base_url"]
        install, owns_studio = installs[0]

        for side in sides:
            if not wait_for_healthz(side["base_url"], 60):
                _log(f"  FATAL: {side['base_url']}/healthz did not answer 200")
                return 2

        # ── THE GATE. Before anything else is measured. Both sides, because an A/B where one side
        # is a development build is worse than no A/B: the 3.2x inflation lands entirely on one arm
        # and reads as a colossal regression or win. ────────────────────
        verdict = None
        for side in sides:
            side_verdict = check_bundle(side["base_url"])
            _log(f"  {side['label']} bundle: {side_verdict.reason}")
            side["verdict"] = side_verdict
            if verdict is None:
                verdict = side_verdict
            if not side_verdict.production and not args.allow_dev_server:
                _log(
                    "  REFUSING TO RUN. A development build inflates the very axis under "
                    "investigation"
                )
                _log(
                    "  by about 3.2x, so a measurement here would confirm any hypothesis brought "
                    "to it."
                )
                _log("  Pass --allow-dev-server only to demonstrate that this gate matters.")
                return 3

        pacer = Pacer().start()
        _log(f"  pacer at {pacer.base_url}")
        model_id = "studiobench-pacer"
        pacer.state.model_ids = [model_id]

        from .runtime.ab import origin_scoped

        init_scripts = []
        # ONE PROVIDER PER ORIGIN, because the provider lives in the BACKEND and localStorage is
        # per-origin. An ATTACHED null control is `--attach U --attach-b U`, which `is_null_control`
        # accepts on purpose: two sides, one Unsloth. Registering per SIDE there registers the pacer
        # twice against the same backend, and `lifecycle.register_provider` is idempotent by DISPLAY
        # NAME -- so the second registration DELETES the id the first side's seed script had already
        # captured, and that script keeps selecting the dead id on every navigation it wins (the
        # order init scripts run in is not defined, and `StudioAuth.rotate` re-adds them mid-run).
        # A base cell that boots with a deleted provider selected renders "No longer offered" and
        # throws `Connection not found` without ever asking for a completion.
        registered: dict = {}
        for index, side in enumerate(sides):
            side_install = installs[index][0]
            side_auth = authenticate(
                side["base_url"],
                args.username,
                side["password"] or (side_install.bootstrap_password if side_install else ""),
            )
            _log(f"  {side['label']}: authenticated as {side_auth.username}")

            # BOTH sides register the SAME pacer, so the bytes on the wire are identical by
            # construction rather than by two configurations that are meant to agree.
            side_origin = side["base_url"].rstrip("/")
            shared = registered.get(side_origin)
            if shared is None:
                side_provider = pacer_provider(pacer.base_url, [model_id])
                # Registered in the BACKEND, and the id it assigns is what the selection names. See
                # lifecycle.register_provider: a provider that exists only in localStorage renders
                # in the picker as "No longer offered" and send throws `Connection not found`
                # without ever asking for a completion.
                register_provider(side["base_url"], side_auth, side_provider)
                side_checkpoint = external_checkpoint_id(side_provider, model_id)
                registered[side_origin] = (side_provider, side_checkpoint)
            else:
                side_provider, side_checkpoint = shared
            _log(
                f"  {side['label']}: provider {side_provider.provider_type} -> "
                f"{side_provider.base_url}, checkpoint {side_checkpoint}"
            )
            side["auth"] = side_auth

            def _side_seed(
                auth_now,
                side = side,
                provider = side_provider,
                cp = side_checkpoint,
            ):
                return origin_scoped(
                    side["base_url"],
                    seed_init_script(
                        auth_now,
                        [provider],
                        extra_local_storage = {
                            # The SELECTION, without which nothing is ever generated. See
                            # lifecycle.external_checkpoint_id.
                            "unsloth_chat_last_external_checkpoint": cp,
                            "unsloth_chat_connections_enabled": "true",
                        },
                    ),
                )

            # Origin-gated even in the single-target case, so the one-build and two-build paths are
            # the same code and the gate cannot rot while unused.
            side["seed_script"] = _side_seed
            init_scripts.append(_side_seed(side_auth))

            if getattr(args, "inject_stream_cost_ms", None) and side is not sides[0]:
                # VALIDATION, not a measurement mode. Burns a known amount of main-thread time per
                # SSE chunk on the TREATMENT side only, so an A/B whose two arms are otherwise the
                # same build has a known answer. It is origin-gated like the seed above, because a
                # context init script fires on every document and burning on both sides would
                # inject the cost into the control as well and read back a recovery of zero.
                from .instruments.selfcheck import stream_cost_injection_init_script
                init_scripts.append(
                    origin_scoped(
                        side["base_url"],
                        stream_cost_injection_init_script(args.inject_stream_cost_ms),
                    )
                )
                _log(
                    f"  {side['label']}: INJECTING {args.inject_stream_cost_ms} ms of main-thread "
                    f"time per SSE chunk. This arm is not a measurement of the build."
                )

        auth = sides[0]["auth"]

        # ONE `add_init_script` CALL PER SCENE SCRIPT, which is what this always did.
        #
        # These were briefly concatenated into a single script, on the reasoning that Playwright
        # says "the order of evaluation of multiple scripts installed via
        # browserContext.addInitScript() and page.addInitScript() is not defined", and surfaces.js
        # reads what dom.js and parity.js put on `window.__sb`. The reasoning is sound and the
        # change was a REGRESSION, caught by CI: joining them means the browser parses and
        # evaluates the three as one unit, so a throw or a parse error in any one of them stops
        # the other two running as well. Separate scripts are separate failure domains, and on the
        # CI fixture that difference cost `message_menu` its More button and turned a green job
        # red. An ordering guarantee is not worth trading fault isolation for when the ordering
        # has been correct in practice for the life of the file.
        #
        # surfaces.js is loaded unconditionally, even without --surfaces. It defines selectors and
        # never runs on its own. Making the page's JS depend on a CLI flag would mean the flag
        # changes what is on the page during the FILM as well, and the film's numbers must not
        # depend on whether a later phase was asked for.
        init_scripts.append(resources.read_text("scene/dom.js"))
        init_scripts.append(resources.read_text("scene/parity.js"))
        init_scripts.append(resources.read_text("scene/surfaces.js"))

        # AN EXTERNAL PROBE OR ABLATION ARM, its own script like the three above. One environment
        # variable rather than a CLI flag per experiment, and WITH THE VARIABLE UNSET NOTHING IS
        # APPENDED: the run is byte-identical to a run of a tree that does not have this hook. That
        # property is the point. A potency probe perturbs the page it observes, so the probe run and
        # the scored run have to be different runs of one harness, and the only safe way to arrange
        # that is for the probe to be absent by default.
        #
        # BECAUSE THE ORDER IS UNDEFINED, A PROBE MUST BE SELF-CONTAINED. It cannot assume the
        # scene scripts have run, so it cannot read `window.__sb` at install time. That is a real
        # constraint and it is written down in CONTRIBUTING-perf.md rather than papered over with
        # a guarantee this file is not in a position to give.
        if extra_init:
            init_scripts.extend(_probe_init_scripts(extra_init, extra_init_source))
            _log(
                f"  EXTRA INIT SCRIPT: {extra_init} -- this run carries an external probe and "
                f"is NOT a clean measurement of the build"
            )

        procs_before = {}
        try:
            from .instruments.rss import new_roots, snapshot_children
            procs_before = snapshot_children(os.getpid())
        except Exception:  # noqa: BLE001
            new_roots = None  # type: ignore[assignment]

        bundle = browser_mod.launch(
            args.engine, headless = not args.headed, init_scripts = init_scripts, log = _log
        )
        # THE RETURN PATH for a probe installed by the hook above. Unsloth ships `connect-src
        # 'self'`, so a beacon to a collector on another port is blocked by CSP before it leaves
        # the page, and the payload schema has no row for a one-off probe. The console is what is
        # left. Lines are filtered on a caller-supplied prefix so they can be recovered from the
        # run log by exact match, and so a probe cannot drown the log in the app's own traffic.
        console_prefix = os.environ.get("SBENCH_PAGE_CONSOLE")
        if console_prefix:
            bundle.page.on(
                "console",
                lambda m: _log(f"  [page] {m.text}") if m.text.startswith(console_prefix) else None,
            )
        if extra_init:
            # A probe that throws on load is the same silence as a probe that was never installed,
            # and the console filter above cannot show it because a failing probe never gets as far
            # as printing its own prefix. Attached only when a probe was asked for, so an ordinary
            # run is unchanged. `console.error` from the isolation wrapper arrives here as a
            # console message rather than a page error, so both channels are listened to.
            bundle.page.on("pageerror", lambda err: _log(f"  [page error] {err}"))
            bundle.page.on(
                "console",
                lambda m: _log(f"  [page error] {m.text}")
                if m.type == "error" and "SBENCH_EXTRA_INIT_SCRIPT" in m.text
                else None,
            )

        procs = []
        if new_roots is not None:
            time.sleep(1.0)
            procs = new_roots(os.getpid(), procs_before)

        # AN INIT SCRIPT IS A SNAPSHOT AND THE RUN OUTLIVES IT. The scripts above carry the access
        # token this run logged in with, they re-run on EVERY navigation, and there is one
        # navigation per cell -- so after `StudioAuth` re-mints a token, the next `page.goto` would
        # write the DEAD one back over whatever the SPA had rotated to for itself, which is the
        # failure this fixes wearing a different hat. Playwright has no way to replace an init
        # script and does not define the order they run in, so the seed script decides for itself
        # whether to write: it compares its token's `exp` against the one in storage and defers to
        # the later one. See `seed_init_script`. At most one extra script per hour of run.
        for side in sides:
            side["auth"].on_rotate = lambda auth_now, side = side: bundle.context.add_init_script(
                side["seed_script"](auth_now)
            )

        # WHAT WAS IN THE FILE BEFORE THIS SESSION COULD WRITE A BYTE. `make_context` opens the
        # `Recorder` on the next line, and the only check left after that point -- the ladder ratio,
        # which is not known until the corpus has been measured -- has to be able to put the payload
        # back the way it found it. See `rollback_session_rows`.
        mark = payload_mark(paths.payload_jsonl)
        ctx, session = make_context(
            bundle,
            base_url,
            args.tier,
            args.instrument_level,
            paths,
            _log,
            procs,
            # ADOPTED, NOT TAKEN AGAIN. `run()` has held this directory since before the payload
            # was archived; the `Recorder` writes its session id into the marker it already holds.
            out_lock = out_lock,
        )
        rec = ctx.recorder
        # The ladder this run PROMISED, resolved before it is announced. Recorded rather than
        # left to be re-derived from the tier later: `--report` reads it back to decide which rungs
        # a payload owes, and a `--rungs` override the payload did not carry made that answer wrong.
        rungs = planned_rungs(args)
        meta_row = {
            "row_type": "run_meta",
            "tier": args.tier,
            "tool_version": TOOL_VERSION,
            "corpus_hash": corpus.corpus_hash,
            "studio_ref": args.branch if owns_studio else f"attached:{base_url}",
            # WHICH COMMIT THAT REF NAMED, so a later `--resume` can tell a continuation from
            # a branch that moved. Empty for an attached Unsloth, whose build is not visible
            # from here. See `commit_problems`.
            "studio_commit": sides[0].get("commit") or "",
            "bundle": verdict.as_dict(),
            "platform": {
                "system": platform.system(),
                "machine": platform.machine(),
                # WHICH PHYSICAL MACHINE, which `system` and `machine` do not answer between
                # them. `platform.machine()` is the ARCHITECTURE -- "the machine type, e.g.
                # 'AMD64'" -- so two ordinary Linux x86_64 boxes report `Linux` and `x86_64`
                # alike, and every number this tool produces is machine-local: `report/render.py`
                # says so in its own words, "machine-local; does not travel between machines".
                # `platform.node()` is the computer's network name, which is the only thing
                # recorded here that differs between two such hosts. Empty when it cannot be
                # determined, exactly as the other two are.
                "node": platform.node(),
                "python": sys.version.split()[0],
                "engine": bundle.engine,
                "engine_note": bundle.engine_note,
            },
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "pacer_base_url": pacer.base_url,
            "cadence": args.cadence,
            "rungs": rungs,
            "tier_rungs": TIER_RUNGS[args.tier],
            "reps": args.reps,
            "instrument_level": args.instrument_level,
            # In the payload, not only in the log. Two runs with different fixtures are
            # not comparable, and a fixture difference that is not recorded is one a later
            # reader has no way to notice before quoting a ratio across it.
            "stream_tail_chars": args.stream_tail_chars,
            "corpus_dollars": bool(args.corpus_dollars),
            # WHICH PROBE, IF ANY, WAS IN THE PAGE. Recorded next to the corpus hash and for
            # the same reason: a payload nobody can audit against the page it measured has to
            # be taken on trust. `null` is the normal case and the only scorable one.
            "probe_init_script": extra_init or None,
            # WHETHER THE PROBE RAN BEFORE THE FILM, for the same reason and with the same
            # consequence: it changes what the cell measures without moving the cell id, so
            # `--resume` needs it on the record to be able to refuse a toggle. See
            # `IDENTITY_AXES`.
            "click_probe": bool(getattr(args, "click_probe", False)),
            # AN ARM RUNNING THIS IS NOT A MEASUREMENT OF THE BUILD, and the payload has to
            # say so itself. A reader who finds a treatment arm 40% slower has no other way
            # to discover that the harness put the 40% there on purpose, and `--resume` reads
            # it back as an identity axis so a calibration cannot be continued as an ordinary
            # run or the other way round. See `IDENTITY_AXES`.
            "inject_stream_cost_ms": getattr(args, "inject_stream_cost_ms", None),
            # WHICH BROWSER ACTUALLY RENDERED THIS, which the engine name does not settle.
            #
            # `--headed` flips `headless` at launch, and since Playwright 1.57 the two modes
            # default to DIFFERENT BINARIES -- headed resolves to `chrome`, headless to
            # `chrome-headless-shell`. This workflow pins `playwright>=1.45,<2`, so that split is
            # in range. It is not a mode flag with a mode-shaped effect: one report measures
            # headless-shell at two to three times the processing time of chrome-stable, headless
            # falls back to software rendering for GPU-accelerated work, and the compositor still
            # ticks at 60 Hz in headless unless BeginFrameControl is driven by hand.
            #
            # This tool measures frames, jank and time-to-settle. A payload recorded headed and one
            # recorded headless are two different renderers, and the payload has to say which it
            # was before anything compares them.
            "headed": bool(getattr(args, "headed", False)),
        }
        rec.emit(meta_row)

        from .scoring import payload_rules

        # THE COMPARABILITY KEY, emitted as its own row so a prose comparison can be checked later.
        #
        # `floor_table.load` already refuses to POOL payloads across tiers and corpora, and that
        # guard works. What it cannot reach is a comparison made in PROSE between two separately
        # published runs, which is how a wrong number nearly propagated here twice.
        #
        # Computed over the run_meta row that was actually emitted, not over a second dict built
        # to look like it. A hand-copied duplicate is one more value whose provenance is not
        # stable, which is the whole failure this module exists to stop: the key has to describe
        # the payload, not a parallel description of it that can drift.
        #
        # Keyed on the COMPUTED corpus hash rather than on the harness commit, because a commit is
        # a claim about provenance and the hash is the thing itself: one tree in this campaign ran
        # a corpus that matched neither the branch's pre-fix nor its post-fix hash, silently and
        # self-consistently, and only printing the hash revealed it.
        rec.emit(
            {
                "row_type": "comparability",
                "key": payload_rules.comparability_key(meta_row),
                "fields": payload_rules.comparability_fields(meta_row),
                "note": (
                    "quote this key beside any number taken from this payload. Two numbers with "
                    "different keys are not comparable, and `--compare` will name the field that "
                    "differs."
                ),
            }
        )
        _log(f"  comparability {payload_rules.comparability_key(meta_row)}")

        rec.gate("production_build", verdict.production, verdict.as_dict())

        if args.tier == "fast":
            # Said in the log AND recorded in the payload. A fast-tier reading is a DIRECTION, not
            # a number: it runs one rung, a 47 s film and however few repetitions the caller asked
            # for, so its detection floor is wider than the standard tier's and it has no null
            # control of its own unless one is run alongside. The gate exists so the analysis layer
            # can refuse to pool a fast payload with a standard one -- a fast reading quoted against
            # a standard floor is the single most likely way this tier gets somebody a wrong answer.
            _log("")
            _log("  FAST TIER: for iteration while you are changing something, not for reporting.")
            _log(
                "  One rung (100K), a 47s film. Use it to see whether a fix moved anything at all,"
            )
            _log("  then confirm with --tier standard and a null control before quoting a number.")
            _log("")
        rec.gate(
            "reportable_tier",
            args.tier != "fast",
            {
                "tier": args.tier,
                "scene": TIER_RUNGS[args.tier],
                "reason": (
                    "the fast tier is an iteration loop: one rung, a compressed film, and a "
                    "wider floor than the standard tier"
                )
                if args.tier == "fast"
                else "standard measurement protocol",
            },
        )
        # THE SAME SHAPE OF GATE, for the same reason. A probe forces layout on a schedule of its
        # own, so its payload is perturbed, and "the docs say probe payloads are never scored" is a
        # convention rather than a gate. A convention is what this subsystem keeps being wrong
        # about: the run still records ordinary cells and still renders an A/B table, so a probe
        # invocation on the standard tier produces something that reads exactly like a result. The
        # gate is what lets `floor_table` refuse it instead of trusting the caller to remember.
        rec.gate(
            "probe_free",
            not extra_init,
            {
                "probe_init_script": extra_init or None,
                "reason": (
                    "an external init script was installed via SBENCH_EXTRA_INIT_SCRIPT, so this "
                    "page was instrumented while it was measured"
                )
                if extra_init
                else "no external init script was installed",
            },
        )
        if extra_init:
            _log("")
            _log("  PROBE RUN: this payload is NOT scorable. floor_table will refuse it.")
            _log("")

        image_path = ensure_probe_image(paths)
        # WHICH SIDE, IF ANY, IS ALLOWED THE WINDOWED READINESS GATE.
        #
        # Per side and never global, so an A/B of the shipped build against a virtualised one runs
        # the base arm on the strict full-mount gate it has always had. A flag that relaxed both
        # sides would let a base arm that failed to finish mounting sail through and be compared
        # against a treatment that did the same, which is the exact "flattering garbage" the gate
        # exists to refuse -- on both arms at once, so the ratio would still look fine.
        #
        # `windowed` was parsed and validated at the top of this function, before the installs, the
        # pacer and the browser existed. Only the application of it is left here.
        for side in sides:
            side["readiness_mode"] = MODE_WINDOWED if side["label"] in windowed else MODE_FULL
            if side["readiness_mode"] == MODE_WINDOWED:
                _log(
                    f"  {side['label']}: WINDOWED readiness gate. This arm is permitted to mount "
                    "fewer messages than the thread contains; it must publish aria-setsize matching "
                    "the seeded count, mount the end of the thread, settle, and pass the "
                    "scroll-to-top completeness probe. See runtime/readiness.py."
                )
                rec.gate(
                    f"windowed_readiness:{side['label']}",
                    True,
                    {
                        "arm": side["label"],
                        "reason": "declared on the command line with --windowed-arm",
                        "note": "structural UI parity is NOT APPLICABLE to this arm; run "
                        "sweep/ui_parity.py --mode behaviour",
                    },
                )
        for side in sides:
            side_seeder = Seeder(
                base_url = side["base_url"], auth = side["auth"], model_id = model_id, log = _log
            )
            side["seeder"] = side_seeder
            side["runner"] = CellRunner(
                session = session,
                pacer = pacer,
                seeder = side_seeder,
                corpus = corpus,
                click_probe = bool(getattr(args, "click_probe", False)),
                readiness_mode = side["readiness_mode"],
                base_url = side["base_url"],
                model_id = model_id,
                tier = args.tier,
                paths = paths,
                log = _log,
                cadence = args.cadence,
                image_path = image_path,
                parity_raw = args.parity_raw,
                parity_shots = args.parity_shots,
                arm_label = side["label"],
            )

        seeder = sides[0]["seeder"]
        runner = sides[0]["runner"]

        if args.surfaces:
            _sweep_surfaces(sides, ctx, paths)

        # The ladder is sized by the ratio MEASURED on this corpus, not by the provisional 4.0.
        # Measured here, before any cell is built, because a rung named in tokens is planned in
        # characters and the ratio is what makes those the same claim.
        cells = build_cells(
            rungs,
            corpus,
            args.tier,
            ctx.session_id,
            args.instrument_level,
            reps = args.reps,
            base_url = sides[0]["base_url"],
            auth = seeder.auth,
            model_id = model_id,
            log = _log,
            stream_tail_chars = args.stream_tail_chars,
            corpus_dollars = args.corpus_dollars,
        )
        if args.stream_tail_chars or args.corpus_dollars:
            # Loud, because both change the fixture. A payload produced under either of them is
            # not comparable with one produced without, and the pair that says so is printed here
            # and written into the run manifest above.
            _log(
                f"  FIXTURE CHANGED: stream tail {args.stream_tail_chars or 'default'}, "
                f"dollars {'on' if args.corpus_dollars else 'off'}. Compare only against a run "
                f"with the same pair."
            )
        if cells:
            ladder_ratio = cells[0][0].meta["ladder_chars_per_token"]
            rec.gate("ladder_ratio_measured", not ladder_ratio["provisional"], ladder_ratio)

            # THE RATIO IS PART OF WHAT THE PAYLOAD MEASURED, and this is the first moment it is
            # known -- the same shape as the commit check above, and for the same reason: the
            # rungs already in the file were built at the ratio they record, the rungs still owed
            # would be built at the one measured now, and one report prints both as one ladder.
            if args.resume:
                ratio_issues: list = []
                for recorded in recorded_identities(paths.payload_jsonl):
                    for problem in ladder_ratio_problems(recorded, ladder_ratio["chars_per_token"]):
                        if problem not in ratio_issues:
                            ratio_issues.append(problem)
                if ratio_issues:
                    # AND NOTHING THIS SESSION WROTE SURVIVES THE REFUSAL. Unlike the commit check
                    # above, this one runs after the `Recorder` has opened the payload, so refusing
                    # is not enough on its own: `run_meta` is already in the file and
                    # `recorded_ladder` folds every one of them, so a refused
                    # `--resume --rungs 1K,10K` would leave 10K promised and unrecorded in a
                    # payload that was complete. See `rollback_session_rows`.
                    rec.close()
                    rollback_session_rows(paths.payload_jsonl, mark)
                    raise SystemExit(
                        f"refusing to resume {paths.payload_jsonl}: the ladder is sized by a "
                        "different chars-per-token ratio than the cells already in it.\n  "
                        + "\n  ".join(ratio_issues)
                        + "\nA rung is named in tokens and built in characters, so the same rung "
                        "label would stand over two different character loads in one table. "
                        "Resume where the tokeniser answers as it did, or re-run into a fresh "
                        "--out."
                    )

        done = _resume_set(paths) if args.resume else set()
        if done:
            _log(f"  resuming: {len(done)} cells already in {paths.payload_jsonl.name}")

        if ab_ref:
            from .runtime.ab import Target, interleave, order_is_balanced

            targets = [
                Target(
                    label = s["label"],
                    ref = s["ref"],
                    base_url = s["base_url"],
                    seeder = s["seeder"],
                    runner = s["runner"],
                )
                for s in sides
            ]
            work = interleave(cells, targets)
            if not order_is_balanced(work):
                # Said out loud rather than silently absorbed: with an odd number of reps one side
                # always runs first, so anything drifting monotonically through the session lands on
                # the other one instead of cancelling.
                _log(
                    "  WARNING: the run order is not balanced (use an even --reps). Linear drift "
                    "within the session is charged to whichever side runs second."
                )
            rec.emit(
                {
                    "row_type": "ab_plan",
                    "base_ref": sides[0]["ref"],
                    "treatment_ref": sides[1]["ref"],
                    # WHICH SERVER THE TREATMENT WAS, when the caller attached one. `run_meta`
                    # already records the base that way in `studio_ref`; without the same for the
                    # treatment a resume can only compare the `--ab` label, which says nothing about
                    # the build that answered on the port. See `requested_identity`. Empty when this
                    # run installed the treatment itself: then the ref above is the identity.
                    "treatment_url": "" if sides[1]["owns"] else sides[1]["base_url"],
                    # The treatment's half of `studio_commit`. Same reason, same emptiness rule.
                    "treatment_commit": sides[1].get("commit") or "",
                    "balanced": order_is_balanced(work),
                    "order": [c.cell_id for _t, c, _p in work],
                }
            )
        else:
            work = [(None, cell, plan) for cell, plan in cells]

        if done:
            # AT PAIR GRANULARITY. An A/B pair whose two arms are not both recorded is re-run
            # whole, so the resumed session never measures one arm on its own -- see
            # `ab.skippable_cells`. For a run without --ab every pair holds one cell and this is the
            # set it already was.
            from .runtime.ab import skippable_cells
            done = skippable_cells(work, done)
            _log(f"  resuming: {len(done)} cells already in {paths.payload_jsonl.name}")
        setup_complete = True
    finally:
        if not setup_complete:
            stop_owned_sides(installs, stop_studio, keep = args.keep_studio)

    rows = []
    resumed = 0
    try:
        for target, cell, plan in work:
            if cell.cell_id in done:
                _log(f"  skipping {cell.cell_id} (already recorded)")
                resumed += 1
                continue
            active = target.runner if target is not None else runner
            if target is not None:
                _log(f"\n### arm {target.label} ({target.ref}) at {target.base_url}")
            rows.append(active.run(cell, plan))
    finally:
        for inst in session.instruments:
            session._safe(inst, "detach")
        try:
            watchdog.cancel()
        except Exception:  # noqa: BLE001
            pass
        bundle.close()
        pacer.stop()
        stop_owned_sides(installs, stop_studio, keep = args.keep_studio)
        rec.close()

    if ab_ref:
        # The cells THIS session was asked to measure. A resumed A/B skips whole pairs or none
        # (`ab.skippable_cells`), so this is either every planned cell or none of them, and it is
        # what `_render_ab` checks the payload against before it is allowed to print a verdict.
        _render_ab(
            paths,
            sides,
            ctx.session_id,
            corpus.corpus_hash,
            planned = [c.cell_id for _t, c, _p in work if c.cell_id not in done],
        )

    _summarise(rows, paths)
    completed = sum(1 for r in rows if r.get("completed"))
    _log(
        f"\n{completed} of {len(rows)} cells completed"
        + (f", {resumed} already complete in the payload" if resumed else "")
        + f". payload: {paths.payload_jsonl}"
    )
    return completion_exit_code(rows, resumed)


def _sweep_surfaces(sides: list, ctx, paths) -> None:
    """The optional surface phase: one sweep per arm, BEFORE the cells.

    Before, not after, and on an empty chat rather than a seeded one. Several surface roots
    contain the keep-alive chat page, so a sweep taken after the film would carry that film's
    thread -- and the two messages its last actions deleted -- into the digest of every route and
    every menu. Running first makes the surface digests about the surfaces.

    A failure here never costs the run. The sweep is additional evidence about the UI; the cells
    are the measurement, and a broken selector in the registry must not stop them.
    """
    from .scene.surface_sweep import render_manifest, sweep
    for side in sides:
        label = side["label"]
        _log(f"\n### surface sweep: {label} at {side['base_url']}")
        try:
            rows, manifest = sweep(
                ctx.page,
                side["base_url"],
                log = _log,
                cell_id = f"surfaces.{label}",
                recorder = ctx.recorder,
            )
        except Exception as exc:  # noqa: BLE001
            # Recorded as a failed gate rather than swallowed. A sweep that raised and a sweep
            # that found nothing look identical in a payload that only carries the rows.
            _log(f"  the surface sweep failed: {type(exc).__name__}: {exc}")
            ctx.recorder.gate(
                f"surface_sweep:{label}", False, {"error": f"{type(exc).__name__}: {exc}"}
            )
            continue
        for row in rows:
            row["arm"] = label
        text = render_manifest(manifest)
        print("\n" + text)
        out = paths.out / f"surfaces_{label}.md"
        out.write_text(text, encoding = "utf-8")
        _log(f"surface coverage manifest written to {out}")
        # PASSES only when every non-conditional surface was reached AND the digests were scoped.
        # An unscoped sweep reports one page-wide digest per surface, which agrees everywhere for
        # reasons that have nothing to do with the surfaces.
        passed = manifest["not_reached_hard"] == 0 and manifest["digests_scoped"]
        ctx.recorder.gate(f"surface_sweep:{label}", passed, manifest)


def _probe_init_scripts(path: str, source: str) -> list[str]:
    """The external script AS SOURCE, plus a separate script that says when it did not install.

    NO `eval`, AND THAT IS THE WHOLE POINT. This used to hand the file to indirect eval as a
    string, so that a malformed probe degraded to a caught `SyntaxError` instead of taking the
    scene scripts with it. Unsloth serves `script-src 'self'` with no `'unsafe-eval'`
    (`studio/backend/main.py::_build_csp`), and `runtime/browser.py::default_engine` picks WEBKIT
    on both Linux and macOS, so on the DEFAULT engine that eval was refused by CSP and the probe
    never installed at all. Measured against a page served with Unsloth's own header:

        chromium   indirect eval runs;      a bad init script leaves the other init scripts alone
        firefox    indirect eval runs;      a bad init script leaves the other init scripts alone
        webkit     indirect eval REFUSED;   a bad init script kills every other init script

    So the isolation the wrapper was bought for does not exist on webkit either way -- Playwright
    installs webkit's init scripts as one bootstrap unit -- and on the two engines where it does
    exist, separate `add_init_script` calls already provide it without evaluating a string. The
    source is therefore installed as its own script, opening it, exactly as it reads on disk.

    THE STAMP GOES AFTER THE SOURCE, NOT BEFORE IT. A directive prologue is the run of expression
    statements a Script or FunctionBody OPENS with (ECMA-262, "Directive Prologues and the Use
    Strict Directive"), so a statement in front of a probe's leading `"use strict"` demotes it to a
    string expression that does nothing. The probe then runs sloppy: an undeclared assignment
    silently creates a global instead of throwing, and the harness is no longer executing the file
    as it reads on disk. Playwright wraps every init script in `(() => { ... })();`
    (`playwright-core/src/server/page.ts`, `class InitScript`), which keeps the file's own prologue
    working as a FunctionBody prologue -- until something is prepended to it.

    THE EMPTY STATEMENT BETWEEN THEM IS THE ATTESTATION. This used to append the stamp on a bare
    newline, on the reasoning that ASI made that safe because the appended line opens with an
    identifier. That reasoning only covers a source whose last line is COMPLETE. ASI does not
    terminate a source that ends mid-expression, so a probe truncated after an assignment operator
    -- `var result =` -- is a SyntaxError on its own and STOPS BEING ONE once the stamp is
    concatenated onto it: the stamp becomes that variable's initializer, `window.__sbExtraInitScript`
    is set by the very assignment that was supposed to prove the probe finished, and the deferred
    check below returns early. The probe reported nothing and the harness called that a clean run,
    which is the one conclusion this pair of scripts exists to prevent. Measured on webkit and
    chromium alike: stamp set, no `pageerror`, console silent, byte-identical to a healthy probe.

    A bare `;` closes any pending operand slot, so that class stays a parse error. It goes AFTER the
    source and never in front of it, because a leading empty statement is not a StringLiteral
    ExpressionStatement and would end the directive prologue before the probe's own `"use strict"`
    was reached.

    WHAT THIS STILL DOES NOT CLOSE, stated so nobody reads the `;` as more than it is: a source
    truncated after a STATEMENT HEAD rather than inside an expression -- `if (x)`, `while (a)`,
    `for (;;)` -- takes the empty statement as its body and parses either way, so the stamp still
    lands. Closing that needs the probe to be PARSED, which this harness deliberately does not do
    (that is the whole point of the no-`eval` note above). The `;` narrows the hole; it does not
    seal it, and the attestation is a smoke alarm rather than a proof.

    The second script is the report. The last line of the probe script stamps
    `window.__sbExtraInitScript`, so a probe that failed to PARSE, or that THREW on the way down,
    leaves it unset and the deferred check names it on the console. A throw also arrives as a
    `pageerror`, which `bundle.page.on("pageerror", ...)` already logs, and that is what separates
    the two cases. On webkit the check dies in the same bootstrap unit as the probe, and the
    `pageerror` is what reports there.
    """
    where = json.dumps(path)
    return [
        f"{source}\n;\nwindow.__sbExtraInitScript = {where};\n",
        (
            "(function () {\n"
            "  setTimeout(function () {\n"
            "    if (window.__sbExtraInitScript) { return; }\n"
            "    try {\n"
            "      window.console.error(\n"
            f"        'SBENCH_EXTRA_INIT_SCRIPT ' + {where} + ' never installed: it did not "
            "parse, or it threw before it finished. '" + " +\n"
            "        'This probe reported nothing, which is NOT the same as an arm that did not '"
            " +\n"
            "        'fire.'\n"
            "      );\n"
            "    } catch (ignored) {}\n"
            "  }, 0);\n"
            "})();\n"
        ),
    ]


def _render_ab(
    paths,
    sides,
    session_id: str,
    corpus_hash: str,
    planned = (),
) -> None:
    """Render the A/B table from the payload the run just wrote.

    Read back from disk rather than kept in memory on purpose: it is the same path a tester takes
    with `--report`, so the table nobody checks and the table everybody reads are produced by one
    piece of code.

    NO VERDICT OVER A PLAN WITH A HOLE IN IT. `planned` is the cells this session was asked to
    measure, and a failed cell does not stop the run -- `CellRunner.run` records the failure and
    returns -- so without this check the comparison is rendered over whatever pairs survived. See
    `ab.unmeasured_planned_cells`: the loss is not the failed cell but its healthy partner, which
    the arm intersection removes silently.
    """
    from .report.render import render_ab_table
    from .runtime.ab import compare_arms, unmeasured_planned_cells

    records = []
    with paths.payload_jsonl.open(encoding = "utf-8") as fh:
        for line in fh:
            try:
                records.append(json.loads(line))
            except ValueError:
                continue

    out = paths.out / "ab.md"

    # NO TABLE AT ALL for a probe run, rather than a table with a warning printed above it. The
    # warning scrolls off; `ab.md` sits in the output directory and gets pasted into a pull
    # request. This is the same refusal `--report` and `floor_table` make, on the same evidence,
    # and it is the entry point that actually runs at the end of every session.
    from .scoring.from_payload import probe_scripts

    probes = probe_scripts(records)
    if probes:
        reason = (
            f"NO A/B TABLE: this run carried an external init script ({', '.join(probes)}), so "
            f"its timings measure the page and the instrument together. The payload is kept "
            f"for the probe's own output and for --assert-liveness; it is not scorable."
        )
        _log("")
        _log(reason)
        # OVERWRITTEN, not left alone and not deleted. `--resume` reuses the output directory, so
        # an `ab.md` from an earlier clean run of the same directory would survive this refusal
        # and sit at the standard artifact path, where it reads as this run's result. Deleting it
        # would leave whoever opens the path with nothing to explain the absence, so the file is
        # replaced by the refusal itself.
        stale = paths.out / "ab.md"
        if stale.exists():
            stale.write_text(f"# No A/B table\n\n{reason}\n", encoding = "utf-8")
            _log(f"  a previous {stale} was replaced by this refusal")
        _log("")
        return

    # A FULLY RESUMED A/B MEASURED NOTHING OF ITS OWN. `--resume` against a finished output skips
    # every cell and is a success, but the ratio is scoped to THIS session, so re-rendering here
    # replaced a real table with NO READING and exited 0 while doing it. The run that measured
    # keeps its report; a run with no prior table still gets one, since there is nothing to lose.
    #
    # AFTER THE PROBE REFUSAL, NOT BEFORE IT. The payload the resume keeps is the payload this
    # check is deciding on behalf of, and a probed one is not scorable no matter which session
    # recorded it. Run first, this returned early for the one sequence that needs the refusal
    # most: a fresh PROBE run into a directory that already holds a clean `ab.md`, killed after
    # its last cell was fsynced and before it rendered -- `archive_payload` moves `payload.jsonl`
    # and nothing else, so the clean table stays at the standard artifact path -- and then
    # resumed, which finds every cell complete, records none of its own and left that table
    # standing over an unscorable probe payload.
    if not any(r.get("row_type") == "cell" and r.get("session_id") == session_id for r in records):
        if out.exists():
            _log(f"\nno cell ran in this session; keeping the A/B table already at {out}")
            return

    # Detected, not declared. `--ab main` IS a null control whether or not the caller says so, and
    # a null control that renders as an ordinary A/B invites somebody to quote "7.7% faster" from a
    # build compared with itself. See `is_null_control`.
    is_null = is_null_control(sides)
    label = _ab_label(sides, is_null)

    try:
        result = compare_arms(
            records,
            sides[0]["label"],
            sides[1]["label"],
            bench_version = TOOL_VERSION,
            corpus_hash = corpus_hash,
            session_id = session_id,
            label = label,
            is_null_control = is_null,
        )
    except Exception as exc:  # noqa: BLE001
        _log(f"\nA/B table could not be built: {type(exc).__name__}: {exc}")
        return

    missing = unmeasured_planned_cells(records, planned, session_id = session_id)
    if missing:
        result.void = True
        result.void_reason = (
            f"{len(missing)} of {len(planned)} planned cells left no usable reading, so what "
            "remains is a selection rather than the plan: " + ", ".join(missing)
        )

    text = render_ab_table(result)
    print("\n" + text)
    out.write_text(text, encoding = "utf-8")
    _log(f"A/B table written to {out}")
    if missing:
        # A noise floor derived from a partial null control is a number about the cells that did
        # not die, quoted afterwards at every real A/B on this machine.
        _log("the plan did not complete, so no noise floor is derived from it")
        return
    if is_null:
        from .scoring.ab import noise_floor_from_null_control
        try:
            floor, source = noise_floor_from_null_control(result)
            _log(
                f"THIS MACHINE'S NOISE FLOOR: {floor:.1f}% ({source}). Pass it to a real A/B; "
                f"a difference smaller than this is not a difference."
            )
        except Exception as exc:  # noqa: BLE001
            _log(f"could not derive a noise floor from the null control: {exc}")
    else:
        _log(
            "NOTE: no null control (base vs base) was run, so the noise floor here is the "
            "declared default and not this machine's. A win inside that floor is not a win."
        )


#: THE PAYLOAD IDENTITY. The axes that decide whether a cell already in a payload measures the
#: same thing a new invocation is asking for.
#:
#: `cell_id` is `r{rung}.{arm}.rep{rep}` and nothing else, so every one of these can change under a
#: cell id that stays identical: the tier picks the FILM (the standard film runs 243 s, the quick
#: one 77.5 s and the fast one 47 s, with different budgets), the cadence picks the rate the reply
#: streams at, the instrument level decides how much of the number is the instrument, the corpus
#: hash is the fixture, the engine is what rendered every frame, and the two refs are the builds
#: under test.
#:
#: `rungs` and `reps` are deliberately NOT here. Resuming with more repetitions or another rung is
#: a legitimate continuation -- it ADDS cells rather than reinterpreting the ones already recorded.
IDENTITY_AXES = (
    "tier",
    "cadence",
    "engine",
    # THE INSTRUMENTS THEMSELVES. `TOOL_VERSION` is bumped when what an instrument MEASURES
    # changes and for no other reason -- this commit bumped it to 0.2.0 because
    # `reasoning_toggle.open_ms` now terminates on a settled mount rather than on the `data-state`
    # flip, which makes it a different quantity under the same name. None of that moves a cell id,
    # so `--resume` into a half-finished 0.1.0 payload from an upgraded tree skipped the completed
    # cells and appended the rest measured by 0.2.0: one ladder built from two instruments, which
    # is the same failure the tier and the corpus are on this list for. `merged_run_meta` names the
    # mixture afterwards, but only `--compare` and a floor-gated `floor_table.render` call it --
    # plain `--report` takes the FIRST header and pools the two quantities without a word. Refused
    # here instead, before anything is installed and before anything is measured.
    "tool_version",
    "instrument_level",
    "corpus_hash",
    "studio_ref",
    "treatment_ref",
    "treatment_url",
    # Added by this branch with `--stream-tail-chars` / `--corpus-dollars`: both change the reply
    # the cell streams without moving its id, so they belong to the identity for the same reason
    # the tier does. Recorded already, so no schema change and a payload from before them reads
    # back as the defaults (see `HISTORICAL_DEFAULTS`).
    "stream_tail_chars",
    "corpus_dollars",
    # `--click-probe`, whose own help text says it "makes the cell's timings incomparable with a
    # cell that did not run it". That is the definition of an identity axis: the probe runs a full
    # `page.click`, a real mouse click, a dispatch, a focus and a hover over the thread before the
    # film starts, and the cell then records a `composer_click_ms` measured on a composer those
    # paths have already been through plus a `click_attribution` block a cell without the flag
    # does not have at all. None of it moves the cell id, so a resume that toggles the flag skips
    # the completed cells and appends the rest under the same ids -- the one ladder built from two
    # films this check exists to refuse.
    "click_probe",
    # `--inject-stream-cost-ms`, on the same rule and for a larger perturbation than either of
    # those two. It burns a known amount of main-thread time per SSE chunk on the TREATMENT arm,
    # so a treatment cell recorded with it is a different reading from one recorded without it,
    # under a cell id that cannot tell -- the id carries the rung, the arm and the repetition and
    # stops there. Off it, `--resume` had two ways to lie about a calibration: against a FINISHED
    # uninjected payload every pair is skippable, the run exits 0 having measured nothing, and the
    # recovery gate is answered from cells that were never injected; against a HALF-FINISHED
    # injected one the resume drops the flag and the ladder ends up part injected and part not.
    # Both land as a recovery fraction near zero, which reads as "the metric is blind" and is the
    # single verdict this flag exists to make trustworthy.
    "inject_stream_cost_ms",
    # `SBENCH_EXTRA_INIT_SCRIPT`, the external probe. Unlike the axes above it cannot end in a
    # wrong NUMBER -- `refuse_if_probed` reads every `run_meta` in the file and every scoring entry
    # point calls it, so one probed session makes the whole payload unscorable and there is no flag
    # to override that. It is here for what the refusal costs instead. The variable is an
    # ENVIRONMENT variable rather than a flag, so it survives in a shell after the experiment that
    # set it is over, and `--resume` into a half-finished clean payload then installs both sides,
    # runs the rungs that are still owed with the probe in the page, and appends a probed
    # `run_meta` to the file. The payload is append-only and the refusal is whole-file, so the
    # cells that were recorded cleanly before it are unscorable from then on and nothing this tool
    # offers takes it back. Refusing here costs a millisecond and happens before the first install.
    "probe_init_script",
    # `--headed`. On the same rule as `engine` directly above the others, and for the same reason
    # the engine is there at all: it decides which browser binary renders the film. Since
    # Playwright 1.57 headed and headless resolve to different executables (`chrome` against
    # `chrome-headless-shell`), headless falls back to software rendering for GPU-accelerated
    # work, and the compositor's own pacing differs. A resume that toggles it would skip the
    # completed cells and append the rest under the same ids, building one ladder from two
    # renderers -- exactly what this check exists to refuse.
    "headed",
)

#: The axes that describe the SECOND side, which only exist when a run has one. See
#: `identity_problems`: an A/B judged against a run that is not one may not differ on them.
TREATMENT_AXES = ("treatment_ref", "treatment_url")

#: The axes whose ABSENCE from a payload is itself a reading, and what it reads as.
#:
#: `identity_problems` otherwise skips an axis the payload never declared, because an axis a row
#: never declared cannot be a difference. That is right for the axes `run_meta` has always carried:
#: a payload missing one of those is a payload this check has nothing to say about.
#:
#: It is WRONG for the ones below. They arrived with the flag or the variable that sets them, so a
#: payload written before them did not decline to record a value -- there was no way to ask for
#: anything but the default, and it ran under `stream_tail_chars = None`, `corpus_dollars = False`,
#: `click_probe = False`, `inject_stream_cost_ms = None` and `probe_init_script = None` by
#: construction. Skipping them therefore accepted `--resume --stream-tail-chars 24000` against such
#: a payload, skipped its completed cells, and recorded the rest under a different streamed fixture
#: beneath the same cell ids: one ladder built from two films, which is what this check exists to
#: refuse. Absence proves the value here, so it is read as the value.
HISTORICAL_DEFAULTS = {
    "stream_tail_chars": None,
    "corpus_dollars": False,
    "click_probe": False,
    "inject_stream_cost_ms": None,
    "probe_init_script": None,
    "headed": False,
}

#: THE RATIO THE LADDER WAS SIZED BY, carried on every cell's `meta` by `session.build_cells`.
#: Not in `IDENTITY_AXES`: those are checked by `prepare_payload` before anything is installed,
#: and this one is not known until `build_cells` has measured the corpus. Checked instead by
#: `ladder_ratio_problems`, on the same footing as `COMMIT_AXES` -- late enough to have the answer,
#: early enough that nothing has been measured under it.
LADDER_RATIO_AXIS = "ladder_chars_per_token"

#: How close two ratios must be to be the same ratio. `measure_chars_per_token` rounds its answer
#: to three decimals and `PROVISIONAL_CHARS_PER_TOKEN` is exact, so every value that reaches a
#: payload sits on a 0.001 grid: half a grid step separates "the same measurement, re-read through
#: JSON" from "a different one". Wide enough that 5.0 never refuses 5.0 for a float's sake, narrow
#: enough that the case this exists for -- a provisional 4.0 against tiktoken's 3.336 on this
#: corpus, a fifth of the character load of every rung -- can never be inside it.
LADDER_RATIO_TOLERANCE = 5e-4

#: The arm a run without `--ab` records every cell under. `Cell.arm` in `runtime.types`; an A/B
#: overwrites it with the target's label, `base` or `treatment`. Named here because it is what
#: tells `recorded_identities` which of the two a session already in the payload was.
SINGLE_ARM = "A0"

#: THE BUILD, as opposed to the name it was asked for by. A ref is a POINTER: `main`, a topic
#: branch and a movable tag all resolve afresh on every install, and `checkout_ref` is the only
#: thing that ever knows which commit one named. So these are not in `IDENTITY_AXES` and are not
#: checked by `prepare_payload`: that refusal is deliberately made BEFORE anything is installed,
#: and a commit cannot be known before the fetch that resolves it. They are checked instead by
#: `commit_problems`, once the sides are up and before the first cell -- late enough to have the
#: answer, early enough that nothing has been measured under it. See `run`.
COMMIT_AXES = ("studio_commit", "treatment_commit")


def requested_identity(args, ab_ref, corpus_hash: str) -> dict:
    """The payload identity THIS invocation is asking for.

    `studio_ref` is spelled exactly as `run_meta` records it, so the requested value and the
    recorded one are comparable without a second convention to keep in step.
    """
    from .runtime.browser import default_engine

    base_ref = f"attached:{args.attach.rstrip('/')}" if args.attach else args.branch
    attach_b = (getattr(args, "attach_b", "") or "").rstrip("/")
    return {
        "tier": args.tier,
        "cadence": args.cadence,
        # THE INSTRUMENTS THIS TREE CARRIES, as a module constant rather than anything the caller
        # can ask for. A payload recorded by another version was measured by other instruments.
        "tool_version": TOOL_VERSION,
        # THE ENGINE THAT WILL RENDER, not the flag that asked for it. `--engine` defaults to
        # nothing and `browser.launch` resolves the platform's desktop webview family, so the
        # requested value has to be resolved the same way `launch` resolves it -- otherwise the
        # commonest engine change of all, one run naming `--engine chromium` and the resume
        # naming none, reads as no difference at all. Resolvable here, before anything is
        # installed, because `default_engine` reads `platform.system()` and nothing else.
        "engine": getattr(args, "engine", "") or default_engine()[0],
        "instrument_level": args.instrument_level,
        "corpus_hash": corpus_hash,
        "studio_ref": base_ref,
        "treatment_ref": ab_ref or "",
        # THE ATTACHED TREATMENT IS THE SERVER, NOT THE LABEL. `studio_ref` folds an attached base
        # down to its URL because that is the only thing about it this harness can see; the
        # treatment carried nothing but the name typed after `--ab`, and that name is free. So
        # `--attach A --attach-b B --ab fix --resume`, re-run later against `--attach-b C`, passed
        # the identity check, skipped every completed treatment cell and reported B's measurements
        # as C's result. Empty for a treatment this run installs itself, whose ref IS its identity.
        "treatment_url": attach_b if (ab_ref and attach_b) else "",
        "stream_tail_chars": args.stream_tail_chars,
        "corpus_dollars": bool(args.corpus_dollars),
        "click_probe": bool(getattr(args, "click_probe", False)),
        "inject_stream_cost_ms": getattr(args, "inject_stream_cost_ms", None),
        # Spelled the same way `run_meta` records it, through `bool`, so the recorded False and
        # the requested value are the same type. Adding the axis without this line made every
        # resume refuse itself -- the payload declared `False` and the request declared `None` --
        # which is the failure mode an identity axis has when it is only half wired up.
        "headed": bool(getattr(args, "headed", False)),
        # FROM THE ENVIRONMENT, because that is where this one is asked for: the probe hook is a
        # variable rather than a flag on purpose, so `args` never sees it and there is nothing to
        # read it from. Spelled exactly as `run_meta` records it -- the path as it was given -- for
        # the same reason `studio_ref` is, so the requested value and the recorded one compare
        # without a second convention to keep in step.
        "probe_init_script": os.environ.get("SBENCH_EXTRA_INIT_SCRIPT") or None,
    }


def recorded_identities(payload_path) -> list:
    """One identity per session already in the payload, from the rows those sessions wrote.

    Read out of `run_meta` (including its nested `platform.engine`), `ab_plan` and the cells
    rather than out of a new field, so a payload written before this check existed is judged on
    exactly the axes it DID record: an axis a row never declared cannot be a difference, and an
    older output therefore still resumes.

    `mode` is the one entry here that is not an axis of any single row. It is what the session's
    cells were recorded under, which is the thing `identity_problems` has to compare and the only
    thing the payload states about it. See `SINGLE_ARM`.
    """
    by_session: dict = {}
    order: list = []
    path = Path(payload_path)
    if not path.exists():
        return []
    with path.open(encoding = "utf-8") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            row_type = row.get("row_type")
            if row_type not in ("run_meta", "ab_plan", "cell"):
                continue
            session = str(row.get("session_id"))
            if session not in by_session:
                by_session[session] = {}
                order.append(session)
            if row_type == "cell":
                # THE MODE A SESSION MEASURED IN, declared by the cells it actually wrote rather
                # than by a header it wrote before it measured anything. A run that died between
                # `run_meta` and `ab_plan` recorded no cells and so declares no mode, and resuming
                # it is not a transition; a session holding cells declares the mode those cells
                # were recorded under, whichever row types the version that wrote them emitted.
                arm = str((row.get("cell") or {}).get("arm") or row.get("arm") or "")
                if arm:
                    by_session[session].setdefault("mode", "single" if arm == SINGLE_ARM else "ab")
                # THE RATIO THE SESSION'S RUNGS WERE SIZED BY, on the same rule as the mode: read
                # off a cell the session actually wrote, so a session that recorded none declares
                # none. See `ladder_ratio_problems`.
                sized = ((row.get("cell") or {}).get("meta") or {}).get("ladder_chars_per_token")
                if isinstance(sized, dict) and sized.get("chars_per_token") is not None:
                    by_session[session].setdefault(
                        LADDER_RATIO_AXIS, float(sized["chars_per_token"])
                    )
                continue
            for axis in IDENTITY_AXES + COMMIT_AXES:
                if axis in row:
                    by_session[session][axis] = row[axis]
            # THE ENGINE IS NESTED, under `run_meta.platform`, so the loop above cannot see it.
            # Read from there rather than from a new top-level field for the same reason as
            # everything else here: a payload written before this check existed still declares
            # what it rendered with, and is judged on it. `run_meta` is emitted AFTER
            # `browser.launch` returns, so what is recorded is the RESOLVED engine and a session
            # that never got a browser up declares none -- and an axis a row never declared
            # cannot be a difference, so that session still resumes.
            if row_type == "run_meta":
                engine = str((row.get("platform") or {}).get("engine") or "")
                if engine:
                    by_session[session]["engine"] = engine
            # `ab_plan` is where the treatment ref is recorded; `run_meta` names only the base.
            if row_type == "ab_plan" and row.get("treatment_ref") is not None:
                by_session[session]["treatment_ref"] = row["treatment_ref"]
    return [by_session[s] for s in order if by_session[s]]


def identity_problems(recorded: dict, requested: dict) -> list:
    """Every axis on which a recorded session and this invocation disagree."""
    problems = []
    # AN A/B AND A RUN THAT IS NOT ONE MAY NOT SHARE A PAYLOAD. The arm in the cell id ("A0"
    # against "base"/"treatment") keeps the new cells from being SKIPPED, which is why this used
    # to be waved through -- but it does not keep them out of each other's REPORT. `score_payload`
    # hands every cell to `measures_from_records`, which keys by rung and keeps the first reading,
    # and `_completion_by_rung` keeps a failure over a success at the same rung. So a single-build
    # run that died at 10K, resumed as `--ab fix` and re-measured successfully on both arms, still
    # printed `INCOMPLETE: the base died at 10K`, scored the rung 0 and reported ONSET RUNG: none,
    # while the identical resume WITHOUT `--ab` -- same crash, same repair, same cell id, so the
    # dead attempt is superseded -- scored it 82.2. The mode is part of what a payload measures,
    # so a run that changes it resumes nothing: it starts a payload of its own.
    requested_mode = "ab" if requested.get("treatment_ref") else "single"
    recorded_mode = recorded.get("mode")
    if recorded_mode is not None and recorded_mode != requested_mode:
        names = {"ab": "an A/B (base against treatment)", "single": "one build on its own"}
        problems.append(
            f"mode: the payload was recorded by a run measuring {names[recorded_mode]}, "
            f"this run measures {names[requested_mode]}"
        )
    # Is there a second side on BOTH sides of this comparison? When there is not, the treatment
    # axes describe a side one of them does not have, and the mode difference above is the whole
    # of what they disagree on. Decided from the ref, so that an A/B whose treatment this run
    # installed -- which records no treatment URL -- is still judged against an attached one on
    # the axis where the two of them do differ.
    both_ab = bool(requested.get("treatment_ref")) and bool(recorded.get("treatment_ref"))
    for axis in IDENTITY_AXES:
        declared = axis in recorded
        if declared:
            got = recorded[axis]
        elif axis in HISTORICAL_DEFAULTS:
            # Not declared, but not silent either: this axis postdates the payload, so the payload
            # ran under the default. See `HISTORICAL_DEFAULTS`.
            got = HISTORICAL_DEFAULTS[axis]
        else:
            # An axis this payload never declared. See `recorded_identities`.
            continue
        if axis in TREATMENT_AXES and not both_ab:
            continue
        want = requested.get(axis)
        if str(want) != str(got):
            where = (
                f"the payload was recorded with {got!r}"
                if declared
                else f"the payload predates this axis and therefore ran with {got!r}"
            )
            problems.append(f"{axis}: {where}, this run asks {want!r}")
    return problems


def resolved_commits(sides: list) -> dict:
    """The commits the sides of THIS run were actually installed from. `""` when unknowable.

    Unknowable for an attached Unsloth: the caller pointed this harness at a URL and nothing about
    what is deployed behind it is visible from here, which is why `studio_ref` folds an attached
    base down to that URL instead.
    """
    out = {axis: "" for axis in COMMIT_AXES}
    for axis, side in zip(COMMIT_AXES, sides):
        if side.get("owns"):
            out[axis] = str(side.get("commit") or "")
    return out


def commit_problems(recorded: dict, resolved: dict) -> list:
    """Every side on which a recorded session and this invocation installed a different BUILD.

    `--resume` skips a completed `cell_id`, and a cell id is the rung, the arm and the repetition.
    The identity check in `prepare_payload` keeps the REFS in step, and a ref is enough right up
    until it moves: `--branch main --ab fix --resume` into a payload recorded yesterday passes
    every axis while `main` has advanced, so the cells already in the file were measured on one
    build and the rungs still owed are measured on another, and `report.assemble_rows` prints the
    mixture under a single header naming one ref. Movable tags and any live topic branch do the
    same thing; `unsloth/main` does it several times a day.

    An empty commit on EITHER side is not a difference. A payload written before this was recorded
    never declared it -- the same rule `recorded_identities` applies to every other axis -- and an
    attached side has no commit to declare, so attaching does not start failing against a payload
    that a self-managed run wrote.
    """
    problems = []
    for axis in COMMIT_AXES:
        want, got = str(resolved.get(axis) or ""), str(recorded.get(axis) or "")
        if not want or not got or want == got:
            continue
        side = "the base" if axis == "studio_commit" else "the treatment"
        problems.append(
            f"{axis}: {side} was recorded at commit {got[:12]}, this run installed {want[:12]}"
        )
    return problems


def ladder_ratio_problems(recorded: dict, measured: float) -> list:
    """Whether the rungs this run is about to build carry the character load the payload's do.

    A rung is NAMED in tokens and BUILT in characters, and the chars-per-token ratio is the only
    thing that makes those the same claim. It is measured afresh on every invocation, `--resume`
    included, and it is not fixed by anything the identity check already pins: `corpus_hash` is the
    frozen fixture's own hash and says nothing about how many tokens a tokeniser reads out of it,
    and `session.ladder_chars_per_token` falls back to `PROVISIONAL_CHARS_PER_TOKEN` whenever no
    real tokeniser answers. tiktoken is not a dependency of this harness, and even where it is
    installed `get_encoding` fetches `cl100k_base` over the network on first use, so "no tokeniser
    answered" is one absent package or one unlucky minute, not a hypothetical.

    So: a run on a machine with no tokeniser sizes every rung at 4.0 and dies at 100K; the resume
    finds tiktoken and sizes at 3.336. `_resume_set` skips the completed cells by `cell_id`, which
    is `r{rung}.{arm}.rep{rep}` and carries no ratio, so 1K and 10K stay at 4,000 and 40,000
    characters while 100K and 1M are built at 333,600 and 3,336,000 -- a sixth less load per rung
    on the second half of the ladder. Nothing downstream can see the mixture: `score_payload` keys
    by rung, the report prints one ladder, and ONSET RUNG names a token label that means two
    different amounts of work in the same table.

    A ratio the payload never recorded is not a difference, the rule `recorded_identities` applies
    to every other axis: a payload written before the ratio travelled on `meta` still resumes. See
    `LADDER_RATIO_TOLERANCE` for what counts as the same ratio.
    """
    got = recorded.get(LADDER_RATIO_AXIS)
    if got is None or measured is None:
        return []
    if abs(float(got) - float(measured)) <= LADDER_RATIO_TOLERANCE:
        return []
    return [
        f"{LADDER_RATIO_AXIS}: the payload's rungs were sized at {float(got)!r} characters per "
        f"token, this run measures {float(measured)!r}"
    ]


def archive_payload(paths, log = _log):
    """Move an existing payload aside so a FRESH run starts a file of its own. `None` when empty.

    APPEND MODE IS FOR VALIDATED RESUMES ONLY. `Recorder` opens the payload with `"a"`, so a second
    invocation into the same `--out` used to write its rows behind the first run's. That is exactly
    right for `--resume`, which re-runs the cells that died under the same deterministic `cell_id`
    so `latest_attempt_rows` can supersede them -- and it is wrong for every other reuse, because
    superseding only reaches the cell ids the new run REACHED. A fresh run that is interrupted
    leaves the previous run's cells standing in the rungs it never got to, `report.assemble_rows`
    takes its header from the FIRST `run_meta` in the file, and `--report` then scores one ladder
    whose rungs came from two builds under two different films without a word about it.

    Moved rather than truncated: the previous run's payload is the previous run's evidence, and the
    fix for reporting a mixture may not be to delete half of it.
    """
    src = Path(paths.payload_jsonl)
    try:
        if not src.exists() or src.stat().st_size == 0:
            return None
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(src.stat().st_mtime))
    except OSError:
        return None
    dest = src.with_name(f"{src.stem}-{stamp}{src.suffix}")
    index = 1
    while dest.exists():
        dest = src.with_name(f"{src.stem}-{stamp}.{index}{src.suffix}")
        index += 1
    src.rename(dest)
    log(f"  a payload was already in this output directory; moved it to {dest.name}")
    log("  (this run starts a payload of its own. Pass --resume to CONTINUE the previous one.)")
    return dest


def payload_mark(payload_path) -> int:
    """How long the payload was BEFORE this session was allowed to append to it.

    Taken before the `Recorder` opens the file, and paired with `rollback_session_rows`. See there
    for what the pair is for; `0` for a payload that does not exist yet, which is the length it
    has.
    """
    try:
        return Path(payload_path).stat().st_size
    except OSError:
        return 0


def rollback_session_rows(
    payload_path,
    mark: int,
    log = _log,
) -> int:
    """Undo everything this session appended, back to `mark`. The bytes dropped.

    THE RULE `prepare_payload` STATES: a refusal has to leave the payload it refused exactly as it
    found it. `prepare_payload` and `commit_problems` keep it by running before the `Recorder`
    exists -- the identity axes are known from the CLI, and the commit is known once the sides are
    installed, both still before the first recorded row. `ladder_ratio_problems` cannot: the ratio
    is not known until `build_cells` has measured the corpus, which is after `make_context` has
    opened the file, written `run_meta` and possibly a failed `instrument_unavailable` gate, and
    after an optional `--surfaces` sweep. So that check keeps the same rule from the other end.

    What a refused resume costs if it does not. `recorded_ladder` folds the `rungs` of EVERY
    `run_meta` in the file, deliberately -- a session killed after its header still owes what it
    promised -- so a refused `--resume --rungs 1K,10K` over a finished 1K payload leaves 10K
    promised and never recorded, and `--report` scores that payload INCOMPLETE from then on.
    `excluded_from_rows` turns every failed gate into an excluded cell, so the refused session's
    own `ladder_ratio_measured` gate is charged to the payload it never touched. Neither is
    recoverable by re-running: the rows are in somebody else's evidence file.

    TRUNCATED, NOT ARCHIVED, which is the opposite of `archive_payload` and for the reason that
    makes it the opposite: an archive preserves a run's evidence, and a refused resume produced no
    evidence. There is one writer -- `Recorder` is the only thing that opens this file, on the main
    thread, and it is closed by the caller before this runs -- so the tail being dropped is this
    session's rows and nothing else.
    """
    path = Path(payload_path)
    try:
        size = path.stat().st_size
    except OSError:
        return 0
    if size <= mark:
        return 0
    try:
        os.truncate(path, mark)
    except OSError as exc:
        log(f"  could not roll back {path.name}: {exc}")
        return 0
    dropped = size - mark
    log(f"  rolled back {dropped} bytes this refused session had appended to {path.name}")
    return dropped


def invalidate_stale_reports(
    out,
    *,
    archived,
    extra_init,
    log = _log,
) -> list:
    """Replace `summary.md` and `ab.md` when the payload they describe is no longer the one there.

    `archive_payload` moves `payload.jsonl` and nothing else, so a `summary.md` written by an
    earlier `--report` of this directory -- step three of the README quickstart, into the same
    `--out` -- stays at the standard artifact path while the payload underneath it is replaced.
    Nothing later puts that right, because the report that would overwrite it is a command the
    next run has no reason to issue: a plain run writes `summary.md` never (only `--report` does)
    and `ab.md` only under `--ab`, so a single-arm run produces no report-shaped file at all and
    both stale ones survive it.

    TWO INDEPENDENT INVALIDATIONS, and each is sufficient on its own.

    `archived` is the general case: the payload these reports described has been moved aside, so
    they describe a file that is no longer at the path they name, whatever the incoming run is.
    This is the half that was missing. Keying only on `extra_init` closed the clean-then-probed
    direction and left probed-then-clean open, where a probe run's refusal text -- which says in
    so many words that the payload beside it is not scorable -- survives into a directory whose
    payload is now perfectly scorable. A refusal that outlives its reason is worse than no file,
    because it is read as a finding about the run that is actually there.

    `extra_init` is the narrower case and is NOT covered by `archived`: a `--resume` that
    `prepare_payload` accepts archives nothing, so `archived` is `None` while the continued
    payload is being extended by a probed run and becomes unscorable under the old summary.

    OVERWRITTEN, not deleted, for the reason `_render_ab` gives: whoever opens the path gets the
    reason rather than a missing file. HERE, not at the probe banner further down, for the reason
    52fc3e848 moved the `ab.md` refusal above its early return: everything between this point and
    the banner can `return`, raise or be killed by the wall-clock watchdog, and every one of those
    paths has already archived the payload the summary belongs to. After `prepare_payload`,
    though, because a `--resume` it refuses touched nothing.

    BOTH ARTIFACTS, and `ab.md` is not already covered by the refusal inside `_render_ab`. That
    function runs only under `if ab_ref`, so a fresh SINGLE-ARM run into a directory left by an
    earlier `--ab` run never reaches it, and the old table survives beside the new payload for as
    long as the directory lasts.

    Returns the paths it rewrote, so a caller can assert on them.
    """
    if not archived and not extra_init:
        return []
    if extra_init:
        why = (
            f"this output directory was reused by a run carrying an external init script "
            f"({extra_init}), so its timings measure the page and the instrument together. The "
            f"payload beside it now is not scorable. Re-run with SBENCH_EXTRA_INIT_SCRIPT unset."
        )
    else:
        why = (
            f"this output directory was reused by a later run, which moved the payload this "
            f"described to {Path(archived).name} and recorded a new one beside it. Re-report the "
            f"payload that is there now, or read the archived one directly."
        )

    rewritten = []
    for name, heading, what in (
        ("summary.md", "# No summary", "summary"),
        ("ab.md", "# No A/B table", "table"),
    ):
        stale = Path(out) / name
        if not stale.exists():
            continue
        stale.write_text(f"{heading}\n\nNO {what.upper()}: {why}\n", encoding = "utf-8")
        log(f"  a previous {stale} was replaced by this refusal")
        rewritten.append(stale)
    return rewritten


def prepare_payload(
    paths,
    requested: dict,
    *,
    resume: bool,
    log = _log,
):
    """What happens to an `--out` that already holds a payload. Called BEFORE anything is installed.

    Before, because both answers are worthless afterwards: the refusal below has to arrive before
    the caller has paid for a clone and a build -- an A/B installs TWO -- and the archive has to
    happen before this run's `Recorder` opens the file it would otherwise append to.

    TWO REUSES, TWO ANSWERS.

    A fresh run archives (see `archive_payload`).

    A `--resume` continues, and is REFUSED when the payload was recorded under a different identity.
    `--resume` skips every `cell_id` the payload already completed, and a cell id encodes the rung,
    the arm and the repetition -- not the tier, the cadence, the instrument level, the corpus or
    either ref. So resuming after changing one of those skips cells that measured something else:
    at the extreme, `--branch main --ab other --resume` into a directory holding a finished
    `main -> fix` run installs and launches two Unsloth instances, skips every cell, exits 0, and leaves the
    OLD comparison standing in `ab.md` for somebody to read as the answer for `other`.
    """
    if not resume:
        return archive_payload(paths, log = log)

    problems: list = []
    for recorded in recorded_identities(paths.payload_jsonl):
        for problem in identity_problems(recorded, requested):
            if problem not in problems:
                problems.append(problem)
    if problems:
        raise SystemExit(
            f"refusing to resume {paths.payload_jsonl}: it was not recorded by a run of this "
            "configuration.\n  "
            + "\n  ".join(problems)
            + "\nA cell id is the rung, the arm and the repetition, so resuming here would skip "
            "cells that measured something else and report the mixture as one run. Re-run into a "
            "fresh --out, or resume with the configuration the payload was recorded under."
        )
    return None


def _resume_set(paths) -> set:
    """The cells `--resume` may skip: the ones whose LATEST attempt completed.

    THE LATEST ATTEMPT DECIDES, which is the rule `latest_attempt_rows` already applies for the
    score (`report.build.score_payload`), the ratio (`ab.readings_by_arm`), the surface parity
    sweep and `--assert-liveness`. Read raw, this loop was the last reader in which a superseded
    row still counted -- and it counted in the direction that skips work.

    How that happens without anybody doing anything unusual: an A/B pair is re-run WHOLE
    (`ab.skippable_cells`), so a resume re-runs an arm that had already succeeded. If that retry
    fails while its partner succeeds, the payload holds a completed row and a LATER failed row
    under the same deterministic `cell_id`. The next `--resume` found the old success, skipped the
    whole pair and exited 0, while `--report` scored the failed retry INCOMPLETE and
    `--assert-liveness` failed on it. A resume that can never re-run the cell that is broken is a
    gate nobody can satisfy by fixing the run.
    """
    from .scoring.from_payload import latest_attempt_rows

    done = set()
    if not paths.payload_jsonl.exists():
        return done
    records = []
    with paths.payload_jsonl.open(encoding = "utf-8") as fh:
        for line in fh:
            try:
                records.append(json.loads(line))
            except ValueError:
                continue
    for row in latest_attempt_rows(records):
        # Only a COMPLETED cell is skipped. A cell that died is re-run, because its failure
        # may have been the machine and not the build.
        if row.get("row_type") == "cell" and row.get("completed"):
            done.add(row.get("cell_id"))
    return done


def _summarise(rows: list, paths) -> None:
    if not rows:
        return
    _log("\n" + "=" * 78)
    _log(
        f"{'cell':<16} {'chars':>10} {'elems':>8} {'spans':>8} {'c/span':>7} "
        f"{'ran':>6} {'miss':>5} {'exp!':>5} {'busy%':>7}"
    )
    _log("-" * 78)
    for r in rows:
        actions = r.get("actions") or []
        ran = sum(1 for a in actions if a.get("ran"))
        # The PEAK, not the end state: the film's last two actions reopen the thread and delete
        # a message, so an end-of-film census describes a thread that is no longer there.
        census = r.get("census_peak") or r.get("census_after") or {}
        _log(
            f"{r['cell_id']:<16} {r.get('assistant_chars_in_dom') or 0:>10,} "
            f"{census.get('elements') or 0:>8,} {census.get('highlight_spans') or 0:>8,} "
            f"{str(r.get('chars_per_span') or '-'):>7} "
            f"{ran}/{len(actions):>4} {r.get('slots_missed', 0):>5} "
            f"{r.get('expect_failures', 0):>5} "
            f"{'-' if not r.get('completed') else 'ok':>7}"
        )
        if not r.get("completed"):
            f = r.get("failure") or {}
            _log(f"    FAILED: {f.get('kind')}: {str(f.get('message'))[:100]}")
    # SAID UNDER THE TABLE THAT PRINTS THEM. `elems` and `spans` come from `census_peak`, whose
    # action is chosen by a max() over per-action censuses that race their own teardown, so it is
    # not the same moment on two arms. Differencing these two columns between arms is what
    # produced a withdrawn "43% more standing DOM" claim. The columns stay because the high-water
    # mark is a useful diagnostic; the warning stays with them so it cannot be read without it.
    _log(
        "\n  elems/spans are census_peak: a diagnostic high-water mark, taken at whichever "
        "action mounted most.\n  Do NOT difference them between arms. For a cross-arm census use "
        "a settled measure."
    )


def _rung_tokens(labels: list) -> list:
    """`1K` -> 1000. The scoring ladder is indexed by tokens; the CLI speaks in rung labels."""
    mult = {"K": 1_000, "M": 1_000_000}
    out = []
    for label in labels:
        text = str(label).strip().upper()
        out.append(int(float(text[:-1]) * mult[text[-1]]) if text[-1] in mult else int(text))
    return out


def recorded_ladder(path) -> list:
    """The rungs the RUN promised, folded over EVERY `run_meta` the payload carries. `[]` when it
    never said.

    A payload knows which ladder it was collecting; the CLI only knows which tier the caller
    happened to type. Reporting a standard run that was killed before its top cell under the
    default tier scored the surviving low rungs and never mentioned the missing one, which is the
    crash-beats-limp failure `report/build.py` exists to refuse, arriving through the front door.

    FOLDED, NOT THE FIRST HEADER, because a resume is allowed to ADD rungs -- `rungs` is
    deliberately not a payload identity axis, and `--resume --rungs 1K,10K` over a finished 1K run
    appends a second `run_meta` promising both. Reading the first one alone meant a continuation
    killed after that header and before the 10K cell reported the 1K ladder it had already finished
    and scored COMPLETE, which is the same truncated run passing as a whole one. Every rung any
    session in this file promised is owed by it, and one it never reached is scored INCOMPLETE
    rather than dropped.
    """
    ladder: list = []
    try:
        with Path(path).open(encoding = "utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if row.get("row_type") != "run_meta":
                    continue
                rungs = row.get("rungs") or TIER_RUNGS.get(str(row.get("tier"))) or []
                for rung in rungs:
                    if str(rung) not in ladder:
                        ladder.append(str(rung))
    except OSError:
        return []
    return ladder


def report_only(args) -> int:
    """Score and render a payload that already exists. No browser, no Unsloth, no network.

    Separate from `run` so a payload produced on somebody else's desktop reports identically here,
    which is the whole point of shipping a single-file benchmark: the numbers come back as a file
    and the analysis happens where the analyst is.
    """
    from .report.build import build_report

    path = Path(args.report)
    if not path.exists():
        _log(f"no payload at {path}")
        return 2

    # `--rungs` first, then the payload's own ladder, then the tier. An explicit `--tier` still
    # wins over the recording, so a reader can deliberately score a payload against another tier's
    # ladder; what is gone is the DEFAULT tier silently shortening somebody else's run.
    recorded = [] if getattr(args, "tier_explicit", True) else recorded_ladder(path)
    if args.rungs:
        declared = _rung_tokens(args.rungs.split(","))
    elif recorded:
        declared = _rung_tokens(recorded)
        _log(f"scoring against the ladder this run recorded: {','.join(recorded)}")
    else:
        declared = _rung_tokens(TIER_RUNGS[args.tier])
    out = path.parent / "summary.md"
    try:
        text, ladder, _payload = build_report(path, declared)
    except SystemExit as exc:
        # A REFUSAL, not a crash, and it has to reach the artefact rather than only the terminal.
        # `SystemExit` is not an `Exception`, so without this clause it would leave the process
        # before the write below. `--resume` reuses the output directory, so a `summary.md` from
        # an earlier clean report of the same directory would survive the refusal and sit next to
        # a now-probed payload, reading as its result. Same reasoning as the stale `ab.md` in
        # `_render_ab`: overwritten rather than deleted, so opening the path gives the reason.
        _log(str(exc))
        if out.exists():
            out.write_text(f"# No summary\n\n{exc}\n", encoding = "utf-8")
            _log(f"  a previous {out} was replaced by this refusal")
        return 2
    except Exception as exc:  # noqa: BLE001
        # A payload that cannot be scored is reported as such rather than half-rendered: a
        # partial report is exactly the artefact that gets quoted without its caveats.
        _log(f"could not build a report from {path}: {type(exc).__name__}: {exc}")
        return 1

    print(text)
    out.write_text(text, encoding = "utf-8")
    _log(f"summary written to {out}")
    return 0


def compare_payloads(args) -> int:
    """Are these two payloads comparable, and if not, exactly which field stops them?

    The guard in `floor_table.load` refuses to POOL payloads across tiers and corpora and it
    works. It cannot reach a comparison made in prose across two separately published runs, which
    is how a withdrawn number nearly propagated twice in one campaign. Quoting the comparability
    key beside a number turns that check from an act of memory into one command.
    """
    from .report.payload import read_records
    from .scoring import payload_rules

    metas = []
    for raw in args.compare:
        path = Path(raw)
        if not path.exists():
            print(f"no such payload: {path}")
            return 2
        # EVERY header, not the first. `--resume` appends a second `run_meta` describing the run
        # that continued the file, and it may legitimately have extended the ladder: `rungs` is
        # deliberately outside `IDENTITY_AXES` because adding a rung ADDS cells rather than
        # reinterpreting the recorded ones. Keying on the first header therefore hashes a ladder
        # the file has since outgrown, and declares a resumed payload comparable with the one-rung
        # run it started as. `floor_table.tiers_of` and `corpora_of` were both fixed for exactly
        # this; `--compare` was the reader left behind.
        #
        # READ THE WAY EVERY OTHER READER IN THIS TOOL READS. The payload is append-only and every
        # row is flushed as it is written, so a run killed during its last append leaves valid rows
        # followed by a torn one -- the failure this format is chosen to survive, and the reason
        # `report.read_records` counts a malformed line instead of raising. `recorded_identities`
        # and `_resume_set` skip one too. Parsing it here with a bare `json.loads` made `--compare`
        # the only reader that answered an interrupted payload with a JSONDecodeError traceback
        # rather than with the check it was asked for.
        records, discarded = read_records(path)
        if discarded:
            print(
                f"  {path}: {discarded} malformed line(s) skipped, which is what a run killed "
                f"mid-append leaves. The check below reads the intact records."
            )
        meta, conflicts = payload_rules.merged_run_meta(records)
        if meta is None:
            print(f"{path} carries no run_meta row, so it cannot be checked at all")
            return 2
        if conflicts:
            # A file whose own headers disagree is not one measurement, so no single key can stand
            # for it and the honest answer is a refusal rather than a token.
            print(f"{path} disagrees with ITSELF across its own run_meta rows:")
            for line in conflicts:
                print(f"  {line}")
            print(
                "\nThis payload holds more than one run and they were not measuring the same "
                "thing, so no comparability key describes it. Score the sessions apart."
            )
            return 2
        metas.append((path, meta))

    (pa, a), (pb, b) = metas
    ka = payload_rules.comparability_key(a)
    kb = payload_rules.comparability_key(b)
    print(f"  {pa}  {ka}")
    print(f"  {pb}  {kb}")
    if ka == kb:
        print("\ncomparable: every field the key covers matches.")
        return 0
    print("\nNOT COMPARABLE. These differ:")
    for line in payload_rules.explain_incomparable(a, b):
        print(f"  {line}")
    print(
        "\nA number from one of these must not be quoted against a number from the other. "
        "Re-run the older side."
    )
    return 1


def assert_liveness(args) -> int:
    """Fail unless every scheduled action in a payload ran, kept its slot and proved its effect.

    THE FAILURE THIS CATCHES. The most expensive wrong answers this harness has produced were not
    wrong numbers, they were absent ones reported as "no effect": four scene actions recorded NOT
    RUN on 312 of 312 attempts because their slots opened while a follow-up turn was still
    streaming, and read as fast, stable and meaningless. A surface crawler walked 53 surfaces that
    would all have digested the same mounted root. An overlay walk could never fire.

    Every one of those is invisible to a test that only checks the run exited 0, and every one of
    them is visible here, because `session.py` already counts `actions_not_run` and `slots_missed`
    per cell. This turns the README's advice to check `ran` before reading a timing into something
    a machine does, which is the only way it gets done every time.

    TWO KINDS OF NOT RUN, and they are not the same finding.

    A SCENE problem is the harness lying: the action was never planned, the button was not there,
    the thread was shorter than the viewport. That is always a failure, on any machine, because it
    means a column of the report is empty and nothing said so.

    A MISSED SLOT is a fact about the machine. The scene is a fixed-duration film on the wall
    clock (see `scene/schedule.py`), so a machine too slow to reach a slot records `slot_missed`
    and the film rolls on BY DESIGN, precisely so a slow machine does not silently take a
    different path through a different-length session. Failing on that turns an honest reading
    into an error, and on a two-core shared CI runner it makes the gate a speed test of the runner.

    So they are counted apart. Scene problems always fail. Missed slots are always PRINTED, and
    fail once they pass `--allow-slot-misses`, which defaults to 0 so a measurement run on a quiet
    machine keeps the strict behaviour and only a caller who knows its machine is contended
    loosens it, in one visible place.

    Offline, so a payload from anyone's laptop or from CI checks identically.
    """
    path = Path(args.assert_liveness)
    if not path.exists():
        _log(f"no payload at {path}")
        return 2

    allowed = {a.strip() for a in (args.allow_not_run or "").split(",") if a.strip()}
    slack = max(0, int(getattr(args, "allow_slot_misses", 0) or 0))
    rows, problems, missed = [], [], []
    for line in path.read_text(encoding = "utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            problems.append("a payload line is not valid JSON")

    # A CELL THAT WAS RE-RUN IS JUDGED ON THE RUN THAT FINISHED IT, which is the same rule the
    # report and the A/B already apply and the same helper that applies it. `--resume` appends to
    # the payload it continues and re-runs the cells that died under the SAME deterministic
    # `cell_id`, so the dead attempt's `completed: false` and its NOT RUN actions are still in the
    # file after a resumed run has succeeded. Read raw, this loop found them forever: the resumed
    # run exited 0 and `--report` scored the retry, while `--assert-liveness` on the same payload
    # failed permanently on a cell that had already been re-run. A gate that cannot be satisfied
    # by fixing the run is a gate people learn to pass with `--allow-not-run`.
    from .scoring.from_payload import ATTEMPT_ROW_TYPES, latest_attempt_rows

    # AN ATTEMPT THAT NEVER REACHED ITS CELL ROW IS A FAILURE, NOT AN ABSENCE. `latest_attempt_rows`
    # names the latest attempt from ANY attempt-keyed row, because the Recorder flushes and fsyncs
    # every action and window row while `CellRunner.run` writes the terminal `cell` row in a
    # `finally` a SIGKILL, an OOM kill or a lost machine never reaches. So the killed attempt
    # supersedes the completed one and then contributes no cell row of its own -- and this loop,
    # reading cell rows only, dropped the cell out of the payload entirely. A resume killed inside a
    # cell whose first attempt had already recorded `completed: false` with NOT RUN actions turned
    # `1` into `0`: the gate stopped reporting a failure that is still written in the file.
    attempted, recorded = set(), set()
    kept = latest_attempt_rows(rows)
    for row in kept:
        if row.get("row_type") in ATTEMPT_ROW_TYPES and row.get("cell_id") is not None:
            attempted.add(str(row.get("cell_id")))
        if row.get("row_type") == "cell" and row.get("cell_id") is not None:
            recorded.add(str(row.get("cell_id")))
    truncated = sorted(attempted - recorded)
    for where in truncated:
        problems.append(f"{where}: an attempt wrote rows but never recorded a cell row")

    cells = len(truncated)
    for row in kept:
        if row.get("row_type") != "cell":
            continue
        cells += 1
        where = row.get("cell_id", "?")
        if row.get("completed") is False:
            problems.append(f"{where}: the cell did not complete")
        for action in row.get("actions") or []:
            name = action.get("action") or action.get("name") or "?"
            if action.get("slot_missed"):
                # A MISSED SLOT IS NOT AN ACTION THE PLATFORM CANNOT PERFORM, so it is classified
                # before the allowance rather than inside it. `scene/schedule.py` records an
                # overrun as `ran = False, slot_missed = True`, so under an allowance-first order a
                # listed name never reached this branch at all and its slot miss left the run
                # silently: an excuse written for "the fixture cannot mount this" swallowed a fact
                # about the machine. Ordering it first is what keeps the two buckets below honest.
                # It is a machine-speed fact whether or not the action also reports ran=False, so
                # it lands in `missed` and is judged against `--allow-slot-misses`, never here.
                missed.append(f"{where}: {name} missed its slot ({action.get('reason') or '?'})")
            elif not action.get("ran"):
                # THE ALLOWANCE IS EXACTLY WHAT ITS NAME SAYS. `--allow-not-run` excuses an action
                # the fixture cannot mount at all; it is not a blanket exemption from the gate.
                # `image_upload` is listed in studiobench-ci.yml only because the fixture cannot
                # mount it, so if it ever does mount, an upload that produces no attachment must
                # be a failure rather than an excuse inherited from a different reason.
                if name in allowed:
                    continue
                problems.append(f"{where}: {name} NOT RUN ({action.get('reason') or 'no reason'})")
            elif action.get("expect_ok") is False:
                # RAN IS NOT DID WHAT IT CLAIMED. `scoring/from_payload.py` and
                # `report/payload.py` both already refuse a timing whose own assertion failed;
                # this loop did not, so `ran = True` with `expect_ok = False` fell past both
                # branches above and the gate exited 0. A selector regression that fails EVERY
                # cell's assertion left the workflow green with the surface non-functional --
                # the same "absent reported as no effect" this gate exists for, one branch over.
                # It is a scene problem, not machine speed, so no allowance and no slack reach it.
                problems.append(
                    f"{where}: {name} ran but its own assertion failed "
                    f"({action.get('reason') or 'no reason'})"
                )

    if cells == 0:
        # An empty payload passing every check is the same false negative in a different costume.
        _log(f"REFUSING: {path} contains no cell rows, so there is nothing to assert about")
        return 2
    for line in problems:
        _log(f"  {line}")
    for line in missed:
        _log(f"  {line}")
    over = len(missed) > slack
    _log(
        f"{cells} cell(s), {len(problems)} scene problem(s), {len(missed)} missed slot(s) "
        f"against a slack of {slack}"
        + (f", {len(allowed)} action(s) allowed not to run" if allowed else "")
    )
    if missed and not over:
        # Said out loud rather than passed over: a run with missed slots has holes in its table,
        # and the only thing this exit code claims is that the harness was not the cause.
        _log(
            "  the missed slots above are machine speed, not a harness fault, but every one of "
            "them is a hole in this run's table. Do not quote a number from this payload."
        )
    return 1 if (problems or over) else 0


def parse_args(argv: list):
    """The CLI surface, split out of `main` so the option contract can be asserted directly."""
    ap = argparse.ArgumentParser(
        prog = "studiobench", description = "A real-path performance benchmark for Unsloth Studio."
    )
    ap.add_argument(
        "--tier",
        choices = TIERS,
        # No default here, and `quick` is applied below instead. `--report` needs to tell "the
        # caller asked for this ladder" apart from "the caller said nothing", because a payload
        # records the ladder its run promised and that answer beats a CLI default.
        default = None,
        help = (
            "fast ~5min (100K only, the iteration loop), quick ~5min (1K,10K, a wiring check), "
            "standard ~20min (1K,10K,100K), full ~60min (+500K,1M)"
        ),
    )
    ap.add_argument(
        "--doctor",
        action = "store_true",
        help = "report what is installed and what each missing piece costs",
    )
    ap.add_argument(
        "--attach",
        metavar = "URL",
        help = "drive an Unsloth that is already running instead of installing one",
    )
    ap.add_argument(
        "--resume", action = "store_true", help = "skip cells already completed in the output payload"
    )
    ap.add_argument(
        "--ab",
        metavar = "REF",
        help = "A/B a second ref, interleaved within one session; with --attach also pass --attach-b",
    )
    ap.add_argument(
        "--attach-b",
        metavar = "URL",
        dest = "attach_b",
        help = "the treatment side's already-running Unsloth, when --ab is used "
        "together with --attach",
    )
    ap.add_argument(
        "--report",
        metavar = "PAYLOAD",
        help = "score and render an existing payload.jsonl, then exit. Runs offline, "
        "so a payload mailed in from another machine reports here",
    )
    ap.add_argument(
        "--compare",
        metavar = ("PAYLOAD_A", "PAYLOAD_B"),
        nargs = 2,
        dest = "compare",
        help = "offline. Say whether two payloads may be compared at all, and if not, name the "
        "field that differs. `floor_table` already refuses to POOL across tiers and corpora; "
        "this is for the case it cannot reach, a comparison made in PROSE between two "
        "separately published runs",
    )
    ap.add_argument(
        "--assert-liveness",
        metavar = "PAYLOAD",
        dest = "assert_liveness",
        help = "exit non-zero unless every scheduled action in an existing "
        "payload.jsonl actually ran, kept its slot and passed its own assertion. "
        "Offline. This is the gate that catches an action which never fired, or "
        "never did what it claimed, reporting as 'no effect'",
    )
    ap.add_argument(
        "--allow-not-run",
        metavar = "ACTIONS",
        dest = "allow_not_run",
        help = "comma-separated action names --assert-liveness may excuse for NOT RUNNING "
        "only. A listed action that does run is still held to its slot and its own "
        "assertion. Use only for an action a platform genuinely cannot perform, and say "
        "which in the pull request: every name here is a hole in the gate",
    )
    ap.add_argument(
        "--windowed-arm",
        metavar = "ARMS",
        dest = "windowed_arm",
        # An ENV FALLBACK as well as the flag. `scripts/pr_perf_sweep.py` builds studiobench's
        # command line itself and is shared with other people's in-flight sweeps, so adding an
        # argument there mid-run is a hazard; this lets the sweep driver select the gate through
        # the environment without anybody editing the driver.
        default = os.environ.get("SBENCH_WINDOWED_ARM", ""),
        help = "comma-separated arm labels (base, treatment) that mount a WINDOW of the thread "
        "rather than all of it, and are therefore gated on the windowed readiness signal "
        "instead of on every message being mounted. Not a relaxation: the named arm must "
        "publish aria-setsize equal to the seeded message count, mount the end of the "
        "thread, settle, and pass a scroll-to-top completeness probe. Naming an arm that "
        "does NOT virtualise makes no difference to it beyond the extra conditions. "
        "Structural UI parity is not applicable to a windowed arm; score it with "
        "sweep/ui_parity.py --mode behaviour",
    )
    ap.add_argument(
        "--click-probe",
        dest = "click_probe",
        action = "store_true",
        help = "before the film starts, split the composer click into what a USER pays and "
        "what Playwright's actionability check pays, plus a hover-only reading. Off by "
        "default: it costs seconds at large rungs and makes the cell's timings "
        "incomparable with a cell that did not run it",
    )
    ap.add_argument(
        "--allow-slot-misses",
        metavar = "N",
        dest = "allow_slot_misses",
        type = int,
        default = 0,
        help = "how many MISSED SLOTS --assert-liveness tolerates before failing. A "
        "missed slot is a fact about the machine, not about the harness, and the "
        "film is designed to roll on through one. Default 0, which is right for a "
        "quiet measurement machine; raise it only on a contended runner, where the "
        "gate is proving the plumbing works rather than that the runner is fast",
    )
    ap.add_argument(
        "--stream-tail-chars",
        type = int,
        dest = "stream_tail_chars",
        help = "override how many characters of the last turn STREAM. The rung ladder "
        "pins this at 6,000 on every rung so that the thread is the only thing that "
        "varies, which means a cost scaling with the length of the reply being streamed "
        "is constant across the whole ladder and reads as a floor. This is the axis that "
        "can see one. Raising it makes the film's after-generation slots run mid-stream, "
        "so check the payload with --assert-liveness rather than trusting the labels",
    )
    ap.add_argument(
        "--inject-stream-cost-ms",
        type = float,
        dest = "inject_stream_cost_ms",
        help = "VALIDATION. Burn this many milliseconds of main-thread time per SSE chunk on the "
        "treatment side, inside the task chain the chunk starts. Needs --ab. The point is to "
        "check that the streaming-cost metric reads back a cost this harness injected itself: a "
        "metric that cannot see a known cost cannot see an unknown one, and the recovery fraction "
        "is what says which of the two a null result was. An arm running this is not a "
        "measurement of the build",
    )
    ap.add_argument(
        "--corpus-dollars",
        action = "store_true",
        dest = "corpus_dollars",
        help = "give the STREAMED turns the CURRENCY AND SHELL dollars a real reply has "
        "($HOME, $12.99). Not the same thing as the LaTeX the frozen corpus carries since "
        "corpus v2: that is well-formed math in the SEEDED thread, which exercises the "
        "renderer, and this is malformed-on-purpose dollars in the turn that STREAMS, "
        "which exercises preprocessLaTeX's currency-escape and code-region heuristics. "
        "Measured over one 96,000 character reply, the cheap regime is 15.3 ms and the "
        "expensive one 281.3 ms. The frozen units on disk and their hashes are untouched",
    )
    ap.add_argument("--rungs", help = "comma-separated rung override, e.g. 1K,10K")
    ap.add_argument("--reps", type = int, default = 1)
    ap.add_argument(
        "--instrument-level",
        type = int,
        default = 0,
        choices = [0, 1, 2, 3],
        help = "0 is the only level headline numbers may come from",
    )
    ap.add_argument(
        "--cadence",
        default = "field",
        choices = ["field", "fast"],
        help = "field is 24 chars every 73ms, the rate of the captured reply",
    )
    ap.add_argument(
        "--engine",
        choices = ["chromium", "webkit", "firefox"],
        help = "default matches the platform's desktop webview family",
    )
    ap.add_argument("--branch", default = "main", help = "Unsloth ref to install when not attaching")
    ap.add_argument("--home", help = "UNSLOTH_STUDIO_HOME for an install")
    ap.add_argument("--port", type = int, default = 5399)
    # `unsloth`, not `admin`. Unsloth's first run prints "DEFAULT ADMIN ACCOUNT CREATED / username:
    # unsloth", and the wrong one answers 401 with a message about resetting the PASSWORD, which
    # sends you looking in the wrong place.
    ap.add_argument("--username", default = "unsloth")
    ap.add_argument("--password", default = "")
    ap.add_argument(
        "--password-b",
        dest = "password_b",
        default = "",
        help = "the treatment Unsloth's password, when --ab is used together with --attach-b. "
        "Two Unsloth instances booted separately mint two different bootstrap passwords, so one "
        "--password cannot authenticate both. Defaults to --password",
    )
    ap.add_argument("--out", help = "output directory")
    ap.add_argument(
        "--surfaces",
        action = "store_true",
        help = "additionally sweep every registered UI surface -- the other routes, "
        "the settings tabs, the sidebar menus, the model picker -- and take a "
        "parity digest of each. The film covers the chat thread; this covers "
        "the rest of the app. Off by default: it costs about a minute per arm "
        "and it does not measure performance",
    )
    ap.add_argument(
        "--parity-shots",
        metavar = "DIR",
        dest = "parity_shots",
        help = "write a viewport PNG per action per arm into DIR, taken at the same instant as "
        "the parity digest, so a mismatch can be SEEN rather than read as a hex pair. Off by "
        "default",
    )
    ap.add_argument(
        "--parity-raw",
        action = "store_true",
        dest = "parity_raw",
        help = "record the NORMALISED signature text beside every parity digest, so "
        "`sweep/parity_null_control.py --hunt` can name which bytes moved between two arms "
        "instead of only that they did. Off by default: it multiplies a payload's size by "
        "roughly a hundred, and only the hunt reads it",
    )
    ap.add_argument("--headed", action = "store_true")
    ap.add_argument("--keep-studio", action = "store_true")
    ap.add_argument(
        "--allow-dev-server",
        action = "store_true",
        help = "run against a development build anyway. ONLY to demonstrate that the "
        "production gate matters: React's dev build inflates the axis under "
        "investigation by about 3.2x",
    )
    args = ap.parse_args(argv)
    # Whether the ladder was ASKED FOR or merely defaulted. Only `--report` cares; everything else
    # sees the tier it always saw.
    args.tier_explicit = args.tier is not None
    if args.tier is None:
        args.tier = "quick"
    return args


def main(argv: list) -> int:
    args = parse_args(argv)

    if args.doctor:
        return doctor(args)
    if args.report:
        return report_only(args)
    if args.compare:
        return compare_payloads(args)
    if args.assert_liveness:
        return assert_liveness(args)
    if args.ab:
        return run(args, ab_ref = args.ab)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
