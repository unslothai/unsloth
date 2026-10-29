# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Entry point; heavy imports are lazy, so --help and --doctor work with nothing installed."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Optional

# Bump whenever what an instrument MEASURES changes; it is the only key field separating such runs.
TOOL_VERSION = "0.2.0"
TIERS = ("fast", "quick", "standard", "full")
TIER_RUNGS = {
    # 100K on purpose: 10K separated none of six PRs from the null control.
    "fast": ["100K"],
    "quick": ["1K", "10K"],
    "standard": ["1K", "10K", "100K"],
    "full": ["1K", "10K", "100K", "500K", "1M"],
}
# The deficit-scheduled pacer makes a slow machine miss slots rather than overrun the tier.
TIER_BUDGET_S = {"fast": 5 * 60, "quick": 5 * 60, "standard": 20 * 60, "full": 60 * 60}


def _log(msg: str = "") -> None:
    print(msg, flush = True)


def engines_installed(probe_text: str) -> list:
    """Engines the doctor's probe reports present; a name with a parenthesised note is not installed."""
    out = []
    for part in str(probe_text).split(","):
        name = part.strip()
        if name and "(" not in name:
            out.append(name)
    return out


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

        # In a subprocess: starting Playwright's driver here leaves a TargetClosedError traceback.
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
        # The package is not the engine: `playwright install` is a separate step. Reported, not fatal.
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


def _windowed_arms(spec: str, labels: list) -> set:
    """Checks --windowed-arm names before anything starts, so a typo cannot leak running Unsloth servers."""
    names = {name.strip() for name in (spec or "").split(",") if name.strip()}
    unknown = names - set(labels)
    if unknown:
        raise SystemExit(
            f"--windowed-arm names {sorted(unknown)}, which is not an arm in this run "
            f"({sorted(labels)})"
        )
    return names


def side_home(explicit, out, label: str, *, ab: bool) -> Path:
    """A/B sides never share UNSLOTH_STUDIO_HOME: a shared one lets the treatment overwrite the base."""
    if not explicit:
        return Path(out) / f"studio_home_{label}"
    return Path(explicit) / label if ab else Path(explicit)


def side_specs(args, ab_ref) -> list:
    """Each A/B side has its own password (--password, --password-b); Unsloth mints one per home."""
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
    """Checks --rungs before install; unknown labels fail, whitespace and lowercase suffixes pass."""
    if not args.rungs:
        return list(TIER_RUNGS[args.tier])
    # Imported here so `--help` works on a machine with nothing installed.
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


# Per-cell cost besides the film (seeding, navigation, calibration, drain, censuses).
CELL_OVERHEAD_S = 60
SURFACE_SWEEP_S = 120
WATCHDOG_MARGIN = 3
# Absolute cap so `--reps 1000` cannot arm a deadline measured in months.
WATCHDOG_MAX_MEASUREMENT_S = 24 * 60 * 60


def planned_work_s(
    tier: str,
    rungs: list,
    reps: int,
    arms: int,
    surfaces: bool = False,
) -> float:
    """Measurement wall clock from the plan: fixed scene duration times one cell per rung, rep and arm."""
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
    """3x the larger of tier and planned measurement time, capped, plus install budget per owned side."""
    from .runtime.lifecycle import INSTALL_TIMEOUT_S

    owned = sum(1 for spec in specs if not spec[2])
    arms = max(1, len(specs))
    planned = planned_work_s(
        tier, list(rungs) if rungs else list(TIER_RUNGS[tier]), reps, arms, surfaces
    )
    measurement = max(TIER_BUDGET_S[tier] * WATCHDOG_MARGIN, planned * WATCHDOG_MARGIN)
    return min(measurement, WATCHDOG_MAX_MEASUREMENT_S) + INSTALL_TIMEOUT_S * owned


def completion_exit_code(rows: list, resumed: int = 0) -> int:
    """Exit 0 when every cell is complete, resumed or new; an empty run with nothing resumed fails."""
    completed = sum(1 for r in rows if r.get("completed"))
    if not rows and not resumed:
        return 1
    return 0 if completed == len(rows) else 1


def is_null_control(sides: list) -> bool:
    """Same URL is a null control; owned sides are null when their commits match, not their ref names."""
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
    """A/B table title; names both commits when two installs of one ref resolved to different builds."""
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
    """Each side's browser origin; spellings like http://studio:80 and http://studio are one origin."""
    from .runtime.ab import browser_origin
    return [
        (browser_origin(attach) if attach else f"http://127.0.0.1:{port}")
        for _label, _ref, attach, port, _password in specs
    ]


def stream_cost_injection_problem(specs: list, inject_ms) -> str | None:
    """Refuses --inject-stream-cost-ms when both arms resolve to one origin; the difference reads zero."""
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
    """Stops only this run's own Unsloth instances; attached ones stay up, and --keep-studio stops none."""
    if keep:
        return
    for side_install, side_owns in installs:
        if side_owns and side_install is not None:
            stop(side_install)


def run(args, ab_ref = None) -> int:
    """Takes the OutDirLock before any clone, build or launch, so a busy --out is refused untouched."""
    from .runtime.types import OutDirLock, Paths

    # Check arm names before any process or directory exists. Pinned by test_studiobench_windowed_arm_names.
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

    # Read before anything starts: raising after launch would leave a detached server on its port.
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

    # Every A/B argument check precedes `prepare_payload`, or a refusal costs the previous payload's path.
    if ab_ref:
        if args.attach and not args.attach_b:
            _log("  --ab with --attach needs --attach-b URL: the second build has to be somewhere.")
            return 2
        injection_problem = stream_cost_injection_problem(
            specs, getattr(args, "inject_stream_cost_ms", None)
        )
        if injection_problem:
            _log(f"  {injection_problem}")
            return 2
        # ONE HOME CANNOT HOLD TWO BUILDS: `install_studio` derives the checkout from the home, so two
        # arms sharing a home share one checkout: the second install overwrites the first and both
        # arms then serve whichever build was installed last, so the A/B compares a build with itself.
        # Measured: two runs of the same pair, equal within each and 3.6x apart between them.
        # The pair read 716 ms and 718 ms within one run.
        if not args.attach and args.home:
            _log(
                "  --home cannot be used with --ab: both arms would install into that one "
                "directory, the second install would overwrite the first, and both arms would "
                "then serve the same build. Drop --home and each arm gets its own "
                "studio_home_<label> under --out."
            )
            return 2

    # Before the first install, so a refusal costs milliseconds, not two builds.
    archived = prepare_payload(
        paths, requested_identity(args, ab_ref, corpus.corpus_hash), resume = bool(args.resume)
    )

    # Stale reports the archive left behind are invalidated here on every output dir reuse.
    invalidate_stale_reports(paths.out, archived = archived, extra_init = extra_init)

    # Armed after the sides are known; the budget is `watchdog_deadline_s`.
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
    # Stop every launched side if setup fails or returns: a stale server would answer the next run's
    # `wait_for_healthz` from the wrong build.
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

        # `checkout_ref` is the only place a ref becomes a commit.
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

        # THE GATE, on both sides: a development build on one arm inflates it ~3.2x.
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
        # One provider per origin: `register_provider` is idempotent by display name, so registering per side
        # on one Unsloth would delete the id the first side's seed captured. See `is_null_control`.
        registered: dict = {}
        for index, side in enumerate(sides):
            side_install = installs[index][0]
            side_auth = authenticate(
                side["base_url"],
                args.username,
                side["password"] or (side_install.bootstrap_password if side_install else ""),
            )
            _log(f"  {side['label']}: authenticated as {side_auth.username}")

            # Both sides register the same pacer, so wire bytes are identical by construction.
            side_origin = side["base_url"].rstrip("/")
            shared = registered.get(side_origin)
            if shared is None:
                side_provider = pacer_provider(pacer.base_url, [model_id])
                # Must be registered in the backend: a localStorage-only provider throws
                # `Connection not found`.
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
                            # See lifecycle.external_checkpoint_id.
                            "unsloth_chat_last_external_checkpoint": cp,
                            "unsloth_chat_connections_enabled": "true",
                        },
                    ),
                )

            # Origin-gated even single-target, so the gate cannot rot while unused.
            side["seed_script"] = _side_seed
            init_scripts.append(_side_seed(side_auth))

            if getattr(args, "inject_stream_cost_ms", None) and side is not sides[0]:
                # Validation only: known per-chunk main-thread cost on the treatment side. Origin-gated.
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

        # One `add_init_script` per scene script: a parse error in a concatenation stops all three.
        init_scripts.append(resources.read_text("scene/dom.js"))
        init_scripts.append(resources.read_text("scene/parity.js"))
        init_scripts.append(resources.read_text("scene/surfaces.js"))

        # External probe or ablation arm; with it unset nothing is appended. Script order is undefined.
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
        # Unsloth's `connect-src 'self'` CSP blocks beacons, so probes report via a console prefix.
        console_prefix = os.environ.get("SBENCH_PAGE_CONSOLE")
        if console_prefix:
            bundle.page.on(
                "console",
                lambda m: _log(f"  [page] {m.text}") if m.text.startswith(console_prefix) else None,
            )
        if extra_init:
            # A wrapper's `console.error` is not a page error, so listen on both channels.
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

        # Init scripts carry the access token and re-run on every navigation, so the seed script defers
        # to the later `exp` after a re-mint.
        for side in sides:
            side["auth"].on_rotate = lambda auth_now, side = side: bundle.context.add_init_script(
                side["seed_script"](auth_now)
            )

        # The ladder-ratio check must be able to restore the payload to this mark.
        mark = payload_mark(paths.payload_jsonl)
        ctx, session = make_context(
            bundle,
            base_url,
            args.tier,
            args.instrument_level,
            paths,
            _log,
            procs,
            # Adopted: `run()` already holds this directory's lock.
            out_lock = out_lock,
        )
        rec = ctx.recorder
        # Recorded, not re-derived: `--report` reads it to decide which rungs a payload owes.
        rungs = planned_rungs(args)
        meta_row = {
            "row_type": "run_meta",
            "tier": args.tier,
            "tool_version": TOOL_VERSION,
            "corpus_hash": corpus.corpus_hash,
            "studio_ref": args.branch if owns_studio else f"attached:{base_url}",
            # Lets `--resume` tell a continuation from a moved branch. See `commit_problems`.
            "studio_commit": sides[0].get("commit") or "",
            "bundle": verdict.as_dict(),
            "platform": {
                "system": platform.system(),
                "machine": platform.machine(),
                # `platform.machine()` is the architecture; `platform.node()` is what tells machines apart.
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
            "stream_tail_chars": args.stream_tail_chars,
            "corpus_dollars": bool(args.corpus_dollars),
            "probe_init_script": extra_init or None,
            # Changes what the cell measures without moving its id, so `--resume` must refuse a toggle.
            "click_probe": bool(getattr(args, "click_probe", False)),
            # Marks an arm the harness perturbed; `--resume` reads it as an identity axis.
            "inject_stream_cost_ms": getattr(args, "inject_stream_cost_ms", None),
            # Since Playwright 1.57 headed is `chrome` and headless is `chrome-headless-shell`, which renders
            # 2-3x slower and in software, so record which.
            "headed": bool(getattr(args, "headed", False)),
        }
        rec.emit(meta_row)

        from .scoring import payload_rules

        # The comparability key, keyed on the computed corpus hash. `floor_table.load` is the refusal.
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
            # A fast-tier reading is a direction, not a number, so analysis must refuse to pool it.
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
        # A probe perturbs the payload, so it is gated rather than left to convention.
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
        # Windowed readiness is per side, never global.
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

        # Sized by the measured ratio, before cells are built: rungs in tokens are planned in characters.
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
            _log(
                f"  FIXTURE CHANGED: stream tail {args.stream_tail_chars or 'default'}, "
                f"dollars {'on' if args.corpus_dollars else 'off'}. Compare only against a run "
                f"with the same pair."
            )
        if cells:
            ladder_ratio = cells[0][0].meta["ladder_chars_per_token"]
            rec.gate("ladder_ratio_measured", not ladder_ratio["provisional"], ladder_ratio)

            if args.resume:
                ratio_issues: list = []
                for recorded in recorded_identities(paths.payload_jsonl):
                    for problem in ladder_ratio_problems(recorded, ladder_ratio["chars_per_token"]):
                        if problem not in ratio_issues:
                            ratio_issues.append(problem)
                if ratio_issues:
                    # Runs after the Recorder opened the payload, so the refusal must roll back this session's rows.
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
                # With odd reps one side always runs first, so monotonic drift lands on the other.
                _log(
                    "  WARNING: the run order is not balanced (use an even --reps). Linear drift "
                    "within the session is charged to whichever side runs second."
                )
            rec.emit(
                {
                    "row_type": "ab_plan",
                    "base_ref": sides[0]["ref"],
                    "treatment_ref": sides[1]["ref"],
                    # Identifies an attached treatment server; empty when this run installed it.
                    # See `requested_identity`.
                    "treatment_url": "" if sides[1]["owns"] else sides[1]["base_url"],
                    "treatment_commit": sides[1].get("commit") or "",
                    "balanced": order_is_balanced(work),
                    "order": [c.cell_id for _t, c, _p in work],
                }
            )
        else:
            work = [(None, cell, plan) for cell, plan in cells]

        if done:
            # A resumed A/B re-runs incomplete pairs whole.
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
        # Cells this session was asked to measure: all planned cells or none, which _render_ab checks.
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
    """Runs before the cells on an empty chat, so surface digests never include the film's messages."""
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
            # A raising sweep is recorded as a failed gate: rows alone make it look the same as an empty sweep.
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
        # Passes only if every non-conditional surface was reached and the digests are scoped.
        passed = manifest["not_reached_hard"] == 0 and manifest["digests_scoped"]
        ctx.recorder.gate(f"surface_sweep:{label}", passed, manifest)


def _probe_init_scripts(path: str, source: str) -> list[str]:
    """Not eval'd (WebKit's CSP refuses it); the stamp goes after the source to keep 'use strict' live."""
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
    """Renders the A/B table from the payload on disk; no verdict while any planned cell is unmeasured."""
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

    # No table at all for a probe run; a warning would scroll off when ab.md is pasted.
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
        # Overwritten: `--resume` reuses the directory, so an old clean ab.md would read as this run's.
        stale = paths.out / "ab.md"
        if stale.exists():
            stale.write_text(f"# No A/B table\n\n{reason}\n", encoding = "utf-8")
            _log(f"  a previous {stale} was replaced by this refusal")
        _log("")
        return

    # A fully resumed A/B measured nothing this session. Must stay after the probe refusal.
    if not any(r.get("row_type") == "cell" and r.get("session_id") == session_id for r in records):
        if out.exists():
            _log(f"\nno cell ran in this session; keeping the A/B table already at {out}")
            return

    # Detected, not declared: `--ab main` is a null control either way.
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


# Payload identity: `cell_id` is only `r{rung}.{arm}.rep{rep}`, so these axes decide whether a
# recorded cell matches. `rungs`/`reps` only add cells.
IDENTITY_AXES = (
    "tier",
    "cadence",
    "engine",
    # `TOOL_VERSION` moves only when an instrument's measurement changes.
    "tool_version",
    "instrument_level",
    "corpus_hash",
    "studio_ref",
    "treatment_ref",
    "treatment_url",
    # Both change the streamed reply without moving the id.
    "stream_tail_chars",
    "corpus_dollars",
    # `--click-probe` makes timings incomparable.
    "click_probe",
    # Treatment-only perturbation; otherwise a resume could answer the recovery gate from uninjected cells.
    "inject_stream_cost_ms",
    # An env var survives in a shell; a probed resume would make prior clean cells unscorable.
    "probe_init_script",
    # Headed and headless are different executables with different rendering.
    "headed",
)

TREATMENT_AXES = ("treatment_ref", "treatment_url")

# Absence is a reading for these: older payloads ran under the default by construction.
HISTORICAL_DEFAULTS = {
    "stream_tail_chars": None,
    "corpus_dollars": False,
    "click_probe": False,
    "inject_stream_cost_ms": None,
    "probe_init_script": None,
    "headed": False,
}

# Not an identity axis: unknown until `build_cells` measures the corpus; see `ladder_ratio_problems`.
LADDER_RATIO_AXIS = "ladder_chars_per_token"

# Half a step of the 0.001 grid separates a re-read from a different measurement.
LADDER_RATIO_TOLERANCE = 5e-4

# `Cell.arm` in `runtime.types`.
SINGLE_ARM = "A0"

# A ref resolves afresh on every install, so `commit_problems` checks these after launch.
COMMIT_AXES = ("studio_commit", "treatment_commit")


def requested_identity(args, ab_ref, corpus_hash: str) -> dict:
    """studio_ref spelling must match run_meta so requested and recorded identities compare."""
    from .runtime.browser import default_engine

    base_ref = f"attached:{args.attach.rstrip('/')}" if args.attach else args.branch
    attach_b = (getattr(args, "attach_b", "") or "").rstrip("/")
    return {
        "tier": args.tier,
        "cadence": args.cadence,
        "tool_version": TOOL_VERSION,
        # The engine that will render, resolved as `browser.launch` does via `default_engine`.
        "engine": getattr(args, "engine", "") or default_engine()[0],
        "instrument_level": args.instrument_level,
        "corpus_hash": corpus_hash,
        "studio_ref": base_ref,
        "treatment_ref": ab_ref or "",
        # The attached treatment is identified by server URL, not its free-form label.
        "treatment_url": attach_b if (ab_ref and attach_b) else "",
        "stream_tail_chars": args.stream_tail_chars,
        "corpus_dollars": bool(args.corpus_dollars),
        "click_probe": bool(getattr(args, "click_probe", False)),
        "inject_stream_cost_ms": getattr(args, "inject_stream_cost_ms", None),
        # Through `bool`, as `run_meta` records it, or every resume refuses itself.
        "headed": bool(getattr(args, "headed", False)),
        # From the environment: the probe hook is a variable, not a flag.
        "probe_init_script": os.environ.get("SBENCH_EXTRA_INIT_SCRIPT") or None,
    }


def recorded_identities(payload_path) -> list:
    """Identity per recorded session, read from existing rows; axes a row never declared are skipped."""
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
                # Declared by the cells a session wrote; a run that died before `ab_plan` declares no mode.
                arm = str((row.get("cell") or {}).get("arm") or row.get("arm") or "")
                if arm:
                    by_session[session].setdefault("mode", "single" if arm == SINGLE_ARM else "ab")
                # Ratio the session's rungs were sized by, read off a cell it wrote; see ladder_ratio_problems.
                sized = ((row.get("cell") or {}).get("meta") or {}).get("ladder_chars_per_token")
                if isinstance(sized, dict) and sized.get("chars_per_token") is not None:
                    by_session[session].setdefault(
                        LADDER_RATIO_AXIS, float(sized["chars_per_token"])
                    )
                continue
            for axis in IDENTITY_AXES + COMMIT_AXES:
                if axis in row:
                    by_session[session][axis] = row[axis]
            # The engine is nested under `run_meta.platform`; a session that never launched declares none.
            if row_type == "run_meta":
                engine = str((row.get("platform") or {}).get("engine") or "")
                if engine:
                    by_session[session]["engine"] = engine
            if row_type == "ab_plan" and row.get("treatment_ref") is not None:
                by_session[session]["treatment_ref"] = row["treatment_ref"]
    return [by_session[s] for s in order if by_session[s]]


def identity_problems(recorded: dict, requested: dict) -> list:
    """Every axis on which a recorded session and this invocation disagree."""
    problems = []
    # An A/B and a single run may not share a payload: reports keep the first reading per rung.
    requested_mode = "ab" if requested.get("treatment_ref") else "single"
    recorded_mode = recorded.get("mode")
    if recorded_mode is not None and recorded_mode != requested_mode:
        names = {"ab": "an A/B (base against treatment)", "single": "one build on its own"}
        problems.append(
            f"mode: the payload was recorded by a run measuring {names[recorded_mode]}, "
            f"this run measures {names[requested_mode]}"
        )
    both_ab = bool(requested.get("treatment_ref")) and bool(recorded.get("treatment_ref"))
    for axis in IDENTITY_AXES:
        declared = axis in recorded
        if declared:
            got = recorded[axis]
        elif axis in HISTORICAL_DEFAULTS:
            # This axis postdates the payload, so it ran under the default. See `HISTORICAL_DEFAULTS`.
            got = HISTORICAL_DEFAULTS[axis]
        else:
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
    """Commits this run installed from; empty for attached sides, whose build the harness cannot see."""
    out = {axis: "" for axis in COMMIT_AXES}
    for axis, side in zip(COMMIT_AXES, sides):
        if side.get("owns"):
            out[axis] = str(side.get("commit") or "")
    return out


def commit_problems(recorded: dict, resolved: dict) -> list:
    """Resume must not mix builds: an empty commit on either side is not a difference."""
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
    """Refuses a resume when chars-per-token differs from the payload's, so each rung keeps its load."""
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
    """Moves a non-empty payload aside, never truncates it: append mode is for validated resumes only."""
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
    """Byte length of the payload before this session appends; pairs with rollback_session_rows."""
    try:
        return Path(payload_path).stat().st_size
    except OSError:
        return 0


def rollback_session_rows(
    payload_path,
    mark: int,
    log = _log,
) -> int:
    """Truncates to mark: a refused resume's rows would otherwise leave the payload scored incomplete."""
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
    """Overwrites summary.md and ab.md when their payload was archived or a probe ran."""
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
    """Runs before install: fresh runs archive the payload, resumes refuse an identity mismatch."""
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
    """Skips only cells whose latest attempt completed, so a later failed retry is re-run."""
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
        # Only completed cells are skipped: a death may have been the machine, not the build.
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
        # The peak: the film ends by reopening and deleting, so the end-state census is stale.
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
    # `census_peak` is chosen by a max() over racing censuses, so it is not the same moment on two arms.
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
    """Folds every run_meta, since a resume can add rungs the first header never promised; [] if none."""
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
    """Scores an existing payload offline, so one produced on another machine reports identically here."""
    from .report.build import build_report

    path = Path(args.report)
    if not path.exists():
        _log(f"no payload at {path}")
        return 2

    # An explicit `--tier` still wins; the default tier must not shorten a recorded ladder.
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
        # `SystemExit` is not an `Exception`; the refusal must reach the artefact, overwriting summary.md.
        _log(str(exc))
        if out.exists():
            out.write_text(f"# No summary\n\n{exc}\n", encoding = "utf-8")
            _log(f"  a previous {out} was replaced by this refusal")
        return 2
    except Exception as exc:  # noqa: BLE001
        _log(f"could not build a report from {path}: {type(exc).__name__}: {exc}")
        return 1

    print(text)
    out.write_text(text, encoding = "utf-8")
    _log(f"summary written to {out}")
    return 0


def compare_payloads(args) -> int:
    """Names the field that stops two payloads being comparable; floor_table.load guards only pooling."""
    from .report.payload import read_records
    from .scoring import payload_rules

    metas = []
    for raw in args.compare:
        path = Path(raw)
        if not path.exists():
            print(f"no such payload: {path}")
            return 2
        # Every header: `--resume` appends `run_meta` rows and may extend the ladder. Torn last lines are
        # counted by `report.read_records`, not raised.
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
    """Scene problems always fail; missed slots are machine facts, failing only past --allow-slot-misses."""
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

    # Judge a re-run cell on the attempt that finished it.
    from .scoring.from_payload import ATTEMPT_ROW_TYPES, latest_attempt_rows

    # A SIGKILL skips the `finally` that writes the cell row, so an attempt without one is a failure.
    # `latest_attempt_rows` is what has to notice.
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
                # A missed slot (`ran=False, slot_missed=True`) is judged against `--allow-slot-misses`, not
                # `--allow-not-run`.
                missed.append(f"{where}: {name} missed its slot ({action.get('reason') or '?'})")
            elif not action.get("ran"):
                # `--allow-not-run` only excuses actions the fixture cannot mount at all.
                if name in allowed:
                    continue
                problems.append(f"{where}: {name} NOT RUN ({action.get('reason') or 'no reason'})")
            elif action.get("expect_ok") is False:
                # `ran=True` with `expect_ok=False` is a failed assertion; scoring and report refuse it,
                # so must this.
                problems.append(
                    f"{where}: {name} ran but its own assertion failed "
                    f"({action.get('reason') or 'no reason'})"
                )

    if cells == 0:
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
        # No default: `--report` must distinguish an explicit ladder from a CLI default.
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
        # Env fallback: `scripts/pr_perf_sweep.py` builds this command line and is shared by live sweeps.
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
    # `unsloth`, not `admin`: the wrong username answers 401 about the password.
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
