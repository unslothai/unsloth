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

TOOL_VERSION = "0.1.0"
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
        return got.stdout.strip()

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


def run(args, ab_ref = None) -> int:
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
    from .runtime.session import CellRunner, build_cells, ensure_probe_image, make_context
    from .runtime import resources
    from .runtime.types import Paths

    out = Path(args.out or f"studiobench-{args.tier}-{int(time.time())}").resolve()
    paths = Paths.under(out)
    _log(f"studiobench {TOOL_VERSION}  tier={args.tier}  out={paths.out}")

    watchdog = browser_mod.install_wall_clock_watchdog(
        TIER_BUDGET_S[args.tier] * 3, "studiobench", _log
    )

    corpus = Corpus.load()
    _log(f"  corpus_hash {corpus.corpus_hash}")

    # READ NOW, BEFORE ANYTHING IS STARTED. The probe's source is not needed until the browser is
    # launched, but reading it there means a path typo raises after Studio and the pacer are up
    # and before the cleanup `finally` that would stop them is entered, so a detached Studio keeps
    # running and holds its port. The cheapest correct fix is to fail while there is nothing to
    # clean up: a missing, unreadable or non-UTF-8 file is a mistake in the invocation, and the
    # right moment to say so is the first second of the run.
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

    # One spec per side. Without --ab there is exactly one and everything below is the old path.
    specs = [("base", args.branch, args.attach, args.port)]
    if ab_ref:
        specs.append(("treatment", ab_ref, args.attach_b, args.port + 1))
        if args.attach and not args.attach_b:
            _log("  --ab with --attach needs --attach-b URL: the second build has to be somewhere.")
            return 2
        _log(f"  A/B: base={args.branch} vs treatment={ab_ref}, interleaved in ONE session")

    installs = []
    sides = []
    for label, ref, attach, port in specs:
        if attach:
            side_url = attach.rstrip("/")
            side_install, owns = None, False
            _log(f"  {label}: attaching to {side_url}")
        else:
            home = Path(args.home or (out / f"studio_home_{label}"))
            _log(f"  {label}: installing Studio from {ref} into {home} (this takes a while)")
            side_install = install_studio(ref, home)
            launch_studio(side_install, port, out / "logs" / f"studio_{label}.log")
            side_url, owns = side_install.base_url, True
            _log(f"  {label}: Studio up at {side_url}")
        installs.append((side_install, owns))
        sides.append({"label": label, "ref": ref, "base_url": side_url})

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
    for index, side in enumerate(sides):
        side_install = installs[index][0]
        side_auth = authenticate(
            side["base_url"],
            args.username,
            args.password or (side_install.bootstrap_password if side_install else ""),
        )
        _log(f"  {side['label']}: authenticated as {side_auth.username}")

        # BOTH sides register the SAME pacer, so the bytes on the wire are identical by
        # construction rather than by two configurations that are meant to agree.
        side_provider = pacer_provider(pacer.base_url, [model_id])
        # Registered in the BACKEND, and the id it assigns is what the selection names. See
        # lifecycle.register_provider: a provider that exists only in localStorage renders in the
        # picker as "No longer offered" and send throws `Connection not found` without ever asking
        # for a completion.
        register_provider(side["base_url"], side_auth, side_provider)
        side_checkpoint = external_checkpoint_id(side_provider, model_id)
        _log(
            f"  {side['label']}: provider {side_provider.provider_type} -> "
            f"{side_provider.base_url}, checkpoint {side_checkpoint}"
        )
        side["auth"] = side_auth

        seed = seed_init_script(
            side_auth,
            [side_provider],
            extra_local_storage = {
                # The SELECTION, without which nothing is ever generated. See
                # lifecycle.external_checkpoint_id.
                "unsloth_chat_last_external_checkpoint": side_checkpoint,
                "unsloth_chat_connections_enabled": "true",
            },
        )
        # Origin-gated even in the single-target case, so the one-build and two-build paths are
        # the same code and the gate cannot rot while unused.
        init_scripts.append(origin_scoped(side["base_url"], seed))

        if getattr(args, "inject_stream_cost_ms", None) and side is not sides[0]:
            # VALIDATION, not a measurement mode. Burns a known amount of main-thread time per SSE
            # chunk on the TREATMENT side only, so an A/B whose two arms are otherwise the same
            # build has a known answer. It is origin-gated like the seed above, because a context
            # init script fires on every document and burning on both sides would inject the cost
            # into the control as well and read back a recovery of zero.
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

    # ONE init script, not four, and the order inside it is the order of this list.
    #
    # Playwright is explicit that "the order of evaluation of multiple scripts installed via
    # browserContext.addInitScript() and page.addInitScript() is not defined", so a dependency
    # between two entries of `init_scripts` is a dependency on nothing. That was already load
    # bearing before any probe existed: surfaces.js reads what dom.js and parity.js put on
    # `window.__sb`, and it was relying on list order to find them. Concatenation makes the
    # sequence a property of the string rather than of an undocumented scheduler.
    #
    # AN EXTERNAL PROBE OR ABLATION ARM goes LAST in that string, so it can wrap anything the
    # scene scripts define. One environment variable rather than a CLI flag per experiment, and
    # WITH THE VARIABLE UNSET NOTHING IS APPENDED: the run is byte-identical to a run of a tree
    # that does not have this hook. That property is the point. A potency probe perturbs the page
    # it observes, so the probe run and the scored run have to be different runs of one harness,
    # and the only safe way to arrange that is for the probe to be absent by default.
    #
    # surfaces.js is loaded unconditionally, even without --surfaces. It defines selectors and
    # never runs on its own. Making the page's JS depend on a CLI flag would mean the flag changes
    # what is on the page during the FILM as well, and the film's numbers must not depend on
    # whether a later phase was asked for.
    page_scripts = [
        resources.read_text("scene/dom.js"),
        resources.read_text("scene/parity.js"),
        resources.read_text("scene/surfaces.js"),
    ]
    if extra_init:
        page_scripts.append(_isolated_probe(extra_init, extra_init_source))
        _log(
            f"  EXTRA INIT SCRIPT: {extra_init} -- this run carries an external probe and is NOT "
            f"a clean measurement of the build"
        )
    # `;` between them: every one of these files is an IIFE, and a file whose last line has no
    # terminator would otherwise splice into the next one's opening parenthesis.
    init_scripts.append("\n;\n".join(page_scripts))

    procs_before = {}
    try:
        from .instruments.rss import new_roots, snapshot_children
        procs_before = snapshot_children(os.getpid())
    except Exception:  # noqa: BLE001
        new_roots = None  # type: ignore[assignment]

    bundle = browser_mod.launch(
        args.engine, headless = not args.headed, init_scripts = init_scripts, log = _log
    )
    # THE RETURN PATH for a probe installed by the hook above. Studio ships `connect-src 'self'`,
    # so a beacon to a collector on another port is blocked by CSP before it leaves the page, and
    # the payload schema has no row for a one-off probe. The console is what is left. Lines are
    # filtered on a caller-supplied prefix so they can be recovered from the run log by exact
    # match, and so a probe cannot drown the log in the app's own console traffic.
    console_prefix = os.environ.get("SBENCH_PAGE_CONSOLE")
    if console_prefix:
        bundle.page.on(
            "console",
            lambda m: _log(f"  [page] {m.text}") if m.text.startswith(console_prefix) else None,
        )
    if extra_init:
        # A probe that throws on load is the same silence as a probe that was never installed, and
        # the console filter above cannot show it because a failing probe never gets as far as
        # printing its own prefix. Attached only when a probe was asked for, so an ordinary run is
        # unchanged. `console.error` from the isolation wrapper arrives here as a console message
        # rather than a page error, so both channels are listened to.
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

    ctx, session = make_context(
        bundle, base_url, args.tier, args.instrument_level, paths, _log, procs
    )
    rec = ctx.recorder
    rec.emit(
        {
            "row_type": "run_meta",
            "tier": args.tier,
            "tool_version": TOOL_VERSION,
            "corpus_hash": corpus.corpus_hash,
            "studio_ref": args.branch if owns_studio else f"attached:{base_url}",
            "bundle": verdict.as_dict(),
            "platform": {
                "system": platform.system(),
                "machine": platform.machine(),
                "python": sys.version.split()[0],
                "engine": bundle.engine,
                "engine_note": bundle.engine_note,
            },
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "pacer_base_url": pacer.base_url,
            "cadence": args.cadence,
            "rungs": TIER_RUNGS[args.tier],
            "instrument_level": args.instrument_level,
            # In the payload, not only in the log. Two runs with different fixtures are
            # not comparable, and a fixture difference that is not recorded is one a later
            # reader has no way to notice before quoting a ratio across it.
            "stream_tail_chars": args.stream_tail_chars,
            "corpus_dollars": bool(args.corpus_dollars),
            # WHICH PROBE, IF ANY, WAS IN THE PAGE. Recorded next to the corpus hash and for the
            # same reason: a payload nobody can audit against the page it measured has to be
            # taken on trust. `null` is the normal case and the only scorable one.
            "probe_init_script": extra_init or None,
        }
    )
    rec.gate("production_build", verdict.production, verdict.as_dict())

    if args.tier == "fast":
        # Said in the log AND recorded in the payload. A fast-tier reading is a DIRECTION, not a
        # number: it runs one rung, a 47 s film and however few repetitions the caller asked for,
        # so its detection floor is wider than the standard tier's and it has no null control of
        # its own unless one is run alongside. The gate exists so the analysis layer can refuse to
        # pool a fast payload with a standard one -- a fast reading quoted against a standard
        # floor is the single most likely way this tier gets somebody a wrong answer.
        _log("")
        _log("  FAST TIER: for iteration while you are changing something, not for reporting.")
        _log("  One rung (100K), a 47s film. Use it to see whether a fix moved anything at all,")
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
            base_url = side["base_url"],
            model_id = model_id,
            tier = args.tier,
            paths = paths,
            log = _log,
            cadence = args.cadence,
            image_path = image_path,
        )

    seeder = sides[0]["seeder"]
    runner = sides[0]["runner"]

    if args.surfaces:
        _sweep_surfaces(sides, ctx, paths)

    rungs = args.rungs.split(",") if args.rungs else TIER_RUNGS[args.tier]
    cells = build_cells(
        rungs,
        corpus,
        args.tier,
        ctx.session_id,
        args.instrument_level,
        reps = args.reps,
        stream_tail_chars = args.stream_tail_chars,
        corpus_dollars = args.corpus_dollars,
    )
    if args.stream_tail_chars or args.corpus_dollars:
        # Loud, because both change the fixture. A payload produced under either of them is not
        # comparable with one produced without, and the pair that says so is printed here and
        # written into the run manifest above.
        _log(
            f"  FIXTURE CHANGED: stream tail {args.stream_tail_chars or 'default'}, "
            f"dollars {'on' if args.corpus_dollars else 'off'}. Compare only against a run "
            f"with the same pair."
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
                "balanced": order_is_balanced(work),
                "order": [c.cell_id for _t, c, _p in work],
            }
        )
    else:
        work = [(None, cell, plan) for cell, plan in cells]

    rows = []
    try:
        for target, cell, plan in work:
            if cell.cell_id in done:
                _log(f"  skipping {cell.cell_id} (already recorded)")
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
        for side_install, side_owns in installs:
            if side_owns and side_install is not None and not args.keep_studio:
                stop_studio(side_install)
        rec.close()

    if ab_ref:
        _render_ab(paths, sides, ctx.session_id, corpus.corpus_hash)

    _summarise(rows, paths)
    completed = sum(1 for r in rows if r.get("completed"))
    _log(f"\n{completed} of {len(rows)} cells completed. payload: {paths.payload_jsonl}")
    return 0 if completed == len(rows) and rows else 1


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


def _isolated_probe(path: str, source: str) -> str:
    """The external script, ordered with the scene scripts but PARSED ON ITS OWN.

    Concatenation bought a guarantee Playwright does not give (it says the evaluation order of
    several init scripts is undefined) and sold a worse one in exchange: the browser parses the
    joined text as ONE unit, so a single syntax error anywhere in the external file means dom.js,
    parity.js and surfaces.js never execute either. The run then has no `window.__sb` at all, the
    probe prints nothing, and the two failures are indistinguishable from an arm that did not fire.

    So the source is embedded as a STRING and evaluated at runtime. The outer unit parses whatever
    the file contains, the scene scripts run first because they are earlier in the same unit, and a
    malformed probe degrades to a caught, reported `SyntaxError` instead of taking the harness with
    it. Indirect eval (`(0, eval)`) so the source evaluates in global scope, exactly as a separate
    init script would.
    """
    return (
        "(function () {\n"
        "  try {\n"
        f"    (0, eval)({json.dumps(source)});\n"
        "  } catch (err) {\n"
        f"    var where = {json.dumps(path)};\n"
        "    try {\n"
        "      window.console.error(\n"
        "        'SBENCH_EXTRA_INIT_SCRIPT ' + where + ' failed to evaluate: ' + err +\n"
        "        '. The scene scripts are unaffected, but this probe reported nothing, which is '"
        " +\n"
        "        'NOT the same as an arm that did not fire.'\n"
        "      );\n"
        "    } catch (ignored) {}\n"
        "  }\n"
        "})();\n"
    )


def _render_ab(paths, sides, session_id: str, corpus_hash: str) -> None:
    """Render the A/B table from the payload the run just wrote.

    Read back from disk rather than kept in memory on purpose: it is the same path a tester takes
    with `--report`, so the table nobody checks and the table everybody reads are produced by one
    piece of code.
    """
    from .report.render import render_ab_table
    from .runtime.ab import compare_arms

    records = []
    with paths.payload_jsonl.open(encoding = "utf-8") as fh:
        for line in fh:
            try:
                records.append(json.loads(line))
            except ValueError:
                continue

    # Detected, not declared. `--ab main` against the same Studio IS a null control whether or not
    # the caller says so, and a null control that renders as an ordinary A/B invites somebody to
    # quote "7.7% faster" from a build compared with itself.
    is_null = sides[0]["ref"] == sides[1]["ref"] and sides[0]["base_url"] == sides[1]["base_url"]
    label = (
        f"null control: {sides[0]['ref']} vs itself"
        if is_null
        else f"{sides[0]['ref']} -> {sides[1]['ref']}"
    )

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

    text = render_ab_table(result)
    print("\n" + text)
    out = paths.out / "ab.md"
    out.write_text(text, encoding = "utf-8")
    _log(f"A/B table written to {out}")
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


def _resume_set(paths) -> set:
    done = set()
    if not paths.payload_jsonl.exists():
        return done
    with paths.payload_jsonl.open(encoding = "utf-8") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except ValueError:
                continue
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


def _rung_tokens(labels: list) -> list:
    """`1K` -> 1000. The scoring ladder is indexed by tokens; the CLI speaks in rung labels."""
    mult = {"K": 1_000, "M": 1_000_000}
    out = []
    for label in labels:
        text = str(label).strip().upper()
        out.append(int(float(text[:-1]) * mult[text[-1]]) if text[-1] in mult else int(text))
    return out


def report_only(args) -> int:
    """Score and render a payload that already exists. No browser, no Studio, no network.

    Separate from `run` so a payload produced on somebody else's desktop reports identically here,
    which is the whole point of shipping a single-file benchmark: the numbers come back as a file
    and the analysis happens where the analyst is.
    """
    from .report.build import build_report

    path = Path(args.report)
    if not path.exists():
        _log(f"no payload at {path}")
        return 2

    declared = (
        _rung_tokens(args.rungs.split(",")) if args.rungs else _rung_tokens(TIER_RUNGS[args.tier])
    )
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


def assert_liveness(args) -> int:
    """Fail unless every scheduled action in a payload actually ran.

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
    cells, problems, missed = 0, [], []
    for line in path.read_text(encoding = "utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            problems.append("a payload line is not valid JSON")
            continue
        if row.get("row_type") != "cell":
            continue
        cells += 1
        where = row.get("cell_id", "?")
        if row.get("completed") is False:
            problems.append(f"{where}: the cell did not complete")
        for action in row.get("actions") or []:
            name = action.get("action") or action.get("name") or "?"
            if name in allowed:
                continue
            if action.get("slot_missed"):
                # A machine-speed fact, whether or not the action also reports ran=False.
                missed.append(f"{where}: {name} missed its slot ({action.get('reason') or '?'})")
            elif not action.get("ran"):
                problems.append(f"{where}: {name} NOT RUN ({action.get('reason') or 'no reason'})")

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


def main(argv: list) -> int:
    ap = argparse.ArgumentParser(
        prog = "studiobench", description = "A real-path performance benchmark for Unsloth Studio."
    )
    ap.add_argument(
        "--tier",
        choices = TIERS,
        default = "quick",
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
        help = "drive a Studio that is already running instead of installing one",
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
        help = "the treatment side's already-running Studio, when --ab is used "
        "together with --attach",
    )
    ap.add_argument(
        "--report",
        metavar = "PAYLOAD",
        help = "score and render an existing payload.jsonl, then exit. Runs offline, "
        "so a payload mailed in from another machine reports here",
    )
    ap.add_argument(
        "--assert-liveness",
        metavar = "PAYLOAD",
        dest = "assert_liveness",
        help = "exit non-zero unless every scheduled action in an existing "
        "payload.jsonl actually ran. Offline. This is the gate that catches an "
        "action which never fired reporting as 'no effect'",
    )
    ap.add_argument(
        "--allow-not-run",
        metavar = "ACTIONS",
        dest = "allow_not_run",
        help = "comma-separated action names --assert-liveness may excuse. Use only "
        "for an action a platform genuinely cannot perform, and say which in "
        "the pull request: every name here is a hole in the gate",
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
    ap.add_argument("--branch", default = "main", help = "Studio ref to install when not attaching")
    ap.add_argument("--home", help = "UNSLOTH_STUDIO_HOME for an install")
    ap.add_argument("--port", type = int, default = 5399)
    # `unsloth`, not `admin`. Studio's first run prints "DEFAULT ADMIN ACCOUNT CREATED / username:
    # unsloth", and the wrong one answers 401 with a message about resetting the PASSWORD, which
    # sends you looking in the wrong place.
    ap.add_argument("--username", default = "unsloth")
    ap.add_argument("--password", default = "")
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

    if args.doctor:
        return doctor(args)
    if args.report:
        return report_only(args)
    if args.assert_liveness:
        return assert_liveness(args)
    if args.ab:
        return run(args, ab_ref = args.ab)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
