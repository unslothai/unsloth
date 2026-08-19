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
TIERS = ("quick", "standard", "full")
TIER_RUNGS = {
    "quick": ["1K", "10K"],
    "standard": ["1K", "10K", "100K"],
    "full": ["1K", "10K", "100K", "500K", "1M"],
}
# Wall clock budgets, from the design. The deficit-scheduled pacer is what makes them honest: a
# slow machine gets the same stream duration, so it misses SLOTS rather than overrunning the tier.
TIER_BUDGET_S = {"quick": 5 * 60, "standard": 20 * 60, "full": 60 * 60}


def _log(msg: str = "") -> None:
    print(msg, flush = True)


# ── doctor ──────────────────────────────────────────────────────────

def doctor(args) -> int:
    """What is here, what is missing, and what each missing thing costs. Never raises."""
    ok = True
    _log(f"studiobench {TOOL_VERSION}")
    _log(f"  python      {sys.version.split()[0]} on {platform.system()} "
         f"{platform.machine()}")

    def check(label: str, fn, *, fatal: bool = True, cost: str = "") -> bool:
        nonlocal ok
        try:
            detail = fn()
            _log(f"  [ok]   {label}: {detail}")
            return True
        except Exception as exc:                                    # noqa: BLE001
            mark = "FAIL" if fatal else "warn"
            _log(f"  [{mark}] {label}: {type(exc).__name__}: {exc}")
            if cost:
                _log(f"         -> {cost}")
            if fatal:
                ok = False
            return False

    def _playwright():
        import subprocess
        import playwright                                           # noqa: F401
        # In a SUBPROCESS. Starting and stopping Playwright's node driver just to read the engine
        # paths leaves asyncio tearing down a cancelled task, and it prints a TargetClosedError
        # traceback to stderr after the check has already passed. A doctor that prints a
        # traceback next to `[ok]` is a doctor nobody believes.
        probe = ("from playwright.sync_api import sync_playwright\n"
                 "with sync_playwright() as pw:\n"
                 "    import pathlib\n"
                 "    out = []\n"
                 "    for n in ('chromium', 'webkit', 'firefox'):\n"
                 "        try:\n"
                 "            p = getattr(pw, n).executable_path\n"
                 "            out.append(n if pathlib.Path(p).exists() else n + ' (not installed)')\n"
                 "        except Exception:\n"
                 "            out.append(n + ' (unavailable)')\n"
                 "    print(', '.join(out))\n")
        got = subprocess.run([sys.executable, "-c", probe], capture_output = True, text = True,
                             timeout = 120)
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
        return (f"corpus_hash {c.corpus_hash[:16]}, {len(c.manifest['units'])} units "
                f"({' '.join(lines)})")

    def _registries():
        from .instruments import available, import_errors
        from .scene import action_names
        errs = import_errors()
        note = "" if not errs else f"; {len(errs)} module(s) failed to import: {list(errs)}"
        return (f"{len(action_names())} actions, "
                f"instruments {[n for n, _ in available()]}{note}")

    def _engine():
        from .runtime.browser import default_engine
        name, _, note = default_engine()
        return f"{name} -- {note}"

    check("playwright", _playwright,
          cost = "no browser can be driven; install with `pip install playwright` then "
                 "`playwright install`")
    check("psutil", _psutil, fatal = False,
          cost = "RSS is reported as null with a reason instead of a number; everything else runs")
    check("frozen corpus", _corpus,
          cost = "the corpus cannot be loaded, so no cell can be built")
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

def run(args) -> int:
    from .fixture.corpus import Corpus
    from .instruments import build as build_instruments                # noqa: F401
    from .pacer import Pacer
    from .runtime import browser as browser_mod
    from .runtime.bundle_guard import check_bundle
    from .runtime.lifecycle import (authenticate, external_checkpoint_id, install_studio,
                                    launch_studio,
                                    pacer_provider, register_provider, seed_init_script,
                                    stop_studio,
                                    wait_for_healthz)
    from .runtime.seeder import Seeder
    from .runtime.session import CellRunner, build_cells, ensure_probe_image, make_context
    from .runtime import resources
    from .runtime.types import Paths

    out = Path(args.out or f"studiobench-{args.tier}-{int(time.time())}").resolve()
    paths = Paths.under(out)
    _log(f"studiobench {TOOL_VERSION}  tier={args.tier}  out={paths.out}")

    watchdog = browser_mod.install_wall_clock_watchdog(
        TIER_BUDGET_S[args.tier] * 3, "studiobench", _log)

    corpus = Corpus.load()
    _log(f"  corpus_hash {corpus.corpus_hash}")

    install = None
    owns_studio = False
    if args.attach:
        base_url = args.attach.rstrip("/")
        _log(f"  attaching to {base_url}")
    else:
        home = Path(args.home or (out / "studio_home"))
        _log(f"  installing Studio from {args.branch} into {home} (this takes a while)")
        install = install_studio(args.branch, home)
        launch_studio(install, args.port, out / "logs" / "studio.log")
        base_url = install.base_url
        owns_studio = True
        _log(f"  Studio up at {base_url}")

    if not wait_for_healthz(base_url, 60):
        _log(f"  FATAL: {base_url}/healthz did not answer 200")
        return 2

    # ── THE GATE. Before anything else is measured. ─────────────────
    verdict = check_bundle(base_url)
    _log(f"  bundle: {verdict.reason}")
    if not verdict.production and not args.allow_dev_server:
        _log("  REFUSING TO RUN. A development build inflates the very axis under investigation")
        _log("  by about 3.2x, so a measurement here would confirm any hypothesis brought to it.")
        _log("  Pass --allow-dev-server only to demonstrate that this gate matters.")
        return 3

    pacer = Pacer().start()
    _log(f"  pacer at {pacer.base_url}")
    model_id = "studiobench-pacer"
    pacer.state.model_ids = [model_id]

    auth = authenticate(base_url, args.username,
                        args.password or (install.bootstrap_password if install else ""))
    _log(f"  authenticated as {auth.username}")

    provider = pacer_provider(pacer.base_url, [model_id])
    # Registered in the BACKEND, and the id it assigns is what the selection names. See
    # lifecycle.register_provider: a provider that exists only in localStorage renders in the
    # picker as "No longer offered" and send throws `Connection not found` without ever asking
    # for a completion.
    register_provider(base_url, auth, provider)
    checkpoint = external_checkpoint_id(provider, model_id)
    _log(f"  provider {provider.provider_type} -> {provider.base_url}, checkpoint {checkpoint}")
    init_scripts = [
        seed_init_script(auth, [provider], extra_local_storage = {
            # The SELECTION, without which nothing is ever generated. See
            # lifecycle.external_checkpoint_id.
            "unsloth_chat_last_external_checkpoint": checkpoint,
            "unsloth_chat_connections_enabled": "true",
        }),
        resources.read_text("scene/dom.js"),
    ]

    procs_before = {}
    try:
        from .instruments.rss import new_roots, snapshot_children
        procs_before = snapshot_children(os.getpid())
    except Exception:                                               # noqa: BLE001
        new_roots = None                                            # type: ignore[assignment]

    bundle = browser_mod.launch(args.engine, headless = not args.headed,
                                init_scripts = init_scripts, log = _log)
    procs = []
    if new_roots is not None:
        time.sleep(1.0)
        procs = new_roots(os.getpid(), procs_before)

    ctx, session = make_context(bundle, base_url, args.tier, args.instrument_level, paths,
                               _log, procs)
    rec = ctx.recorder
    rec.emit({"row_type": "run_meta", "tier": args.tier, "tool_version": TOOL_VERSION,
              "corpus_hash": corpus.corpus_hash,
              "studio_ref": args.branch if owns_studio else f"attached:{base_url}",
              "bundle": verdict.as_dict(), "platform": {
                  "system": platform.system(), "machine": platform.machine(),
                  "python": sys.version.split()[0], "engine": bundle.engine,
                  "engine_note": bundle.engine_note},
              "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "pacer_base_url": pacer.base_url, "cadence": args.cadence,
              "rungs": TIER_RUNGS[args.tier], "instrument_level": args.instrument_level})
    rec.gate("production_build", verdict.production, verdict.as_dict())

    seeder = Seeder(base_url = base_url, auth = auth, model_id = model_id, log = _log)
    runner = CellRunner(session = session, pacer = pacer, seeder = seeder, corpus = corpus,
                        base_url = base_url, model_id = model_id, tier = args.tier,
                        paths = paths, log = _log, cadence = args.cadence,
                        image_path = ensure_probe_image(paths))

    rungs = args.rungs.split(",") if args.rungs else TIER_RUNGS[args.tier]
    cells = build_cells(rungs, corpus, args.tier, ctx.session_id, args.instrument_level,
                        reps = args.reps)

    done = _resume_set(paths) if args.resume else set()
    if done:
        _log(f"  resuming: {len(done)} cells already in {paths.payload_jsonl.name}")

    rows = []
    try:
        for cell, plan in cells:
            if cell.cell_id in done:
                _log(f"  skipping {cell.cell_id} (already recorded)")
                continue
            rows.append(runner.run(cell, plan))
    finally:
        for inst in session.instruments:
            session._safe(inst, "detach")
        try:
            watchdog.cancel()
        except Exception:                                           # noqa: BLE001
            pass
        bundle.close()
        pacer.stop()
        if owns_studio and install is not None and not args.keep_studio:
            stop_studio(install)
        rec.close()

    _summarise(rows, paths)
    completed = sum(1 for r in rows if r.get("completed"))
    _log(f"\n{completed} of {len(rows)} cells completed. payload: {paths.payload_jsonl}")
    return 0 if completed == len(rows) and rows else 1


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
    _log(f"{'cell':<16} {'chars':>10} {'elems':>8} {'spans':>8} {'c/span':>7} "
         f"{'ran':>6} {'miss':>5} {'exp!':>5} {'busy%':>7}")
    _log("-" * 78)
    for r in rows:
        actions = r.get("actions") or []
        ran = sum(1 for a in actions if a.get("ran"))
        # The PEAK, not the end state: the film's last two actions reopen the thread and delete
        # a message, so an end-of-film census describes a thread that is no longer there.
        census = r.get("census_peak") or r.get("census_after") or {}
        _log(f"{r['cell_id']:<16} {r.get('assistant_chars_in_dom') or 0:>10,} "
             f"{census.get('elements') or 0:>8,} {census.get('highlight_spans') or 0:>8,} "
             f"{str(r.get('chars_per_span') or '-'):>7} "
             f"{ran}/{len(actions):>4} {r.get('slots_missed', 0):>5} "
             f"{r.get('expect_failures', 0):>5} "
             f"{'-' if not r.get('completed') else 'ok':>7}")
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

    declared = _rung_tokens(args.rungs.split(",")) if args.rungs else _rung_tokens(
        TIER_RUNGS[args.tier])
    try:
        text, ladder, _payload = build_report(path, declared)
    except Exception as exc:                                        # noqa: BLE001
        # A payload that cannot be scored is reported as such rather than half-rendered: a
        # partial report is exactly the artefact that gets quoted without its caveats.
        _log(f"could not build a report from {path}: {type(exc).__name__}: {exc}")
        return 1

    print(text)
    out = path.parent / "summary.md"
    out.write_text(text, encoding = "utf-8")
    _log(f"summary written to {out}")
    return 0


def main(argv: list) -> int:
    ap = argparse.ArgumentParser(
        prog = "studiobench",
        description = "A real-path performance benchmark for Unsloth Studio.")
    ap.add_argument("--tier", choices = TIERS, default = "quick",
                    help = "quick ~5min (1K,10K), standard ~20min (+100K), full ~60min (+500K,1M)")
    ap.add_argument("--doctor", action = "store_true",
                    help = "report what is installed and what each missing piece costs")
    ap.add_argument("--attach", metavar = "URL",
                    help = "drive a Studio that is already running instead of installing one")
    ap.add_argument("--resume", action = "store_true",
                    help = "skip cells already completed in the output payload")
    ap.add_argument("--ab", metavar = "REF",
                    help = "A/B a second ref, interleaved within one session (not yet wired)")
    ap.add_argument("--report", metavar = "PAYLOAD",
                    help = "score and render an existing payload.jsonl, then exit. Runs offline, "
                           "so a payload mailed in from another machine reports here")
    ap.add_argument("--rungs", help = "comma-separated rung override, e.g. 1K,10K")
    ap.add_argument("--reps", type = int, default = 1)
    ap.add_argument("--instrument-level", type = int, default = 0, choices = [0, 1, 2, 3],
                    help = "0 is the only level headline numbers may come from")
    ap.add_argument("--cadence", default = "field", choices = ["field", "fast"],
                    help = "field is 24 chars every 73ms, the rate of the captured reply")
    ap.add_argument("--engine", choices = ["chromium", "webkit", "firefox"],
                    help = "default matches the platform's desktop webview family")
    ap.add_argument("--branch", default = "main", help = "Studio ref to install when not attaching")
    ap.add_argument("--home", help = "UNSLOTH_STUDIO_HOME for an install")
    ap.add_argument("--port", type = int, default = 5399)
    # `unsloth`, not `admin`. Studio's first run prints "DEFAULT ADMIN ACCOUNT CREATED / username:
    # unsloth", and the wrong one answers 401 with a message about resetting the PASSWORD, which
    # sends you looking in the wrong place.
    ap.add_argument("--username", default = "unsloth")
    ap.add_argument("--password", default = "")
    ap.add_argument("--out", help = "output directory")
    ap.add_argument("--headed", action = "store_true")
    ap.add_argument("--keep-studio", action = "store_true")
    ap.add_argument("--allow-dev-server", action = "store_true",
                    help = "run against a development build anyway. ONLY to demonstrate that the "
                           "production gate matters: React's dev build inflates the axis under "
                           "investigation by about 3.2x")
    args = ap.parse_args(argv)

    if args.doctor:
        return doctor(args)
    if args.report:
        return report_only(args)
    if args.ab:
        _log("--ab is declared in the CLI but not yet wired. Layer 3 owns the arm ladder.")
        return 2
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
