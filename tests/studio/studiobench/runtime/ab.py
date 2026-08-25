# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Run two Studio builds against each other INSIDE ONE SESSION.

WHY THIS CANNOT BE TWO RUNS. Cross-session drift on this app measured 8%, which is larger than
most of the wins anybody argues about. `scoring.ab.assert_comparable` refuses two different
session ids outright, and that refusal is correct: running the base today and the treatment
tomorrow produces a ratio whose dominant term is the machine, not the change. So both builds are
installed up front, one browser drives both, and the cells alternate.

WHY THE ORDER FLIPS. Anything that drifts monotonically within a session -- thermal throttling, a
browser heap that never quite shrinks, another process ramping up -- is charged entirely to
whichever side runs second if the order is fixed. Alternating (base, treatment) on even reps and
(treatment, base) on odd ones cancels the linear part of that term instead of measuring it. With
`--reps 1` the order cannot be balanced, which is why a single-rep A/B prints the warning it does.

WHAT VOIDS THE RESULT. The null control -- the base build compared against ITSELF, interleaved the
same way -- runs first and must land inside its own noise band. If comparing a build to itself
produces a difference, then a difference between two builds means nothing, and no table is worth
printing. That check is the whole reason this file interleaves at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Any, Callable, Optional

from ..fixture.corpus import Corpus, RungPlan
from .types import Cell


@dataclass
class Target:
    """One side of the comparison: a Studio to drive and everything needed to drive it."""

    label: str  # "base" or "treatment"
    ref: str  # the git ref, for the report
    base_url: str
    seeder: Any
    runner: Any  # a CellRunner bound to this target's base_url and seeder
    install: Any = None  # StudioInstall, when we own it
    owns_studio: bool = False


#: The port a scheme does not spell out. `window.location.origin` omits it, so an attach URL that
#: writes it is the same origin under a different name.
DEFAULT_PORTS = {"http": 80, "https": 443, "ws": 80, "wss": 443}


def browser_origin(url: str) -> str:
    """The ORIGIN a browser computes for `url`, spelled the way `window.location.origin` spells it.

    THE ONLY THING THAT DISCRIMINATES THE TWO ARMS. Both are driven by one browser context and one
    page, so `origin_scoped`'s `window.location.origin !== <url>` is the whole of the gate, and the
    right-hand side of that comparison has to be what the browser will actually produce rather than
    what the caller typed. Measured in chromium against real documents:
    `http://studio:80` and `HTTP://STUDIO` and `http://studio/app` all report an origin of
    `http://studio`, so every one of those spellings gates a script onto a document that does not
    exist. The failure is silent in both directions -- the base's seed then runs on the treatment's
    documents as well, and the treatment's injection runs on neither -- and it reaches
    `evaluate_stream_cost_recovery_gate` as a recovery of zero blamed on the accumulator.

    WHAT IS NOT FOLDED TOGETHER: `localhost` and `127.0.0.1`. A browser treats those as two
    origins, chromium reports them as two (`http://localhost:8000` stays `http://localhost:8000`),
    and a check that called them one would refuse a perfectly good pair of arms. Only the four
    canonicalisations the URL standard itself performs are applied: the scheme and host are
    lower-cased, a port the scheme implies is dropped, and the path, query, fragment and any
    userinfo are discarded.

    A string this cannot parse as an absolute URL is returned trailing-slash-stripped, which is
    what the acquisition loop does with `--attach` anyway: an unparseable URL is the caller's
    problem to see at `wait_for_healthz`, not a reason for this to guess.
    """
    from urllib.parse import urlsplit

    try:
        split = urlsplit(url.strip())
        host, port = split.hostname, split.port
    except ValueError:
        return url.rstrip("/")
    scheme = split.scheme.lower()
    if not scheme or not host:
        return url.rstrip("/")
    host = host.lower()
    if ":" in host:  # IPv6, which serialises with its brackets
        host = f"[{host}]"
    if port is None or port == DEFAULT_PORTS.get(scheme):
        return f"{scheme}://{host}"
    return f"{scheme}://{host}:{port}"


def origin_scoped(base_url: str, script: str) -> str:
    """Run `script` only on its own Studio's origin.

    `add_init_script` fires on every document in the context, and localStorage is per-origin under
    the SAME KEY NAMES on both builds. Seeding both unconditionally means whichever script runs
    last writes the other build's auth token into this build's storage, and the failure shows up
    much later as a logged-out SPA or a provider that renders as "No longer offered" -- neither of
    which points back here.

    Gated on the CANONICAL origin rather than on the URL as typed: see `browser_origin` for the
    spellings that otherwise gate a script onto a document no browser will ever produce.
    """
    import json as _json
    return (
        "(() => { if (window.location.origin !== "
        + _json.dumps(browser_origin(base_url))
        + ") return; "
        + script
        + " })();"
    )


def interleave(
    cells: list[tuple[Cell, RungPlan]], targets: list[Target]
) -> list[tuple[Target, Cell, RungPlan]]:
    """Order the work so the two sides sit next to each other in time, not in separate halves.

    Adjacent in time is the point: the closer two paired readings are, the less of whatever the
    machine is doing separates them. Returned as a flat list so the caller's loop stays a loop
    and the ordering decision lives in one testable function.
    """
    out: list[tuple[Target, Cell, RungPlan]] = []
    for cell, plan in cells:
        order = list(targets) if cell.rep % 2 == 0 else list(reversed(targets))
        for target in order:
            out.append((target, cell.derive(arm = target.label), plan))
    return out


def skippable_cells(work: list[tuple[Any, Cell, RungPlan]], done: set) -> set:
    """Of the cells `--resume` COULD skip, the ones it may: only whole `(rung, rep)` pairs.

    A PAIR IS THE UNIT OF AN A/B, NOT A CELL. An interruption between the two adjacent cells of one
    pair is the ordinary way a run stops, and skipping the arm that completed then measured its
    partner ALONE in the new session. That lone reading can never be used: `readings_by_arm` scopes
    the comparison to one session on purpose -- cross-session drift measured 8%, larger than most
    wins anybody argues about -- so the completed arm from the old session is dropped, the new
    arm has nothing to pair with, and the run pays a full cell for a number no table can contain.
    What it renders instead is that repetition missing from the table, or NO READING with an exit
    code of 0 underneath it.

    So a pair is skipped only when EVERY arm of it is already complete, and otherwise both arms are
    re-run -- adjacent in time, in one session, which is the only arrangement `interleave` exists to
    produce. The old attempt stays in the payload and `latest_attempt_rows` supersedes it, exactly
    as it already does for a cell that died.

    AND A COMPARISON IS ALL OF ITS PAIRS, which is the same argument one step further out. The
    session filter that drops a lone completed arm drops a WHOLE completed pair for exactly the
    same reason, and `_render_ab` then prints a headline and a VERDICT over whatever is left
    without saying which rungs are missing: an interrupted standard tier whose 10K pair had
    recorded a 30% regression published `VERDICT: IMPROVED (20.0% faster)` off the 100K pair alone,
    with nothing in `ab.md` mentioning 10K at all. `render.py` rules out fixing that with a note
    above the table -- the table gets screenshotted and the warning does not -- and declining to
    publish costs the same wall clock as re-running while yielding no table. So an A/B with ANY
    work left re-runs EVERY pair in the new session, at most one tier budget, which is the budget
    the interrupted run was already paying.

    A FINISHED A/B still skips everything: nothing runs, and `_render_ab` keeps the table that run
    already wrote. And a legitimate extension -- `--resume --rungs 1K,10K` over a finished 1K run
    -- now produces a complete two-rung table instead of a 10K-only one.

    Single-target work is a degenerate case of the same rule: each pair holds one cell, so nothing
    changes for a run without `--ab`, whose ladder `report.build.score_payload` reads across
    sessions anyway.
    """
    by_pair: dict[tuple[str, int], list[str]] = {}
    for _target, cell, _plan in work:
        by_pair.setdefault((str(cell.rung), int(cell.rep)), []).append(str(cell.cell_id))
    out: set = set()
    for cell_ids in by_pair.values():
        if all(cell_id in done for cell_id in cell_ids):
            out.update(cell_ids)
    # A pair carrying more than one arm is a comparison, and a comparison is scoped to one session.
    # Partial skipping is right for a single-target ladder and wrong for a ratio.
    if any(len(cell_ids) > 1 for cell_ids in by_pair.values()):
        planned = sum(len(cell_ids) for cell_ids in by_pair.values())
        if len(out) != planned:
            return set()
    return out


def order_is_balanced(plan: list[tuple[Target, Cell, RungPlan]]) -> bool:
    """True when each side ran first equally often, so linear drift cancels rather than lands.

    Reported rather than enforced: an unbalanced plan is still worth running, it just carries a
    drift term that the reader has to be told about instead of discovering later.
    """
    labels = {target.label for target, _cell, _plan in plan}
    first_counts: dict[str, int] = {label: 0 for label in labels}
    seen: set[str] = set()
    for target, cell, _plan in plan:
        key = f"{cell.rung}:{cell.rep}"
        if key in seen:
            continue
        seen.add(key)
        first_counts[target.label] += 1
    # Every label is seeded at zero first. Counting only the labels that DID run first reports a
    # single-rep plan -- where one side always goes first and nothing cancels -- as balanced,
    # which is the one answer this function exists to prevent.
    return len(labels) > 1 and len(set(first_counts.values())) == 1


#: The per-cell gates whose failure means the cell's TIMINGS ARE NOT A READING OF THE BUILD, and
#: therefore the only ones that may take the whole cell out of the ratios.
#:
#: NAMED RATHER THAN "ANY FAILED GATE", because a per-cell gate is not automatically fatal and the
#: one that is not says so itself. `timer_clamp` fails whenever idle calibration cannot establish a
#: floor -- an overloaded machine, or simply the frames instrument not being loaded -- and
#: `session.py` is explicit that this is "NOT fatal, and NOT silently zero": blocked time is a
#: subtraction against that floor, so `busy_pct` is null with the reason attached AND EVERY OTHER
#: COLUMN STANDS. Excluding the cell for it would delete keystroke latency, frame and census
#: readings that were measured correctly, and would do it most often on exactly the machines least
#: able to spare a repetition.
#:
#: The two below are different in kind: both say the FILM ITSELF was wrong. A thread that lost
#: messages and a reply that stopped being rendered do not produce a suspect column, they produce a
#: cheaper cell, and there is no metric in it that can be trusted afterwards.
INVALIDATING_CELL_GATES: frozenset[str] = frozenset({"thread_complete", "follows_the_stream"})


def gate_detail_is_unmeasured(detail: Mapping[str, Any]) -> bool:
    """Did this failed gate row report a MISSING READING rather than a FAILING BUILD?

    ONE DEFINITION FOR BOTH ADMISSION LISTS. `INVALIDATING_CELL_GATES` above was centralised so
    the scorers could not drift into disagreeing about what invalidates a cell; the predicate that
    waives a row is the same decision one level down, and `sweep/ui_parity.py` applies it to the
    same rows for the DOM side. Two copies of it drift the same way the gate names would have, and
    the drift is invisible because each copy looks locally correct.

    THREE PRODUCERS, all meaning "the instrument did not answer", none meaning "the arm lost
    something":

    `follow_attempted: False` is `_read_follow` reporting that the page-side sampler is not
    installed, and `probe_attempted: False` is `probe_thread_completeness` reporting the same for
    `window.__sb.dom`. `pinned`/`coverage` are then None and the row says `passed: False`. That is
    an absent INSTRUMENT, not a film that went wrong -- the same thing `timer_clamp` is kept off
    the list above for -- and reading it as fatal would mark every cell of every run unusable
    wherever the harness is not loaded, a far larger blast radius than the defect being closed.

    `stream_coverage_unmeasured: True` is the third and the one that was standing open.
    `attached_fraction_of_stream` is fixed by the scene schedule rather than by the build, so it
    is identical on both arms by construction and cancels in every comparison drawn from these
    cells; see the block in `session.py` that writes it for the measurements. Left fatal it voided
    the entire A/B table -- VERDICT: NO READING -- for a reason that has nothing to do with the two
    builds being compared, and did so on the null control as readily as on a real pair. The cell's
    timings still stand: both arms rendered the same share of the same film.

    NARROWED TO THE INSTRUMENT, because `probe_attempted: False` has two producers and only one of
    them is an absent instrument. `window.__sb.dom is not installed` is the harness not being
    loaded and is waived. `no thread viewport` is the ARM missing the surface the film measures,
    which is a defect about the build, and waiving it let a real failure ride the instrument
    allowance.

    This does NOT relax the `unmeasured` COVERAGE VERDICT of the completeness probe, which is a
    different value and stays fatal: `record_completeness_gate` refuses to score a cell whose probe
    RAN and could not answer, because "we could not tell" must not be recorded as "it was fine".
    The case waived here is the probe never having run at all.
    """

    unmeasured = (
        detail.get("follow_attempted") is False
        or detail.get("probe_attempted") is False
        or detail.get("stream_coverage_unmeasured") is True
    )
    return unmeasured and "viewport" not in str(detail.get("reason") or "").lower()


def failed_invalidating_gates(records: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """`{cell_id: why}` for every cell carrying a FAILED INVALIDATING per-cell gate row.

    Shared with `report/build.py` and `sweep/floor_table.py`, which are the other two scorers that
    admit a cell, so the three cannot drift into disagreeing about what invalidates one.

    RUN-LEVEL GATES ARE NOT IN HERE. `production_build` and `reportable_tier` are emitted without a
    `cell_id`, and they are properties of the whole run: reading them as per-cell would empty both
    arms and turn a fast-tier A/B into an empty table rather than the table it asked for. Only a
    gate that named a cell can disqualify that cell.

    ATTEMPTS ARE SCOPED BY HAND because `latest_attempt_rows` cannot do it: `ATTEMPT_ROW_TYPES` is
    `{cell, action, window}`, so a gate row survives the filter that drops the rest of a superseded
    attempt. `--resume` reuses the cell id, so without this a cell that failed its gate, was re-run
    and PASSED would stay disqualified by the dead attempt's row -- the retry silently unable to
    count, which is the mirror of the defect this function is fixing. The winning attempt is the
    session the surviving cell row carries; a row without a session id predates the stamp and is
    kept, as `latest_attempt_rows` keeps it.
    """
    winning: dict[str, Any] = {}
    for row in records:
        if row.get("row_type") == "cell" and row.get("cell_id") is not None:
            winning[str(row.get("cell_id"))] = row.get("session_id")

    failed: dict[str, str] = {}
    for row in records:
        if row.get("row_type") != "gate" or row.get("passed") is not False:
            continue
        name = str(row.get("name"))
        if name not in INVALIDATING_CELL_GATES or row.get("cell_id") is None:
            continue
        cell_id = str(row.get("cell_id"))
        keep = winning.get(cell_id)
        if keep is not None and row.get("session_id") not in (None, keep):
            continue
        detail = row.get("detail") if isinstance(row.get("detail"), dict) else {}
        # NOT MEASURED IS NOT FAILED. See `gate_detail_is_unmeasured`, which is shared with
        # `sweep/ui_parity.py` so the two admission lists cannot drift apart on it. Readiness now
        # refuses a cell with no thread viewport outright, so that narrowing is the second of two
        # doors on the same hole.
        if gate_detail_is_unmeasured(detail):
            continue
        why = detail.get("reason") or detail.get("coverage_reason") or "the cell's own self-check"
        failed.setdefault(cell_id, f"gate {name}: {why}")
    return failed


def unmeasured_planned_cells(
    records: list[dict],
    planned: Sequence[str],
    session_id: Optional[str] = None,
) -> list[str]:
    """The planned cells this session has no completed reading for, in plan order.

    A COMPARISON IS ALL OF ITS PAIRS, and a cell that failed removes its HEALTHY PARTNER from the
    table too: `readings_by_arm` drops the incomplete cell, and `compare_arms` intersects the two
    arms' keys, so the surviving pairs are a subset chosen by which cell happened to die. The
    hazard is the one `skippable_cells` describes for an interrupted resume, arriving by the other
    road -- `CellRunner.run` catches the exception and returns an incomplete row, so the run
    continues and `_render_ab` is reached with a hole in the plan. A 10K base cell that died
    published `VERDICT: IMPROVED (20.0% faster)` off the 100K pair alone while the completed 10K
    pair underneath it was a 26.5% regression, and `ab.md` named neither the missing rung nor the
    failure.

    The exit code is already nonzero when a cell fails, and it is not enough: `ab.md` outlives the
    process and is what gets quoted. `render_ab_table` prints no numbers at all when a result is
    void, for the reason stated there -- the table gets screenshotted and the warning does not --
    and an incomplete plan is voided on the same grounds.
    """
    from ..scoring.from_payload import latest_attempt_rows

    # THE SAME TWO FILTERS `readings_by_arm` APPLIES, because this function exists to notice the
    # holes that one punches. It drops a cell for `completed is not True` AND for a failed
    # invalidating gate; reading only the first left the second kind of hole invisible. A cell that
    # completed but lost its thread's middle, or whose reply stopped being rendered, is removed
    # from the ratios here and takes its healthy partner with it through the arm intersection in
    # `compare_arms`, while this said the plan was whole -- so `ab.md` published a verdict over
    # the rungs that survived instead of the VOID that is the point of the guard. That is the
    # partial-plan selection bias, arriving by the gate road instead of the crash road.
    failed = failed_invalidating_gates(records)
    complete: set = set()
    for row in latest_attempt_rows(records):
        if row.get("row_type") != "cell" or row.get("completed") is not True:
            continue
        if session_id is not None and row.get("session_id") not in (None, session_id):
            continue
        if str(row.get("cell_id")) in failed:
            continue
        complete.add(str(row.get("cell_id")))
    return [str(cell_id) for cell_id in planned if str(cell_id) not in complete]


def readings_by_arm(
    records: list[dict], session_id: Optional[str] = None
) -> dict[str, dict[int, dict]]:
    """Split one payload's cell rows into `{arm: {rung_tokens: {metric: Measure}}}`.

    A CELL THAT DID NOT COMPLETE IS NOT AN ARM'S READING. The ladder scores an incomplete cell on
    purpose -- a build that dies at 500K is the most important thing the run has to say -- but a
    ratio is a different question. An arm that crashed after emitting one action row still carries
    that row's timings, and pairing them against a completed cell on the other side turns a crash
    into a win: a treatment cell holding nothing but a 50 ms keystroke, measured against a
    completed 100 ms base cell, reports IMPROVED. The crash is still in the payload, in the
    summary and in `excluded_cells`; it is only kept out of the ratios.

    A CELL THAT FAILED A PER-CELL GATE IS NOT ONE EITHER, for the same reason and by a shorter
    route. `thread_complete` and `follows_the_stream` are advisory at the point they are emitted:
    `record_completeness_gate`'s verdict is discarded by its caller and the film runs on, so the
    cell reaches this function with `completed=True` and a full set of timings. Those timings are
    CHEAPER THAN A CORRECT CELL'S, and cheaper in the direction that flatters the arm -- a thread
    that lost its middle renders fewer rows, and a streamed reply that left the viewport and was
    unmounted stops costing anything to paint. Pairing one against a complete cell on the other
    side reports the defect as an improvement, which is the crash-into-a-win failure again with a
    gate row instead of a missing one.

    `excluded_from_rows` does not cover this path. It reads the same failed gate rows into
    `excluded_cells`, but that block is derived, rendered and schema-checked and nothing filters
    on it: stripping the failing gate rows out of a payload and re-scoring produces byte-identical
    metric lines. `ab.md` is scored here, from `readings_by_arm` and `measures_by_cell`, and
    neither consulted a gate row before this.

    `session_id`, when given, keeps the comparison inside ONE session. `--resume` appends to the
    payload a previous run wrote, so an interrupted A/B resumed into the same directory otherwise
    hands `compare_arms` cells from two browser sessions -- the 8% cross-session drift term that
    `assert_comparable` exists to refuse, arriving through the back door because both sides are
    labelled with the CURRENT session id.

    Deferred import: `scoring` pulls in the anchor table and this module is imported by the CLI
    before a run, where that cost buys nothing.
    """
    from ..scoring.from_payload import latest_attempt_rows, measures_by_cell

    # The session filter below scopes the CELL rows, but `action` and `window` rows are collected
    # by `cell_id` alone, and a resumed retry reuses the cell id of the attempt that died. Without
    # this the completed-cell filter admitted the dead attempt's windows into the retry's reading.
    records = list(latest_attempt_rows(records))
    failed_gates = failed_invalidating_gates(records)

    arms: dict[str, list[dict]] = {}
    cell_ids: dict[str, set[str]] = {}
    for row in records:
        if row.get("row_type") == "cell":
            if row.get("completed") is not True:
                continue
            if str(row.get("cell_id")) in failed_gates:
                continue
            if session_id is not None and row.get("session_id") not in (None, session_id):
                continue
            arm = str((row.get("cell") or {}).get("arm") or row.get("arm") or "")
            if arm:
                cell_ids.setdefault(arm, set()).add(str(row.get("cell_id")))

    for arm, ids in cell_ids.items():
        subset = [
            r
            for r in records
            if r.get("row_type") not in {"cell", "action", "window"} or str(r.get("cell_id")) in ids
        ]
        arms[arm] = subset

    return {arm: measures_by_cell(rows) for arm, rows in arms.items()}


def compare_arms(
    records: list[dict],
    base_label: str,
    treatment_label: str,
    *,
    bench_version: str,
    corpus_hash: str,
    session_id: str,
    label: str,
    noise_floor_pct: Optional[float] = None,
    noise_floor_source: str = "declared default",
    is_null_control: bool = False,
) -> Any:
    """Build the A/B result for one pair of arms out of an already-recorded payload."""
    from ..scoring.ab import DEFAULT_NOISE_FLOOR_PCT, Pair, RunIdentity, compare
    from ..scoring.anchors import METRIC_BY_KEY, weights_id

    by_arm = readings_by_arm(records, session_id = session_id)
    base = by_arm.get(base_label, {})
    treatment = by_arm.get(treatment_label, {})

    rung_ladder_id = _ladder_id(sorted({rung for rung, _rep in set(base) | set(treatment)}))
    identity_kwargs = dict(
        bench_version = bench_version,
        corpus_hash = corpus_hash,
        rung_ladder_id = rung_ladder_id,
        weights_id = weights_id() if callable(weights_id) else str(weights_id),
        session_id = session_id,
    )
    # Paired PER REPETITION, matching (rung, rep) on both sides. Repetition r of the base and
    # repetition r of the treatment ran adjacent in time, so pairing them is what makes the
    # comparison paired at all; pooling reps into one reading per rung throws away every
    # observation but the first and leaves the bootstrap with nothing to resample.
    pairs = []
    for key in sorted(set(base) & set(treatment)):
        rung, _rep = key
        for metric_key in METRIC_BY_KEY:
            base_measure = base[key].get(metric_key)
            treatment_measure = treatment[key].get(metric_key)
            if base_measure is None or treatment_measure is None:
                continue
            pairs.append(
                Pair(
                    rung_tokens = int(rung),
                    metric_key = metric_key,
                    base = base_measure,
                    treatment = treatment_measure,
                )
            )

    return compare(
        label,
        pairs,
        RunIdentity(**identity_kwargs),
        RunIdentity(**identity_kwargs),
        noise_floor_pct = (DEFAULT_NOISE_FLOOR_PCT if noise_floor_pct is None else noise_floor_pct),
        noise_floor_source = noise_floor_source,
        is_null_control = is_null_control,
    )


def _ladder_id(rungs: list) -> str:
    import hashlib
    digest = hashlib.sha256(",".join(str(int(r)) for r in rungs).encode()).hexdigest()[:12]
    return f"r-{digest}"


def make_target(
    label: str,
    ref: str,
    base_url: str,
    *,
    pacer,
    model_id: str,
    corpus: Corpus,
    tier: str,
    paths,
    log: Callable[[str], None],
    cadence: str,
    image_path,
    session,
    parity_raw: bool = False,
    parity_shots = None,
    username: str,
    password: str,
) -> Target:
    """Authenticate against one Studio, register the shared pacer on it, and bind a runner.

    Both sides talk to the SAME pacer, so the bytes on the wire are identical by construction
    rather than by two configurations that are meant to match.
    """
    from .lifecycle import authenticate, external_checkpoint_id, pacer_provider, register_provider
    from .seeder import Seeder
    from .session import CellRunner

    auth = authenticate(base_url, username, password)
    provider = pacer_provider(pacer.base_url, [model_id])
    register_provider(base_url, auth, provider)
    checkpoint = external_checkpoint_id(provider, model_id)
    log(f"  {label}: {base_url} -> pacer {pacer.base_url}, checkpoint {checkpoint}")

    seeder = Seeder(base_url = base_url, auth = auth, model_id = model_id, log = log)
    runner = CellRunner(
        session = session,
        pacer = pacer,
        seeder = seeder,
        corpus = corpus,
        base_url = base_url,
        model_id = model_id,
        tier = tier,
        paths = paths,
        log = log,
        cadence = cadence,
        image_path = image_path,
        parity_raw = parity_raw,
        parity_shots = parity_shots,
        arm_label = label,
    )
    target = Target(label = label, ref = ref, base_url = base_url, seeder = seeder, runner = runner)
    target.auth = auth  # type: ignore[attr-defined]
    target.checkpoint = checkpoint  # type: ignore[attr-defined]
    return target
