# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The EMITTER of `cell_aborted`, driven through the real `CellRunner` and a real `Recorder`.

Registering a row type in `ROW_TYPES` and emitting a hand-built copy of it in a test proves the
schema accepts the row. It does not prove the harness still WRITES it, or still writes it with the
keys the schema demands, and those are the two ways this fix can rot.

Both were confirmed reachable by mutation, against the whole suite:

    emitter omits the required `reason` key    449 tests pass, and the real CellRunner then dies
                                               on the first failed cell with `ValueError:
                                               cell_aborted row is missing required keys`
    emitter stops writing the row entirely     449 tests pass, and the orphan windows are silent
                                               again

The first is defect 10 reintroduced one field lower: a guard that crashes the run it protects,
invisible to every test. The second is defect 4 reintroduced whole -- window rows are written as
the film runs and the `cell` row when it ends, so a cell that dies leaves a complete-looking set of
windows behind with nobody owning them. Reading those without a guard reported the 1M rung at
28.7 fps against a 46.7 fps baseline, drawn entirely from a cell that never finished.

So this drives the real thing: a real `Recorder` writing a real payload, the real `CellRunner.run`,
and a `_run_inner` that raises, which is exactly the aborted-cell path. Then it reads the payload
back the way the analysis that published 28.7 fps read it -- scanning FORWARD -- and asks whether a
forward reader could have discarded the orphans without joining backwards to the cell row.
"""

from __future__ import annotations

import json
from pathlib import Path

from tests.studio.studiobench.runtime import session as session_mod
from tests.studio.studiobench.runtime.types import (
    BenchContext,
    Cell,
    Recorder,
    new_session_id,
)


class Boom(Exception):
    """Stands in for anything that kills a film halfway: a crashed tab, a watchdog, a Ctrl-C."""


def _aborted_payload(tmp_path: Path) -> list[dict]:
    """Run one cell that writes seven window rows and then dies. Return the payload rows in order."""
    rec = Recorder(tmp_path / "payload.jsonl", new_session_id())
    ctx = BenchContext(recorder = rec, log = lambda *_a, **_k: None)
    sess = session_mod.Session(ctx = ctx, instruments = [])
    cell = Cell(cell_id = "r1M.treatment.rep0", rung = "r1M", rung_tokens = 1_000_000)

    runner = object.__new__(session_mod.CellRunner)
    runner.session = sess
    runner.log = lambda *_a, **_k: None
    runner.paths = type("P", (), {"logs": tmp_path})()

    # The film gets some way in and writes its windows, exactly as the real one does.
    for i in range(7):
        rec.emit(
            {
                "row_type": "window",
                "cell_id": cell.cell_id,
                "name": f"stream:gap{i}",
                "kind": "gap",
                "t_open_ms": float(i * 100),
                "duration_ms": 33.0,
                "instruments": {"frames": {"fps": 28.7}},
            }
        )

    def _boom(_cell, _plan, _row):
        raise Boom("the film did not reach the end")

    runner._run_inner = _boom
    plan = type(
        "Plan",
        (),
        {
            "rung": "r1M",
            "seeded_chars": 1,
            "streamed_chars": 1,
            "target_chars": 2,
            "target_tokens": 1_000_000,
        },
    )()

    row = session_mod.CellRunner.run(runner, cell, plan)
    assert row["completed"] is False, "the cell was supposed to die, so the rest proves nothing"
    rec.close()
    return [
        json.loads(line)
        for line in (tmp_path / "payload.jsonl").read_text(encoding = "utf-8").splitlines()
        if line.strip()
    ]


def test_a_cell_that_dies_writes_a_terminal_cell_aborted_row(tmp_path):
    """The emitter runs at all, and the row lands AFTER the cell row it disowns."""
    rows = _aborted_payload(tmp_path)
    kinds = [r["row_type"] for r in rows]
    assert "cell_aborted" in kinds, (
        "the real CellRunner finished an aborted cell without announcing it. Every window row it "
        "already wrote is now an orphan that a forward-scanning reader will pool with completed "
        "cells, which is how the 1M rung was reported at 28.7 fps from a cell that never finished."
    )
    assert kinds.index("cell_aborted") > kinds.index("cell"), (
        "cell_aborted must be TERMINAL, after the cell row, or a reader scanning forward meets it "
        "before the thing it is disowning."
    )


def test_the_emitted_row_carries_the_keys_the_schema_demands(tmp_path):
    """Written through a real `Recorder`, so the schema check the run would hit is the one here.

    This is the mutation that survived the whole suite: drop `reason` from the emitter's dict and
    every test still passes, while the first failed cell of a real run raises out of its own
    failure handler. A guard that crashes the run it protects is worse than no guard.
    """
    rows = _aborted_payload(tmp_path)
    aborted = [r for r in rows if r["row_type"] == "cell_aborted"]
    assert len(aborted) == 1
    got = aborted[0]
    assert got["cell_id"] == "r1M.treatment.rep0"
    assert got["reason"] == "the film did not reach the end"
    assert got["kind"] == "Boom"


def test_a_forward_reader_can_discard_the_orphans_without_joining_backwards(tmp_path):
    """The property the row exists for, asserted the way the wrong number was produced.

    `floor_table.cell_metrics` guarded the orphans by joining back to the cell row. The analysis
    that published 28.7 fps did not, because it read window rows in one forward pass. This asserts
    that the same one-pass reader now has what it needs.
    """
    rows = _aborted_payload(tmp_path)
    disowned: set[str] = set()
    windows: list[dict] = []
    for r in rows:
        if r["row_type"] == "window":
            windows.append(r)
        elif r["row_type"] == "cell_aborted":
            disowned.add(r["cell_id"])

    assert len(windows) == 7
    keep = [w for w in windows if w.get("cell_id") not in disowned]
    assert keep == [], (
        "a forward reader still pooled the windows of a cell that never finished. An incomplete "
        "cell is not a shorter film but a different one: at 1M a completed cell emitted 6 "
        "stream:gap windows against 17 at 100K, so an in-flight cell contributes whichever phase "
        "it had reached rather than a uniform sample."
    )
