# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The watchdog may not kill a healthy run during the setup it is waiting for.

`--branch REF` is the advertised self-managed path: studiobench clones this repository and runs
`install.sh`, a multi-gigabyte download it allows 45 minutes for, and an A/B does that twice
before the first cell. The deadline was three times the TIER budget -- 15 minutes on fast and
quick, the two tiers people iterate with -- armed before the first install, so a slow download
tripped `os._exit(2)` during setup. It exits through `os._exit`, so the `finally` that stops the
Unsloth instances and the pacer does not run either: the run dies, leaves its processes behind, and never
reaches the measurement it advertised.

The deadline is asserted at the real entry point, by driving `run()` until the first install.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from studiobench import __main__ as cli  # noqa: E402
from studiobench.runtime import browser as browser_mod  # noqa: E402
from studiobench.runtime import lifecycle  # noqa: E402

#: What the README documents for one `install.sh`, and what `install_studio` allows it.
INSTALL_BUDGET_S = 45 * 60


class _StopBeforeInstalling(RuntimeError):
    pass


class _FakeWatchdog:
    def cancel(self) -> None:
        pass


def _armed_deadline(monkeypatch, tmp_path, argv):
    """Run the CLI for real up to the first install, and report the deadline it armed."""

    seen: list[float] = []

    def _capture(
        deadline_s,
        label = "studiobench",
        log = print,
    ):
        seen.append(float(deadline_s))
        return _FakeWatchdog()

    def _no_install(*_args, **_kwargs):
        raise _StopBeforeInstalling("the test stops here")

    def _no_attach(base_url, *_args, **_kwargs):
        raise _StopBeforeInstalling("the test stops here")

    monkeypatch.setattr(browser_mod, "install_wall_clock_watchdog", _capture)
    monkeypatch.setattr(lifecycle, "install_studio", _no_install)
    monkeypatch.setattr(lifecycle, "wait_for_healthz", _no_attach)

    args = cli.parse_args([*argv, "--out", str(tmp_path / "out")])
    with pytest.raises(_StopBeforeInstalling):
        cli.run(args, ab_ref = args.ab)
    assert len(seen) == 1
    return seen[0]


def test_a_self_managed_ab_watchdog_covers_both_installs(monkeypatch, tmp_path):
    deadline = _armed_deadline(
        monkeypatch, tmp_path, ["--tier", "fast", "--branch", "main", "--ab", "fix"]
    )
    assert deadline >= cli.TIER_BUDGET_S["fast"] * 3 + 2 * INSTALL_BUDGET_S


def test_an_attached_run_adds_no_install_budget(monkeypatch, tmp_path):
    """The control: a run that installs nothing keeps the measurement deadline it always had.

    A fast-tier A/B is two 57 second films, so the planned work is nowhere near the tier's own
    900 seconds and the tier budget still decides. The fast tier stays fast.
    """

    deadline = _armed_deadline(
        monkeypatch,
        tmp_path,
        [
            "--tier",
            "fast",
            "--attach",
            "http://127.0.0.1:5401",
            "--attach-b",
            "http://127.0.0.1:5402",
            "--ab",
            "fix",
        ],
    )
    assert deadline == cli.TIER_BUDGET_S["fast"] * 3


def _planned_film_s(tier: str, rungs: int, reps: int, arms: int) -> float:
    """The FILM alone, with no allowance for anything around it: the floor the deadline must clear."""
    from studiobench.scene.schedule import SCENES
    return SCENES[tier].duration_ms / 1000.0 * rungs * reps * arms


def test_a_standard_ab_at_four_reps_is_not_hard_exited_part_way_through(monkeypatch, tmp_path):
    """The defect. Three rungs, four reps, two arms is 24 cells of the 243 second standard film.

    That is 5,832 seconds of film before a thread is seeded, and the deadline was
    `TIER_BUDGET_S["standard"] * 3` = 3,600 seconds with nothing added for an attached pair -- so
    the watchdog fired 40% of the way through a perfectly healthy run, through `os._exit`, taking
    the `finally` that stops the Unsloth instances with it.
    """

    deadline = _armed_deadline(
        monkeypatch,
        tmp_path,
        [
            "--tier",
            "standard",
            "--reps",
            "4",
            "--attach",
            "http://127.0.0.1:5401",
            "--attach-b",
            "http://127.0.0.1:5402",
            "--ab",
            "fix",
        ],
    )
    film_s = _planned_film_s("standard", rungs = 3, reps = 4, arms = 2)
    assert film_s == pytest.approx(5832)
    assert deadline > cli.TIER_BUDGET_S["standard"] * 3
    assert deadline > film_s


def test_a_rung_override_the_tier_never_had_is_covered(monkeypatch, tmp_path):
    """`--rungs` is a ladder the tier does not describe, so the tier budget cannot describe it."""

    deadline = _armed_deadline(
        monkeypatch,
        tmp_path,
        [
            "--tier",
            "standard",
            "--rungs",
            "1K,10K,100K,500K,1M",
            "--reps",
            "3",
            "--attach",
            "http://127.0.0.1:5401",
            "--attach-b",
            "http://127.0.0.1:5402",
            "--ab",
            "fix",
        ],
    )
    assert deadline > _planned_film_s("standard", rungs = 5, reps = 3, arms = 2)


def test_the_scaled_deadline_is_still_capped(monkeypatch, tmp_path):
    """A deadline that scales without limit is not a deadline."""

    deadline = _armed_deadline(
        monkeypatch,
        tmp_path,
        [
            "--tier",
            "full",
            "--reps",
            "500",
            "--attach",
            "http://127.0.0.1:5401",
            "--attach-b",
            "http://127.0.0.1:5402",
            "--ab",
            "fix",
        ],
    )
    assert deadline == cli.WATCHDOG_MAX_MEASUREMENT_S


def test_a_single_arm_tier_ladder_keeps_its_documented_budget(monkeypatch, tmp_path):
    """Every tier's own ladder, run once against one attached Unsloth, is unchanged.

    This is the regression guard on the change: the deadline may only grow where the caller asked
    for more work than the tier describes.
    """

    for tier in ("fast", "quick", "standard", "full"):
        deadline = _armed_deadline(
            monkeypatch, tmp_path, ["--tier", tier, "--attach", "http://127.0.0.1:5401"]
        )
        assert deadline == cli.TIER_BUDGET_S[tier] * 3, tier


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
