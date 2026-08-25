# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""What the kernel builder generates and what the launcher can read back.

The transport, not the payloads: the generated driver and payload cells are
EXECUTED here with Kaggle replaced by a stub, so a control-flow mistake in a
cell that only runs on a Kaggle T4 is caught on a runner. No network call, no
credential and no GPU -- `subprocess` is swapped out wholesale for the driver
cells, and the payload cells executed stop before they reach torch.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import types
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_DIR = REPO_ROOT / "tests" / "kaggle" / "t4_smoke"
CI_DIR = REPO_ROOT / ".github" / "scripts" / "kaggle_t4_ci"

sys.path.insert(0, str(SMOKE_DIR))
sys.path.insert(0, str(CI_DIR))

import build_kernel  # noqa: E402
import gate  # noqa: E402
import launch  # noqa: E402
from legs import KERNELS, LEGS  # noqa: E402


# ------------------------------------------------------------------ driver


class _Stub:
    """Stands in for `subprocess` while the generated driver cells run.

    Answers the four commands the driver issues (GPU probe, `which uv`, venv
    build, papermill) and records what papermill was handed, which is where the
    per-payload isolation is either present or not.
    """

    def __init__(
        self,
        *,
        gpus: int,
        venv_ok: bool = True,
    ):
        self.gpus = gpus
        self.venv_ok = venv_ok
        self.papermill: list[dict] = []
        # Every `pip install --target ...` the driver issued, which is the only
        # place an overlay's CONTENTS are decided. Recorded as the full command
        # so a guard can ask what was installed, not merely that something was.
        self.overlay_installs: list[list[str]] = []
        # What the resolver is pretended to have found. Deliberately mixed: one
        # ordinary pure-Python distribution and one native one, so the driver's
        # deny-list has something real to reject. A stub closure of only safe
        # packages would let a driver with no deny-list at all pass.
        self.resolver_closure = [
            ("transformers", "4.57.6"),
            ("trl", "0.22.2"),
            ("torch", "2.99.0"),
        ]
        self.TimeoutExpired = subprocess.TimeoutExpired
        self.CalledProcessError = subprocess.CalledProcessError
        self.STDOUT = subprocess.STDOUT

    def run(self, cmd, **kw):
        cmd = [str(c) for c in cmd]
        if cmd[0] == "nvidia-smi":
            if self.gpus < 0:
                raise OSError("nvidia-smi is not on this box")
            out = "".join("Tesla T4, 15360 MiB\n" for _ in range(self.gpus))
            return types.SimpleNamespace(returncode = 0, stdout = out, stderr = "")
        if cmd[0] == "which":
            return types.SimpleNamespace(returncode = 0, stdout = "/usr/bin/uv\n", stderr = "")
        if "--report" in cmd:
            # `pip install --dry-run --report FILE` writes the resolved closure
            # to FILE and prints nothing useful, so a stub that only returns a
            # returncode leaves the driver with an empty manifest -- which it
            # handles by installing nothing, and the overlay guard would then
            # pass while proving the overlay never happened.
            report = Path(cmd[cmd.index("--report") + 1])
            report.parent.mkdir(parents = True, exist_ok = True)
            report.write_text(
                json.dumps(
                    {
                        "install": [
                            {"metadata": {"name": n, "version": v}}
                            for n, v in self.resolver_closure
                        ]
                    }
                ),
                encoding = "utf-8",
            )
            return types.SimpleNamespace(returncode = 0, stdout = "", stderr = "")
        if "--target" in cmd:
            self.overlay_installs.append(list(cmd))
            Path(cmd[cmd.index("--target") + 1]).mkdir(parents = True, exist_ok = True)
            return types.SimpleNamespace(returncode = 0, stdout = "", stderr = "")
        if "papermill" in cmd:
            env = kw.get("env") or {}
            self.papermill.append(
                {
                    "notebook": Path(cmd[cmd.index("papermill") + 1]).name,
                    "cuda": env.get("CUDA_VISIBLE_DEVICES"),
                    "kernel": cmd[cmd.index("-k") + 1],
                    "compile_location": env.get("UNSLOTH_COMPILE_LOCATION"),
                    # The whole env, because the caches this file now asserts
                    # on are several variables and a recorder that names them
                    # one at a time goes stale the moment another is added.
                    "env": dict(env),
                }
            )
            Path(cmd[cmd.index("papermill") + 2]).write_text("{}", encoding = "utf-8")
            return types.SimpleNamespace(returncode = 0, stdout = "", stderr = "")
        if not self.venv_ok:
            raise subprocess.CalledProcessError(1, cmd)
        return types.SimpleNamespace(returncode = 0, stdout = "", stderr = "")


def _drive(
    tmp_path: Path,
    leg_names,
    *,
    gpus: int,
    venv_ok: bool = True,
) -> dict:
    """Run the generated driver's setup and runner cells against the stub.

    The only edit to the generated source is the `/kaggle/working` literal,
    rewritten to a temp directory so the cells run off a Kaggle box.
    """
    driver = build_kernel.build_kernel(
        SMOKE_DIR,
        leg_names,
        unsloth_ref = "main",
        zoo_ref = "main",
        extra_args = (),
        per_run_timeout = 60,
        skip_reference = True,
    )
    stub = _Stub(gpus = gpus, venv_ok = venv_ok)
    saved = sys.modules["subprocess"]
    sys.modules["subprocess"] = stub
    namespace: dict = {}
    raised = None
    try:
        for cell in driver["cells"][:2]:
            source = (
                "".join(cell["source"])
                # Venvs moved off /kaggle/working (19.5 GB, and the artifact)
                # onto the ~1 TB overlay when two legs per card made four of
                # them possible at once. Both roots are rewritten here, or the
                # stub counts venvs in a directory nothing ever writes to.
                .replace("/tmp/t4ci_venvs", str(tmp_path / "venvs"))
                .replace("/kaggle/working", str(tmp_path))
            )
            try:
                exec(compile(source, "<driver-cell>", "exec"), namespace)
            except SystemExit as exc:
                raised = exc
                break
    finally:
        sys.modules["subprocess"] = saved
    return {
        "stood_down": raised,
        "n_gpu": namespace.get("N_GPU"),
        "papermill": stub.papermill,
        "results": namespace.get("results") or {},
    }


def test_a_gpu_shortfall_stands_the_kernel_down(tmp_path):
    """A missing GPU is infrastructure, not a result.

    `max(1, len(GPUS))` put both payloads on device 0, where each child still
    sees one card and passes its own visibility assertion, so a contended OOM
    came back looking like a code failure.
    """
    driven = _drive(tmp_path, ["control", "canary"], gpus = -1)
    assert driven["stood_down"] is not None, "a 1-GPU allocation ran both payloads anyway"
    assert driven["papermill"] == []


def test_two_gpus_still_run_both_payloads_one_per_card(tmp_path):
    driven = _drive(tmp_path, ["control", "canary"], gpus = 2)
    assert driven["stood_down"] is None
    assert sorted(p["cuda"] for p in driven["papermill"]) == ["0", "1"]


class _PackedStub(_Stub):
    """`_Stub`, but observable in the two ways a PACKED kernel can go wrong.

    A kernel now carries more legs than it has cards, so the legs queue. Two
    things that used to be structurally impossible become possible and have to
    be watched:

    * two legs on the SAME card at the same time, which is the contended OOM
      the shortfall guard was written for, reached by a route it cannot see;
    * every leg's virtualenv alive at once, each carrying its own torch, on a
      `/kaggle/working` that is not sized for it.

    So papermill HOLDS for a moment (instant calls cannot overlap, and a test
    that cannot observe the failure is not a test), `uv venv` really creates
    its directory, and both the live-payload and live-venv counts are sampled
    while the run is in flight.
    """

    def __init__(
        self,
        *,
        gpus,
        durations = None,
        hold = 0.05,
        vram = None,
    ):
        super().__init__(gpus = gpus)
        self.durations = durations or {}
        self.hold = hold
        self._live_on_card: dict = {}
        self._lock = threading.Lock()
        self.same_card_overlaps: list = []
        self.peak_card_gb: dict = {}
        self.peak_card_legs: dict = {}
        self.vram = vram or {}
        self.max_live_venvs = 0
        self.root: Path | None = None
        self.venv_root: Path | None = None
        self.venvs_created: list = []

    def run(self, cmd, **kw):
        cmd = [str(c) for c in cmd]
        if len(cmd) > 2 and cmd[1] == "venv":
            # Recorded at CREATION. Looking for leftover venv_* after the run
            # cannot tell where they were built: the teardown removes them, so
            # a kernel building every venv on the 19.5 GB artifact volume ends
            # just as clean as one building them on the big overlay.
            self.venvs_created.append(Path(cmd[2]))
            Path(cmd[2]).mkdir(parents = True, exist_ok = True)
        if "papermill" in cmd:
            notebook = Path(cmd[cmd.index("papermill") + 1]).name
            card = (kw.get("env") or {}).get("CUDA_VISIBLE_DEVICES")
            with self._lock:
                live = self._live_on_card.setdefault(card, set())
                live.add(notebook)
                # Two legs on one card is now LEGAL when their measured VRAM
                # fits, so the overlap itself is no longer the finding. What is
                # recorded is the peak SUM, which is the thing that has to stay
                # under budget -- an overlap of two 0.7 GB legs is the feature,
                # and an overlap involving gptoss at 12.78 GB is the bug.
                self.peak_card_gb[card] = max(
                    self.peak_card_gb.get(card, 0.0),
                    sum(self.vram.get(n, 1.0) for n in live),
                )
                self.peak_card_legs[card] = max(self.peak_card_legs.get(card, 0), len(live))
                if len(live) > 1:
                    self.same_card_overlaps.append((card, sorted(live)))
                if self.venv_root is not None:
                    self.max_live_venvs = max(
                        self.max_live_venvs, len(list(self.venv_root.glob("venv_*")))
                    )
            time.sleep(self.durations.get(notebook, self.hold))
            with self._lock:
                self._live_on_card[card].discard(notebook)
        return super().run(cmd, **kw)


class _HubStub(types.ModuleType):
    """Records `snapshot_download` calls in order, with a hold.

    The hold is not decoration. The prefetch runs on a thread nobody joins, so
    an instant stub would let it finish before the first card even starts and
    every ordering question this file asks would answer itself trivially.
    """

    def __init__(
        self,
        hold = 0.02,
        fail_for = (),
    ):
        super().__init__("huggingface_hub")
        self.calls: list = []
        self.hold = hold
        self.fail_for = set(fail_for)
        self.hf_home_at_call: list = []
        # What the hub was asked to FILTER on, per call. Recorded because the
        # patterns are computed a long way from here and reported in the
        # summary, and a version that worked out the right glob and then never
        # passed it would look identical in every artifact.
        self.patterns_at_call: list = []
        self._lock = threading.Lock()

    def snapshot_download(
        self,
        repo_id = None,
        **kw,
    ):
        with self._lock:
            self.calls.append(repo_id)
            self.hf_home_at_call.append(os.environ.get("HF_HOME"))
            self.patterns_at_call.append(kw.get("allow_patterns"))
        time.sleep(self.hold)
        if repo_id in self.fail_for:
            raise RuntimeError(f"stub refuses {repo_id}")


def _drive_packed(
    tmp_path,
    leg_names,
    *,
    gpus,
    durations = None,
    studio = None,
    prefetch_repos = (),
    hub = None,
    after_gpu_concurrent = False,
    venv_fallback = False,
):
    driver = build_kernel.build_kernel(
        SMOKE_DIR,
        leg_names,
        unsloth_ref = "main",
        zoo_ref = "main",
        extra_args = (),
        per_run_timeout = 60,
        skip_reference = True,
        studio = studio,
        prefetch_repos = prefetch_repos,
        after_gpu_concurrent = after_gpu_concurrent,
    )
    stub = _PackedStub(
        gpus = gpus,
        durations = durations,
        vram = {f"t4_{n}.ipynb": LEGS[n].vram_gb for n in leg_names},
    )
    stub.root = tmp_path
    # On the fallback path the venvs land in WORK itself, so that is where the
    # stub has to count them.
    stub.venv_root = tmp_path if venv_fallback else tmp_path / "venvs"
    hub = hub if hub is not None else _HubStub()
    saved = sys.modules["subprocess"]
    saved_hub = sys.modules.get("huggingface_hub")
    sys.modules["subprocess"] = stub
    sys.modules["huggingface_hub"] = hub
    namespace: dict = {}
    raised = None
    try:
        for cell in driver["cells"][:2]:
            source = (
                "".join(cell["source"])
                # Venvs moved off /kaggle/working (19.5 GB, and the artifact)
                # onto the ~1 TB overlay when two legs per card made four of
                # them possible at once. Both roots are rewritten here, or the
                # stub counts venvs in a directory nothing ever writes to.
                # venv_fallback points the preferred root at a path whose
                # parent is a regular file, so `mkdir` raises OSError and the
                # kernel takes its own fallback branch. Rewriting it straight
                # to WORK would test an assignment; this tests the branch.
                .replace(
                    "/tmp/t4ci_venvs",
                    str(tmp_path / "blocked" / "t4ci_venvs")
                    if venv_fallback
                    else str(tmp_path / "venvs"),
                )
                .replace("/kaggle/working", str(tmp_path))
            )
            try:
                exec(compile(source, "<driver-cell>", "exec"), namespace)
            except SystemExit as exc:
                raised = exc
                break
        # JOIN the lane before handing back. The kernel deliberately does not
        # (it is a daemon thread, so a slow download cannot hold the session
        # open), but a test that merely SAMPLES it leaks: the lane resolves
        # `huggingface_hub` out of sys.modules at call time, so one still
        # running after its test returns records into the NEXT test's stub.
        # That is not hypothetical -- it is how this harness first went order
        # dependent, passing alone and failing inside the suite.
        lane = namespace.get("prefetch_thread")
        if lane is not None:
            lane.join(30.0)
            assert not lane.is_alive(), "the prefetch lane outlived its test"
    finally:
        sys.modules["subprocess"] = saved
        if saved_hub is None:
            sys.modules.pop("huggingface_hub", None)
        else:
            sys.modules["huggingface_hub"] = saved_hub
    return {
        "stood_down": raised,
        "stub": stub,
        "hub": hub,
        "results": namespace.get("results") or {},
        "card_load": namespace.get("card_load") or {},
        "card_count": namespace.get("card_count") or {},
    }


# Derived from KERNELS, not a second copy of it. As a literal this silently
# went on describing the OLD longest-first order after legs.py moved to the
# second-wave one, so every test driving it was exercising an order the kernel
# no longer builds -- including the test that exists to assert the order.
# Derived from KERNELS, so it follows the registry rather than restating it.
# Renamed from ALL_LEGS when the Default leg made the kernel five: a name that
# counts is a name that goes stale silently, and two assertions below had
# already hardcoded the 4 to match it.
ALL_LEGS = list(KERNELS[0])


def test_losing_tmp_drops_the_kernel_back_to_one_leg_per_card(tmp_path):
    """The venv fallback described an intention nothing implemented.

    Venvs moved to /tmp because co-scheduling made four torch-bearing venvs
    possible at once and four do not fit in the 19.5 GB /kaggle/working. The
    fallback for a box with no writable /tmp says it "keeps a one-leg-per-card
    run working" -- but MAX_LEGS_PER_CARD was a constant, so the fallback put
    the venvs back on the small partition and went right on building two per
    card. It would have surfaced as an install dying halfway through, which
    reads like anything except a full disk.

    The fallback BRANCH is exercised, not simulated: the preferred root is
    pointed at a path whose parent is a regular file, so mkdir raises exactly
    as it would there.
    """
    (tmp_path / "blocked").write_text("not a directory")
    driven = _drive_packed(tmp_path, ALL_LEGS, gpus = 2, venv_fallback = True)
    assert driven["stood_down"] is None
    stub = driven["stub"]
    assert stub.venv_root is not None
    for card, count in stub.peak_card_legs.items():
        assert count <= 1, (
            f"card {card} ran {count} legs at once with the venvs back on "
            f"/kaggle/working: {stub.same_card_overlaps}"
        )
    assert stub.max_live_venvs <= 2, stub.max_live_venvs
    assert len(stub.papermill) == len(ALL_LEGS), stub.papermill


def test_a_seeds_seat_is_taken_before_any_worker_can_look_at_the_card(tmp_path):
    """The race that put 13.48 GB on a 13.0 GB card, on real hardware.

    `test_no_card_is_ever_asked_to_hold_more_than_it_has` asserts the same
    budget and passed throughout, because with default stub durations every
    leg finishes before the 5s start stagger elapses and no overlap is ever
    recorded. Run 32667451396 was not so lucky: gpu1's seed sat unreserved for
    those 5s, a free worker saw an empty card and put gptoss on it, and when
    the seed worker finally woke, `_admit` correctly refused and the caller
    threw the answer away. control and gptoss then shared one card for 691s.

    So the durations here are chosen to hold the window open rather than to be
    fast: gptoss must outlive the stagger, or the second leg lands after it has
    already finished and the test goes green on a schedule that never happened.
    """
    driven = _drive_packed(
        tmp_path,
        ALL_LEGS,
        gpus = 2,
        durations = {
            "t4_canary.ipynb": 2.0,
            "t4_control.ipynb": 2.0,
            "t4_frontier.ipynb": 0.2,
            "t4_gptoss.ipynb": 7.0,
        },
    )
    stub = driven["stub"]
    for card, peak in stub.peak_card_gb.items():
        assert peak <= 13.0, f"card {card} peaked at {peak} GB. Overlaps: {stub.same_card_overlaps}"
    # gptoss is 12.78 of a 13.0 budget, so it can only ever run alone. Asserted
    # on the overlap record as well as the sum: a VRAM table that silently
    # under-priced it would satisfy the sum check while the card burned.
    for card, live in stub.same_card_overlaps:
        assert "t4_gptoss.ipynb" not in live, (card, live)


def test_no_card_is_ever_asked_to_hold_more_than_it_has(tmp_path):
    """Was "never two legs on one card"; is now "never over the VRAM budget".

    Two legs on a card is the FEATURE, not the bug: measured on run
    32611343797 the three Qwen legs peak at 0.70 GB each on a 14.56 GB card,
    so one leg per card left it 95% empty. What must still never happen is the
    thing that produced the OOM this file's shortfall guard was written for --
    payloads whose summed appetite exceeds the card. gptoss peaks at 12.78 GB,
    so it is excluded by the arithmetic rather than by a special case.

    Asserted on the summed GB and not on the overlap, because after this change
    an overlap is exactly what success looks like.
    """
    driven = _drive_packed(tmp_path, ALL_LEGS, gpus = 2)
    assert driven["stood_down"] is None
    stub = driven["stub"]
    for card, peak in stub.peak_card_gb.items():
        assert peak <= 13.0, f"card {card} peaked at {peak} GB: {stub.same_card_overlaps}"
    for card, count in stub.peak_card_legs.items():
        assert count <= 2, f"card {card} held {count} legs at once"
    assert len(stub.papermill) == len(ALL_LEGS), stub.papermill
    # Both cards are used, and every leg ran. The SPLIT is deliberately not
    # asserted: how many legs each card ends up with is a function of how long
    # the legs take relative to the 5s venv stagger, not something the
    # scheduler promises. Under the measured durations (gptoss 384.1s,
    # frontier 312.2s, canary 265.3s, control 262.2s) the stagger is under 2%
    # and the split is 2/2; under the sub-second stubs here the first card
    # legitimately drains most of the queue before the second clears its
    # stagger. Pinning 2/2 would be pinning the stub's timing.
    assert set(p["cuda"] for p in stub.papermill) == {"0", "1"}, stub.papermill


def test_gptoss_starts_in_the_second_wave_so_the_prefetch_has_a_window(tmp_path):
    """Was longest-first; is now second-wave, and the change is the point.

    A prefetch only pays for what it finishes BEFORE the leg that wants the
    model starts, and gptoss is the only leg with a ~12 GB download. Starting
    it at t=0 leaves no window in front of it, which is why prefetching without
    this reorder measures WORSE than doing neither (603.1s against 563.1s).

    Third is first pick of the second wave: with two cards, positions 0 and 1
    are seeded and 2 is the first to be taken off the pending queue, so gptoss
    starts at ~190-220s under the measured durations. That is lead time the
    prefetch spends, and a small leg is still running beside it.

    Asserted on POSITION in the order rather than on a start timestamp: the
    timestamp is a function of the stub's durations, and pinning it would pin
    the stub. legs.KERNELS carries the full table.
    """
    order = list(KERNELS[0])
    assert order.index("gptoss") == 2, (
        f"gptoss is at position {order.index('gptoss')} of {order}; first means "
        "the prefetch has no window and second-wave is what buys the saving"
    )
    # ...and it must not be LAST either, which is the other intuitive answer.
    # gptoss is the longest leg, so ending on it idles the other card for its
    # whole ~284s: simulated at 651.1s worst case against 528.1s here.
    assert order[-1] != "gptoss", order

    driven = _drive_packed(tmp_path, ALL_LEGS, gpus = 2)
    started = [p["notebook"] for p in driven["stub"].papermill]
    assert started[0] != "t4_gptoss.ipynb", started
    assert started != sorted(started), "payloads are running in alphabetical order"


def test_each_leg_keeps_its_own_venv_compile_cache_and_ipykernel(tmp_path):
    """Packing must not let two legs share an interpreter.

    The legs exist to install DIFFERENT library sets. They are separated by a
    per-payload virtualenv, a per-payload ipykernel spec and a per-payload
    `UNSLOTH_COMPILE_LOCATION`; all three are keyed by the payload's index, so
    an index reused across a wave would silently merge two legs' trees and the
    last writer would win.
    """
    driven = _drive_packed(tmp_path, ALL_LEGS, gpus = 2)
    calls = driven["stub"].papermill
    for field in ("kernel", "compile_location", "notebook"):
        values = [c[field] for c in calls]
        assert len(set(values)) == len(calls), (field, values)


def test_a_finished_leg_gives_its_virtualenv_back(tmp_path):
    """Each venv carries its own torch and its own NVIDIA runtime.

    The tail cell prunes `venv_*`, but only after every payload has finished,
    so the PEAK is what matters and freeing at the end of each leg is what
    bounds it. The bound is one venv per concurrent LEG, which co-scheduling
    raised from 2 to 4.

    That raise is precisely why the venvs no longer live on `/kaggle/working`.
    That path is 19.5 GB and is also what Kaggle ships home; four torch trees
    do not fit in it, and the failure would arrive as an install dying midway
    for reasons that look nothing like a full disk. They go on the ~1 TB
    overlay instead, and only the evidence stays where Kaggle collects it.
    """
    driven = _drive_packed(tmp_path, ALL_LEGS, gpus = 2)
    stub = driven["stub"]
    ceiling = 2 * 2  # cards x MAX_LEGS_PER_CARD
    assert (
        stub.max_live_venvs <= ceiling
    ), f"{stub.max_live_venvs} virtualenvs were alive at once, ceiling {ceiling}"
    assert list((tmp_path / "venvs").glob("venv_*")) == [], "a payload left its virtualenv behind"
    # ...and none of them was ever created on the artifact volume.
    assert stub.venvs_created, "no virtualenv was built at all"
    for created in stub.venvs_created:
        assert (tmp_path / "venvs") in created.parents, (
            f"a virtualenv was built at {created}, on /kaggle/working -- which "
            "is 19.5 GB and is also the artifact Kaggle ships home"
        )


# --------------------------------------------------- Studio in the same kernel

STUDIO = {
    "unsloth_ref": "main",
    "repo_url": "https://github.com/unslothai/unsloth",
    "payload_args": "--max-steps 8",
}
STUDIO_INSTALL = build_kernel.STUDIO_INSTALL_NOTEBOOK
STUDIO_TEST = build_kernel.STUDIO_TEST_NOTEBOOK
# A value no card index can be confused with, so "the driver left this alone"
# and "the driver pinned a card" are distinguishable. An unpinned lane inherits
# whatever the ambient environment has; a pinned one is overwritten with a
# single index.
AMBIENT_CUDA = "0,1"


def _drive_with_studio(
    tmp_path,
    monkeypatch,
    leg_names,
    *,
    gpus = 2,
    durations = None,
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", AMBIENT_CUDA)
    return _drive_packed(tmp_path, leg_names, gpus = gpus, durations = durations, studio = STUDIO)


def test_the_studio_install_never_takes_a_card_and_the_legs_never_wait_for_it(
    tmp_path, monkeypatch
):
    """The whole point of carrying Studio here: its install is free time.

    Checkout, `install.sh --local`, the frontend build and the Playwright
    browser are network and CPU and touch no GPU, so they run beside the
    training legs rather than after them. If this lane ever queued for a card
    it would displace a leg and the merge would cost more than it saves.
    """
    driven = _drive_with_studio(
        tmp_path,
        monkeypatch,
        ALL_LEGS,
        durations = {f"t4_{LEGS[n].name}.ipynb": 0.30 for n in ALL_LEGS},
    )
    assert driven["stood_down"] is None
    calls = {c["notebook"]: c for c in driven["stub"].papermill}
    assert STUDIO_INSTALL in calls, sorted(calls)

    # Unpinned. install.sh --local resolves torch, and an installer that cannot
    # see a device resolves a CPU-only one -- which is the exact regression
    # Studio's verify cell exists to catch, so hiding the cards here would
    # manufacture it.
    assert calls[STUDIO_INSTALL]["cuda"] == AMBIENT_CUDA, calls[STUDIO_INSTALL]
    # ...while every leg is still pinned to exactly one card. The SPLIT is not
    # asserted, for the reason given in
    # test_four_legs_on_two_cards_never_put_two_legs_on_one_card_at_once: how
    # many legs each card takes depends on leg duration against the 5s venv
    # stagger, and under sub-second stubs the first card legitimately drains
    # most of the queue.
    leg_cards = [c["cuda"] for n, c in calls.items() if n.startswith("t4_")]
    assert len(leg_cards) == len(ALL_LEGS), calls
    assert set(leg_cards) <= {"0", "1"}, leg_cards
    assert set(leg_cards) == {"0", "1"}, leg_cards
    # Two legs on a card is legal now (see the VRAM budget); what must hold is
    # that the summed appetite never exceeds what the card has.
    for card, peak in driven["stub"].peak_card_gb.items():
        assert peak <= 13.0, (card, peak, driven["stub"].same_card_overlaps)


def test_the_studio_assertions_wait_for_both_cards_rather_than_borrowing_one(tmp_path, monkeypatch):
    """Studio keeps both T4s visible, and that is deliberate upstream.

    Its own driver says so: "Studio's own device selection is part of what is
    under test; masking one would test a machine nobody has." So the GPU half
    runs once the leg queue has drained, unpinned, rather than being handed a
    single card out of the queue.
    """
    driven = _drive_with_studio(tmp_path, monkeypatch, ALL_LEGS)
    calls = [c["notebook"] for c in driven["stub"].papermill]
    assert STUDIO_TEST in calls, calls
    # Last, after every leg.
    assert calls[-1] == STUDIO_TEST, calls
    by_name = {c["notebook"]: c for c in driven["stub"].papermill}
    assert by_name[STUDIO_TEST]["cuda"] == AMBIENT_CUDA, by_name[STUDIO_TEST]


def test_a_failed_studio_install_skips_its_assertions_with_the_reason(tmp_path, monkeypatch):
    """Otherwise the missing venv is reported as a Studio regression.

    The install half is what puts the interpreter, the frontend and the
    llama.cpp on disk. Running the assertions against a half-built tree fails
    on `no interpreter at ...`, which reads like the code under test broke
    rather than like the install did.
    """
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", AMBIENT_CUDA)

    class _InstallFails(_PackedStub):
        def run(self, cmd, **kw):
            cmd = [str(c) for c in cmd]
            if "papermill" in cmd and STUDIO_INSTALL in " ".join(cmd):
                self.papermill.append(
                    {
                        "notebook": STUDIO_INSTALL,
                        "cuda": None,
                        "kernel": None,
                        "compile_location": None,
                    }
                )
                Path(cmd[cmd.index("papermill") + 2]).write_text("{}", encoding = "utf-8")
                return types.SimpleNamespace(returncode = 1, stdout = "", stderr = "")
            return super().run(cmd, **kw)

    driver = build_kernel.build_kernel(
        SMOKE_DIR,
        ALL_LEGS,
        unsloth_ref = "main",
        zoo_ref = "main",
        extra_args = (),
        per_run_timeout = 60,
        skip_reference = True,
        studio = STUDIO,
    )
    stub = _InstallFails(gpus = 2)
    stub.root = tmp_path
    stub.venv_root = tmp_path / "venvs"
    saved = sys.modules["subprocess"]
    sys.modules["subprocess"] = stub
    namespace: dict = {}
    try:
        for cell in driver["cells"][:2]:
            source = (
                "".join(cell["source"])
                # Venvs moved off /kaggle/working (19.5 GB, and the artifact)
                # onto the ~1 TB overlay when two legs per card made four of
                # them possible at once. Both roots are rewritten here, or the
                # stub counts venvs in a directory nothing ever writes to.
                .replace("/tmp/t4ci_venvs", str(tmp_path / "venvs"))
                .replace("/kaggle/working", str(tmp_path))
            )
            exec(compile(source, "<driver-cell>", "exec"), namespace)
    finally:
        sys.modules["subprocess"] = saved

    ran = [c["notebook"] for c in stub.papermill]
    assert STUDIO_TEST not in ran, ran
    # Every leg still ran: a broken Studio install must not take the notebook
    # signal down with it.
    assert sorted(n for n in ran if n.startswith("t4_")) == sorted(
        f"t4_{LEGS[leg].name}.ipynb" for leg in ALL_LEGS
    )
    recorded = (namespace.get("results") or {}).get(STUDIO_TEST)
    assert recorded is not None, "the skip was not recorded at all"
    assert recorded["returncode"] is None
    assert "install lane did not succeed" in recorded["error"]


def test_studio_is_not_in_the_card_queue(tmp_path, monkeypatch):
    """ORDER is the legs. Either Studio half in it would be handed a card."""
    driver = build_kernel.build_kernel(
        SMOKE_DIR,
        ALL_LEGS,
        unsloth_ref = "main",
        zoo_ref = "main",
        extra_args = (),
        per_run_timeout = 60,
        skip_reference = True,
        studio = STUDIO,
    )
    setup = "".join(driver["cells"][0]["source"])
    order = next(l for l in setup.splitlines() if l.startswith("ORDER = "))
    assert STUDIO_INSTALL not in order, order
    assert STUDIO_TEST not in order, order
    assert order.count("t4_") == len(ALL_LEGS), order
    # ...but both are carried, or the kernel would have nothing to run.
    payloads = set(driver["metadata"]["kaggle_t4_ci"]["payloads"])
    assert {STUDIO_INSTALL, STUDIO_TEST} <= payloads, sorted(payloads)


def test_a_one_card_allocation_still_stands_a_packed_kernel_down(tmp_path):
    """The shortfall guard survives the change that made it stop counting legs.

    It used to compare GPUs against the payload count. There are deliberately
    more payloads than cards now, so that comparison would stand every healthy
    run down; it compares against the width the packing was built for instead.
    What must NOT change is that a genuinely short allocation is still called
    infrastructure, because one card silently serialises the whole kernel and
    doubles its wall clock while looking like a slow but healthy run.
    """
    driven = _drive_packed(tmp_path, ALL_LEGS, gpus = 1)
    assert driven["stood_down"] is not None, "a 1-GPU allocation ran the packed kernel anyway"
    assert driven["stub"].papermill == []


def test_a_payload_whose_venv_failed_is_not_run_in_the_system_kernel(tmp_path):
    """Falling back to `python3` puts both legs' installs in one tree.

    The legs deliberately install different library sets, so a shared
    site-packages destroys the comparison rather than merely risking
    corruption, and the resulting import error reads as a code regression.
    """
    driven = _drive(tmp_path, ["control", "canary"], gpus = 2, venv_ok = False)
    assert [
        p["kernel"] for p in driven["papermill"]
    ] == [], "a payload ran in the shared system kernel after its venv failed"
    assert driven["results"], "the skipped payloads left no record"
    for entry in driven["results"].values():
        assert entry["error"]


def test_each_payload_compiles_into_its_own_cache(tmp_path):
    """Concurrent legs must not share `unsloth_compiled_cache`.

    It is a relative path resolved against the working directory that both
    papermill children inherit, and the legs compile the same modules against
    deliberately different transformers/TRL versions.
    """
    driven = _drive(tmp_path, ["control", "canary"], gpus = 2)
    locations = [p["compile_location"] for p in driven["papermill"]]
    assert all(locations), "no per-payload UNSLOTH_COMPILE_LOCATION was set"
    assert len(set(locations)) == len(locations), f"shared compile cache: {locations}"


def test_the_prune_still_reaches_the_per_payload_directories():
    """Whatever the per-payload names are, the tail cell must still drop them.

    `kernels output` ships the whole of /kaggle/working back over the wire, and
    a sweep that missed the venvs once shipped 371MB.
    """
    driver = build_kernel.build_driver({"t4_control.ipynb": {"cells": []}}, 60)
    tail = "".join(driver["cells"][2]["source"])
    assert '"unsloth_compiled_cache*"' in tail or "'unsloth_compiled_cache*'" in tail
    assert '"t4_smoke_src*"' in tail or "'t4_smoke_src*'" in tail


# ----------------------------------------------------------------- payload


def _payload_cells(leg, **kw) -> list[str]:
    notebook = build_kernel.build_payload_notebook(
        SMOKE_DIR, leg, unsloth_ref = "main", zoo_ref = "main", reference = "", **kw
    )
    return ["".join(cell["source"]) for cell in notebook["cells"]]


def test_each_payload_materialises_into_its_own_directory():
    """Two payloads writing one directory can truncate a file the other reads.

    `write_bytes` truncates first and the legs carry byte-identical copies of
    the same sources, so the loser of that race imports a partial file and dies
    for a reason unrelated to the commit.
    """
    roots = set()
    for name in ("control", "canary"):
        materialise = _payload_cells(LEGS[name])[0]
        roots.add(materialise.split("ROOT = pathlib.Path(")[1].split(")")[0])
    assert len(roots) == 2, f"both payloads materialise into {roots}"


def test_a_shared_argument_does_not_override_a_legs_own_option():
    """`--smoke-args` is shared so control and canary stay comparable.

    It must not reach a leg that already sets that option: the gpt-oss leg's 3
    steps are a measured fit for a 16GB card, and argparse takes the LAST value,
    so appending the SFT legs' 10 silently retrained the 20B leg.
    """
    run_cell = _payload_cells(LEGS["gptoss"], extra_args = ("--max-steps", "10"))[3]
    argv = run_cell.split("cmd += [")[1].split("]")[0]
    assert argv.count('"--max-steps"') == 1, argv
    assert '"3"' in argv and '"10"' not in argv

    # A leg that does NOT set it still receives the shared value.
    canary = _payload_cells(LEGS["canary"], extra_args = ("--max-steps", "10"))[3]
    assert '"--max-steps", "10"' in canary.split("cmd += [")[1]


def test_a_probe_failure_is_reported_as_a_failed_payload(tmp_path, monkeypatch):
    """A commit that breaks `import unsloth` must not exit green.

    The probe raises before the run cell can write a report, so without one of
    its own the launcher extracts nothing, calls the run `infra` and passes on a
    deterministic import regression.
    """
    monkeypatch.setattr(build_kernel, "KERNEL_ROOT", str(tmp_path / "src"))
    leg = LEGS["control"]
    broken = type(leg)(
        **{
            **{field: getattr(leg, field) for field in leg.__dataclass_fields__},
            "imports": ("unsloth_module_the_broken_commit_cannot_import",),
        }
    )
    cells = _payload_cells(broken)

    outputs = []
    for index in (0, 2):
        script = tmp_path / f"cell{index}.py"
        script.write_text(cells[index], encoding = "utf-8")
        proc = subprocess.run(
            [sys.executable, str(script)], capture_output = True, text = True, timeout = 600
        )
        outputs.append(proc.stdout + proc.stderr)
    assert "KAGGLE_T4_CI_PAYLOAD MISSING" in outputs[1]

    # What Kaggle brings back, and what the launcher makes of it.
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "t4_control_output.ipynb").write_text(
        json.dumps(
            {
                "cells": [
                    {"cell_type": "code", "outputs": [{"output_type": "stream", "text": text}]}
                    for text in outputs
                ]
            }
        ),
        encoding = "utf-8",
    )
    reports = launch.extract_reports(evidence)
    assert reports, "the import failure produced no report at all"
    assert reports[0]["passed"] is False
    assert reports[0]["label"] == "control"
    assert any(
        "unsloth_module_the_broken_commit_cannot_import" in f for f in reports[0]["failures"]
    )

    # The resolved versions are printed one line above the failure, and
    # report.version_table reads them off the REPORT rather than the log, so
    # without them the summary cannot say which release the red leg had.
    import report as report_module

    assert report_module.resolved_versions(reports[0]), reports[0].keys()
    table = report_module.version_table(
        [reports[0], {"label": "control", "versions_flat": {"transformers": "4.57.6"}}]
    )
    assert any("| package | control |" in line for line in table), table


def test_an_install_that_cannot_be_resolved_is_reported_as_a_failed_payload(tmp_path, monkeypatch):
    """Three exhausted pip attempts used to raise without reporting anything.

    The launcher then sees a leg with no report, calls the run `partial` or
    `infra` and exits GREEN, so the job added to catch a broken distribution
    could not fail on one.
    """
    monkeypatch.setattr(build_kernel, "KERNEL_ROOT", str(tmp_path / "src"))
    install = _payload_cells(LEGS["control"])[1]
    script = tmp_path / "install.py"
    script.write_text(
        "import subprocess, time, types\n"
        "subprocess.run = lambda cmd, **kw: types.SimpleNamespace(\n"
        "    returncode=1, stdout='', stderr='ERROR: ResolutionImpossible')\n"
        "time.sleep = lambda _s: None\n" + install,
        encoding = "utf-8",
    )
    proc = subprocess.run(
        [sys.executable, str(script)], capture_output = True, text = True, timeout = 600
    )
    assert proc.returncode != 0
    assert "KAGGLE_T4_CI_PAYLOAD INSTALL FAILED" in proc.stdout

    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "kernel.log").write_text(proc.stdout + proc.stderr, encoding = "utf-8")
    reports = launch.extract_reports(evidence)
    assert reports, "the exhausted install produced no report at all"
    assert reports[0]["label"] == "control"
    assert reports[0]["passed"] is False
    assert any("ResolutionImpossible" in f for f in reports[0]["failures"])


def test_the_install_backs_off_between_attempts():
    """Three immediate retries all land inside the same upstream blip.

    The third failure is what the failed-payload report above rests on, so it
    has to mean "this cannot be resolved" rather than "one bad minute".
    """
    install = _payload_cells(LEGS["control"])[1]
    assert "time.sleep(15 * attempt)" in install


# ------------------------------------------------------------------ launcher


@pytest.mark.parametrize("plain", [False, True])
def test_a_report_reaches_the_launcher_through_kaggles_structured_log(tmp_path, plain):
    """The log fallback exists for the run whose notebook never came back.

    Kaggle hands the log over as a JSON array of stream records, so scanning it
    as text finds no line starting with the report prefix and files a real
    failure as `infra`.
    """
    payload = {
        "label": "control",
        "model": "unsloth/Qwen2.5-0.5B",
        "passed": False,
        "failures": ["reference band: out of band at step 3"],
    }
    line = launch.RESULT_PREFIX + json.dumps(payload) + "\n"
    body = (
        line
        if plain
        else json.dumps(
            [
                {"stream_name": "stdout", "time": 12.0, "data": "install done\n"},
                {"stream_name": "stdout", "time": 13.0, "data": line},
            ]
        )
    )
    kernel_dir = tmp_path / "unsloth-t4-ci-deadbeef"
    kernel_dir.mkdir()
    (kernel_dir / "kernel.log").write_text(body, encoding = "utf-8")

    reports = launch.extract_reports(tmp_path)
    assert [r["label"] for r in reports] == ["control"]
    assert reports[0]["passed"] is False


def test_a_log_record_that_splits_the_report_is_still_read(tmp_path):
    """Record boundaries are not line boundaries; join before scanning."""
    payload = {"label": "canary", "model": "unsloth/Qwen2.5-0.5B", "passed": True}
    line = launch.RESULT_PREFIX + json.dumps(payload) + "\n"
    half = len(line) // 2
    kernel_dir = tmp_path / "unsloth-t4-ci-cafe"
    kernel_dir.mkdir()
    (kernel_dir / "kernel.log").write_text(
        json.dumps(
            [
                {"stream_name": "stdout", "data": line[:half]},
                {"stream_name": "stdout", "data": line[half:]},
            ]
        ),
        encoding = "utf-8",
    )
    assert [r["label"] for r in launch.extract_reports(tmp_path)] == ["canary"]


def test_every_push_attempt_gets_its_own_slug(tmp_path, monkeypatch):
    """Retrying onto one slug pushes a SECOND session and hides the first.

    A push to an existing id creates a new VERSION and starts another batch
    session rather than superseding the running one, and `kernels/output` and
    `kernels status` never pass a version label, so they answer for the latest
    session only. A retry after a lost response therefore reads the wrong
    execution's evidence while the first keeps billing unseen.
    """
    # The accepted push records its slug in the in-flight registry, so point
    # that at the tmp dir: otherwise this test files a kernel that does not
    # exist into the real registry and every later launcher's orphan sweep
    # tries, and fails, to delete it.
    monkeypatch.setattr(launch, "INFLIGHT", tmp_path / "inflight.json")
    attempts: list[list[str]] = []
    deleted: list[str] = []

    def fake_run(cmd, **kw):
        cmd = [str(c) for c in cmd]
        attempts.append(cmd)
        if cmd[1:3] == ["kernels", "delete"]:
            deleted.append(cmd[3])
            return types.SimpleNamespace(returncode = 0, stdout = "", stderr = "")
        metadata = json.loads((Path(cmd[cmd.index("-p") + 1]) / "kernel-metadata.json").read_text())
        attempts[-1] = ["push", metadata["id"]]
        if len(deleted) + 1 < 3:
            return types.SimpleNamespace(returncode = 1, stdout = "", stderr = "Connection reset")
        return types.SimpleNamespace(returncode = 0, stdout = "Successfully pushed", stderr = "")

    monkeypatch.setattr(launch.subprocess, "run", fake_run)
    monkeypatch.setattr(launch.time, "sleep", lambda _s: None)
    pushed = launch.push(Path(__file__), "someuser", 3600)

    slugs = [a[1] for a in attempts if a[0] == "push"]
    assert len(slugs) == 3, slugs
    assert len(set(slugs)) == 3, f"every retry reused one slug: {slugs}"
    assert pushed["ok"] and pushed["slug"] == slugs[-1]
    # Each earlier attempt may have landed despite its error, so it is deleted
    # before the next adds a second concurrent session.
    assert deleted == [s for s in slugs[:-1]]
    assert pushed["attempts"] == slugs
    # Only the accepted attempt is left for release to reclaim; the discarded
    # ones are gone, and registering those would leave the next launcher
    # sweeping for kernels that never existed.
    assert [e["slug"] for e in launch._inflight_read()] == [slugs[-1]]


def _drive_main(
    monkeypatch,
    tmp_path,
    *,
    push_seconds,
    pushes,
    extra_argv = (),
    api_seconds = 0.0,
):
    """Run `launch.main()` end to end with Kaggle replaced by stubs.

    Returns the per-kernel wait budgets, the slugs deleted on the way out and
    the launch result. The clock is fake, so a push can burn arbitrary wall
    time without the test taking any.

    ``api_seconds`` is what authentication costs. It is not free: with
    KAGGLE_API_TOKEN set, which is the only credential the workflow passes,
    `authenticate()` introspects the token over the network.
    """
    clock = {"t": 1_000_000.0}
    monkeypatch.setattr(launch.time, "time", lambda: clock["t"])
    monkeypatch.setattr(launch.time, "sleep", lambda _s: None)

    def fake_api():
        clock["t"] += api_seconds
        return object()

    monkeypatch.setattr(launch, "_api", fake_api)

    outcomes = list(pushes)

    def fake_push(
        notebook,
        user,
        kernel_timeout_sec,
        accelerator = "NvidiaTeslaT4",
        attempted = None,
    ):
        clock["t"] += push_seconds
        outcome = outcomes.pop(0)
        # Like the real one: the caller's list is filled as the slugs are filed.
        if attempted is not None:
            attempted.extend(outcome.get("attempts") or [])
        return outcome

    waits: list[int] = []

    def fake_wait(api, slug, poll_every, max_wait):
        waits.append(max_wait)
        return "COMPLETE"

    deleted: list[str] = []

    def fake_run(cmd, **kw):
        cmd = [str(c) for c in cmd]
        if cmd[1:3] == ["kernels", "delete"]:
            deleted.append(cmd[3])
        return types.SimpleNamespace(returncode = 0, stdout = "", stderr = "")

    monkeypatch.setattr(launch, "push", fake_push)
    monkeypatch.setattr(launch, "wait", fake_wait)
    monkeypatch.setattr(
        launch, "fetch_evidence", lambda slug, outdir, **kw: {"notebooks": [], "log": None}
    )
    monkeypatch.setattr(
        launch,
        "extract_reports",
        lambda outdir: [{"label": "control", "model": "m", "passed": True}],
    )
    monkeypatch.setattr(launch.subprocess, "run", fake_run)
    monkeypatch.delenv("GITHUB_OUTPUT", raising = False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launch.py",
            *[a for i in range(len(pushes)) for a in ("--notebook", f"k{i}.ipynb")],
            "--user",
            "someuser",
            "--outdir",
            str(tmp_path),
            "--expect",
            "1",
            "--max-wait",
            "5400",
            *extra_argv,
        ],
    )
    assert launch.main() == 0
    result = json.loads((tmp_path / "launch_result.json").read_text(encoding = "utf-8"))
    return waits, deleted, result


_TWO_PUSHES = [
    {
        "ok": True,
        "slug": "someuser/unsloth-t4-ci-aaaa",
        "attempts": ["someuser/unsloth-t4-ci-aaaa"],
    },
    {
        "ok": True,
        "slug": "someuser/unsloth-t4-ci-bbbb",
        "attempts": ["someuser/unsloth-t4-ci-bbbb"],
    },
]


def test_the_launcher_will_not_push_what_it_may_not_live_to_delete(monkeypatch, tmp_path):
    """A window one second short of the worst case pushes nothing.

    The job that runs this launcher is killed at a fixed time and killing it
    takes release() with it, so a kernel pushed with less than the launcher's
    own worst case left can be left up billing quota to its own ceiling. The
    steps before it -- a checkout, a pip install, the harness suite -- have no
    deadline of their own, so the job timeout leaving room for them is an
    assumption rather than a fact about the run; this measures what is actually
    left.
    """
    _, deleted, result = _drive_main(
        monkeypatch,
        tmp_path,
        push_seconds = 0.0,
        pushes = _TWO_PUSHES,
        extra_argv = (
            "--deadline-epoch",
            str(int(1_000_000 + launch.worst_case_seconds(5400, 2)) - 1),
        ),
    )
    assert not result.get("kernels"), "a kernel was pushed with no room to delete it"
    assert result["slug"] is None and deleted == []
    assert result["verdict"] == "infra"
    assert "could be killed during cleanup" in result["reason"]


def test_a_window_that_fits_still_launches(monkeypatch, tmp_path):
    """The guard has to stand down for a short window and ONLY for one.

    One second more than the worst case is the whole of it, so this is the
    ordinary run: a guard that refused here, or that read the deadline as an
    absolute duration rather than the moment the job dies, would stand every
    invocation down and the workflow would never test anything again -- green
    every time, which is exactly how it would go unnoticed.
    """
    waits, _, result = _drive_main(
        monkeypatch,
        tmp_path,
        push_seconds = 0.0,
        pushes = _TWO_PUSHES,
        extra_argv = (
            "--deadline-epoch",
            str(int(1_000_000 + launch.worst_case_seconds(5400, 2))),
        ),
    )
    assert [k["slug"] for k in result["kernels"]] == [p["slug"] for p in _TWO_PUSHES]
    assert waits == [5400, 5400]
    assert result["verdict"] == "pass"


def test_the_window_is_measured_again_after_authenticating(monkeypatch, tmp_path):
    """Authentication sits between the guard and the first push, and costs time.

    `_api()` calls `KaggleApi.authenticate()`, and the only credential this
    workflow passes is KAGGLE_API_TOKEN, so kaggle 2.2.4 takes the access-token
    branch: `_authenticate_with_access_token` -> `_introspect_token`, an HTTP
    round trip whose only bound is the process-wide SOCKET_TIMEOUT_SEC. Checked
    once, before that call, a window that fitted by less than the timeout is
    already gone by the time the first kernel is pushed, and being killed during
    release() is what leaves kernels billing.
    """
    _, deleted, result = _drive_main(
        monkeypatch,
        tmp_path,
        push_seconds = 0.0,
        pushes = _TWO_PUSHES,
        api_seconds = float(launch.SOCKET_TIMEOUT_SEC),
        extra_argv = (
            "--deadline-epoch",
            str(int(1_000_000 + launch.worst_case_seconds(5400, 2)) + 60),
        ),
    )
    assert not result.get("kernels"), "a kernel was pushed after the window went"
    assert result["slug"] is None and deleted == []
    assert result["verdict"] == "infra"
    assert "could be killed during cleanup" in result["reason"]


def test_a_window_that_survives_authentication_still_launches(monkeypatch, tmp_path):
    """The recheck must cost the ordinary run nothing.

    Authentication that returns well inside the slack leaves the worst case
    covered, and a second guard that stood down here would make every run green
    without testing anything, which is the failure mode this whole guard is
    least able to notice.
    """
    _, _, result = _drive_main(
        monkeypatch,
        tmp_path,
        push_seconds = 0.0,
        pushes = _TWO_PUSHES,
        api_seconds = 30.0,
        extra_argv = (
            "--deadline-epoch",
            str(int(1_000_000 + launch.worst_case_seconds(5400, 2)) + 60),
        ),
    )
    assert [k["slug"] for k in result["kernels"]] == [p["slug"] for p in _TWO_PUSHES]
    assert result["verdict"] == "pass"


def test_no_deadline_is_no_guard(monkeypatch, tmp_path):
    """Run by hand, with no job to be killed by, there is nothing to check.

    The flag defaults to 0 and the launcher then pushes exactly as it always
    did, so a local reproduction does not have to invent a deadline to get a
    kernel.
    """
    _, _, result = _drive_main(
        monkeypatch,
        tmp_path,
        push_seconds = 0.0,
        pushes = _TWO_PUSHES,
    )
    assert [k["slug"] for k in result["kernels"]] == [p["slug"] for p in _TWO_PUSHES]


def test_the_deletion_deadline_covers_the_time_spent_pushing(monkeypatch, tmp_path):
    """A kernel bills from the moment Kaggle accepts it, not from the last push.

    Started after the push loop, the deadline gave the first kernel --max-wait
    on top of the SECOND push's retries: 45 minutes of throttling turned a 90
    minute ceiling into 135 minutes of billing, past the budget the gate
    reserved for the run.
    """
    waits, _, _ = _drive_main(
        monkeypatch,
        tmp_path,
        push_seconds = 1800.0,
        pushes = [
            {
                "ok": True,
                "slug": "someuser/unsloth-t4-ci-aaaa",
                "attempts": ["someuser/unsloth-t4-ci-aaaa"],
            },
            {
                "ok": True,
                "slug": "someuser/unsloth-t4-ci-bbbb",
                "attempts": ["someuser/unsloth-t4-ci-bbbb"],
            },
        ],
    )
    # 5400s of invocation deadline, 3600s of it spent pushing.
    assert waits == [1800, 1800]


def test_every_slug_a_push_filed_is_deleted_on_the_way_out(monkeypatch, tmp_path):
    """An attempt whose response was lost may be running, and is untracked.

    `push()` records every slug it filed because Kaggle answers an accepted push
    with a 5xx or a reset connection often enough to be a known issue. Cleanup
    read only the ACCEPTED slug, so a failed final attempt (which has none) and
    any earlier attempt whose discard was refused kept a session slot and billed
    quota unseen.
    """
    _, deleted, result = _drive_main(
        monkeypatch,
        tmp_path,
        push_seconds = 0.0,
        pushes = [
            # Accepted on the third attempt; the first two may still be up.
            {
                "ok": True,
                "slug": "someuser/unsloth-t4-ci-cccc",
                "attempts": [
                    "someuser/unsloth-t4-ci-aaaa",
                    "someuser/unsloth-t4-ci-bbbb",
                    "someuser/unsloth-t4-ci-cccc",
                ],
            },
            # Never accepted, so no `slug`, and the last attempt is the
            # ambiguous one.
            {
                "ok": False,
                "reason": "push_failed",
                "detail": "Connection reset by peer",
                "attempts": ["someuser/unsloth-t4-ci-dddd", "someuser/unsloth-t4-ci-eeee"],
            },
        ],
    )
    assert sorted(deleted) == [
        "someuser/unsloth-t4-ci-aaaa",
        "someuser/unsloth-t4-ci-bbbb",
        "someuser/unsloth-t4-ci-cccc",
        "someuser/unsloth-t4-ci-dddd",
        "someuser/unsloth-t4-ci-eeee",
    ]
    assert all(k["released"] for k in result["kernels"])


def test_the_temp_dir_is_left_alone_when_the_log_is_not_json(tmp_path):
    """A plain-text log, and a JSON object that is not a record array."""
    kernel_dir = tmp_path / "unsloth-t4-ci-beef"
    kernel_dir.mkdir()
    (kernel_dir / "kernel.log").write_text(json.dumps({"log": "nothing here"}), encoding = "utf-8")
    assert launch.extract_reports(tmp_path) == []


def test_a_push_that_runs_out_of_wall_clock_is_a_recorded_failure(monkeypatch):
    """`subprocess.run(timeout=...)` RAISES; it does not return a bad result.

    That exception used to leave `push()` entirely, taking the slugs it had
    filed. Those are the most ambiguous slugs there are, the client having been
    killed mid-call so that whether Kaggle accepted the kernel is unknowable,
    and losing them means nothing can delete the session one may have started.
    """
    deleted: list[str] = []

    def fake_run(cmd, **kw):
        cmd = [str(c) for c in cmd]
        if cmd[1:3] == ["kernels", "delete"]:
            deleted.append(cmd[3])
            return types.SimpleNamespace(returncode = 0, stdout = "", stderr = "")
        raise subprocess.TimeoutExpired(cmd, launch.PUSH_SUBPROCESS_TIMEOUT_SEC)

    monkeypatch.setattr(launch.subprocess, "run", fake_run)
    monkeypatch.setattr(launch.time, "sleep", lambda _s: None)

    pushed = launch.push(Path(__file__), "someuser", 3600)
    assert pushed["ok"] is False
    # A timeout is Kaggle under load, so it retries like any throttle, and every
    # slug filed comes back for the caller to reconcile.
    assert len(pushed["attempts"]) == launch.PUSH_ATTEMPTS
    assert len(set(pushed["attempts"])) == launch.PUSH_ATTEMPTS
    assert "timed out" in pushed["detail"]
    # Each attempt discards the previous one before adding another session.
    assert deleted == pushed["attempts"][:-1]


def test_a_push_that_times_out_does_not_abandon_the_kernel_already_accepted(monkeypatch, tmp_path):
    """The case that costs quota: kernel 1 is up when kernel 2's push hangs.

    Nothing caught that exception, so `main()` exited without `release()` and
    without writing launch_result.json, and the workflow has no cleanup step of
    its own. The accepted kernel then billed to its ceiling with nobody reading
    its result, and Kaggle's push-time timeout has been measured not to stop a
    wedged one.
    """
    deleted: list[str] = []
    pushes = {"n": 0}

    def fake_run(cmd, **kw):
        cmd = [str(c) for c in cmd]
        if cmd[1:3] == ["kernels", "delete"]:
            deleted.append(cmd[3])
            return types.SimpleNamespace(returncode = 0, stdout = "", stderr = "")
        pushes["n"] += 1
        if pushes["n"] == 1:
            return types.SimpleNamespace(
                returncode = 0, stdout = "Kernel version 1 successfully pushed", stderr = ""
            )
        raise subprocess.TimeoutExpired(cmd, launch.PUSH_SUBPROCESS_TIMEOUT_SEC)

    monkeypatch.setattr(launch.subprocess, "run", fake_run)
    monkeypatch.setattr(launch.time, "sleep", lambda _s: None)
    monkeypatch.setattr(launch, "_api", lambda: object())
    monkeypatch.setattr(launch, "wait", lambda api, slug, poll_every, max_wait: "COMPLETE")
    monkeypatch.setattr(
        launch, "fetch_evidence", lambda slug, outdir, **kw: {"notebooks": [], "log": None}
    )
    monkeypatch.setattr(
        launch,
        "extract_reports",
        lambda outdir: [{"label": "control", "model": "m", "passed": True}],
    )
    monkeypatch.delenv("GITHUB_OUTPUT", raising = False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launch.py",
            "--notebook",
            str(tmp_path / "k0.ipynb"),
            "--notebook",
            str(tmp_path / "k1.ipynb"),
            "--user",
            "someuser",
            "--outdir",
            str(tmp_path / "ev"),
            "--expect",
            "2",
        ],
    )
    (tmp_path / "k0.ipynb").write_text("{}", encoding = "utf-8")
    (tmp_path / "k1.ipynb").write_text("{}", encoding = "utf-8")

    assert launch.main() == 0
    result = json.loads((tmp_path / "ev" / "launch_result.json").read_text(encoding = "utf-8"))
    accepted = result["kernels"][0]["slug"]
    assert accepted and accepted in deleted
    # Including every slug the timed-out push filed, any of which may be the
    # session Kaggle accepted and never reported.
    for slug in result["kernels"][1]["attempted"]:
        assert slug in deleted, slug
    assert all(k["released"] for k in result["kernels"])


@pytest.mark.parametrize(
    "boom",
    [
        # `text=True` wraps the pipes in a TextIOWrapper with STRICT error
        # handling, so one byte the locale encoding cannot decode raises out of
        # `subprocess.run` itself, after the request was filed.
        UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte"),
        # And the runner's own answers, which are not the foreseen ones either.
        OSError("cannot allocate memory"),
        MemoryError("the runner ran out"),
    ],
    ids = ["decode", "oserror", "memory"],
)
def test_a_push_that_raises_outside_the_timeout_still_gives_up_its_slug(
    monkeypatch, tmp_path, boom
):
    """The slug is filed BEFORE `kaggle kernels push` is invoked.

    So the failure that loses it is not the push reporting an error -- that is
    reported and reconciled -- but the push raising something the retry loop
    does not handle. Only `TimeoutExpired` was, and every other raise unwound
    past the line that filed the entry, so `release()` iterated a list with no
    entry for this notebook at all and a kernel Kaggle may have accepted was
    left billing to its own ceiling with nobody reading it.

    Reconciliation must therefore not depend on `push()` RETURNING.
    """
    deleted: list[str] = []
    pushes = {"n": 0}

    def fake_run(cmd, **kw):
        cmd = [str(c) for c in cmd]
        if cmd[1:3] == ["kernels", "delete"]:
            deleted.append(cmd[3])
            return types.SimpleNamespace(returncode = 0, stdout = "", stderr = "")
        pushes["n"] += 1
        if pushes["n"] == 1:
            return types.SimpleNamespace(
                returncode = 0, stdout = "Kernel version 1 successfully pushed", stderr = ""
            )
        raise boom

    monkeypatch.setattr(launch.subprocess, "run", fake_run)
    monkeypatch.setattr(launch.time, "sleep", lambda _s: None)
    monkeypatch.setattr(launch, "_api", lambda: object())
    monkeypatch.setattr(launch, "wait", lambda api, slug, poll_every, max_wait: "COMPLETE")
    monkeypatch.delenv("GITHUB_OUTPUT", raising = False)
    (tmp_path / "k0.ipynb").write_text("{}", encoding = "utf-8")
    (tmp_path / "k1.ipynb").write_text("{}", encoding = "utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launch.py",
            "--notebook",
            str(tmp_path / "k0.ipynb"),
            "--notebook",
            str(tmp_path / "k1.ipynb"),
            "--user",
            "someuser",
            "--outdir",
            str(tmp_path / "ev"),
            "--expect",
            "2",
        ],
    )

    assert launch.main() == 0
    result = json.loads((tmp_path / "ev" / "launch_result.json").read_text(encoding = "utf-8"))
    assert result["verdict"] == "infra"
    assert len(result["kernels"]) == 2, "the notebook whose push raised left no entry to reconcile"
    raised_on = result["kernels"][1]
    assert raised_on["attempted"], "the slug that push filed was lost with the exception"
    for slug in raised_on["attempted"]:
        assert slug in deleted, slug
    # And the kernel already accepted goes too, as it did for the timeout.
    assert result["kernels"][0]["slug"] in deleted
    assert all(k["released"] for k in result["kernels"])
    assert result["unreleased"] == []


def _accepting_push(slug: str):
    """A `push` stub Kaggle accepted, with the real one's calling convention.

    Including `attempted`, the caller-owned list the real push fills as it files
    each slug: a stub that only returned the slugs would let a caller that never
    reads the return value pass here and leak a kernel on Kaggle.
    """

    def fake_push(
        notebook,
        user,
        kernel_timeout_sec,
        accelerator = "NvidiaTeslaT4",
        attempted = None,
    ):
        if attempted is not None:
            attempted.append(slug)
        return {"ok": True, "slug": slug, "attempts": [slug]}

    return fake_push


def test_an_abort_anywhere_in_the_launcher_still_deletes_what_it_pushed(monkeypatch, tmp_path):
    """The outer guard, not the timeout specifically.

    Every line after the first push can leave a kernel running if it raises, and
    the runner is the only thing that would delete it. An abort is `infra` by
    this file's contract, nothing having been learned about the code under test,
    so it exits 0 rather than colouring a pull request red.
    """

    def boom(outdir):
        raise MemoryError("the runner ran out")

    deleted: list[str] = []

    def fake_run(cmd, **kw):
        cmd = [str(c) for c in cmd]
        if cmd[1:3] == ["kernels", "delete"]:
            deleted.append(cmd[3])
        return types.SimpleNamespace(returncode = 0, stdout = "", stderr = "")

    monkeypatch.setattr(launch, "_api", lambda: object())
    monkeypatch.setattr(launch, "push", _accepting_push("someuser/unsloth-t4-ci-abcd"))
    monkeypatch.setattr(launch, "wait", lambda api, slug, poll_every, max_wait: "COMPLETE")
    monkeypatch.setattr(
        launch, "fetch_evidence", lambda slug, outdir, **kw: {"notebooks": [], "log": None}
    )
    monkeypatch.setattr(launch, "extract_reports", boom)
    monkeypatch.setattr(launch.subprocess, "run", fake_run)
    monkeypatch.delenv("GITHUB_OUTPUT", raising = False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launch.py",
            "--notebook",
            "k0.ipynb",
            "--user",
            "someuser",
            "--outdir",
            str(tmp_path),
            "--expect",
            "1",
        ],
    )

    assert launch.main() == 0
    assert deleted == ["someuser/unsloth-t4-ci-abcd"]
    result = json.loads((tmp_path / "launch_result.json").read_text(encoding = "utf-8"))
    assert result["verdict"] == "infra"
    assert "MemoryError" in result["reason"]


# --------------------------------------------------------- kernel cleanup


def _drive_one_kernel(monkeypatch, tmp_path, fake_run):
    """`main()` over one kernel that pushes, completes and reports.

    Everything but the delete calls is stubbed, so what comes back reads the
    release path and nothing else.
    """
    monkeypatch.setattr(launch, "_api", lambda: object())
    monkeypatch.setattr(launch, "push", _accepting_push("someuser/unsloth-t4-ci-abcd"))
    monkeypatch.setattr(launch, "wait", lambda api, slug, poll_every, max_wait: "COMPLETE")
    monkeypatch.setattr(
        launch, "fetch_evidence", lambda slug, outdir, **kw: {"notebooks": [], "log": None}
    )
    monkeypatch.setattr(
        launch,
        "extract_reports",
        lambda outdir: [{"label": "control", "model": "m", "passed": True}],
    )
    monkeypatch.setattr(launch.subprocess, "run", fake_run)
    monkeypatch.setattr(launch.time, "sleep", lambda _s: None)
    monkeypatch.delenv("GITHUB_OUTPUT", raising = False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launch.py",
            "--notebook",
            "k0.ipynb",
            "--user",
            "someuser",
            "--outdir",
            str(tmp_path),
            "--expect",
            "1",
        ],
    )
    code = launch.main()
    return code, json.loads((tmp_path / "launch_result.json").read_text(encoding = "utf-8"))


def _refusing_run(
    returncode: int,
    message: str,
    succeed_from: int = 10**6,
):
    """A `subprocess.run` whose deletes fail until the nth attempt."""
    calls: list[str] = []

    def fake_run(cmd, **kw):
        cmd = [str(c) for c in cmd]
        if cmd[1:3] == ["kernels", "delete"]:
            calls.append(cmd[3])
            if len(calls) >= succeed_from:
                return types.SimpleNamespace(returncode = 0, stdout = "", stderr = "")
            return types.SimpleNamespace(returncode = returncode, stdout = "", stderr = message)
        return types.SimpleNamespace(returncode = 0, stdout = "", stderr = "")

    return fake_run, calls


def test_a_delete_kaggle_refused_does_not_count_as_released(monkeypatch, tmp_path, capsys):
    """`subprocess.run` does not raise on a nonzero exit.

    So the release loop used to record every slug as released whatever came
    back, while cleanup is the budget control rather than a tidy-up. The refusal
    below is the one that was live: the client the workflow pinned had no
    `kernels delete` subcommand at all, so argparse answered every delete with
    exit 2 and the run still reported the kernel released while it billed on to
    its own ceiling.
    """
    fake_run, calls = _refusing_run(
        2, "kaggle kernels: error: argument command: invalid choice: 'delete'"
    )
    code, result = _drive_one_kernel(monkeypatch, tmp_path, fake_run)

    assert code == 0
    entry = result["kernels"][0]
    assert entry["released"] is False
    assert entry["released_slugs"] == []
    assert result["unreleased"] == ["someuser/unsloth-t4-ci-abcd"]
    # And it is said out loud, or nobody goes and deletes it by hand.
    out = capsys.readouterr().out
    assert "::warning title=Kaggle kernels may still be running::" in out
    assert "someuser/unsloth-t4-ci-abcd" in out


def test_a_refused_delete_is_retried_before_it_is_given_up_on(monkeypatch, tmp_path):
    """A 5xx or a reset connection is exactly as transient here as it is on
    the push side, and giving up on the first one abandons a live kernel."""
    fake_run, calls = _refusing_run(1, "503 Service Unavailable")
    _code, result = _drive_one_kernel(monkeypatch, tmp_path, fake_run)

    assert len(calls) == launch.DELETE_ATTEMPTS
    assert result["kernels"][0]["released"] is False


def test_a_delete_that_succeeds_on_a_retry_is_released(monkeypatch, tmp_path, capsys):
    """The other direction: the retry has to be able to end in success, or
    the check is just a slower way of always reporting a leak."""
    fake_run, calls = _refusing_run(1, "502 Bad Gateway", succeed_from = 2)
    _code, result = _drive_one_kernel(monkeypatch, tmp_path, fake_run)

    assert len(calls) == 2
    entry = result["kernels"][0]
    assert entry["released"] is True
    assert entry["released_slugs"] == ["someuser/unsloth-t4-ci-abcd"]
    assert result["unreleased"] == []
    assert "::warning title=Kaggle kernels may still be running::" not in capsys.readouterr().out


def test_a_delete_that_never_ran_is_not_a_deletion(monkeypatch, tmp_path):
    """`subprocess.run` raising is the one case the old code did notice, and
    it is still not a released kernel."""

    def fake_run(cmd, **kw):
        cmd = [str(c) for c in cmd]
        if cmd[1:3] == ["kernels", "delete"]:
            raise subprocess.TimeoutExpired(cmd, 180)
        return types.SimpleNamespace(returncode = 0, stdout = "", stderr = "")

    _code, result = _drive_one_kernel(monkeypatch, tmp_path, fake_run)
    assert result["kernels"][0]["released"] is False
    assert result["unreleased"] == ["someuser/unsloth-t4-ci-abcd"]


def test_a_kernel_kaggle_says_is_not_there_is_a_freed_slot_not_a_leak(
    monkeypatch, tmp_path, capsys
):
    """Most slugs release() reconciles were never accepted, or already went.

    push() files a fresh slug per attempt and keeps every one, and a retry's
    _discard() deletes the previous attempt without recording that it worked,
    so reconciliation asks Kaggle a second time about a kernel that is gone.
    Reading that as a failed cleanup spends DELETE_ATTEMPTS on an absent kernel
    -- ahead of the accepted one, which is the only one still billing -- and
    then tells a human to go and delete a slug that does not exist.

    The stderr below is what the pinned client prints for a 404: kagglesdk
    calls `raise_for_status`, and cli.py prints the HTTPError and exits 1.
    """
    fake_run, calls = _refusing_run(
        1,
        "404 Client Error: Not Found for url: "
        "https://api.kaggle.com/v1/kernels.KernelsApiService/DeleteKernel",
    )
    _code, result = _drive_one_kernel(monkeypatch, tmp_path, fake_run)

    assert calls == ["someuser/unsloth-t4-ci-abcd"], "an absent kernel was asked about again"
    entry = result["kernels"][0]
    assert entry["released"] is True
    assert entry["released_slugs"] == ["someuser/unsloth-t4-ci-abcd"]
    assert result["unreleased"] == []
    assert "::warning title=Kaggle kernels may still be running::" not in capsys.readouterr().out


@pytest.mark.parametrize("marker", list(gate.GONE_MARKERS))
def test_cleanup_reads_a_missing_kernel_in_the_gate_s_words(monkeypatch, marker):
    """One vocabulary, taken FROM the gate rather than copied beside it.

    Both files ask the same account the same question through the same client:
    the gate to tell a deleted kernel from an unreadable one before it spends
    quota, cleanup to tell a freed slot from one still billing. A second list
    would drift out of agreement with the first without either being wrong on
    its own, so this parametrises over the gate's own tuple.
    """
    calls: list[list[str]] = []

    def fake_run(cmd, **kw):
        calls.append([str(c) for c in cmd])
        return types.SimpleNamespace(returncode = 1, stdout = "", stderr = f"delete refused: {marker}")

    monkeypatch.setattr(launch.subprocess, "run", fake_run)
    monkeypatch.setattr(launch.time, "sleep", lambda _s: None)

    assert launch.delete_kernel("someuser/gone") is True
    assert len(calls) == 1, calls


def test_a_nonzero_delete_that_is_not_a_missing_kernel_still_retries(monkeypatch):
    """The other half of the same branch.

    A 5xx, a reset connection or an argparse refusal says nothing about whether
    the kernel is up, so trusting the exit code alone would turn a transient
    into a silently abandoned session.
    """
    calls: list[list[str]] = []

    def fake_run(cmd, **kw):
        calls.append([str(c) for c in cmd])
        return types.SimpleNamespace(returncode = 1, stdout = "", stderr = "503 Service Unavailable")

    monkeypatch.setattr(launch.subprocess, "run", fake_run)
    monkeypatch.setattr(launch.time, "sleep", lambda _s: None)

    assert launch.delete_kernel("someuser/maybe-live") is False
    assert len(calls) == launch.DELETE_ATTEMPTS


def test_a_payload_that_cannot_see_its_gpu_reports_instead_of_vanishing(tmp_path, monkeypatch):
    """A CPU-only torch wheel must not exit this job green.

    `device_count() == 0` used to abort the verify cell with a bare assert. The
    run cell is the only other thing that emits a report and is never reached
    from there, so the launcher extracted nothing and called the run `infra`,
    which exits 0, making a dependency regression that breaks CUDA invisible.
    """
    monkeypatch.setattr(build_kernel, "KERNEL_ROOT", str(tmp_path / "src"))
    leg = LEGS["control"]
    # Import probe satisfied, so the cell reaches the GPU check.
    trivial = type(leg)(
        **{
            **{field: getattr(leg, field) for field in leg.__dataclass_fields__},
            "imports": ("json",),
        }
    )
    cells = _payload_cells(trivial)

    # A torch that installed and imports, and sees no card.
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    (stubs / "torch.py").write_text(
        "class _Cuda:\n"
        "    @staticmethod\n"
        "    def device_count():\n"
        "        return 0\n"
        "    @staticmethod\n"
        "    def is_available():\n"
        "        return False\n"
        "cuda = _Cuda()\n",
        encoding = "utf-8",
    )

    outputs = []
    for index in (0, 2):
        script = tmp_path / f"cell{index}.py"
        script.write_text(cells[index], encoding = "utf-8")
        proc = subprocess.run(
            [sys.executable, str(script)],
            capture_output = True,
            text = True,
            timeout = 600,
            env = {**os.environ, "PYTHONPATH": str(stubs)},
        )
        outputs.append(proc.stdout + proc.stderr)
    assert "KAGGLE_T4_CI_PAYLOAD GPU_UNUSABLE" in outputs[1]

    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "t4_control_output.ipynb").write_text(
        json.dumps(
            {
                "cells": [
                    {"cell_type": "code", "outputs": [{"output_type": "stream", "text": text}]}
                    for text in outputs
                ]
            }
        ),
        encoding = "utf-8",
    )
    reports = launch.extract_reports(evidence)
    assert reports, "an unusable GPU produced no report at all"
    assert reports[0]["passed"] is False
    assert any("could not use its GPU" in f for f in reports[0]["failures"])


def test_a_payload_that_writes_malformed_utf8_still_reports(tmp_path, monkeypatch):
    """Bytes that are not UTF-8 are output, not a reason to lose the verdict.

    `subprocess.run(text=True)` decodes strictly, so one malformed byte from a
    native crash handler raises UnicodeDecodeError inside the run cell, before
    the synthetic report below it is printed. Papermill then aborts the cell,
    the launcher extracts no report for this leg and calls the run `partial` or
    `infra`, both of which are green -- on a payload that died.
    """
    monkeypatch.setattr(build_kernel, "KERNEL_ROOT", str(tmp_path / "src"))
    leg = LEGS["control"]
    root = Path(build_kernel._kernel_root(leg))
    root.mkdir(parents = True, exist_ok = True)
    (root / leg.entry).write_text(
        "import sys\n"
        "sys.stdout.buffer.write(b'trained \\xff\\xfe then died\\n')\n"
        "sys.stderr.buffer.write(b'terminate called \\xff\\n')\n"
        "sys.exit(134)\n",
        encoding = "utf-8",
    )

    run_cell = _payload_cells(leg)[3].replace("/kaggle/working", str(tmp_path))
    script = tmp_path / "run_cell.py"
    script.write_text(run_cell, encoding = "utf-8")
    proc = subprocess.run(
        [sys.executable, str(script)],
        capture_output = True,
        text = True,
        errors = "replace",
        timeout = 600,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "UnicodeDecodeError" not in proc.stderr

    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "t4_control_output.ipynb").write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "code",
                        "outputs": [{"output_type": "stream", "text": proc.stdout}],
                    }
                ]
            }
        ),
        encoding = "utf-8",
    )
    reports = launch.extract_reports(evidence)
    assert reports, "a payload that died with undecodable output produced no report"
    assert reports[0]["passed"] is False
    assert reports[0]["returncode"] == 134


# ------------------------------------------------------- evidence budget
#
# The phase between the last poll and release(). The kernels are still billing
# here and nothing else deletes them, so a collection that outlasts the job
# deadline is how a runner gets killed with kernels up. What follows is that
# bound, exercised rather than restated.


class _SlowPages:
    """A `kernels/output` endpoint that paginates forever and answers slowly.

    Both halves of the worst case in one stub: `hasNextPageToken` never clears,
    so the listing walks its whole page limit, and every call sits until its
    OWN timeout expires, which is what a socket at Kaggle's ceiling does.
    """

    def __init__(self, clock):
        self.clock = clock
        self.calls: list[int] = []

    def __call__(
        self,
        req,
        timeout = None,
    ):
        self.calls.append(timeout)
        self.clock.advance(timeout)
        body = json.dumps(
            {"files": [], "log": "", "hasNextPageToken": True, "nextPageToken": "more"}
        ).encode()
        return _Response(body)


class _Response:
    """A whole body, in one piece, however it is asked for.

    ``read(amt)`` and ``read1`` are what an ``HTTPResponse`` offers and what the
    chunked reader uses; a fake that only answers ``read()`` would make the
    reader untestable rather than the code wrong.
    """

    def __init__(self, body: bytes):
        self.body = body
        self.pos = 0

    def read(self, amt = None):
        if amt is None or amt < 0:
            chunk, self.pos = self.body[self.pos :], len(self.body)
            return chunk
        return self.read1(amt)

    def read1(self, amt = -1):
        end = len(self.body) if amt is None or amt < 0 else self.pos + amt
        chunk = self.body[self.pos : end]
        self.pos += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Clock:
    """A monotonic stand-in for time.time() that only moves when told."""

    def __init__(self, start: float = 1000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_a_paginating_output_endpoint_cannot_outlast_the_evidence_budget(monkeypatch, tmp_path):
    """The P1 this constant exists for.

    Unbounded, one kernel's listing is OUTPUT_PAGE_LIMIT pages at the socket
    ceiling -- 2400s -- and two kernels are 4800s against the 600s the job
    deadline budgets for the whole phase. The runner is then killed here,
    taking finish() -> release() with it, and the kernels it pushed keep
    billing: the single outcome that deadline exists to prevent.
    """
    clock = _Clock()
    monkeypatch.setattr(launch.time, "time", clock)
    monkeypatch.setenv("KAGGLE_API_TOKEN", "not-a-real-token")
    slow = _SlowPages(clock)
    monkeypatch.setattr(launch.urllib.request, "urlopen", slow)

    started = clock()
    deadline = started + launch.EVIDENCE_BUDGET_SEC
    listing = launch.list_outputs("someuser/k", timeout = 120, deadline = deadline)

    spent = clock() - started
    assert (
        spent <= launch.EVIDENCE_BUDGET_SEC
    ), f"the listing spent {spent}s against a {launch.EVIDENCE_BUDGET_SEC}s budget"
    # Unbounded this is OUTPUT_PAGE_LIMIT x 120s; the budget is what stopped it.
    assert len(slow.calls) < launch.OUTPUT_PAGE_LIMIT
    assert all(t <= 120 for t in slow.calls)
    assert listing["truncated"] is True, "an incomplete listing must say so"


def test_the_evidence_budget_is_shared_by_every_kernel(monkeypatch, tmp_path):
    """One budget for the phase, not one per kernel.

    Per kernel the term scales with the kernel count, and the job deadline is
    derived from a single number; the second kernel of a run whose first
    kernel spent the budget collects nothing rather than doubling the bound.
    """
    clock = _Clock()
    monkeypatch.setattr(launch.time, "time", clock)
    monkeypatch.setenv("KAGGLE_API_TOKEN", "not-a-real-token")
    monkeypatch.setattr(launch.urllib.request, "urlopen", _SlowPages(clock))

    started = clock()
    deadline = started + launch.EVIDENCE_BUDGET_SEC
    for slug in ("someuser/a", "someuser/b"):
        launch.fetch_evidence(slug, tmp_path / slug.split("/")[-1], deadline = deadline)
    assert clock() - started <= launch.EVIDENCE_BUDGET_SEC


def test_a_slow_notebook_download_cannot_outlast_the_evidence_budget(monkeypatch, tmp_path):
    """Downloads are the other half: Kaggle caps neither their size nor count.

    Each executed notebook is a 300s call and the listing decides how many
    there are, so an endpoint offering twenty of them is 6000s of downloads
    with nothing to stop them.
    """
    clock = _Clock()
    monkeypatch.setattr(launch.time, "time", clock)
    monkeypatch.setenv("KAGGLE_API_TOKEN", "not-a-real-token")

    files = [
        {"fileName": f"nb{i}{launch.OUTPUT_SUFFIX}", "url": f"https://example.invalid/{i}"}
        for i in range(20)
    ]

    def urlopen(req, timeout = None):
        url = getattr(req, "full_url", "")
        clock.advance(timeout)
        if "kernels/output" in url:
            return _Response(json.dumps({"files": files, "log": "x"}).encode())
        return _Response(b"{}")

    monkeypatch.setattr(launch.urllib.request, "urlopen", urlopen)
    started = clock()
    evidence = launch.fetch_evidence(
        "someuser/k", tmp_path / "k", deadline = started + launch.EVIDENCE_BUDGET_SEC
    )
    spent = clock() - started
    assert spent <= launch.EVIDENCE_BUDGET_SEC, f"downloads spent {spent}s"
    assert len(evidence["notebooks"]) < len(files)
    assert evidence["truncated"] is True


class _Socket:
    """The live socket under a response, recording every re-clamp."""

    def __init__(self):
        self.timeouts: list[float] = []

    def settimeout(self, seconds):
        self.timeouts.append(seconds)


class _Trickle:
    """A response that keeps returning bytes instead of stalling.

    The case a socket timeout cannot see. `urlopen(timeout=...)` bounds each
    blocking socket operation, so an endpoint that answers every read -- slowly,
    but with data -- renews it forever and never trips it. Each chunk here
    advances the clock by `per_chunk`; a whole-body `read()` advances by all of
    them at once, which is what one unbounded `resp.read()` costs.
    """

    def __init__(self, clock, body: bytes, chunks: int, per_chunk: float):
        self.clock = clock
        self.body = body
        self.size = max(1, len(body) // chunks)
        self.per_chunk = per_chunk
        self.pos = 0
        self.reads = 0
        self.fp = type("fp", (), {"raw": type("raw", (), {"_sock": _Socket()})()})()

    def read(self, amt = None):
        if amt is None or amt < 0:
            # The unbounded read: the socket kept feeding it, so it returned
            # only once the whole body was through.
            self.clock.advance(self.per_chunk * (len(self.body) / self.size))
            self.pos = len(self.body)
            return self.body
        return self.read1(amt)

    def read1(self, amt = -1):
        self.reads += 1
        if self.pos >= len(self.body):
            return b""
        take = self.size if amt is None or amt < 0 else min(amt, self.size)
        chunk = self.body[self.pos : self.pos + take]
        self.pos += len(chunk)
        self.clock.advance(self.per_chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _trickled_listing(
    clock,
    files,
    chunks = 20,
    per_chunk = 60.0,
):
    body = json.dumps({"files": files, "log": "x"}).encode()
    # JSON tolerates trailing whitespace, so padding buys chunks without
    # changing what parses.
    return _Trickle(clock, body + b" " * (chunks * len(body)), chunks, per_chunk)


def test_a_trickling_output_listing_cannot_outlast_the_evidence_budget(monkeypatch, tmp_path):
    """The deadline has to hold DURING the read, not only before it.

    Every check was on the near side of `urlopen`, and the timeout it takes is
    a per-socket-operation one, so a body arriving slowly enough renews it
    indefinitely: 20 chunks a minute apart is 1200s of `resp.read()` against a
    600s budget, spent before release() gets to delete anything. The kernels
    bill for all of it.
    """
    clock = _Clock()
    monkeypatch.setattr(launch.time, "time", clock)
    monkeypatch.setenv("KAGGLE_API_TOKEN", "not-a-real-token")
    resp = _trickled_listing(clock, [])
    monkeypatch.setattr(launch.urllib.request, "urlopen", lambda req, timeout = None: resp)

    started = clock()
    listing = launch.list_outputs(
        "someuser/k", timeout = 120, deadline = started + launch.EVIDENCE_BUDGET_SEC
    )
    spent = clock() - started
    assert spent <= launch.EVIDENCE_BUDGET_SEC, f"the listing read spent {spent}s"
    assert listing["truncated"] is True, "an abandoned listing must say it is incomplete"
    assert resp.pos < len(resp.body), "the read was abandoned, not completed"
    # And the live socket was re-clamped as the budget drained, so a read that
    # starts just inside the deadline cannot block for a full socket timeout
    # past it.
    clamps = resp.fp.raw._sock.timeouts
    assert clamps and all(t <= launch.EVIDENCE_BUDGET_SEC for t in clamps), clamps
    assert clamps == sorted(clamps, reverse = True), clamps


def test_a_trickling_notebook_download_cannot_outlast_the_evidence_budget(monkeypatch, tmp_path):
    """The same hole on the download, where the bodies are unbounded.

    Kaggle caps neither the size nor the count of a kernel's outputs, so this
    is the read most able to run long, and a partial file must not be published
    as evidence either.
    """
    clock = _Clock()
    monkeypatch.setattr(launch.time, "time", clock)
    monkeypatch.setenv("KAGGLE_API_TOKEN", "not-a-real-token")
    files = [{"fileName": f"nb{launch.OUTPUT_SUFFIX}", "url": "https://example.invalid/nb"}]
    payload = json.dumps({"cells": []}).encode()
    download = _Trickle(clock, payload + b" " * (20 * len(payload)), 20, 60.0)

    def urlopen(req, timeout = None):
        if "kernels/output" in getattr(req, "full_url", ""):
            return _Response(json.dumps({"files": files, "log": "x"}).encode())
        return download

    monkeypatch.setattr(launch.urllib.request, "urlopen", urlopen)
    started = clock()
    evidence = launch.fetch_evidence(
        "someuser/k", tmp_path / "k", deadline = started + launch.EVIDENCE_BUDGET_SEC
    )
    spent = clock() - started
    assert spent <= launch.EVIDENCE_BUDGET_SEC, f"the download spent {spent}s"
    assert evidence["notebooks"] == [], evidence
    assert evidence["truncated"] is True
    assert not list((tmp_path / "k").glob("*.part")), "a half-written download was left behind"


def test_main_bounds_the_whole_evidence_phase_it_is_budgeted_for(monkeypatch, tmp_path):
    """main() must actually hand the budget down, on the real call path.

    A deadline the collection loop does not pass through is the bug with a
    constant added to it, and release() runs AFTER this loop: every second
    overspent here is a second the kernels keep billing with the job deadline
    approaching.
    """
    clock = _Clock()
    monkeypatch.setattr(launch.time, "time", clock)
    seen: list[float | None] = []

    def fake_fetch(
        slug,
        outdir,
        timeout = 300,
        deadline = None,
    ):
        seen.append(deadline)
        # Spend the whole budget on the first kernel.
        clock.advance(launch.EVIDENCE_BUDGET_SEC)
        return {"notebooks": [], "log": None, "truncated": True}

    monkeypatch.setattr(
        launch,
        "push",
        lambda nb, user, t, accelerator = "NvidiaTeslaT4", attempted = None: (
            attempted.append(f"{user}/s{len(attempted)}"),
            {"ok": True, "slug": attempted[-1], "attempts": list(attempted)},
        )[1],
    )
    monkeypatch.setattr(launch, "wait", lambda api, slug, every, remaining: "COMPLETE")
    monkeypatch.setattr(launch, "fetch_evidence", fake_fetch)
    monkeypatch.setattr(
        launch,
        "extract_reports",
        lambda outdir: [{"label": "control", "model": "m", "passed": True}],
    )
    monkeypatch.setattr(launch, "_api", lambda: object())
    monkeypatch.setattr(launch, "delete_kernel", lambda slug: True)
    monkeypatch.delenv("GITHUB_OUTPUT", raising = False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launch.py",
            "--notebook",
            "k0.ipynb",
            "--notebook",
            "k1.ipynb",
            "--user",
            "someuser",
            "--outdir",
            str(tmp_path),
            "--expect",
            "1",
        ],
    )
    started = clock()
    assert launch.main() == 0
    assert len(seen) == 2, "both kernels were collected"
    assert seen[0] is not None, "main() never handed the collection a deadline"
    assert seen[0] == seen[1], "the two kernels must share ONE budget, not get one each"
    assert seen[0] - started <= launch.EVIDENCE_BUDGET_SEC


# --------------------------------------------------------- the merged kernel's
# --------------------------------------------------------- two reporters

NOTEBOOK_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "kaggle-t4-notebook-ci.yml"


def test_the_merged_kernel_runs_both_reporters():
    """One kernel, two experiments, so two report steps over one evidence dir.

    Dropping the Studio one is the failure this guard exists for: the kernel
    would still install Studio, still drive the UI on a T4, and the job would
    still go green with nothing said about it. The T4 reporter FILTERS the
    studio-gpu label out, so its section would look complete while the payload
    that half the wall clock went on is unreported.
    """
    source = NOTEBOOK_WORKFLOW.read_text(encoding = "utf-8")
    assert ".github/scripts/kaggle_t4_ci/report.py" in source
    assert ".github/scripts/kaggle_studio_ci/report.py" in source
    assert ".github/scripts/kaggle_studio_ci/collect_evidence.py" in source


def test_the_shared_wheels_are_the_specs_every_leg_holds_in_common():
    """Built once from the very SHAs the legs name -- main AND the ref tested.

    Every leg installs unsloth_zoo and unsloth from the same two pinned SHAs,
    and pip does not cache a VCS build, so run 32679427416 cloned and built
    both FOUR times: install was 149-191s per leg, the largest single phase of
    each and 41% of gpt-oss.

    The list is DERIVED from the legs' own groups, never declared again, and
    that is what keeps it honest. A hand-written copy of the two SHAs would
    drift silently, because a wheel built from the wrong ref installs perfectly
    and fails nothing at all.

    The intersection is the safety property, not an optimisation: a spec only
    one leg carries is part of what that leg tests, and sharing it would make
    the legs agree about the thing they exist to disagree about.
    """
    common = build_kernel._shared_vcs_specs(
        {
            "a": [["unsloth_zoo @ git+u@S1"], ["transformers==5.5.0"]],
            "b": [["unsloth_zoo @ git+u@S1"], ["--upgrade", "transformers"]],
        }
    )
    assert common == ("unsloth_zoo @ git+u@S1",), common

    # Different refs for the same package share NOTHING. Returning either one
    # would hand a leg a wheel built from the other leg's commit.
    assert (
        build_kernel._shared_vcs_specs(
            {
                "a": [["unsloth @ git+u@S1"]],
                "b": [["unsloth @ git+u@S2"]],
            }
        )
        == ()
    )

    # A spec only one leg carries is never shared.
    assert build_kernel._shared_vcs_specs(
        {
            "a": [["x @ git+u@S1"], ["y @ git+u@S9"]],
            "b": [["x @ git+u@S1"]],
        }
    ) == ("x @ git+u@S1",)

    # And on the real legs: BOTH packages, at the refs asked for.
    driver = build_kernel.build_kernel(
        SMOKE_DIR,
        ALL_LEGS,
        unsloth_ref = "PRSHA",
        zoo_ref = "MAINSHA",
        extra_args = (),
        per_run_timeout = 60,
        skip_reference = True,
        shared_wheels = True,
    )
    src = "".join("".join(c["source"]) for c in driver["cells"])
    specs = re.search(r"SHARED_WHEEL_SPECS = (.+)", src).group(1)
    assert "unsloth @ git+" in specs and "unsloth_zoo @ git+" in specs, specs
    assert "PRSHA" in specs and "MAINSHA" in specs, specs
    # Built before any leg can start. A background build loses the race it
    # exists to win: the legs are admitted and start installing at t=0.
    assert src.index('"pip", "wheel"') < src.index(
        "threads = []"
    ), "the wheels are built after the leg workers start, so no leg can use them"


def test_every_leg_gets_its_own_torch_and_triton_cache(tmp_path):
    """A separate venv never protected these, and nobody noticed.

    UNSLOTH_COMPILE_LOCATION was set per leg and assumed to be the whole story.
    torch and triton key their caches off $TMPDIR rather than off the
    interpreter: on torch 2.9.1 an unset TORCHINDUCTOR_CACHE_DIR resolves to
    `tempfile.gettempdir()/torchinductor_$USER`
    (torch/_inductor/runtime/cache_dir_utils.py:22) and the triton cache lands
    under that same directory. So four legs whose entire purpose is to install
    DIFFERENT transformers/TRL/peft versions and compile the same modules were
    sharing one /tmp/torchinductor_root.

    Asserted as DISTINCT per leg rather than merely present -- one directory
    named once and handed to everybody would satisfy "is set" and reproduce the
    bug exactly.
    """
    driven = _drive_packed(tmp_path, ALL_LEGS, gpus = 2)
    seen: dict[str, set] = {}
    for call in driven["stub"].papermill:
        env = call.get("env") or {}
        for key in (
            "TORCHINDUCTOR_CACHE_DIR",
            "TRITON_CACHE_DIR",
            "TMPDIR",
            "UNSLOTH_COMPILE_LOCATION",
        ):
            assert env.get(key), f"{call['notebook']} has no {key}"
            seen.setdefault(key, set()).add(env[key])
    for key, values in seen.items():
        assert len(values) == len(
            driven["stub"].papermill
        ), f"{key} is shared between legs: {sorted(values)}"


def test_two_dispatches_can_hold_the_two_kaggle_slots_at_once():
    """One concurrency group cannot express "at most two", and one was used.

    A Kaggle account allows 2 concurrent GPU sessions and this job takes one,
    so the cap is 2 -- but GitHub concurrency is 1 per group. The single group
    did not merely serialise: only ONE run may be PENDING in a group, so a
    second queued run CANCELS the first instead of queueing behind it. That
    killed run 32674255736 and made an A/B impossible to run at all, which is
    how two "different" configurations came to be compared against each other
    while executing the same schedule.

    Bounded by construction is the property worth guarding: the input offers
    exactly two slots, so the account can never be asked for a third session.
    """
    source = NOTEBOOK_WORKFLOW.read_text(encoding = "utf-8")
    workflow = yaml.safe_load(source)
    slot = workflow[True]["workflow_dispatch"]["inputs"]["slot"]
    assert slot["type"] == "choice", slot
    assert slot["options"] == ["1", "2"], (
        f"the slot input offers {slot['options']}, so the account could be "
        f"asked for more than its 2 concurrent sessions"
    )
    assert slot.get("default") == "1", slot

    # BOTH levels, because the discard rule applies to both and fixing only the
    # job left the whole thing broken in exactly the same way: the two runs
    # still shared one workflow-level group, so the second dispatch discarded
    # the first while it was pending and never reached the job group at all.
    for scope, block in (
        ("workflow", workflow["concurrency"]),
        ("job", workflow["jobs"]["t4-smoke"]["concurrency"]),
    ):
        group = block["group"]
        assert "inputs.slot" in group, f"{scope}: {group}"
        # A non-dispatch event has no input and must land in a single shared
        # slot, or every push would get a session of its own.
        assert "'1'" in group, f"{scope}: {group}"
        assert block["cancel-in-progress"] is False, scope


def test_the_shared_wheel_build_is_opt_in():
    """Measured once, attributable to nothing, so it ships behind a flag.

    On run 32689629906 the wheels helped the leg that runs ALONE (gpt-oss
    install 191.2s -> 152.6s) and cost the three that run CONCURRENTLY
    (149-163s -> 319-334s). That run also changed the torch/triton cache
    layout, so neither effect can be attributed to either change. A default-on
    optimisation resting on that would be a guess wearing a measurement's
    clothes.
    """
    source = NOTEBOOK_WORKFLOW.read_text(encoding = "utf-8")
    workflow = yaml.safe_load(source)
    inputs = workflow[True]["workflow_dispatch"]["inputs"]
    assert inputs["shared_wheels"].get("default") is False, inputs["shared_wheels"]
    build = source.split("build_kernel.py")[1].split("- name:")[0]
    assert "$SHARED_WHEELS" in build, build
    assert "'--shared-wheels'" in source

    # Off means no wheel build in the kernel at all, not merely an unused one.
    off = build_kernel.build_kernel(
        SMOKE_DIR,
        ALL_LEGS,
        unsloth_ref = "R",
        zoo_ref = "R",
        extra_args = (),
        per_run_timeout = 60,
        skip_reference = True,
    )
    src = "".join("".join(c["source"]) for c in off["cells"])
    assert "SHARED_WHEEL_SPECS = ()" in src, (
        "wheels are off but the kernel still carries specs, so it would spend "
        "the build time and change nothing"
    )


def test_the_workflow_can_actually_reach_studio_concurrent():
    """A CLI flag nothing passes is dead code that reads as a feature.

    This is not hypothetical. Run 32674263571 was dispatched as the VARIANT of
    an A/B on exactly this behaviour. `--studio-concurrent` existed in
    build_kernel's argument parser and was threaded all the way to
    AFTER_GPU_CONCURRENT, the unit tests for it passed, and the workflow never
    passed the flag -- so the kernel built with it False, the "variant" ran the
    control's schedule, and the comparison was a configuration against itself.
    Nothing was red. `AFTER_GPU_SHARED` was simply absent from kernel.log, and
    absence is not something a green tick reports.

    So the chain is asserted end to end: the input exists, something converts
    it into the flag, and the flag reaches the build command.
    """
    source = NOTEBOOK_WORKFLOW.read_text(encoding = "utf-8")
    workflow = yaml.safe_load(source)
    inputs = workflow[True]["workflow_dispatch"]["inputs"]
    assert "studio_concurrent" in inputs, sorted(inputs)
    # Off by default. Sharing costs the coverage property that Studio picks its
    # own card out of two, which the Studio builder keeps deliberately, so this
    # has to be something a dispatch asks for rather than the default shape.
    assert inputs["studio_concurrent"].get("default") is False, inputs["studio_concurrent"]

    build = source.split("build_kernel.py")[1].split("- name:")[0]
    assert "$STUDIO_CONCURRENT" in build, (
        "the build command does not interpolate STUDIO_CONCURRENT, so the "
        "input cannot reach the kernel no matter what it is set to"
    )
    assert "inputs.studio_concurrent" in source
    assert "'--studio-concurrent'" in source or '"--studio-concurrent"' in source

    # And the flag the workflow spells must be one the CLI accepts. A rename on
    # either side would otherwise land as an unrecognised argument at build
    # time, or worse, be silently ignored.
    cli = (CI_DIR / "build_kernel.py").read_text(encoding = "utf-8")
    assert '"--studio-concurrent"' in cli, "build_kernel.py does not define the flag"


def test_the_t4_reporter_is_told_the_leg_count_not_the_payload_count():
    """`payloads` counts Studio; `legs` does not, and this reporter drops it.

    Handing it `payloads` makes a complete four-leg result read as short by one
    forever: it filters the studio-gpu report out and then compares what is
    left against a number that included it.
    """
    source = NOTEBOOK_WORKFLOW.read_text(encoding = "utf-8")
    reporter = source.split(".github/scripts/kaggle_t4_ci/report.py")[1].split("- name:")[0]
    assert "steps.build.outputs.legs" in reporter
    assert "steps.build.outputs.payloads" not in reporter


@pytest.mark.parametrize(
    ("label", "reporter", "expect_red"),
    [
        ("control", "kaggle_t4_ci", True),
        ("control", "kaggle_studio_ci", False),
        ("studio-gpu", "kaggle_t4_ci", False),
        ("studio-gpu", "kaggle_studio_ci", True),
    ],
)
def test_a_failing_payload_only_reddens_the_reporter_that_owns_it(
    tmp_path, label, reporter, expect_red
):
    """The launcher writes ONE verdict for a kernel that now holds two
    unrelated experiments. A reporter reading it directly would announce a
    failure it cannot describe, over a section listing none, and point at the
    wrong half of a 13-minute kernel."""
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    reports = [
        {"label": "control", "passed": label != "control", "steps": []},
        {"label": "studio-gpu", "passed": label != "studio-gpu", "assertions": []},
    ]
    (evidence / "launch_result.json").write_text(
        json.dumps(
            {
                "verdict": "fail",
                "reason": "1 of 2 payload(s) failed their assertions",
                "slug": "u/s",
                "kernel_state": "COMPLETE",
                "reports": reports,
            }
        )
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / ".github" / "scripts" / reporter / "report.py"),
            "--evidence",
            str(evidence),
            "--expect",
            "1",
        ],
        capture_output = True,
        text = True,
    )
    assert (proc.returncode == 1) is expect_red, proc.stdout


def test_the_build_step_actually_packs_studio_in():
    """The whole pipelining claim in this workflow's header rests on one flag.

    Without it the kernel builds four legs, every reporter still renders, the
    Studio section reads NOT RUN with a plausible-sounding reason, and the job
    is green -- which is indistinguishable from a run whose sampling declined.
    """
    source = NOTEBOOK_WORKFLOW.read_text(encoding = "utf-8")
    build = source.split("- name: Build the kernel notebooks")[1].split("- name:")[0]
    assert "--with-studio" in build
    assert "--studio-args" in build


# ------------------------------------------------------------- the prefetch lane


def test_the_prefetch_lane_never_takes_a_card(tmp_path):
    """It is CPU and network work, and a card it held would be a card idle.

    The whole saving is that downloading happens BESIDE training rather than
    in front of it. A prefetch that consumed a GPU slot would move the wait
    rather than remove it, and would also break the packing arithmetic that
    assumes exactly two lanes compete for two cards.
    """
    hub = _HubStub()
    driven = _drive_packed(tmp_path, ALL_LEGS, gpus = 2, prefetch_repos = ("a/big", "b/small"), hub = hub)
    assert driven["stood_down"] is None
    assert hub.calls == ["a/big", "b/small"], hub.calls
    # Every papermill call is a LEG. The prefetch is not one of them, so it
    # cannot have been handed CUDA_VISIBLE_DEVICES.
    assert len(driven["stub"].papermill) == len(ALL_LEGS), driven["stub"].papermill
    # Two legs on a card is legal now (see the VRAM budget); what must hold is
    # that the summed appetite never exceeds what the card has.
    for card, peak in driven["stub"].peak_card_gb.items():
        assert peak <= 13.0, (card, peak, driven["stub"].same_card_overlaps)


def test_the_leg_prefetch_does_not_redirect_hf_home(tmp_path):
    """The legs read the Kaggle image's DEFAULT cache.

    Pointing the lane at a private root is the silent failure this guards: it
    downloads all 12 GB perfectly, into a directory no leg looks in, reports
    success, and the run is green and no faster. Nothing at runtime would say
    so, which is why it is asserted here.
    """
    hub = _HubStub()
    before = os.environ.get("HF_HOME")
    _drive_packed(tmp_path, ALL_LEGS, gpus = 2, prefetch_repos = ("a/big",), hub = hub)
    assert hub.hf_home_at_call == [before], hub.hf_home_at_call
    assert os.environ.get("HF_HOME") == before


def test_a_failing_prefetch_does_not_fail_the_kernel(tmp_path):
    """Graceful degradation is the entire safety argument for shipping this.

    The leg that wants the model still downloads it itself, exactly as it did
    before the lane existed, so a prefetch failure costs seconds. If it could
    fail the kernel it would be a brand new way to go red for something that
    is not under test -- on a payload that is not even the subject of the run.
    """
    hub = _HubStub(fail_for = ("a/big", "b/small"))
    driven = _drive_packed(tmp_path, ALL_LEGS, gpus = 2, prefetch_repos = ("a/big", "b/small"), hub = hub)
    assert driven["stood_down"] is None
    assert len(driven["stub"].papermill) == len(ALL_LEGS), driven["stub"].papermill
    assert all(r.get("returncode") == 0 for r in driven["results"].values()), driven["results"]


def test_no_prefetch_repos_leaves_the_schedule_exactly_as_it_was(tmp_path):
    """The lane is opt-in at the call site, and off means OFF: no thread, no
    huggingface_hub import, no behaviour change for a kernel built without it."""
    hub = _HubStub()
    driven = _drive_packed(tmp_path, ALL_LEGS, gpus = 2, prefetch_repos = (), hub = hub)
    assert hub.calls == [], hub.calls
    assert driven["stood_down"] is None
    assert len(driven["stub"].papermill) == len(ALL_LEGS)


def test_the_prefetch_list_matches_the_models_the_legs_actually_load():
    """A prefetch of the wrong repo downloads happily and warms nothing.

    There is no runtime feedback for this: the lane reports success, the legs
    download for themselves, and the only symptom is a saving that never
    arrives. So the declared list is checked against the DEFAULT_MODEL the
    payload scripts really carry, read out of their source.

    The DEFAULT_MODEL is where an earlier version of this test stopped, and
    stopping there is what let the bug through. What a leg ASKS FOR and what it
    LOADS are different for gpt-oss: `unsloth/gpt-oss-20b` is MXFP4, sm_75
    cannot read MXFP4, and unsloth redirects to `-unsloth-bnb-4bit` at load
    time. The old assertion compared the prefetch list against the declared
    name, so it agreed with a prefetch of 55.1 GB that no leg ever opened. It
    now applies LOAD_REDIRECTS first, and separately pins that the redirect it
    is applying is the one the payload actually documents.
    """
    from legs import LOAD_REDIRECTS, PREFETCH_REPOS

    defaults = set()
    for script in ("run_t4_smoke.py", "run_gptoss_t4.py"):
        text = (SMOKE_DIR / script).read_text(encoding = "utf-8")
        found = re.findall(r'^DEFAULT_MODEL\s*=\s*"([^"]+)"', text, re.MULTILINE)
        assert len(found) == 1, f"{script} declares {found}"
        defaults.add(found[0])

    loaded = {LOAD_REDIRECTS.get(name, name) for name in defaults}
    assert (
        set(PREFETCH_REPOS) == loaded
    ), f"prefetching {sorted(set(PREFETCH_REPOS))} but the legs load {sorted(loaded)}"

    # LOAD_REDIRECTS is only as good as its agreement with the payload. If the
    # redirect ever stops being real, this list must stop claiming it -- or the
    # prefetch goes back to warming a cache nobody reads, in the other
    # direction and just as invisibly.
    gptoss = (SMOKE_DIR / "run_gptoss_t4.py").read_text(encoding = "utf-8")
    for declared, actual in LOAD_REDIRECTS.items():
        assert actual in gptoss, (
            f"LOAD_REDIRECTS says {declared} loads as {actual}, but no payload "
            f"mentions {actual}, so the redirect is asserted and not observed"
        )
    # Qwen FIRST, and the reasoning inverted once the schedule was simulated
    # end to end. gpt-oss is bigger, but it is wanted by ONE leg whose setup
    # does not finish until ~160s anyway, whereas the small model gates THREE
    # legs and costs ~20s. Fetching the big one first pushes the small one out
    # past the moment the first leg is ready and delays three legs to give one
    # a head start it did not need.
    assert "Qwen" in PREFETCH_REPOS[0], PREFETCH_REPOS
    assert "gpt-oss" in PREFETCH_REPOS[-1], PREFETCH_REPOS


def test_the_generated_prefetch_cell_runs_not_merely_compiles():
    """Compiling it is not enough, and that is not hypothetical here.

    The first version interpolated `hf_home` with `json.dumps`, so `None`
    became the JSON literal `null`. It compiled cleanly and died with a
    NameError the first time it RAN -- which on the real thing means minutes
    into a paid Kaggle session.
    """
    prefetch = build_kernel._prefetch_builder()
    hub = _HubStub(hold = 0.0)
    saved = sys.modules.get("huggingface_hub")
    sys.modules["huggingface_hub"] = hub
    try:
        for hf_home in (None, "/tmp/somewhere"):
            source = prefetch.prefetch_cell(
                ["a/b"], hf_home = hf_home, attempt_timeout = 2, total_timeout = 5
            )
            exec(compile(source, "<prefetch>", "exec"), {"__name__": "prefetch"})
    finally:
        if saved is None:
            sys.modules.pop("huggingface_hub", None)
        else:
            sys.modules["huggingface_hub"] = saved
    assert hub.calls == ["a/b", "a/b"], hub.calls


def test_a_repos_allow_patterns_reach_the_hub_and_a_bare_repo_stays_unfiltered():
    """Computing the right glob and not passing it looks identical everywhere.

    The patterns are worked out in the studio builder, carried through
    `_normalise`, interpolated into generated source and echoed into the
    summary. Every one of those steps can be right while the `snapshot_download`
    call omits the keyword, and the only symptom is the 69.1 GB bill this was
    written to stop -- the summary would still print the glob it meant to use.

    The bare-string case is asserted alongside, because "filter everything"
    breaks the opposite way: a small model whose every file is loaded must not
    quietly acquire a filter and arrive incomplete.
    """
    prefetch = build_kernel._prefetch_builder()
    hub = _HubStub(hold = 0.0)
    saved = sys.modules.get("huggingface_hub")
    sys.modules["huggingface_hub"] = hub
    try:
        source = prefetch.prefetch_cell(
            [("big/gguf", ["*UD-Q4_K_XL*"]), "small/model"],
            attempt_timeout = 2,
            total_timeout = 5,
        )
        exec(compile(source, "<prefetch>", "exec"), {"__name__": "prefetch"})
    finally:
        if saved is None:
            sys.modules.pop("huggingface_hub", None)
        else:
            sys.modules["huggingface_hub"] = saved

    assert hub.calls == ["big/gguf", "small/model"], hub.calls
    assert hub.patterns_at_call == [["*UD-Q4_K_XL*"], None], hub.patterns_at_call


def test_the_last_prefetch_attempt_falls_back_to_classic_http():
    """Retrying a STALLING transport is how a retry loop eats the session.

    Xet retries 408/429/5xx itself with backoff (5 attempts, 3s base, a
    six-minute cap per delay), so a throttled transfer can sit inside one call
    for minutes without raising. Repeating the same transport inherits that.
    The escalation to HF_HUB_DISABLE_XET is what makes the last attempt a
    genuinely different thing to try.
    """
    prefetch = build_kernel._prefetch_builder()
    seen: list = []

    class _Recording(_HubStub):
        def snapshot_download(
            self,
            repo_id = None,
            **kw,
        ):
            seen.append(os.environ.get("HF_HUB_DISABLE_XET"))
            raise RuntimeError("always")

    saved = sys.modules.get("huggingface_hub")
    before = os.environ.get("HF_HUB_DISABLE_XET")
    sys.modules["huggingface_hub"] = _Recording(hold = 0.0)
    try:
        source = prefetch.prefetch_cell(["a/b"], attempt_timeout = 1, total_timeout = 30)
        exec(compile(source, "<prefetch>", "exec"), {"__name__": "prefetch"})
    finally:
        if saved is None:
            sys.modules.pop("huggingface_hub", None)
        else:
            sys.modules["huggingface_hub"] = saved
    assert seen[-1] == "1", seen
    assert seen[:-1] == [None] * (len(seen) - 1), seen
    # ...and it is not left set for whatever runs next in this interpreter.
    assert os.environ.get("HF_HUB_DISABLE_XET") == before


def test_the_studio_prefetch_lands_in_studios_own_cache():
    """Studio keeps a private HF_HOME, and the prefetch must follow it there.

    The t4 lane deliberately does the opposite -- it leaves HF_HOME alone so
    the training legs can read what it warms -- so the two are easy to conflate
    and the failure is silent either way: bytes land somewhere real, the
    download reports success, and the payload that wanted them downloads again.

    The install cell passes hf_home=None to INHERIT, which is only correct
    because the setup cell has already exported Studio's root. That ordering is
    what is pinned here; a prefetch cell hoisted above setup would inherit the
    image default and quietly stop helping.
    """
    studio = build_kernel._studio_builder()
    notebook = studio.build_payload_notebook(
        unsloth_ref = "x",
        repo_url = "https://h/r",
        payload_args = "--max-steps 8",
        phase = "install",
    )
    sources = ["".join(cell["source"]) for cell in notebook["cells"]]
    sets_home = [i for i, src in enumerate(sources) if 'os.environ["HF_HOME"]' in src]
    prefetches = [i for i, src in enumerate(sources) if "KAGGLE_CI_PREFETCH" in src]
    assert sets_home, "the install phase never exports Studio's HF_HOME"
    assert prefetches, "the install phase carries no prefetch"
    assert min(sets_home) < min(prefetches), (
        f"HF_HOME is exported at cell {min(sets_home)} but the prefetch runs at "
        f"{min(prefetches)}, so it would warm the image default instead"
    )
    # And it must not hardcode a root of its own alongside the inherited one.
    assert "_HF_HOME = None" in sources[min(prefetches)], sources[min(prefetches)][:400]


def test_the_studio_prefetch_follows_the_dispatched_models():
    """--chat-model and --train-model are dispatch inputs.

    A hardcoded pair here would prefetch the defaults while the payload loaded
    something else -- which downloads happily, warms a cache nobody reads, and
    reports success.

    --chat-variant is read for the same reason and now matters as much. Studio
    loads ONE quant from a GGUF repo that ships many, so an unfiltered snapshot
    is not merely generous: run 32667451396 pulled 69.1 GB of Qwen3.5-2B-GGUF
    to serve a single UD-Q4_K_XL file, and on a 4-core Kaggle box that CPU came
    straight out of the payloads the prefetch exists to speed up.
    """
    studio = build_kernel._studio_builder()
    chat, train = studio._models_from("--chat-model a/b --train-model c/d")
    assert chat == ("a/b", ["*UD-Q4_K_XL*"]), chat
    assert train == "c/d", train
    assert studio._models_from("--chat-model=e/f")[0][0] == "e/f"

    # The filter follows the dispatched variant rather than the default, or a
    # run that overrode it would prefetch a quant it never loads.
    picked, patterns = studio._models_from("--chat-variant Q8_0")[0]
    assert patterns == ["*Q8_0*"], patterns

    # Loose at BOTH ends on purpose. A split GGUF is named
    # `...UD-Q4_K_XL-00001-of-00002.gguf`, so a suffix-anchored glob would
    # match the single-file case and miss every shard of the split one --
    # downloading nothing, reporting success, leaving Studio to fetch it.
    assert patterns[0].startswith("*") and patterns[0].endswith("*"), patterns

    defaults = studio._models_from("--max-steps 8")
    flat = [entry[0] if isinstance(entry, tuple) else entry for entry in defaults]
    payload = (SMOKE_DIR.parent / "studio_gpu" / "run_studio_gpu.py").read_text(encoding = "utf-8")
    for flag in ("--chat-model", "--train-model", "--chat-variant"):
        declared = re.search(rf'ap\.add_argument\("{flag}", default = "([^"]+)"\)', payload)
        assert declared, f"{flag} default not found in run_studio_gpu.py"
        if flag == "--chat-variant":
            assert defaults[0][1] == [f"*{declared.group(1)}*"], defaults[0]
        else:
            assert declared.group(1) in flat, (declared.group(1), flat)


def test_the_report_shows_what_the_prefetch_achieved(tmp_path):
    """The number the leg order is arranged around has to be readable without
    downloading an artifact -- including when it says the lane did not help."""
    import report as t4_report

    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "kernel.log").write_text(
        'KAGGLE_CI_PREFETCH {"repo": "unsloth/gpt-oss-20b", "ok": true, "seconds": 141.0, '
        '"download_seconds": 141.0, "bytes": 12000000000, "mb_per_s": 85.1, '
        '"transport": "auto", "attempts": 1}\n'
        'KAGGLE_CI_PREFETCH {"repo": "unsloth/Qwen2.5-0.5B-Instruct", "ok": false, '
        '"seconds": 9.0, "download_seconds": null, "bytes": 0, "mb_per_s": null, '
        '"transport": "http", "attempts": 3, "error": "nope"}\n',
        encoding = "utf-8",
    )
    lines = "\n".join(t4_report.prefetch_table(evidence))
    assert "unsloth/gpt-oss-20b" in lines
    assert "141.0" in lines and "85.1" in lines and "12.0" in lines
    assert "**NO**" in lines, "a failed prefetch must be visible, not rounded away"
    assert "fallback" in lines, "a failed prefetch must say what it costs the schedule"
    # A kernel built without the lane gets no section at all, rather than a
    # table of zeroes that reads like a lane that ran and achieved nothing.
    bare = tmp_path / "bare"
    bare.mkdir()
    (bare / "kernel.log").write_text("nothing to see", encoding = "utf-8")
    assert t4_report.prefetch_table(bare) == []

    # ...and it is WIRED IN. Calling the renderer directly proves it renders,
    # which is not the same claim: deleting the one line that appends it to the
    # summary left this test green, because a table nobody calls still formats
    # perfectly. So drive main() and read what a human would actually see.
    (evidence / "launch_result.json").write_text(
        json.dumps(
            {
                "verdict": "pass",
                "reason": "all 1 payload(s) passed",
                "slug": "u/s",
                "kernel_state": "COMPLETE",
                "reports": [{"label": "control", "passed": True, "steps": []}],
            }
        ),
        encoding = "utf-8",
    )
    summary = tmp_path / "summary.md"
    proc = subprocess.run(
        [sys.executable, str(CI_DIR / "report.py"), "--evidence", str(evidence), "--expect", "1"],
        capture_output = True,
        text = True,
        env = {**os.environ, "GITHUB_STEP_SUMMARY": str(summary)},
    )
    assert proc.returncode == 0, proc.stdout
    rendered = summary.read_text(encoding = "utf-8")
    assert "model prefetch" in rendered, rendered
    assert "unsloth/gpt-oss-20b" in rendered, rendered


def test_gptoss_never_shares_a_card(tmp_path):
    """12.78 GB of a 14.56 GB card, so it is alone by arithmetic.

    Not by a special case -- there is no `if name == "gptoss"` anywhere. If a
    leg's appetite ever grows past the budget it stops sharing on its own, and
    if gptoss ever shrinks it starts sharing on its own. What must never happen
    is the pairing that put 13.48 GB on a card and came back as an OOM reading
    like a code failure.
    """
    durations = {f"t4_{n}.ipynb": 0.4 for n in ALL_LEGS}
    driven = _drive_packed(tmp_path, ALL_LEGS, gpus = 2, durations = durations)
    for card, together in driven["stub"].same_card_overlaps:
        assert "t4_gptoss.ipynb" not in together, (card, together)


def test_two_small_legs_do_share_a_card(tmp_path):
    """The feature, asserted positively.

    Every other guard here is a bound -- never over budget, never more than two
    -- and every one of them is satisfied by a scheduler that co-schedules
    NOTHING. Without this the whole change could silently do no work at all and
    the suite would stay green.
    """
    durations = {f"t4_{n}.ipynb": 0.4 for n in ALL_LEGS}
    driven = _drive_packed(tmp_path, ALL_LEGS, gpus = 2, durations = durations)
    assert driven[
        "stub"
    ].same_card_overlaps, "no card ever held two legs at once, so the VRAM budget bought nothing"
    assert max(driven["stub"].peak_card_legs.values()) == 2
    # The admission ledger has to balance. A leg that reserves and never
    # releases leaks capacity, and with four legs and four worker slots nothing
    # ever waits, so the leak is invisible until the day a fifth leg is wired
    # and one card silently stops taking work.
    assert driven["card_load"] and all(abs(v) < 1e-9 for v in driven["card_load"].values()), driven[
        "card_load"
    ]
    assert all(v == 0 for v in driven["card_count"].values()), driven["card_count"]


def test_the_declared_vram_matches_what_the_legs_reported():
    """`Leg.vram_gb` decides who may share a card, and nothing checks it at
    runtime: a leg that under-declares gets admitted beside another and the
    contention comes back as an OOM attributed to whichever leg happened to
    allocate last.

    So the declared figures are checked against the peaks the payloads really
    reported, captured in the evidence of run 32611343797 and committed beside
    this test.
    """
    measured = json.loads(
        (Path(__file__).parent / "t4_smoke" / "measured_vram.json").read_text(encoding = "utf-8")
    )
    for name, peak in measured["peak_reserved_gb"].items():
        declared = LEGS[name].vram_gb
        assert declared >= peak, (
            f"{name} declares {declared} GB but peaked at {peak} GB, so the "
            "admission check would let something share a card with it that "
            "does not fit"
        )
        # ...and not so far above it that the budget stops admitting anything.
        assert declared <= peak + 1.5, (name, declared, peak)
    assert measured["card_total_gb"] > 13.0, measured


# ------------------------------------------------- Studio sharing a card


def test_studio_waits_for_the_queue_by_default(tmp_path, monkeypatch):
    """The default keeps both T4s visible to Studio.

    Sharing is faster and narrower, so it must be something someone turned on,
    not something that arrived with an unrelated change.
    """
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", AMBIENT_CUDA)
    durations = {f"t4_{n}.ipynb": 0.4 for n in ALL_LEGS}
    durations[STUDIO_INSTALL] = 0.1
    driven = _drive_packed(tmp_path, ALL_LEGS, gpus = 2, studio = STUDIO, durations = durations)
    calls = {c["notebook"]: c for c in driven["stub"].papermill}
    assert calls[STUDIO_TEST]["cuda"] == AMBIENT_CUDA, calls[STUDIO_TEST]
    legs = [
        (c["notebook"], c["cuda"])
        for c in driven["stub"].papermill
        if c["notebook"].startswith("t4_")
    ]
    assert len(legs) == len(ALL_LEGS)


def test_studio_concurrent_takes_a_card_gptoss_is_not_on(tmp_path):
    """--studio-concurrent trades coverage for time and must pay honestly.

    Two things have to hold. Studio is PINNED to a card, because sharing means
    it is no longer choosing between two. And it is admitted by the same VRAM
    check the legs use, so it can never land beside gptoss: 12.78 + 2.2 is
    14.98 on a card budgeted to 13.0, which is the pairing that came back as an
    OOM reading like a code failure.

    Driven with gptoss as the ONLY leg, so the placement is deterministic
    rather than a race between the stub's durations. An earlier version used
    all four legs and asserted on recorded overlaps; the small legs finished
    while Studio was still building its venv, so no overlap was ever recorded
    and deleting the VRAM check left the test green.
    """
    driven = _drive_packed(
        tmp_path,
        ["gptoss"],
        gpus = 2,
        studio = STUDIO,
        durations = {"t4_gptoss.ipynb": 4.0, STUDIO_INSTALL: 0.05},
        after_gpu_concurrent = True,
    )
    calls = {c["notebook"]: c for c in driven["stub"].papermill}
    assert STUDIO_TEST in calls, sorted(calls)
    assert calls[STUDIO_TEST]["cuda"] in ("0", "1"), calls[STUDIO_TEST]
    assert calls["t4_gptoss.ipynb"]["cuda"] in ("0", "1"), calls["t4_gptoss.ipynb"]
    assert calls[STUDIO_TEST]["cuda"] != calls["t4_gptoss.ipynb"]["cuda"], (
        f"Studio was put on the same card as gptoss "
        f"({calls[STUDIO_TEST]['cuda']}): 12.78 + 2.2 GB on a 13.0 GB budget"
    )
    for card, peak in driven["stub"].peak_card_gb.items():
        assert peak <= 13.0, (card, peak, driven["stub"].same_card_overlaps)


def test_studio_concurrent_still_skips_when_its_install_failed(tmp_path):
    """The dependency survives the faster path.

    Running the assertions against a half-built tree fails on a missing venv,
    which reads like the code under test broke rather than like the install
    did -- and on this path the test half is started from its own thread, so
    the gate had to be re-implemented rather than inherited.
    """

    class _InstallFails(_PackedStub):
        def run(self, cmd, **kw):
            cmd = [str(c) for c in cmd]
            if "papermill" in cmd and STUDIO_INSTALL in " ".join(cmd):
                self.papermill.append(
                    {
                        "notebook": STUDIO_INSTALL,
                        "cuda": None,
                        "kernel": None,
                        "compile_location": None,
                    }
                )
                Path(cmd[cmd.index("papermill") + 2]).write_text("{}", encoding = "utf-8")
                return types.SimpleNamespace(returncode = 1, stdout = "", stderr = "")
            return super().run(cmd, **kw)

    driver = build_kernel.build_kernel(
        SMOKE_DIR,
        ALL_LEGS,
        unsloth_ref = "main",
        zoo_ref = "main",
        extra_args = (),
        per_run_timeout = 60,
        skip_reference = True,
        studio = STUDIO,
        after_gpu_concurrent = True,
    )
    stub = _InstallFails(gpus = 2, durations = {f"t4_{n}.ipynb": 0.3 for n in ALL_LEGS})
    stub.root = tmp_path
    stub.venv_root = tmp_path / "venvs"
    saved = sys.modules["subprocess"]
    sys.modules["subprocess"] = stub
    namespace: dict = {}
    try:
        for cell in driver["cells"][:2]:
            source = (
                "".join(cell["source"])
                .replace("/tmp/t4ci_venvs", str(tmp_path / "venvs"))
                .replace("/kaggle/working", str(tmp_path))
            )
            exec(compile(source, "<driver-cell>", "exec"), namespace)
    finally:
        sys.modules["subprocess"] = saved
    results = namespace.get("results") or {}
    assert STUDIO_TEST in results, sorted(results)
    assert results[STUDIO_TEST]["returncode"] is None, results[STUDIO_TEST]
    assert "install lane did not succeed" in results[STUDIO_TEST]["error"]
    assert STUDIO_TEST not in [c["notebook"] for c in stub.papermill]


def test_a_legs_overlay_reaches_its_payload_and_never_carries_torch(tmp_path):
    """The overlay must WIN over the venv, and must not bring native packages.

    Two failures are being guarded, and they look identical from outside:

    * An overlay built but never put on ``PYTHONPATH``. The leg runs on the base
      versions, trains, passes, and reports a version table nobody reads. This
      is the reason the payload's env is inspected rather than the fact that a
      ``pip install --target`` happened.
    * An overlay that shadows torch. ``pip install --dry-run --report`` resolves
      the FULL closure, and a closure containing transformers frequently
      contains torch too; installing that into the overlay puts a second torch
      ahead of the one already loaded against this box's CUDA runtime. The stub
      resolver therefore returns torch on purpose, so a driver that forgot to
      filter fails here instead of on a Kaggle session.

    Measured basis for the mechanism: kernel unsloth-probe-overlay-t4-r2-38ac4d
    on a real T4 resolved transformers==4.57.6 + trl~=0.22.0 to three packages,
    115.9 MB, in 10.0s, with transformers and trl imported from the overlay and
    torch still from the base.
    """
    leg = "canary"
    overlay = ("transformers==4.57.6", "trl~=0.22.0")
    original = LEGS[leg].overlay
    object.__setattr__(LEGS[leg], "overlay", overlay)
    try:
        stub = _drive_packed(tmp_path, [leg], gpus = 2)["stub"]
    finally:
        object.__setattr__(LEGS[leg], "overlay", original)

    installs = [c for c in stub.overlay_installs if "--target" in c]
    assert installs, "the leg declared an overlay and nothing was installed into one"
    target = installs[0][installs[0].index("--target") + 1]
    assert f"overlay_t4_{leg}" in target, target

    installed = " ".join(installs[0]).lower()
    assert "transformers==4.57.6" in installed, installed
    assert (
        "torch==" not in installed
    ), f"the overlay installed torch, which shadows the base one: {installed}"

    record = [p for p in stub.papermill if p["notebook"] == f"t4_{leg}.ipynb"]
    assert record, [p["notebook"] for p in stub.papermill]
    pythonpath = record[0]["env"].get("PYTHONPATH", "")
    assert target in pythonpath.split(os.pathsep), (
        f"the overlay was built at {target} but the payload's PYTHONPATH is "
        f"{pythonpath!r}, so the child would import the base versions"
    )


def test_a_leg_with_no_overlay_gets_no_pythonpath(tmp_path):
    """The control case, without which the test above proves only that a
    variable exists somewhere.

    A driver that unconditionally set PYTHONPATH -- to the overlay root, to an
    empty directory, to anything -- would satisfy the first guard while giving
    every leg the same environment. The legs' whole purpose is that they differ.
    """
    stub = _drive_packed(tmp_path, ["control"], gpus = 2)["stub"]
    assert not [
        c for c in stub.overlay_installs if "--target" in c
    ], "a leg declaring no overlay had one built for it"
    record = [p for p in stub.papermill if p["notebook"] == "t4_control.ipynb"]
    assert record
    assert "overlay_" not in record[0]["env"].get("PYTHONPATH", "")


def test_every_leg_installs_bitsandbytes_and_probes_that_it_imports():
    """bitsandbytes has to be asked for, and asked for EARLY.

    It is absent from every dependency set the CI resolves: `unsloth_zoo`
    declares 57 requirements and bitsandbytes is not among them, and git-main
    `unsloth` declares only seven unconditional dependencies (typer, rich,
    pydantic, pyyaml, nest-asyncio, structlog, click) with bitsandbytes reachable
    only through its CUDA extras. The released PyPI package DOES carry it
    unconditionally, which is why notebooks installing from PyPI never notice --
    and why this CI, which installs from git SHAs, must ask.

    Without it the run gets a long way before failing: the install succeeds, the
    model downloads, and it dies inside `from_pretrained` at
    unsloth_zoo/patching_utils.py:386. Probing it in the import cell turns that
    into a failure before the session is spent, which is the whole point of the
    probe list.
    """
    for name, leg in LEGS.items():
        flat = [spec for group in leg.install for spec in group]
        assert any("bitsandbytes" in spec for spec in flat), (
            f"leg {name!r} never installs bitsandbytes; on the Kaggle image it "
            f"would fail inside from_pretrained after the model download"
        )
        assert "bitsandbytes" in leg.imports, (
            f"leg {name!r} does not probe bitsandbytes, so a broken or missing "
            f"copy surfaces minutes later as a model-loading error"
        )
