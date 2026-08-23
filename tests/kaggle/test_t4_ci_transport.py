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
import subprocess
import sys
import tempfile
import threading
import time
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_DIR = REPO_ROOT / "tests" / "kaggle" / "t4_smoke"
CI_DIR = REPO_ROOT / ".github" / "scripts" / "kaggle_t4_ci"

sys.path.insert(0, str(SMOKE_DIR))
sys.path.insert(0, str(CI_DIR))

import build_kernel  # noqa: E402
import gate  # noqa: E402
import launch  # noqa: E402
from legs import LEGS  # noqa: E402


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
        if "papermill" in cmd:
            env = kw.get("env") or {}
            self.papermill.append(
                {
                    "notebook": Path(cmd[cmd.index("papermill") + 1]).name,
                    "cuda": env.get("CUDA_VISIBLE_DEVICES"),
                    "kernel": cmd[cmd.index("-k") + 1],
                    "compile_location": env.get("UNSLOTH_COMPILE_LOCATION"),
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
            source = "".join(cell["source"]).replace("/kaggle/working", str(tmp_path))
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
    ):
        super().__init__(gpus = gpus)
        self.durations = durations or {}
        self.hold = hold
        self._live_on_card: dict = {}
        self._lock = threading.Lock()
        self.same_card_overlaps: list = []
        self.max_live_venvs = 0
        self.root: Path | None = None

    def run(self, cmd, **kw):
        cmd = [str(c) for c in cmd]
        if len(cmd) > 2 and cmd[1] == "venv":
            Path(cmd[2]).mkdir(parents = True, exist_ok = True)
        if "papermill" in cmd:
            notebook = Path(cmd[cmd.index("papermill") + 1]).name
            card = (kw.get("env") or {}).get("CUDA_VISIBLE_DEVICES")
            with self._lock:
                if self._live_on_card.get(card):
                    self.same_card_overlaps.append((card, self._live_on_card[card], notebook))
                self._live_on_card[card] = notebook
                if self.root is not None:
                    live = len(list(self.root.glob("venv_*")))
                    self.max_live_venvs = max(self.max_live_venvs, live)
            time.sleep(self.durations.get(notebook, self.hold))
            with self._lock:
                self._live_on_card[card] = None
        return super().run(cmd, **kw)


def _drive_packed(
    tmp_path,
    leg_names,
    *,
    gpus,
    durations = None,
    studio = None,
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
    )
    stub = _PackedStub(gpus = gpus, durations = durations)
    stub.root = tmp_path
    saved = sys.modules["subprocess"]
    sys.modules["subprocess"] = stub
    namespace: dict = {}
    raised = None
    try:
        for cell in driver["cells"][:2]:
            source = "".join(cell["source"]).replace("/kaggle/working", str(tmp_path))
            try:
                exec(compile(source, "<driver-cell>", "exec"), namespace)
            except SystemExit as exc:
                raised = exc
                break
    finally:
        sys.modules["subprocess"] = saved
    return {"stood_down": raised, "stub": stub, "results": namespace.get("results") or {}}


ALL_FOUR = ["gptoss", "frontier", "canary", "control"]


def test_four_legs_on_two_cards_never_put_two_legs_on_one_card_at_once(tmp_path):
    """The property that makes packing safe at all.

    Four payloads across two T4s is only sound because a card takes its next
    leg when the previous one has EXITED. If they overlapped, each child would
    still pass its own `device_count() == 1` assertion and then fight for 15GB,
    which is exactly the failure the shortfall guard was added to prevent and
    exactly the one it cannot see from where it stands.
    """
    driven = _drive_packed(tmp_path, ALL_FOUR, gpus = 2)
    assert driven["stood_down"] is None
    stub = driven["stub"]
    assert stub.same_card_overlaps == [], stub.same_card_overlaps
    assert len(stub.papermill) == 4, stub.papermill
    # Both cards are used, and every leg ran. The SPLIT is deliberately not
    # asserted: how many legs each card ends up with is a function of how long
    # the legs take relative to the 5s venv stagger, not something the
    # scheduler promises. Under the measured durations (gptoss 384.1s,
    # frontier 312.2s, canary 265.3s, control 262.2s) the stagger is under 2%
    # and the split is 2/2; under the sub-second stubs here the first card
    # legitimately drains most of the queue before the second clears its
    # stagger. Pinning 2/2 would be pinning the stub's timing.
    assert set(p["cuda"] for p in stub.papermill) == {"0", "1"}, stub.papermill


def test_the_longest_leg_starts_first_so_the_schedule_can_balance_around_it(tmp_path):
    """Start order is longest-first, and it is load bearing rather than tidy.

    Measured on run 32607621452: gptoss 384.1s, frontier 312.2s, canary 265.3s,
    control 262.2s. Longest-first packs those as 646.3s, which is the optimal
    split of the four; `sorted(PAYLOADS)` would start gptoss LAST, and a greedy
    scheduler cannot balance around the leg that sets the makespan if it picks
    it up at the end.
    """
    driven = _drive_packed(tmp_path, ALL_FOUR, gpus = 2)
    started = [p["notebook"] for p in driven["stub"].papermill]
    assert started[0] == "t4_gptoss.ipynb", started
    assert started != sorted(
        started
    ), "payloads are running in alphabetical order, so the longest leg is last"


def test_each_leg_keeps_its_own_venv_compile_cache_and_ipykernel(tmp_path):
    """Packing must not let two legs share an interpreter.

    The legs exist to install DIFFERENT library sets. They are separated by a
    per-payload virtualenv, a per-payload ipykernel spec and a per-payload
    `UNSLOTH_COMPILE_LOCATION`; all three are keyed by the payload's index, so
    an index reused across a wave would silently merge two legs' trees and the
    last writer would win.
    """
    driven = _drive_packed(tmp_path, ALL_FOUR, gpus = 2)
    calls = driven["stub"].papermill
    for field in ("kernel", "compile_location", "notebook"):
        values = [c[field] for c in calls]
        assert len(set(values)) == len(calls), (field, values)


def test_a_finished_leg_gives_its_virtualenv_back(tmp_path):
    """Otherwise four torch trees sit on /kaggle/working at once.

    The tail cell prunes `venv_*`, but only after every payload has finished,
    which was sufficient when a kernel held one payload per card. Packed, the
    peak is what matters, and it has to stay at one venv per CARD rather than
    one per LEG.
    """
    driven = _drive_packed(tmp_path, ALL_FOUR, gpus = 2)
    stub = driven["stub"]
    assert (
        stub.max_live_venvs <= 2
    ), f"{stub.max_live_venvs} virtualenvs were alive at once on a 2-card kernel"
    assert list(tmp_path.glob("venv_*")) == [], "a payload left its virtualenv behind"


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
        ALL_FOUR,
        durations = {
            n: 0.30
            for n in ("t4_gptoss.ipynb", "t4_frontier.ipynb", "t4_canary.ipynb", "t4_control.ipynb")
        },
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
    assert len(leg_cards) == len(ALL_FOUR), calls
    assert set(leg_cards) <= {"0", "1"}, leg_cards
    assert set(leg_cards) == {"0", "1"}, leg_cards
    assert driven["stub"].same_card_overlaps == [], driven["stub"].same_card_overlaps


def test_the_studio_assertions_wait_for_both_cards_rather_than_borrowing_one(tmp_path, monkeypatch):
    """Studio keeps both T4s visible, and that is deliberate upstream.

    Its own driver says so: "Studio's own device selection is part of what is
    under test; masking one would test a machine nobody has." So the GPU half
    runs once the leg queue has drained, unpinned, rather than being handed a
    single card out of the queue.
    """
    driven = _drive_with_studio(tmp_path, monkeypatch, ALL_FOUR)
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
        ALL_FOUR,
        unsloth_ref = "main",
        zoo_ref = "main",
        extra_args = (),
        per_run_timeout = 60,
        skip_reference = True,
        studio = STUDIO,
    )
    stub = _InstallFails(gpus = 2)
    stub.root = tmp_path
    saved = sys.modules["subprocess"]
    sys.modules["subprocess"] = stub
    namespace: dict = {}
    try:
        for cell in driver["cells"][:2]:
            source = "".join(cell["source"]).replace("/kaggle/working", str(tmp_path))
            exec(compile(source, "<driver-cell>", "exec"), namespace)
    finally:
        sys.modules["subprocess"] = saved

    ran = [c["notebook"] for c in stub.papermill]
    assert STUDIO_TEST not in ran, ran
    # Every leg still ran: a broken Studio install must not take the notebook
    # signal down with it.
    assert sorted(n for n in ran if n.startswith("t4_")) == sorted(
        f"t4_{leg}.ipynb" for leg in ALL_FOUR
    )
    recorded = (namespace.get("results") or {}).get(STUDIO_TEST)
    assert recorded is not None, "the skip was not recorded at all"
    assert recorded["returncode"] is None
    assert "install lane did not succeed" in recorded["error"]


def test_studio_is_not_in_the_card_queue(tmp_path, monkeypatch):
    """ORDER is the legs. Either Studio half in it would be handed a card."""
    driver = build_kernel.build_kernel(
        SMOKE_DIR,
        ALL_FOUR,
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
    assert order.count("t4_") == len(ALL_FOUR), order
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
    driven = _drive_packed(tmp_path, ALL_FOUR, gpus = 1)
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
