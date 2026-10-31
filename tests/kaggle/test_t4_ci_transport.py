# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Runs the generated driver and payload cells with Kaggle stubbed, so T4-only bugs are caught in CI."""

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


def _shared_setup_1(saved):
    if saved is None:
        sys.modules.pop("huggingface_hub", None)
    else:
        sys.modules["huggingface_hub"] = saved


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_DIR = REPO_ROOT / "tests" / "kaggle" / "t4_smoke"
CI_DIR = REPO_ROOT / ".github" / "scripts" / "kaggle_t4_ci"

sys.path.insert(0, str(SMOKE_DIR))
sys.path.insert(0, str(CI_DIR))

import build_kernel  # noqa: E402
import gate  # noqa: E402
import launch  # noqa: E402
from legs import KERNELS, LEGS  # noqa: E402


class _StubKaggleApi:
    """A client that can name its account, since launch.py refuses to push when the owner is unknown."""

    CONFIG_NAME_USER = "username"

    def __init__(self, username = "someuser"):
        self.config_values = {self.CONFIG_NAME_USER: username}


def _stub_api(*_args, **_kwargs):
    return _StubKaggleApi()


class _Stub:
    """Stands in for subprocess and records the papermill calls, where per-payload isolation is checked."""

    def __init__(
        self,
        *,
        gpus: int,
        venv_ok: bool = True,
    ):
        self.gpus = gpus
        self.venv_ok = venv_ok
        self.papermill: list[dict] = []
        # Every `pip install --target` command, in full, so guards can see what was installed.
        self.overlay_installs: list[list[str]] = []
        # Mixed pure-Python and native, so the driver's deny-list has something real to reject.
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
            # --dry-run --report writes the closure to FILE; an empty one makes the driver install
            # nothing and the overlay guard pass vacuously.
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
    """Runs the generated setup and runner cells, rewriting /kaggle/working to a temp directory."""
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
                # Venvs live on the large overlay, not /kaggle/working; rewrite both roots.
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
    """A GPU shortfall is infrastructure; clamping to one card made contended OOMs read as code bugs."""
    driven = _drive(tmp_path, ["control", "canary"], gpus = -1)
    assert driven["stood_down"] is not None, "a 1-GPU allocation ran both payloads anyway"
    assert driven["papermill"] == []


def test_two_gpus_still_run_both_payloads_one_per_card(tmp_path):
    driven = _drive(tmp_path, ["control", "canary"], gpus = 2)
    assert driven["stood_down"] is None
    assert sorted(p["cuda"] for p in driven["papermill"]) == ["0", "1"]


class _PackedStub(_Stub):
    """Packed legs can overlap on one card or keep every venv alive at once; the stub records both."""

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
            # Recorded at creation: teardown removes venvs, so where they lived is lost afterwards.
            self.venvs_created.append(Path(cmd[2]))
            Path(cmd[2]).mkdir(parents = True, exist_ok = True)
        if "papermill" in cmd:
            notebook = Path(cmd[cmd.index("papermill") + 1]).name
            card = (kw.get("env") or {}).get("CUDA_VISIBLE_DEVICES")
            with self._lock:
                live = self._live_on_card.setdefault(card, set())
                live.add(notebook)
                # Two legs per card is legal when VRAM fits; the peak sum must stay under budget.
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
    """Records snapshot_download calls in order; the hold stops an unjoined prefetch finishing first."""

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
        # Per-call filter patterns: computing them but never passing them looks identical.
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
    # On the fallback path venvs land in WORK itself.
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
                # Venvs live on the large overlay; rewrite both roots. venv_fallback points under a
                # regular file so mkdir raises and the kernel's own fallback branch is tested.
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
        # Join the prefetch lane: it resolves huggingface_hub at call time, so a still-running lane
        # records into the next test's stub.
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


# Derived from KERNELS so it follows the registry. Includes all-card legs, because the
# scheduling rules must hold for the set that actually ships.
ALL_LEGS = list(KERNELS[0])


def test_losing_tmp_drops_the_kernel_back_to_one_leg_per_card(tmp_path):
    """The venv fallback must drop to one leg per card; a constant MAX_LEGS_PER_CARD kept packing two."""
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
    """A seed must take its seat before workers look at the card, or a second leg can overcommit it."""
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
    # gptoss nearly fills the budget, so it must run alone; check overlaps as well as the sum.
    for card, live in stub.same_card_overlaps:
        assert "t4_gptoss.ipynb" not in live, (card, live)


def test_no_card_is_ever_asked_to_hold_more_than_it_has(tmp_path):
    """Summed VRAM on a card must never exceed its budget; sharing is fine, overcommitting is not."""
    driven = _drive_packed(tmp_path, ALL_LEGS, gpus = 2)
    assert driven["stood_down"] is None
    stub = driven["stub"]
    for card, peak in stub.peak_card_gb.items():
        assert peak <= 13.0, f"card {card} peaked at {peak} GB: {stub.same_card_overlaps}"
    for card, count in stub.peak_card_legs.items():
        assert count <= 2, f"card {card} held {count} legs at once"
    assert len(stub.papermill) == len(ALL_LEGS), stub.papermill
    # The per-card split depends on stub timing, so it is not asserted. All-card legs are unpinned
    # by design and inherit the ambient CUDA_VISIBLE_DEVICES.
    pinned = [
        p
        for p in stub.papermill
        if p["notebook"] not in {f"t4_{LEGS[n].name}.ipynb" for n in ALL_LEGS if LEGS[n].all_cards}
    ]
    assert set(p["cuda"] for p in pinned) == {"0", "1"}, stub.papermill
    # Conditional: no all-card leg is wired today.
    unpinned = [p for p in stub.papermill if p not in pinned]
    if any(LEGS[n].all_cards for n in ALL_LEGS):
        assert unpinned, "an all-card leg is wired and no unpinned payload ran"
    else:
        assert not unpinned, unpinned


def test_gptoss_starts_in_the_second_wave_so_the_prefetch_has_a_window(tmp_path):
    """gptoss is third in the order, the first pick of the second wave, so the prefetch has lead time."""
    order = list(KERNELS[0])
    assert order.index("gptoss") == 2, (
        f"gptoss is at position {order.index('gptoss')} of {order}; first means "
        "the prefetch has no window and second-wave is what buys the saving"
    )
    # Not last either: gptoss is the longest leg, so ending on it idles the other card.
    assert order[-1] != "gptoss", order

    driven = _drive_packed(tmp_path, ALL_LEGS, gpus = 2)
    started = [p["notebook"] for p in driven["stub"].papermill]
    assert started[0] != "t4_gptoss.ipynb", started
    assert started != sorted(started), "payloads are running in alphabetical order"


def test_each_leg_keeps_its_own_venv_compile_cache_and_ipykernel(tmp_path):
    """Each leg needs its own venv, ipykernel and UNSLOTH_COMPILE_LOCATION, or the legs merge."""
    driven = _drive_packed(tmp_path, ALL_LEGS, gpus = 2)
    calls = driven["stub"].papermill
    for field in ("kernel", "compile_location", "notebook"):
        values = [c[field] for c in calls]
        assert len(set(values)) == len(calls), (field, values)


def test_a_finished_leg_gives_its_virtualenv_back(tmp_path):
    """A finished leg frees its venv so peak disk stays bounded; venvs stay off /kaggle/working."""
    driven = _drive_packed(tmp_path, ALL_LEGS, gpus = 2)
    stub = driven["stub"]
    ceiling = 2 * 2  # cards x MAX_LEGS_PER_CARD
    assert (
        stub.max_live_venvs <= ceiling
    ), f"{stub.max_live_venvs} virtualenvs were alive at once, ceiling {ceiling}"
    assert list((tmp_path / "venvs").glob("venv_*")) == [], "a payload left its virtualenv behind"
    assert stub.venvs_created, "no virtualenv was built at all"
    for created in stub.venvs_created:
        assert (tmp_path / "venvs") in created.parents, (
            f"a virtualenv was built at {created}, on /kaggle/working -- which "
            "is 19.5 GB and is also the artifact Kaggle ships home"
        )


STUDIO = {
    "unsloth_ref": "main",
    "repo_url": "https://github.com/unslothai/unsloth",
    "payload_args": "--max-steps 8",
}
STUDIO_INSTALL = build_kernel.STUDIO_INSTALL_NOTEBOOK
STUDIO_TEST = build_kernel.STUDIO_TEST_NOTEBOOK
# A value no card index can be confused with, so unpinned and pinned are distinguishable.
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
    """Studio's install touches no GPU, so it must run beside the legs and never queue for a card."""
    driven = _drive_with_studio(
        tmp_path,
        monkeypatch,
        ALL_LEGS,
        durations = {f"t4_{LEGS[n].name}.ipynb": 0.30 for n in ALL_LEGS},
    )
    assert driven["stood_down"] is None
    calls = {c["notebook"]: c for c in driven["stub"].papermill}
    assert STUDIO_INSTALL in calls, sorted(calls)

    # Unpinned: an installer that sees no device resolves a CPU-only torch.
    assert calls[STUDIO_INSTALL]["cuda"] == AMBIENT_CUDA, calls[STUDIO_INSTALL]
    # Every leg with a card is pinned to one card; the per-card split is unasserted by design.
    unpinned = {f"t4_{LEGS[n].name}.ipynb" for n in ALL_LEGS if LEGS[n].all_cards}
    leg_cards = [c["cuda"] for n, c in calls.items() if n.startswith("t4_") and n not in unpinned]
    assert len(leg_cards) == len(ALL_LEGS) - len(unpinned), calls
    assert set(leg_cards) == {"0", "1"}, leg_cards
    # Without this the exclusion could cover a leg that never started.
    for notebook in unpinned:
        assert notebook in calls, sorted(calls)
        assert calls[notebook]["cuda"] == AMBIENT_CUDA, calls[notebook]
    # The summed VRAM on a card must never exceed what it has.
    for card, peak in driven["stub"].peak_card_gb.items():
        assert peak <= 13.0, (card, peak, driven["stub"].same_card_overlaps)


def test_the_studio_assertions_wait_for_both_cards_rather_than_borrowing_one(tmp_path, monkeypatch):
    """Studio keeps both T4s visible by design, so its GPU assertions run last, after the queue drains."""
    driven = _drive_with_studio(tmp_path, monkeypatch, ALL_LEGS)
    calls = [c["notebook"] for c in driven["stub"].papermill]
    assert STUDIO_TEST in calls, calls
    assert calls[-1] == STUDIO_TEST, calls
    by_name = {c["notebook"]: c for c in driven["stub"].papermill}
    assert by_name[STUDIO_TEST]["cuda"] == AMBIENT_CUDA, by_name[STUDIO_TEST]


def test_a_failed_studio_install_skips_its_assertions_with_the_reason(tmp_path, monkeypatch):
    """Skip Studio's assertions when its install failed, or a missing venv reads as a code regression."""
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
                # Venvs live on the large overlay, not /kaggle/working; rewrite both roots.
                .replace("/tmp/t4ci_venvs", str(tmp_path / "venvs"))
                .replace("/kaggle/working", str(tmp_path))
            )
            exec(compile(source, "<driver-cell>", "exec"), namespace)
    finally:
        sys.modules["subprocess"] = saved

    ran = [c["notebook"] for c in stub.papermill]
    assert STUDIO_TEST not in ran, ran
    # A broken Studio install must not take the notebook legs down.
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
    payloads = set(driver["metadata"]["kaggle_t4_ci"]["payloads"])
    assert {STUDIO_INSTALL, STUDIO_TEST} <= payloads, sorted(payloads)


def test_a_one_card_allocation_still_stands_a_packed_kernel_down(tmp_path):
    """The shortfall guard compares GPUs to the packing width, so a single card must still stand down."""
    driven = _drive_packed(tmp_path, ALL_LEGS, gpus = 1)
    assert driven["stood_down"] is not None, "a 1-GPU allocation ran the packed kernel anyway"
    assert driven["stub"].papermill == []


def test_a_payload_whose_venv_failed_is_not_run_in_the_system_kernel(tmp_path):
    """A failed venv must not fall back to python3: shared site-packages would destroy the comparison."""
    driven = _drive(tmp_path, ["control", "canary"], gpus = 2, venv_ok = False)
    assert [
        p["kernel"] for p in driven["papermill"]
    ] == [], "a payload ran in the shared system kernel after its venv failed"
    assert driven["results"], "the skipped payloads left no record"
    for entry in driven["results"].values():
        assert entry["error"]


def test_each_payload_compiles_into_its_own_cache(tmp_path):
    """Concurrent legs must not share unsloth_compiled_cache, a relative path resolved against the cwd."""
    driven = _drive(tmp_path, ["control", "canary"], gpus = 2)
    locations = [p["compile_location"] for p in driven["papermill"]]
    assert all(locations), "no per-payload UNSLOTH_COMPILE_LOCATION was set"
    assert len(set(locations)) == len(locations), f"shared compile cache: {locations}"


def test_the_prune_still_reaches_the_per_payload_directories():
    """The tail cell must still prune per-payload directories, which kernels output would ship home."""
    driver = build_kernel.build_driver({"t4_control.ipynb": {"cells": []}}, 60)
    tail = "".join(driver["cells"][2]["source"])
    assert '"unsloth_compiled_cache*"' in tail or "'unsloth_compiled_cache*'" in tail
    assert '"t4_smoke_src*"' in tail or "'t4_smoke_src*'" in tail


def _payload_cells(leg, **kw) -> list[str]:
    notebook = build_kernel.build_payload_notebook(
        SMOKE_DIR, leg, unsloth_ref = "main", zoo_ref = "main", reference = "", **kw
    )
    return ["".join(cell["source"]) for cell in notebook["cells"]]


def test_each_payload_materialises_into_its_own_directory():
    """Payloads sharing a directory can truncate each other's files, since write_bytes truncates first."""
    roots = set()
    for name in ("control", "canary"):
        materialise = _payload_cells(LEGS[name])[0]
        roots.add(materialise.split("ROOT = pathlib.Path(")[1].split(")")[0])
    assert len(roots) == 2, f"both payloads materialise into {roots}"


def test_a_shared_argument_does_not_override_a_legs_own_option():
    """A shared --smoke-args must not override a leg's own option; argparse keeps the last value."""
    run_cell = _payload_cells(LEGS["gptoss"], extra_args = ("--max-steps", "10"))[3]
    argv = run_cell.split("cmd += [")[1].split("]")[0]
    assert argv.count('"--max-steps"') == 1, argv
    assert '"3"' in argv and '"10"' not in argv

    canary = _payload_cells(LEGS["canary"], extra_args = ("--max-steps", "10"))[3]
    assert '"--max-steps", "10"' in canary.split("cmd += [")[1]


def test_a_probe_failure_is_reported_as_a_failed_payload(tmp_path, monkeypatch):
    """An import failure must still write a report, or the launcher calls the run infra and passes."""
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

    # report.version_table reads versions from the report, not the log.
    import report as report_module

    assert report_module.resolved_versions(reports[0]), reports[0].keys()
    table = report_module.version_table(
        [reports[0], {"label": "control", "versions_flat": {"transformers": "4.57.6"}}]
    )
    assert any("| package | control |" in line for line in table), table


def test_an_install_that_cannot_be_resolved_is_reported_as_a_failed_payload(tmp_path, monkeypatch):
    """Exhausted pip retries must report a failed payload, or the launcher exits green with no report."""
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
    """Pip retries sleep 15 * attempt seconds between tries, so three failures outlast one upstream blip."""
    install = _payload_cells(LEGS["control"])[1]
    assert "time.sleep(15 * attempt)" in install


@pytest.mark.parametrize("plain", [False, True])
def test_a_report_reaches_the_launcher_through_kaggles_structured_log(tmp_path, plain):
    """Kaggle's log is a JSON array of stream records; scanning it as text misses the report prefix."""
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
    """A retry onto an existing slug starts a second session and hides the first; each attempt needs one."""
    # Redirect the in-flight registry, or this test files a fake kernel into the real one.
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
    # Each earlier attempt may have landed, so it is deleted before the next push.
    assert deleted == [s for s in slugs[:-1]]
    assert pushed["attempts"] == slugs
    # Only the accepted attempt remains registered for release.
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
    """Runs launch.main() with Kaggle stubbed and a fake clock; api_seconds is the authentication cost."""
    clock = {"t": 1_000_000.0}
    monkeypatch.setattr(launch.time, "time", lambda: clock["t"])
    monkeypatch.setattr(launch.time, "sleep", lambda _s: None)

    def fake_api(*_args, **_kwargs):
        clock["t"] += api_seconds
        return _StubKaggleApi()

    monkeypatch.setattr(launch, "_api", fake_api)

    outcomes = list(pushes)

    def fake_push(
        notebook,
        user,
        kernel_timeout_sec,
        accelerator = "NvidiaTeslaT4",
        attempted = None,
        **kwargs,
    ):
        clock["t"] += push_seconds
        outcome = outcomes.pop(0)
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
    """A window one second short of the worst case must push nothing, or a killed job leaves kernels up."""
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
    """A window one second above the worst case must still launch, or the guard stands down every run."""
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
    """Re-measure the window after authenticating; that round trip is bounded only by SOCKET_TIMEOUT_SEC."""
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
    """The recheck after authentication must not stand down a window that still covers the worst case."""
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
    """No deadline (the default 0) means no window check, so a local run needs no invented deadline."""
    _, _, result = _drive_main(
        monkeypatch,
        tmp_path,
        push_seconds = 0.0,
        pushes = _TWO_PUSHES,
    )
    assert [k["slug"] for k in result["kernels"]] == [p["slug"] for p in _TWO_PUSHES]


def test_the_deletion_deadline_covers_the_time_spent_pushing(monkeypatch, tmp_path):
    """The deletion deadline must start before the pushes, since a kernel bills from acceptance."""
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
    """Cleanup must delete every slug a push filed, including attempts whose response was lost."""
    _, deleted, result = _drive_main(
        monkeypatch,
        tmp_path,
        push_seconds = 0.0,
        pushes = [
            {
                "ok": True,
                "slug": "someuser/unsloth-t4-ci-cccc",
                "attempts": [
                    "someuser/unsloth-t4-ci-aaaa",
                    "someuser/unsloth-t4-ci-bbbb",
                    "someuser/unsloth-t4-ci-cccc",
                ],
            },
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
    """A subprocess timeout raises rather than returning, so push() must still record the slugs it filed."""
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
    # A timeout is treated as a throttle and retried; every filed slug is returned.
    assert len(pushed["attempts"]) == launch.PUSH_ATTEMPTS
    assert len(set(pushed["attempts"])) == launch.PUSH_ATTEMPTS
    assert "timed out" in pushed["detail"]
    assert deleted == pushed["attempts"][:-1]


def test_a_push_that_times_out_does_not_abandon_the_kernel_already_accepted(monkeypatch, tmp_path):
    """A hung push must not skip release() for an accepted kernel, which then bills to its ceiling."""
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
    monkeypatch.setattr(launch, "_api", _stub_api)
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
    # Any slug the timed-out push filed may be the session Kaggle accepted.
    for slug in result["kernels"][1]["attempted"]:
        assert slug in deleted, slug
    assert all(k["released"] for k in result["kernels"])


@pytest.mark.parametrize(
    "boom",
    [
        # `text=True` decodes strictly, so a bad byte raises after the request was filed.
        UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte"),
        OSError("cannot allocate memory"),
        MemoryError("the runner ran out"),
    ],
    ids = ["decode", "oserror", "memory"],
)
def test_a_push_that_raises_outside_the_timeout_still_gives_up_its_slug(
    monkeypatch, tmp_path, boom
):
    """Slugs are filed before the push runs, so a raise other than TimeoutExpired must not lose them."""
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
    monkeypatch.setattr(launch, "_api", _stub_api)
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
    assert result["kernels"][0]["slug"] in deleted
    assert all(k["released"] for k in result["kernels"])
    assert result["unreleased"] == []


def _accepting_push(slug: str):
    """Stub push that fills the caller-owned attempted list as the real one does, not just return slugs."""

    def fake_push(
        notebook,
        user,
        kernel_timeout_sec,
        accelerator = "NvidiaTeslaT4",
        attempted = None,
        **kwargs,
    ):
        if attempted is not None:
            attempted.append(slug)
        return {"ok": True, "slug": slug, "attempts": [slug]}

    return fake_push


def test_an_abort_anywhere_in_the_launcher_still_deletes_what_it_pushed(monkeypatch, tmp_path):
    """Any abort after the first push must still delete what was pushed; an abort is infra and exits 0."""

    def boom(outdir):
        raise MemoryError("the runner ran out")

    deleted: list[str] = []

    def fake_run(cmd, **kw):
        cmd = [str(c) for c in cmd]
        if cmd[1:3] == ["kernels", "delete"]:
            deleted.append(cmd[3])
        return types.SimpleNamespace(returncode = 0, stdout = "", stderr = "")

    monkeypatch.setattr(launch, "_api", _stub_api)
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


def _drive_one_kernel(monkeypatch, tmp_path, fake_run):
    """Runs main() over one kernel with all but the delete calls stubbed, so only release is exercised."""
    monkeypatch.setattr(launch, "_api", _stub_api)
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
    """A nonzero kernels delete exit is a refusal, not a release; subprocess.run does not raise on it."""
    fake_run, calls = _refusing_run(
        2, "kaggle kernels: error: argument command: invalid choice: 'delete'"
    )
    code, result = _drive_one_kernel(monkeypatch, tmp_path, fake_run)

    assert code == 0
    entry = result["kernels"][0]
    assert entry["released"] is False
    assert entry["released_slugs"] == []
    assert result["unreleased"] == ["someuser/unsloth-t4-ci-abcd"]
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
    """A 404 from the Kaggle delete means the kernel is already gone: a freed slot, not a leak."""
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
    """Cleanup reads a missing kernel with the gate's own tuple of words, so the two cannot drift apart."""
    calls: list[list[str]] = []

    def fake_run(cmd, **kw):
        calls.append([str(c) for c in cmd])
        return types.SimpleNamespace(returncode = 1, stdout = "", stderr = f"delete refused: {marker}")

    monkeypatch.setattr(launch.subprocess, "run", fake_run)
    monkeypatch.setattr(launch.time, "sleep", lambda _s: None)

    assert launch.delete_kernel("someuser/gone") is True
    assert len(calls) == 1, calls


def test_a_nonzero_delete_that_is_not_a_missing_kernel_still_retries(monkeypatch):
    """Other nonzero deletes still retry: a 5xx says nothing about whether the kernel is up."""
    calls: list[list[str]] = []

    def fake_run(cmd, **kw):
        calls.append([str(c) for c in cmd])
        return types.SimpleNamespace(returncode = 1, stdout = "", stderr = "503 Service Unavailable")

    monkeypatch.setattr(launch.subprocess, "run", fake_run)
    monkeypatch.setattr(launch.time, "sleep", lambda _s: None)

    assert launch.delete_kernel("someuser/maybe-live") is False
    assert len(calls) == launch.DELETE_ATTEMPTS


def test_a_payload_that_cannot_see_its_gpu_reports_instead_of_vanishing(tmp_path, monkeypatch):
    """A payload with no visible GPU must still report, or the launcher calls it infra and exits green."""
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
    """Malformed UTF-8 from a payload must not abort the run cell before its report is written."""
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


# Kernels still bill between the last poll and release(), so evidence collection must stay
# within the job deadline.


class _SlowPages:
    """An output listing that never stops paginating and answers slowly; each call waits out its timeout."""

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
    """Fake response with read(amt) and read1, as HTTPResponse offers and the chunked reader calls."""

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
    """The output listing must fit the evidence budget, or the runner dies before release() deletes."""
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
    # Unbounded this is OUTPUT_PAGE_LIMIT x 120s.
    assert len(slow.calls) < launch.OUTPUT_PAGE_LIMIT
    assert all(t <= 120 for t in slow.calls)
    assert listing["truncated"] is True, "an incomplete listing must say so"


def test_the_evidence_budget_is_shared_by_every_kernel(monkeypatch, tmp_path):
    """One evidence budget for the whole phase, shared by every kernel, rather than one per kernel."""
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
    """Notebook downloads are unbounded in size and count, so they too must fit the evidence budget."""
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
    """A body that trickles bytes renews the socket timeout forever, so reads must be bounded per chunk."""

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
    # JSON tolerates trailing whitespace, so padding adds chunks without changing the parse.
    return _Trickle(clock, body + b" " * (chunks * len(body)), chunks, per_chunk)


def test_a_trickling_output_listing_cannot_outlast_the_evidence_budget(monkeypatch, tmp_path):
    """The evidence deadline must hold during the read, not only before it, since socket timeouts renew."""
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
    # The socket timeout is re-clamped as the budget drains, so a read cannot overrun the deadline.
    clamps = resp.fp.raw._sock.timeouts
    assert clamps and all(t <= launch.EVIDENCE_BUDGET_SEC for t in clamps), clamps
    assert clamps == sorted(clamps, reverse = True), clamps


def test_a_trickling_notebook_download_cannot_outlast_the_evidence_budget(monkeypatch, tmp_path):
    """The deadline must cover notebook downloads too; a partial file must not be published as evidence."""
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
    """main() must pass the evidence deadline into the collection loop, since release() runs after it."""
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
        clock.advance(launch.EVIDENCE_BUDGET_SEC)
        return {"notebooks": [], "log": None, "truncated": True}

    monkeypatch.setattr(
        launch,
        "push",
        lambda nb, user, t, accelerator = "NvidiaTeslaT4", attempted = None, **kwargs: (
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
    monkeypatch.setattr(launch, "_api", _stub_api)
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


NOTEBOOK_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "kaggle-t4-notebook-ci.yml"


def test_the_merged_kernel_runs_both_reporters():
    """The merged kernel must run both reporters, since dropping the Studio one would still go green."""
    source = NOTEBOOK_WORKFLOW.read_text(encoding = "utf-8")
    assert ".github/scripts/kaggle_t4_ci/report.py" in source
    assert ".github/scripts/kaggle_studio_ci/report.py" in source
    assert ".github/scripts/kaggle_studio_ci/collect_evidence.py" in source


def test_the_shared_wheels_are_the_specs_every_leg_holds_in_common():
    """Shared wheels come only from specs every leg holds, derived from the legs so SHAs cannot drift."""
    common = build_kernel._shared_vcs_specs(
        {
            "a": [["unsloth_zoo @ git+u@S1"], ["transformers==5.5.0"]],
            "b": [["unsloth_zoo @ git+u@S1"], ["--upgrade", "transformers"]],
        }
    )
    assert common == ("unsloth_zoo @ git+u@S1",), common

    # Different refs for the same package share nothing.
    assert (
        build_kernel._shared_vcs_specs(
            {
                "a": [["unsloth @ git+u@S1"]],
                "b": [["unsloth @ git+u@S2"]],
            }
        )
        == ()
    )

    assert build_kernel._shared_vcs_specs(
        {
            "a": [["x @ git+u@S1"], ["y @ git+u@S9"]],
            "b": [["x @ git+u@S1"]],
        }
    ) == ("x @ git+u@S1",)

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
    # Built before any leg starts: legs begin installing at t=0.
    assert src.index('"pip", "wheel"') < src.index(
        "threads = []"
    ), "the wheels are built after the leg workers start, so no leg can use them"


def test_every_leg_gets_its_own_torch_and_triton_cache(tmp_path):
    """torch and triton caches key off TMPDIR, not the interpreter, so each leg needs its own cache dir."""
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
    """One concurrency group cancels the first pending run, so the slot input must offer exactly two."""
    source = NOTEBOOK_WORKFLOW.read_text(encoding = "utf-8")
    workflow = yaml.safe_load(source)
    slot = workflow[True]["workflow_dispatch"]["inputs"]["slot"]
    assert slot["type"] == "choice", slot
    assert slot["options"] == ["1", "2"], (
        f"the slot input offers {slot['options']}, so the account could be "
        f"asked for more than its 2 concurrent sessions"
    )
    assert slot.get("default") == "1", slot

    # Both levels: a shared workflow-level group still discards a pending dispatch.
    for scope, block in (
        ("workflow", workflow["concurrency"]),
        ("job", workflow["jobs"]["t4-smoke"]["concurrency"]),
    ):
        group = block["group"]
        assert "inputs.slot" in group, f"{scope}: {group}"
        # Non-dispatch events must share one slot, or every push gets its own session.
        assert "'1'" in group, f"{scope}: {group}"
        assert block["cancel-in-progress"] is False, scope


def _build_step_body(source):
    """Slices by step name: a comment naming build_kernel.py must not move the window."""
    return source.split("- name: Build the kernel notebooks")[1].split("- name:")[0]


def test_the_shared_wheel_build_is_opt_in():
    """Shared wheels default to False because one measured run could not attribute its gains or losses."""
    source = NOTEBOOK_WORKFLOW.read_text(encoding = "utf-8")
    workflow = yaml.safe_load(source)
    inputs = workflow[True]["workflow_dispatch"]["inputs"]
    assert inputs["shared_wheels"].get("default") is False, inputs["shared_wheels"]
    build = _build_step_body(source)
    assert "$SHARED_WHEELS" in build, build
    assert "'--shared-wheels'" in source

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
    """A flag nothing passes is dead code: the workflow input must reach --studio-concurrent end to end."""
    source = NOTEBOOK_WORKFLOW.read_text(encoding = "utf-8")
    workflow = yaml.safe_load(source)
    inputs = workflow[True]["workflow_dispatch"]["inputs"]
    assert "studio_concurrent" in inputs, sorted(inputs)
    # On by default to take Studio's GPU half off the critical path; the one-card cost is recorded
    # by the payload (see test_studio_server_flags.py).
    assert inputs["studio_concurrent"].get("default") is True, inputs["studio_concurrent"]
    assert 'inputs.studio_concurrent }}" = "false"' in source, (
        "nothing reads the input as a way to turn sharing off, so a dispatch "
        "asking for the two-card coverage would silently get the shared shape"
    )

    build = _build_step_body(source)
    assert "$STUDIO_CONCURRENT" in build, (
        "the build command does not interpolate STUDIO_CONCURRENT, so the "
        "input cannot reach the kernel no matter what it is set to"
    )
    assert "inputs.studio_concurrent" in source
    assert "'--studio-concurrent'" in source or '"--studio-concurrent"' in source

    # The flag the workflow passes must be one the CLI accepts.
    cli = (CI_DIR / "build_kernel.py").read_text(encoding = "utf-8")
    assert '"--studio-concurrent"' in cli, "build_kernel.py does not define the flag"


def test_the_t4_reporter_is_told_the_leg_count_not_the_payload_count():
    """The T4 reporter takes the leg count: payloads counts Studio, which this reporter filters out."""
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
    """Only the reporter owning a failed payload goes red; the launcher's verdict spans two experiments."""
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
    """Build must pass --with-studio, or Studio reads NOT RUN while the job stays green."""
    source = NOTEBOOK_WORKFLOW.read_text(encoding = "utf-8")
    build = source.split("- name: Build the kernel notebooks")[1].split("- name:")[0]
    assert "--with-studio" in build
    assert "--studio-args" in build


def test_the_prefetch_lane_never_takes_a_card(tmp_path):
    """The prefetch lane never takes a card: holding one would idle it and break the two-lane packing."""
    hub = _HubStub()
    driven = _drive_packed(tmp_path, ALL_LEGS, gpus = 2, prefetch_repos = ("a/big", "b/small"), hub = hub)
    assert driven["stood_down"] is None
    assert hub.calls == ["a/big", "b/small"], hub.calls
    # The prefetch is not a papermill leg, so it cannot have been given CUDA_VISIBLE_DEVICES.
    assert len(driven["stub"].papermill) == len(ALL_LEGS), driven["stub"].papermill
    # The summed VRAM on a card must never exceed what it has.
    for card, peak in driven["stub"].peak_card_gb.items():
        assert peak <= 13.0, (card, peak, driven["stub"].same_card_overlaps)


def test_the_leg_prefetch_does_not_redirect_hf_home(tmp_path):
    """Prefetch must leave HF_HOME alone: legs read the default cache, and a private root fails silently."""
    hub = _HubStub()
    before = os.environ.get("HF_HOME")
    _drive_packed(tmp_path, ALL_LEGS, gpus = 2, prefetch_repos = ("a/big",), hub = hub)
    assert hub.hf_home_at_call == [before], hub.hf_home_at_call
    assert os.environ.get("HF_HOME") == before


def test_a_failing_prefetch_does_not_fail_the_kernel(tmp_path):
    """A failed prefetch must not fail the kernel: the leg downloads the model itself, costing seconds."""
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
    """Prefetch list must match what legs load after LOAD_REDIRECTS, not the name they request."""
    from legs import LEGS, LOAD_REDIRECTS, PREFETCH_REPOS

    # The model walk lives in test_prefetch_covers_the_wired_legs.py; this file keeps the
    # redirect-provenance half.
    from test_prefetch_covers_the_wired_legs import models_for

    loaded = set()
    for leg in LEGS.values():
        loaded |= models_for(leg)
    stray = sorted(set(PREFETCH_REPOS) - loaded)
    assert not stray, f"prefetching {stray}, which no leg loads even after LOAD_REDIRECTS"

    # Every LOAD_REDIRECTS entry must be real: either named in a payload source, or citing the
    # kernel report that measured it (the Qwen remap is derived at load time).
    sources = "".join(path.read_text(encoding = "utf-8") for path in sorted(SMOKE_DIR.glob("*.py")))
    legs_src = (Path(build_kernel.__file__).parent / "legs.py").read_text(encoding = "utf-8")
    for declared, actual in LOAD_REDIRECTS.items():
        if actual in sources:
            continue
        assert re.search(rf"#[^\n]*{re.escape(actual)}", legs_src) or any(
            actual in line and line.lstrip().startswith("#") for line in legs_src.splitlines()
        ), (
            f"LOAD_REDIRECTS says {declared} loads as {actual}, no payload "
            f"mentions {actual}, and no comment in legs.py cites the run that "
            f"measured it -- so the redirect is asserted and not observed"
        )
    # Qwen first: the small model gates three legs, while gpt-oss serves one slow-starting leg.
    assert "Qwen" in PREFETCH_REPOS[0], PREFETCH_REPOS
    assert "gpt-oss" in PREFETCH_REPOS[-1], PREFETCH_REPOS


def test_the_generated_prefetch_cell_runs_not_merely_compiles():
    """The generated prefetch cell must run, not just compile; json.dumps turned None into null."""
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
        _shared_setup_1(saved)
    assert hub.calls == ["a/b", "a/b"], hub.calls


def test_a_repos_allow_patterns_reach_the_hub_and_a_bare_repo_stays_unfiltered():
    """Each repo's allow_patterns must reach snapshot_download; a bare repo must stay unfiltered."""
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
        _shared_setup_1(saved)

    assert hub.calls == ["big/gguf", "small/model"], hub.calls
    assert hub.patterns_at_call == [["*UD-Q4_K_XL*"], None], hub.patterns_at_call


def test_the_last_prefetch_attempt_falls_back_to_classic_http():
    """Last attempt sets HF_HUB_DISABLE_XET: retrying a stalling Xet transport just repeats the stall."""
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
        _shared_setup_1(saved)
    assert seen[-1] == "1", seen
    assert seen[:-1] == [None] * (len(seen) - 1), seen
    assert os.environ.get("HF_HUB_DISABLE_XET") == before


def test_the_studio_prefetch_lands_in_studios_own_cache():
    """Studio prefetch must follow Studio's private HF_HOME; the install inherits it from setup."""
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
    assert "_HF_HOME = None" in sources[min(prefetches)], sources[min(prefetches)][:400]


def test_the_studio_prefetch_follows_the_dispatched_models():
    """Studio prefetch follows the dispatched --chat-model, --train-model and --chat-variant inputs."""
    studio = build_kernel._studio_builder()
    chat, train = studio._models_from("--chat-model a/b --train-model c/d")
    assert chat == ("a/b", ["*UD-Q4_K_XL*"]), chat
    assert train == "c/d", train
    assert studio._models_from("--chat-model=e/f")[0][0] == "e/f"

    # The filter follows the dispatched variant, not the default.
    picked, patterns = studio._models_from("--chat-variant Q8_0")[0]
    assert patterns == ["*Q8_0*"], patterns

    # Wildcards at both ends so split GGUF shards (`...-00001-of-00002.gguf`) also match.
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
    # No lane means no section, not a table of zeroes.
    bare = tmp_path / "bare"
    bare.mkdir()
    (bare / "kernel.log").write_text("nothing to see", encoding = "utf-8")
    assert t4_report.prefetch_table(bare) == []

    # Drive main(): calling the renderer directly passes even if nothing appends its output.
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
    """gptoss never shares a card: at 12.78 GB it is alone by VRAM arithmetic, not by a special case."""
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
    # The admission ledger must balance, or a leaked reservation surfaces only with a fifth leg.
    assert driven["card_load"] and all(abs(v) < 1e-9 for v in driven["card_load"].values()), driven[
        "card_load"
    ]
    assert all(v == 0 for v in driven["card_count"].values()), driven["card_count"]


def test_the_declared_vram_matches_what_the_legs_reported():
    """Leg.vram_gb decides card sharing but nothing checks it at runtime, so it must match measured
    peaks."""
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
        assert declared <= peak + 1.5, (name, declared, peak)
    assert measured["card_total_gb"] > 13.0, measured


def test_studio_waits_for_the_queue_by_default(tmp_path, monkeypatch):
    """Studio waits for the leg queue by default, keeping both T4s visible; sharing must be opted into."""
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
    """Studio under --studio-concurrent is pinned to a card and VRAM-checked, never beside gptoss."""
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
    """The concurrent path re-implements the install gate, so a failed install still skips assertions."""

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
    """The overlay must reach PYTHONPATH and never carry torch, which would shadow the loaded CUDA torch."""
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
    """A leg with no overlay must get no PYTHONPATH, or the overlay check passes by setting it always."""
    stub = _drive_packed(tmp_path, ["control"], gpus = 2)["stub"]
    assert not [
        c for c in stub.overlay_installs if "--target" in c
    ], "a leg declaring no overlay had one built for it"
    record = [p for p in stub.papermill if p["notebook"] == "t4_control.ipynb"]
    assert record
    assert "overlay_" not in record[0]["env"].get("PYTHONPATH", "")


def test_every_leg_installs_bitsandbytes_and_probes_that_it_imports():
    """Every leg must install bitsandbytes and probe its import; git-SHA installs omit it."""
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
