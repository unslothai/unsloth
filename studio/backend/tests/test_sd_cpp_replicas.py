# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The native (sd.cpp) engine on two CUDA cards under concurrent requests, CPU only with fake sd-servers: one
sd-server per card once a request waits, FIFO order, the idle single-request path unchanged, a replica dying, the
kill switch, and host RAM pricing (pinned, --mmap, refused, companion split kept)."""

from __future__ import annotations

import importlib
import sys
import threading
import time
import types
from pathlib import Path

import pytest

from core.inference import sd_cpp_backend as bk
from core.inference.diffusion_families import detect_family
from core.inference.sd_cpp_backend import SdCppDiffusionBackend

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_sd_cpp_backend import _FakeServer  # noqa: E402

# An idle Kaggle T4 as nvidia-smi reads it: 14,900 MiB free of 15,360.
T4 = (14_900, 15_360)
# Qwen-Image-2.1 Q4_K_M-like: fits one card resident, nothing spills.
FITS_ONE_CARD = {"diffusion_model": 8_000, "llm": 2_500, "vae": 250}
# Z-Image Q8_0-like: the DiT fits a card, the BF16 encoder does not fit beside it (main takes a companion split).
SPILLS_ENCODER = {"diffusion_model": 6_400, "llm": 7_678, "vae": 320}


def _rp():
    # Imported per test so each one fails on its own on a tree without the module.
    return importlib.import_module("core.inference.sd_cpp_replicas")


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch):
    for name in (
        "UNSLOTH_DIFFUSION_MULTIGPU",
        "UNSLOTH_DIFFUSION_MULTIGPU_ALL",
        "UNSLOTH_SD_CPP_REPLICAS",
        "CUDA_VISIBLE_DEVICES",
    ):
        monkeypatch.delenv(name, raising = False)


class _Rig:
    """Fake sd-servers whose renders block on a semaphore, so tests decide when each one finishes."""

    def __init__(self):
        self.servers: list = []
        self.entered: list = []  # (server, prompt) in the order renders began
        self.gate = threading.Semaphore(0)
        self.fail_on: dict = {}  # server index -> exception raised by its next render
        self.lock = threading.Lock()
        self.start_delay = 0.0

    def factory(self, binary):
        rig = self

        class _Gated(_FakeServer):
            def start(self, files, **kwargs):
                time.sleep(rig.start_delay)
                self.env = kwargs.pop("env", None)
                self.started = dict(files = files, **kwargs)
                self.started["extra_args"] = list(kwargs.get("extra_args") or [])

            def img_gen(self, payload, **kwargs):
                with rig.lock:
                    rig.entered.append((self, payload["prompt"]))
                    index = rig.servers.index(self)
                    err = rig.fail_on.pop(index, None)
                if err is not None:
                    self.alive = False
                    raise err
                cancel = kwargs.get("cancel_event")
                while not rig.gate.acquire(timeout = 0.05):
                    if self.stopped or (cancel is not None and cancel.is_set()):
                        raise bk.SdCppCancelled("cancelled")
                return super().img_gen(payload, **kwargs)

        server = _Gated(binary)
        self.servers.append(server)
        return server

    def wait(
        self,
        cond,
        timeout = 10.0,
    ):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if cond():
                return True
            time.sleep(0.01)
        raise AssertionError("timed out")


def _sparse(path: Path, mib: int) -> str:
    with open(path, "wb") as f:
        f.truncate(mib << 20)
    return str(path)


def _auto_load(
    monkeypatch,
    tmp_path,
    *,
    sizes = FITS_ONE_CARD,
    host_room = 100_000,
    cards = None,
    gpu_ordinals = "automatic",
    rig = None,
):
    """A GGUF load through _run_load on two idle T4s under Automatic, with fake servers."""
    rig = rig or _Rig()
    cards = cards or {0: T4, 1: T4}
    monkeypatch.setattr(
        bk,
        "_sd_cpp_probe_output",
        lambda *_a: "".join(f"CUDA{o}\tTesla T4\n" for o in sorted(cards)) + "CPU\tXeon\n",
    )
    monkeypatch.setattr(bk, "physical_card_name", lambda ordinal: (None, None))
    monkeypatch.setattr(
        "core.inference.diffusion_device.smi_free_total_mib",
        lambda o: cards.get(int(o), (None, None)),
    )
    monkeypatch.setattr("utils.hardware.get_parent_visible_gpu_ids", lambda: sorted(cards))
    monkeypatch.setattr("core.inference.diffusion_device.auto_cuda_ranking_possible", lambda: True)
    monkeypatch.setattr(
        "core.inference.diffusion_device.resolve_diffusion_device_target",
        lambda **kw: types.SimpleNamespace(device = "cuda"),
    )
    monkeypatch.setattr("torch.cuda.current_device", lambda: 0)
    monkeypatch.setattr("torch.cuda.device_count", lambda: len(cards))
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *_a: (7, 5))
    monkeypatch.setattr(bk, "find_sd_server_binary", lambda: "/x/sd-server")
    monkeypatch.setattr(bk, "_server_binary_runnable", lambda *_a, **_k: True)
    monkeypatch.setattr(bk, "SdCppServer", rig.factory)
    try:
        rp = _rp()
        monkeypatch.setattr(rp, "accelerator_ok", lambda binary: True)
        room = host_room if callable(host_room) else (lambda: host_room)
        monkeypatch.setattr(rp, "host_room_mib", room)
    except ModuleNotFoundError:
        pass
    fam = detect_family("z-image")
    paths = {kind: _sparse(tmp_path / f"{kind}.bin", mib) for kind, mib in sizes.items()}
    b = SdCppDiffusionBackend()
    monkeypatch.setattr(b, "_asset_specs", lambda *a, **k: [])
    monkeypatch.setattr(b, "_set_expected_bytes", lambda *a, **k: None)
    monkeypatch.setattr(b, "_fetch_assets", lambda *a, **k: dict(paths))
    monkeypatch.setattr(
        bk, "resolve_diffusion_device_target", lambda: types.SimpleNamespace(device = "cuda")
    )
    b._load_token = 1
    b._run_load(
        repo_id = "unsloth/Z-Image-GGUF",
        gguf_filename = "z-image-Q8_0.gguf",
        base = fam.base_repo,
        fam = fam,
        hf_token = None,
        _load_token = 1,
        gpu_ordinal = None if gpu_ordinals == "automatic" else min(gpu_ordinals),
        _gpu_ordinals = gpu_ordinals,
    )
    assert b._state is not None, b._loading and b._loading.error
    return b, rig


def _backend_arg(flags):
    flags = list(flags or ())
    return flags[flags.index("--backend") + 1] if "--backend" in flags else None


class _Call(threading.Thread):
    def __init__(self, b, prompt):
        super().__init__(daemon = True)
        self.b, self.prompt = b, prompt
        self.result = self.error = None

    def run(self):
        try:
            self.result = self.b.generate(
                prompt = self.prompt, width = 256, height = 256, steps = 4, seed = 1
            )
        except BaseException as exc:  # noqa: BLE001
            self.error = exc


def _start(b, prompt):
    call = _Call(b, prompt)
    call.start()
    return call


def _waiting(b):
    return b.status()["replicas"]["waiting"]


def test_two_waiting_requests_render_on_both_cards_at_once(monkeypatch, tmp_path):
    b, rig = _auto_load(monkeypatch, tmp_path)
    assert len(rig.servers) == 1
    first = _start(b, "a")
    rig.wait(lambda: len(rig.entered) == 1)
    second = _start(b, "b")
    # The second request waits on a busy server: the replica starts, and it renders there while the first still runs.
    rig.wait(lambda: len(rig.entered) == 2)
    (s0, p0), (s1, p1) = rig.entered
    assert (p0, p1) == ("a", "b") and s0 is rig.servers[0] and s1 is rig.servers[1]
    assert first.is_alive() and second.is_alive()
    replica = rig.servers[1]
    assert replica.env["CUDA_VISIBLE_DEVICES"] == "1"
    assert _backend_arg(replica.started["offload"]) is None and replica.started["extra_args"] == []
    assert replica.started["files"] is rig.servers[0].started["files"]
    status = b.status()["replicas"]
    assert status["servers"] == 2 and status["in_flight"] == 2 and status["active"] is True
    assert [c["card"] for c in status["cards"]] == ["cuda:0", "cuda:1"]
    rig.gate.release(2)
    first.join(5), second.join(5)
    assert first.error is None and second.error is None
    assert len(first.result["images"]) == 1 and len(second.result["images"]) == 1
    status = b.status()["replicas"]
    assert [c["dispatched"] for c in status["cards"]] == [1, 1]


def test_waiting_requests_start_in_arrival_order(monkeypatch, tmp_path):
    b, rig = _auto_load(monkeypatch, tmp_path)
    calls = [_start(b, "r0")]
    rig.wait(lambda: len(rig.entered) == 1)
    for i in range(1, 7):
        calls.append(_start(b, f"r{i}"))
        # r1 takes the replica as soon as it is ready; every later one must be in the line before the next arrives.
        if i == 1:
            rig.wait(lambda: len(rig.entered) == 2)
        else:
            rig.wait(lambda i = i: _waiting(b) == i - 1)
    for _ in range(5):
        before = len(rig.entered)
        rig.gate.release(1)
        rig.wait(lambda before = before: len(rig.entered) == before + 1)
    rig.gate.release(2)
    for call in calls:
        call.join(5)
        assert call.error is None
    assert [p for _s, p in rig.entered] == [f"r{i}" for i in range(7)]
    assert {s for s, _p in rig.entered} == set(rig.servers[:2])


def test_a_single_request_keeps_main_path_and_argv(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_SD_CPP_REPLICAS", "off")
    (tmp_path / "off").mkdir()
    ref, ref_rig = _auto_load(monkeypatch, tmp_path / "off")
    ref_rig.gate.release(1)
    ref.generate(prompt = "same", width = 256, height = 256, steps = 4, seed = 9)
    monkeypatch.delenv("UNSLOTH_SD_CPP_REPLICAS")

    b, rig = _auto_load(monkeypatch, tmp_path)
    rig.gate.release(1)
    out = b.generate(prompt = "same", width = 256, height = 256, steps = 4, seed = 9)
    assert len(out["images"]) == 1
    # One server, no replica start, the same argv, env and payload as with replicas off.
    assert len(rig.servers) == 1

    def _argv(server):
        return {k: v for k, v in server.started.items() if k != "files"}

    assert _argv(rig.servers[0]) == _argv(ref_rig.servers[0])
    assert rig.servers[0].env == ref_rig.servers[0].env
    assert rig.servers[0].payloads == ref_rig.servers[0].payloads
    replicas = b.status()["replicas"]
    assert replicas["servers"] == 1 and replicas["cards"][1]["status"] == "planned"
    assert {k: v for k, v in b.status().items() if k != "replicas"} == ref.status()


def test_a_replica_that_dies_mid_render_hands_the_request_to_another_server(monkeypatch, tmp_path):
    b, rig = _auto_load(monkeypatch, tmp_path)
    rig.fail_on[1] = RuntimeError(
        "sd-server connection lost during img_gen poll (process exited, code -9)"
    )
    first = _start(b, "a")
    rig.wait(lambda: len(rig.entered) == 1)
    second = _start(b, "b")
    rig.wait(lambda: len(rig.entered) == 2)
    rig.wait(lambda: b.status()["replicas"]["cards"][1]["status"] == "dead")
    rig.wait(lambda: rig.servers[1].stopped)
    rig.gate.release(2)
    first.join(5), second.join(5)
    assert first.error is None and second.error is None
    # b rendered once more, on the primary, and came back with exactly one image.
    assert [p for s, p in rig.entered if s is rig.servers[0]] == ["a", "b"]
    assert len(second.result["images"]) == 1
    replicas = b.status()["replicas"]
    assert replicas["failovers"] == 1 and replicas["servers"] == 1
    assert "failed a render" in replicas["cards"][1]["reason"]
    rig.gate.release(1)
    assert len(b.generate(prompt = "c", width = 256, height = 256, steps = 4, seed = 1)["images"]) == 1


def test_a_replica_that_exits_while_idle_is_not_given_work(monkeypatch, tmp_path):
    b, rig = _auto_load(monkeypatch, tmp_path)
    first = _start(b, "a")
    rig.wait(lambda: len(rig.entered) == 1)
    second = _start(b, "b")
    rig.wait(lambda: len(rig.entered) == 2)
    rig.gate.release(2)
    first.join(5), second.join(5)
    rig.servers[1].alive = False  # the replica process exits between renders
    third = _start(b, "c")
    rig.wait(lambda: len(rig.entered) == 3)
    fourth = _start(b, "d")
    rig.wait(lambda: b.status()["replicas"]["cards"][1]["status"] == "dead")
    rig.gate.release(2)
    third.join(5), fourth.join(5)
    assert third.error is None and fourth.error is None
    assert [p for s, p in rig.entered if s is rig.servers[0]] == ["a", "c", "d"]


@pytest.mark.parametrize("setting", ["off", "0", "1"])
def test_the_kill_switch_keeps_one_server_and_main_split(monkeypatch, tmp_path, setting):
    monkeypatch.setenv("UNSLOTH_SD_CPP_REPLICAS", setting)
    b, rig = _auto_load(monkeypatch, tmp_path, sizes = SPILLS_ENCODER)
    assert b._replicas is None and "replicas" not in b.status()
    assert _backend_arg(rig.servers[0].started["offload"]) == "diffusion=CUDA0,te=CUDA1,vae=CUDA1"
    first = _start(b, "a")
    rig.wait(lambda: len(rig.entered) == 1)
    second = _start(b, "b")
    time.sleep(0.3)
    assert len(rig.servers) == 1 and len(rig.entered) == 1
    rig.gate.release(2)
    first.join(5), second.join(5)
    assert first.error is None and second.error is None
    assert _rp().replica_setting() == ("off", None)


def test_a_server_count_applies_to_an_explicit_card_pick(monkeypatch, tmp_path):
    b, _rig = _auto_load(monkeypatch, tmp_path, gpu_ordinals = [0, 1])
    assert b._replicas is None  # auto is the Automatic selection only
    monkeypatch.setenv("UNSLOTH_SD_CPP_REPLICAS", "2")
    b, _rig = _auto_load(monkeypatch, tmp_path, gpu_ordinals = [0, 1])
    assert b.status()["replicas"]["mode"] == "count"


def test_host_ram_pricing_pinned_mmap_and_refused():
    rp = _rp()
    kw = dict(ordinal = 1, limit_mib = 13_364, diffusion_mib = 6_400, text_encoder_mib = 7_678, vae_mib = 320)
    pinned = rp.price_replica(host_room_mib = 20_000, **kw)
    assert pinned.fits and not pinned.mmap and pinned.spill_mib == 7_678
    assert pinned.host_mib == 7_678 + rp.PROCESS_HOST_MIB
    mapped = rp.price_replica(host_room_mib = 5_000, **kw)
    assert mapped.fits and mapped.mmap and mapped.host_mib == rp.PROCESS_HOST_MIB
    refused = rp.price_replica(host_room_mib = 1_000, **kw)
    assert not refused.fits and "host RAM" in refused.reason
    # Nothing spills: --mmap would not save anything, so a host that cannot hold the process refuses.
    resident = dict(kw, diffusion_mib = 8_000, text_encoder_mib = 2_500, vae_mib = 250)
    assert rp.price_replica(host_room_mib = 2_000, **resident).spill_mib == 0
    assert not rp.price_replica(host_room_mib = 1_000, **resident).fits
    # A card that cannot hold the diffusion model is never a replica (that load is cut across cards).
    assert not rp.price_replica(host_room_mib = 99_999, **dict(kw, limit_mib = 8_000)).fits


def test_auto_keeps_main_companion_split_for_a_single_user(monkeypatch, tmp_path):
    # Plenty of host RAM: Auto still runs main's split argv and placement, and a queue starts nothing extra.
    monkeypatch.setenv("UNSLOTH_SD_CPP_REPLICAS", "off")
    (tmp_path / "off").mkdir()
    ref, ref_rig = _auto_load(monkeypatch, tmp_path / "off", sizes = SPILLS_ENCODER, host_room = 30_000)
    monkeypatch.delenv("UNSLOTH_SD_CPP_REPLICAS")
    b, rig = _auto_load(monkeypatch, tmp_path, sizes = SPILLS_ENCODER, host_room = 30_000)
    primary = rig.servers[0]
    assert _backend_arg(primary.started["offload"]) == "diffusion=CUDA0,te=CUDA1,vae=CUDA1"
    assert {k: v for k, v in primary.started.items() if k != "files"} == {
        k: v for k, v in ref_rig.servers[0].started.items() if k != "files"
    }
    assert primary.env == ref_rig.servers[0].env
    assert b._replicas is None and b.status() == ref.status()
    assert b.status()["multi_gpu"]["chosen"] == "companion_split"
    rig.gate.release(1)
    assert len(b.generate(prompt = "a", width = 256, height = 256, steps = 4, seed = 1)["images"]) == 1
    first = _start(b, "a")
    rig.wait(lambda: len(rig.entered) == 2)
    second = _start(b, "b")
    time.sleep(0.3)
    assert len(rig.servers) == 1
    rig.gate.release(2)
    first.join(5), second.join(5)
    assert first.error is None and second.error is None


def test_a_server_count_turns_a_companion_split_into_one_server_per_card(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_SD_CPP_REPLICAS", "2")
    b, rig = _auto_load(monkeypatch, tmp_path, sizes = SPILLS_ENCODER, host_room = 30_000)
    primary = rig.servers[0]
    # Run as a one-card Studio would: no --backend pin, masked to its card.
    assert _backend_arg(primary.started["offload"]) is None
    assert primary.env["CUDA_VISIBLE_DEVICES"] == "0"
    assert "multi_gpu" not in b.status()
    assert b.status()["replicas"]["replaced_split"] is True
    assert b._state.physical_gpu_id == 0
    first = _start(b, "a")
    rig.wait(lambda: len(rig.entered) == 1)
    second = _start(b, "b")
    rig.wait(lambda: len(rig.entered) == 2)
    replica = rig.servers[1]
    assert (
        replica.env["CUDA_VISIBLE_DEVICES"] == "1" and "--mmap" not in replica.started["extra_args"]
    )
    rig.gate.release(2)
    first.join(5), second.join(5)
    assert first.error is None and second.error is None


def test_a_replica_uses_mmap_when_the_host_cannot_pin_a_second_spill(monkeypatch, tmp_path):
    # At load: the primary's pinned spill plus a replica under --mmap fit. At spawn the primary holds its share.
    monkeypatch.setenv("UNSLOTH_SD_CPP_REPLICAS", "2")
    rooms = iter([12_000])
    b, rig = _auto_load(
        monkeypatch, tmp_path, sizes = SPILLS_ENCODER, host_room = lambda: next(rooms, 4_000)
    )
    assert b.status()["replicas"]["replaced_split"] is True
    first = _start(b, "a")
    rig.wait(lambda: len(rig.entered) == 1)
    second = _start(b, "b")
    rig.wait(lambda: len(rig.entered) == 2)
    assert rig.servers[1].started["extra_args"] == ["--mmap"]
    card = b.status()["replicas"]["cards"][1]
    assert card["mmap"] is True and card["host_mib"] == _rp().PROCESS_HOST_MIB
    rig.gate.release(2)
    first.join(5), second.join(5)


def test_the_split_stays_when_the_host_cannot_hold_the_one_card_spill(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_SD_CPP_REPLICAS", "2")
    b, rig = _auto_load(monkeypatch, tmp_path, sizes = SPILLS_ENCODER, host_room = 9_000)
    assert b._replicas is None and "replicas" not in b.status()
    assert _backend_arg(rig.servers[0].started["offload"]) == "diffusion=CUDA0,te=CUDA1,vae=CUDA1"


def test_a_second_replica_the_host_cannot_hold_is_refused_and_the_primary_serves(
    monkeypatch, tmp_path
):
    # Nothing spills, so the load prices no host RAM; the replica start finds 1 GB left.
    b, rig = _auto_load(monkeypatch, tmp_path, host_room = 1_000)
    first = _start(b, "a")
    rig.wait(lambda: len(rig.entered) == 1)
    second = _start(b, "b")
    rig.wait(lambda: b.status()["replicas"]["cards"][1]["status"] == "refused")
    assert len(rig.servers) == 1 and "host RAM" in b.status()["replicas"]["cards"][1]["reason"]
    rig.gate.release(2)
    first.join(5), second.join(5)
    assert first.error is None and second.error is None
    assert [p for _s, p in rig.entered] == ["a", "b"]


def test_unload_stops_every_replica_and_ends_their_renders(monkeypatch, tmp_path):
    b, rig = _auto_load(monkeypatch, tmp_path)
    first = _start(b, "a")
    rig.wait(lambda: len(rig.entered) == 1)
    second = _start(b, "b")
    rig.wait(lambda: len(rig.entered) == 2)
    third = _start(b, "c")
    rig.wait(lambda: _waiting(b) == 1)
    b.unload()
    for call in (first, second, third):
        call.join(5)
        assert not call.is_alive()
        assert isinstance(call.error, RuntimeError)
    assert all(s.stopped for s in rig.servers)
    assert b._replicas is None and b._stopping_servers == 0
    assert not bk._tree_in_use(b)
    assert "replicas" not in b.status()


def test_a_dead_primary_takes_its_replicas_with_the_load(monkeypatch, tmp_path):
    b, rig = _auto_load(monkeypatch, tmp_path)
    first = _start(b, "a")
    rig.wait(lambda: len(rig.entered) == 1)
    second = _start(b, "b")
    rig.wait(lambda: len(rig.entered) == 2)
    rig.gate.release(2)
    first.join(5), second.join(5)
    rig.servers[0].alive = False
    assert b.status()["loaded"] is False
    rig.wait(lambda: rig.servers[1].stopped)
    rig.wait(lambda: b._stopping_servers == 0)
    assert b._replicas is None and not bk._tree_in_use(b)


class _AccountCall(_Call):
    def __init__(self, b, prompt, account_id):
        super().__init__(b, prompt)
        self.account_id = account_id

    def run(self):
        from utils.account_context import AccountContext, run_as
        run_as(AccountContext(self.account_id, self.account_id), super().run)


def test_stop_and_account_retire_reach_a_render_on_a_replica(monkeypatch, tmp_path):
    from core.inference import diffusion_engine_router

    b, rig = _auto_load(monkeypatch, tmp_path)
    monkeypatch.setattr(bk, "_sd_cpp_backend", b)
    first = _AccountCall(b, "a", "acct-a")
    first.start()
    rig.wait(lambda: len(rig.entered) == 1)
    second = _AccountCall(b, "b", "acct-b")
    second.start()
    rig.wait(lambda: len(rig.entered) == 2)
    assert rig.entered[1][0] is rig.servers[1]
    assert bk.generation_in_flight()
    # Retiring acct-b stops its render on the replica, and only that one.
    assert diffusion_engine_router.cancel_generation_for_account("acct-b") is True
    second.join(5)
    assert isinstance(second.error, RuntimeError) and "cancelled" in str(second.error)
    assert first.is_alive()
    # A foreign account's replica render blocks an account-scoped unload as a primary render does.
    third = _AccountCall(b, "c", "acct-c")
    third.start()
    rig.wait(lambda: len(rig.entered) == 3 and rig.entered[2][0] is rig.servers[1])
    from core.inference.gpu_arbiter import GpuBusyForAnotherAccountError

    monkeypatch.setattr(
        "hub.services.models.account_access.require_resident_control", lambda *a, **k: None
    )
    with pytest.raises(GpuBusyForAnotherAccountError):
        b.unload(expected_account = "acct-a")
    # Stop with no account scope stops every render, the replica's included.
    assert b.cancel_generate() is True
    first.join(5), third.join(5)
    assert all(isinstance(c.error, RuntimeError) for c in (first, third))


def test_the_status_route_model_carries_the_replicas_block(monkeypatch, tmp_path):
    from models.inference import DiffusionStatusResponse
    b, _rig = _auto_load(monkeypatch, tmp_path)
    assert DiffusionStatusResponse(**b.status()).replicas["cards"][1]["card"] == "cuda:1"


def _two_renders(b, rig):
    first = _start(b, "a")
    rig.wait(lambda: len(rig.entered) == 1)
    second = _start(b, "b")
    rig.wait(lambda: len(rig.entered) == 2)
    return first, second


def test_a_job_error_on_a_healthy_replica_is_the_request_error(monkeypatch, tmp_path):
    b, rig = _auto_load(monkeypatch, tmp_path)
    err = RuntimeError("sd-server generation failed: bad_request: prompt too long")

    def _fail_once(index, exc):
        orig = rig.servers[index].img_gen

        def img_gen(payload, **kw):
            rig.entered.append((rig.servers[index], payload["prompt"]))
            raise exc

        rig.servers[index].img_gen = img_gen
        return orig

    # Spawn the replica with a first pair, then make its next render fail on a live server.
    first, second = _two_renders(b, rig)
    rig.gate.release(2)
    first.join(5), second.join(5)
    _fail_once(1, err)
    third = _start(b, "c")
    rig.wait(lambda: len(rig.entered) == 3)
    fourth = _start(b, "d")
    fourth.join(5)
    assert fourth.error is err
    assert b.status()["replicas"]["cards"][1]["status"] == "ready"
    assert b.status()["replicas"]["failovers"] == 0
    rig.gate.release(1)
    third.join(5)


def test_a_vram_refusal_on_a_live_replica_retires_it_and_the_primary_renders(monkeypatch, tmp_path):
    b, rig = _auto_load(monkeypatch, tmp_path)
    monkeypatch.setattr(
        bk.sd_cpp_args_mod, "is_vram_refusal_text", lambda text: "VRAM" in str(text)
    )
    first, second = _two_renders(b, rig)
    rig.gate.release(2)
    first.join(5), second.join(5)
    replica = rig.servers[1]

    def img_gen(payload, **kw):
        rig.entered.append((replica, payload["prompt"]))
        raise RuntimeError("sd-server generation failed: not enough VRAM for this render")

    replica.img_gen = img_gen
    third = _start(b, "c")
    rig.wait(lambda: len(rig.entered) == 3)
    fourth = _start(b, "d")
    rig.wait(lambda: b.status()["replicas"]["cards"][1]["status"] == "dead")
    rig.gate.release(2)
    third.join(5), fourth.join(5)
    assert third.error is None and fourth.error is None
    assert len(fourth.result["images"]) == 1
    rig.wait(lambda: replica.stopped and b._stopping_servers == 0)


def test_a_primary_restart_takes_the_replicas_down(monkeypatch, tmp_path):
    b, rig = _auto_load(monkeypatch, tmp_path)
    first, second = _two_renders(b, rig)
    rig.gate.release(2)
    first.join(5), second.join(5)
    old = b._state.server
    fresh = rig.factory("/x/sd-server")
    import dataclasses

    b._hand_over_replacement(dataclasses.replace(b._state, server = fresh), fresh, old)
    assert b._replicas is None and "replicas" not in b.status()
    rig.wait(lambda: rig.servers[1].stopped and b._stopping_servers == 0)


def test_replicas_priced_together_never_share_one_reading_of_the_host(monkeypatch, tmp_path):
    # Three cards, room for one more server process: the second replica is refused, not priced against the same RAM.
    rig = _Rig()
    # A replica charges the host once it has loaded; one still loading has not yet.
    started = lambda: sum(1 for s in rig.servers[1:] if s.started is not None)  # noqa: E731
    b, rig = _auto_load(
        monkeypatch,
        tmp_path,
        host_room = lambda: 2_000 - 1_536 * started(),
        cards = {0: T4, 1: T4, 2: T4},
        rig = rig,
    )
    rig.start_delay = 0.3
    assert [c["card"] for c in b.status()["replicas"]["cards"]] == ["cuda:0", "cuda:1", "cuda:2"]
    first = _start(b, "a")
    rig.wait(lambda: len(rig.entered) == 1)
    second = _start(b, "b")
    rig.wait(lambda: {c["status"] for c in b.status()["replicas"]["cards"][1:]} >= {"refused"})
    rig.wait(lambda: len(rig.entered) == 2)
    statuses = sorted(c["status"] for c in b.status()["replicas"]["cards"][1:])
    assert statuses in (["busy", "refused"], ["ready", "refused"])
    assert len(rig.servers) == 2
    rig.gate.release(2)
    first.join(5), second.join(5)


def test_a_second_failing_replica_is_retired_even_past_the_one_retry(monkeypatch):
    from core.inference import sd_cpp_replicas as rp

    retired, released = [], []
    lanes = [types.SimpleNamespace(ordinal = o, server = object()) for o in (1, 2)]
    for lane in lanes:
        lane.backend = types.SimpleNamespace(
            _generate_impl = lambda **kw: (_ for _ in ()).throw(RuntimeError("not enough VRAM"))
        )
    pool = object.__new__(rp.ReplicaPool)
    pool.closed = False
    pool._cond = threading.Condition()
    pool._failovers = 0
    order = iter(lanes)
    monkeypatch.setattr(pool, "_acquire", lambda waiting, front: next(order), raising = False)
    monkeypatch.setattr(pool, "_release", released.append, raising = False)
    monkeypatch.setattr(
        pool, "_retire", lambda lane, why: retired.append(lane.ordinal), raising = False
    )
    monkeypatch.setattr(rp, "_replica_failure", lambda exc, server: True)
    with pytest.raises(RuntimeError, match = "VRAM"):
        pool.run({})
    # One retry only, but both failing servers leave the line.
    assert retired == [1, 2] and pool._failovers == 1 and released == lanes
