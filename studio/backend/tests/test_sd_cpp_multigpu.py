# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The native (sd.cpp) engine on two CUDA cards, CPU only: begin_load carries the multi-card selection, the split
decision, the two-card ``--backend`` assignment the server gets, status, and every way it stays one card (kill
switch, one card, explicit offload, a load that fits one card)."""

from __future__ import annotations

import os
import sys
import time
import types
from pathlib import Path

import pytest

from core.inference import sd_cpp_backend as bk
from core.inference import sd_cpp_multigpu as M
from core.inference.diffusion_families import detect_family
from core.inference.sd_cpp_backend import SdCppDiffusionBackend

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_sd_cpp_backend import _FakeServer  # noqa: E402

GIB = 1024
# An idle Kaggle T4: 14,900 MiB free, less the 1 GiB reserve.
T4_LIMIT = 13_876


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch):
    for name in ("UNSLOTH_DIFFUSION_MULTIGPU", "UNSLOTH_DIFFUSION_MULTIGPU_ALL"):
        monkeypatch.delenv(name, raising = False)


def _plan(**kw):
    args = dict(
        diffusion_mib = 12_247,
        text_encoder_mib = 4_465,
        vae_mib = 242,
        primary = 0,
        primary_limit_mib = T4_LIMIT,
        secondary_limits_mib = {1: T4_LIMIT},
    )
    args.update(kw)
    return M.plan_sd_cpp_split(**args)


def test_a_load_that_does_not_fit_one_card_moves_the_encoders_to_the_second():
    # Qwen-Image Q4_K_M on 2x T4: 12.0 GiB DiT + 4.4 GiB Qwen2.5-VL + VAE does not fit one card resident.
    split, why = _plan()
    assert split is not None and why == ""
    assert (split.primary, split.secondary, split.vae_on_secondary) == (0, 1, True)
    assert split.describe()["placement"] == {
        "diffusion_model": "cuda:0",
        "text_encoders": "cuda:1",
        "vae": "cuda:1",
    }


def test_the_vae_stays_with_the_diffusion_model_when_the_second_card_has_no_decode_room():
    split, _ = _plan(secondary_limits_mib = {1: 5_600})
    assert split is not None and split.vae_on_secondary is False


def test_one_card_whenever_one_card_holds_everything_or_nothing_fits():
    assert _plan(primary_limit_mib = 40_000)[0] is None
    assert "not needed" in _plan(primary_limit_mib = 40_000)[1]
    assert _plan(secondary_limits_mib = {1: 2_000})[0] is None
    assert _plan(secondary_limits_mib = {1: None})[0] is None
    assert _plan(primary_limit_mib = None)[0] is None
    assert _plan(text_encoder_mib = 0)[0] is None


def test_the_roomiest_secondary_is_chosen():
    split, _ = _plan(secondary_limits_mib = {1: 7_000, 2: 9_000, 3: 9_000})
    assert split.secondary == 2


def test_file_sizes_count_index_shards(tmp_path):
    (tmp_path / "a.safetensors").write_bytes(b"\0" * (3 << 20))
    (tmp_path / "b.safetensors").write_bytes(b"\0" * (2 << 20))
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(
        '{"weight_map": {"x": "a.safetensors", "y": "b.safetensors", "z": "a.safetensors"}}'
    )
    assert M.file_mib(str(index)) == 5
    assert M.file_mib(str(tmp_path / "a.safetensors")) == 3
    assert M.file_mib(str(tmp_path / "missing.gguf")) == 0
    assert M.file_mib(None) == 0


def _names(monkeypatch):
    monkeypatch.setattr(
        bk, "_sd_cpp_probe_output", lambda *_a: "CUDA0\tTesla T4\nCUDA1\tTesla T4\nCPU\tXeon\n"
    )
    monkeypatch.setattr(bk, "physical_card_name", lambda ordinal: (None, None))


def test_the_pin_names_a_card_per_module(monkeypatch):
    _names(monkeypatch)
    split = M.SdCppCardSplit(primary = 0, secondary = 1, vae_on_secondary = False, reason = "r")
    assert bk._offload_with_device_pin_impl([], "/sd/sd-cli", 0, split) == [
        "--backend",
        "diffusion=CUDA0,te=CUDA1,vae=CUDA0",
    ]
    both = M.SdCppCardSplit(primary = 0, secondary = 1, vae_on_secondary = True, reason = "r")
    assert bk._offload_with_device_pin_impl([], "/sd/sd-cli", 0, both) == [
        "--backend",
        "diffusion=CUDA0,te=CUDA1,vae=CUDA1",
    ]
    # without a split (and for a split of another primary) the one-card pin is byte-identical to before
    for extra in (
        (),
        (None,),
        (M.SdCppCardSplit(primary = 1, secondary = 0, vae_on_secondary = False, reason = "r"),),
    ):
        assert bk._offload_with_device_pin_impl([], "/sd/sd-cli", 0, *extra) == [
            "--backend",
            "diffusion=CUDA0,te=CUDA0,vae=CUDA0",
        ]


def test_a_two_card_server_is_not_charged_to_one_card(monkeypatch):
    _names(monkeypatch)
    monkeypatch.setattr("utils.hardware.get_parent_visible_gpu_ids", lambda: [0, 1])
    one = ("--backend", "diffusion=CUDA0,te=CUDA0,vae=CUDA0")
    two = ("--backend", "diffusion=CUDA0,te=CUDA1,vae=CUDA0")
    assert bk._resolved_server_physical_gpu_id("/sd/sd-server", "cuda", 0, one) == 0
    assert bk._resolved_server_physical_gpu_id("/sd/sd-server", "cuda", 0, two) is None
    assert bk._backend_devices(two) == {"cuda0", "cuda1"}
    assert bk._backend_devices(("--backend", "diffusion=CUDA0&CUDA1,te=cpu,vae=CUDA0")) == {
        "cuda0",
        "cuda1",
    }


def _begin(monkeypatch, gpu_ids):
    seen: dict = {}
    monkeypatch.setattr(
        "core.inference.diffusion_engine_router.off_torch_sd_cpp_device", lambda *a, **k: None
    )
    monkeypatch.setattr(
        bk, "resolve_diffusion_device_target", lambda **kw: types.SimpleNamespace(device = "cuda")
    )
    monkeypatch.setattr(bk, "resolve_selected_cuda_ordinal", lambda ids: min(ids))
    monkeypatch.setattr(bk, "resolve_selected_cuda_ordinals", lambda ids: sorted(set(ids)))
    monkeypatch.setattr(bk, "auto_cuda_ordinal_for_load", lambda: 0)
    b = SdCppDiffusionBackend()
    monkeypatch.setattr(b, "_run_load", lambda **kw: seen.update(kw, done = True))
    b.begin_load(
        repo_id = "unsloth/Qwen-Image-GGUF", gguf_filename = "qwen-image-Q4_K_M.gguf", gpu_ids = gpu_ids
    )
    deadline = time.monotonic() + 5
    while "done" not in seen and time.monotonic() < deadline:
        time.sleep(0.01)
    return seen


def test_begin_load_carries_the_multi_card_selection(monkeypatch):
    assert _begin(monkeypatch, [0, 1])["_gpu_ordinals"] == [0, 1]
    assert _begin(monkeypatch, None)["_gpu_ordinals"] == "automatic"


def test_one_card_and_the_kill_switch_carry_no_selection(monkeypatch):
    assert "_gpu_ordinals" not in _begin(monkeypatch, [1])
    monkeypatch.setenv("UNSLOTH_DIFFUSION_MULTIGPU_ALL", "0")
    assert "_gpu_ordinals" not in _begin(monkeypatch, [0, 1])
    assert "_gpu_ordinals" not in _begin(monkeypatch, None)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_MULTIGPU_ALL", "1")
    monkeypatch.setenv("UNSLOTH_DIFFUSION_MULTIGPU", "0")
    assert "_gpu_ordinals" not in _begin(monkeypatch, [0, 1])


def _sparse(path: Path, mib: int) -> str:
    with open(path, "wb") as f:
        f.truncate(mib << 20)
    return str(path)


def _server_load(
    monkeypatch,
    tmp_path,
    *,
    gpu_ordinals,
    memory_mode = None,
    limits = None,
):
    _names(monkeypatch)
    monkeypatch.setattr(
        M, "card_limit_mib", lambda o: (limits or {0: T4_LIMIT, 1: T4_LIMIT}).get(int(o))
    )
    fam = detect_family("qwen-image")
    servers: list = []
    monkeypatch.setattr(bk, "find_sd_server_binary", lambda: "/x/sd-server")
    monkeypatch.setattr(bk, "_server_binary_runnable", lambda *_a, **_k: True)
    monkeypatch.setattr(
        bk, "SdCppServer", lambda binary: (servers.append(_FakeServer(binary)), servers[-1])[1]
    )
    paths = {
        "diffusion_model": _sparse(tmp_path / "qwen-image-Q4_K_M.gguf", 12 * GIB),
        "vae": _sparse(tmp_path / "vae.safetensors", 250),
        "qwen2vl": _sparse(tmp_path / "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf", 4 * GIB + 400),
    }
    b = SdCppDiffusionBackend()
    monkeypatch.setattr(b, "_asset_specs", lambda *a, **k: [])
    monkeypatch.setattr(b, "_set_expected_bytes", lambda *a, **k: None)
    monkeypatch.setattr(b, "_fetch_assets", lambda *a, **k: dict(paths))
    monkeypatch.setattr(
        bk, "resolve_diffusion_device_target", lambda: types.SimpleNamespace(device = "cuda")
    )
    b._load_token = 1
    kwargs = dict(
        repo_id = "unsloth/Qwen-Image-GGUF",
        gguf_filename = "qwen-image-Q4_K_M.gguf",
        base = fam.base_repo,
        fam = fam,
        hf_token = None,
        _load_token = 1,
        gpu_ordinal = 0,
        memory_mode = memory_mode,
    )
    if gpu_ordinals is not None:
        kwargs["_gpu_ordinals"] = gpu_ordinals
    b._run_load(**kwargs)
    return b, servers


def _backend_arg(flags):
    flags = list(flags or ())
    return flags[flags.index("--backend") + 1] if "--backend" in flags else None


def test_a_two_card_load_starts_the_server_with_the_encoders_on_the_second_card(
    monkeypatch, tmp_path
):
    b, servers = _server_load(monkeypatch, tmp_path, gpu_ordinals = [0, 1])
    assert _backend_arg(servers[-1].started.get("offload")) == "diffusion=CUDA0,te=CUDA1,vae=CUDA1"
    assert _backend_arg(b._state.offload_flags) == "diffusion=CUDA0,te=CUDA1,vae=CUDA1"
    assert b._state.physical_gpu_id is None
    status = b.status()
    assert status["multi_gpu"]["placement"]["text_encoders"] == "cuda:1"
    assert status["offload_policy"] == "none"


@pytest.mark.parametrize(
    "case",
    ["one_card", "kill_switch", "explicit_offload", "fits_one_card"],
)
def test_every_one_card_case_keeps_main_argv(monkeypatch, tmp_path, case):
    ordinals = [0, 1]
    memory_mode, limits = None, None
    if case == "one_card":
        ordinals = None
    elif case == "kill_switch":
        monkeypatch.setenv("UNSLOTH_DIFFUSION_MULTIGPU_ALL", "0")
    elif case == "explicit_offload":
        memory_mode = "balanced"
    elif case == "fits_one_card":
        limits = {0: 40_000, 1: 40_000}
    b, servers = _server_load(
        monkeypatch, tmp_path, gpu_ordinals = ordinals, memory_mode = memory_mode, limits = limits
    )
    assert _backend_arg(servers[-1].started.get("offload")) == "diffusion=CUDA0,te=CUDA0,vae=CUDA0"
    assert "multi_gpu" not in b.status()
    if case != "explicit_offload":
        # the one-card argv is exactly what a load without any selection gets
        ref, ref_servers = _server_load(monkeypatch, tmp_path, gpu_ordinals = None)
        assert servers[-1].started.get("offload") == ref_servers[-1].started.get("offload")
        assert b._state.offload_flags == ref._state.offload_flags


def test_keep_multiple_models_loaded_keeps_one_card(monkeypatch, tmp_path):
    monkeypatch.setattr("core.inference.diffusion_multigpu.models_kept_loaded", lambda: True)
    b, servers = _server_load(monkeypatch, tmp_path, gpu_ordinals = [0, 1])
    assert _backend_arg(servers[-1].started.get("offload")) == "diffusion=CUDA0,te=CUDA0,vae=CUDA0"
    assert os.environ.get("UNSLOTH_DIFFUSION_MULTIGPU_ALL") is None
