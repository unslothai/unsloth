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
from core.inference.diffusion_families import detect_family
from core.inference.sd_cpp_backend import SdCppDiffusionBackend

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_sd_cpp_backend import _FakeServer  # noqa: E402

GIB = 1024


def _m():
    from core.inference import sd_cpp_multigpu
    return sd_cpp_multigpu


# An idle Kaggle T4: 14,900 MiB free, less the 1.5 GiB reserve.
T4_LIMIT = 13_364


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
    return _m().plan_sd_cpp_split(**args)


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
    split, _ = _plan(vae_mib = 0)
    assert split is not None and split.vae_on_secondary is False


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
    assert _m().file_mib(str(index)) == 5
    assert _m().file_mib(str(tmp_path / "a.safetensors")) == 3
    assert _m().file_mib(str(tmp_path / "missing.gguf")) == 0
    assert _m().file_mib(None) == 0


def _names(monkeypatch):
    monkeypatch.setattr(
        bk, "_sd_cpp_probe_output", lambda *_a: "CUDA0\tTesla T4\nCUDA1\tTesla T4\nCPU\tXeon\n"
    )
    monkeypatch.setattr(bk, "physical_card_name", lambda ordinal: (None, None))


def test_the_pin_names_a_card_per_module(monkeypatch):
    _names(monkeypatch)
    split = _m().SdCppCardSplit(primary = 0, secondary = 1, vae_on_secondary = False, reason = "r")
    assert bk._offload_with_device_pin_impl([], "/sd/sd-cli", 0, split) == [
        "--backend",
        "diffusion=CUDA0,te=CUDA1,vae=CUDA0",
    ]
    both = _m().SdCppCardSplit(primary = 0, secondary = 1, vae_on_secondary = True, reason = "r")
    assert bk._offload_with_device_pin_impl([], "/sd/sd-cli", 0, both) == [
        "--backend",
        "diffusion=CUDA0,te=CUDA1,vae=CUDA1",
    ]
    # without a split (and for a split of another primary) the one-card pin is byte-identical to before
    for extra in (
        (),
        (None,),
        (_m().SdCppCardSplit(primary = 1, secondary = 0, vae_on_secondary = False, reason = "r"),),
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
    gpu_ordinal = 0,
):
    _names(monkeypatch)
    cards = limits or {0: (14_900, 15_360), 1: (14_900, 15_360)}
    monkeypatch.setattr(
        "core.inference.diffusion_device.smi_free_total_mib",
        lambda o: cards.get(int(o), (None, None)),
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
        gpu_ordinal = gpu_ordinal,
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
        limits = {0: (48_000, 49_152), 1: (48_000, 49_152)}
    b, servers = _server_load(
        monkeypatch, tmp_path, gpu_ordinals = ordinals, memory_mode = memory_mode, limits = limits
    )
    assert _backend_arg(servers[-1].started.get("offload")) == "diffusion=CUDA0,te=CUDA0,vae=CUDA0"
    assert "multi_gpu" not in b.status()
    if case != "explicit_offload":
        ref, ref_servers = _server_load(monkeypatch, tmp_path, gpu_ordinals = None)
        assert servers[-1].started.get("offload") == ref_servers[-1].started.get("offload")
        assert b._state.offload_flags == ref._state.offload_flags


def test_keep_multiple_models_loaded_keeps_one_card(monkeypatch, tmp_path):
    monkeypatch.setattr("core.inference.diffusion_multigpu.models_kept_loaded", lambda: True)
    b, servers = _server_load(monkeypatch, tmp_path, gpu_ordinals = [0, 1])
    assert _backend_arg(servers[-1].started.get("offload")) == "diffusion=CUDA0,te=CUDA0,vae=CUDA0"
    assert os.environ.get("UNSLOTH_DIFFUSION_MULTIGPU_ALL") is None


def test_automatic_with_an_unpinned_card_still_splits_and_pins_the_primary(monkeypatch, tmp_path):
    # identical idle cards: Automatic's ranker leaves the load unpinned, the selection's primary is cuda:0
    monkeypatch.setattr("core.inference.diffusion_device.auto_cuda_ranking_possible", lambda: True)
    monkeypatch.setattr(
        "core.inference.diffusion_device.resolve_diffusion_device_target",
        lambda **kw: types.SimpleNamespace(device = "cuda"),
    )
    monkeypatch.setattr("torch.cuda.current_device", lambda: 0)
    monkeypatch.setattr("torch.cuda.device_count", lambda: 2)
    b, servers = _server_load(monkeypatch, tmp_path, gpu_ordinals = "automatic", gpu_ordinal = None)
    assert _backend_arg(servers[-1].started.get("offload")) == "diffusion=CUDA0,te=CUDA1,vae=CUDA1"
    assert b.status()["multi_gpu"]["primary"] == "cuda:0"


def test_helpers_keep_to_the_split_diffusion_card_and_status_follows_a_cpu_restart(
    monkeypatch, tmp_path
):
    monkeypatch.setattr("utils.hardware.get_parent_visible_gpu_ids", lambda: [2, 3])
    b, _servers = _server_load(monkeypatch, tmp_path, gpu_ordinals = [0, 1])
    assert b._state.physical_gpu_id is None
    assert b._state.placed_ordinal == 0
    assert "multi_gpu" in b.status()
    b._cpu_backend_forced = True
    assert "multi_gpu" not in b.status()


def test_a_diffusion_model_no_card_holds_is_cut_across_both(monkeypatch):
    split, _ = _plan(layer_ok = True)
    assert split.layer is True and split.vae_on_secondary is True
    assert split.budgets_mib == (T4_LIMIT, T4_LIMIT)
    assert split.describe()["placement"]["diffusion_model"] == "cuda:0+cuda:1"
    # a model that fits the primary keeps the companion split; without a capable build there is no block split
    assert _plan(diffusion_mib = 9_000, layer_ok = True)[0].layer is False
    assert _plan(layer_ok = False)[0].layer is False
    assert _plan(diffusion_mib = 22_000, layer_ok = True)[0].layer is False
    # the secondary's room is what is left beside the encoders and the VAE: no blocks fit there, no block split
    crowded, _ = _plan(
        primary_limit_mib = 20_000,
        secondary_limits_mib = {1: 14_000},
        diffusion_mib = 23_000,
        text_encoder_mib = 10_000,
        vae_mib = 250,
        layer_ok = True,
    )
    assert crowded is None or crowded.layer is False


def test_the_block_split_pin_lists_both_cards_and_their_budgets(monkeypatch):
    monkeypatch.setattr(
        bk,
        "_sd_cpp_probe_output",
        lambda _b, *args: (
            "  --max-vram  budget\n  --auto-fit  on|off\n"
            if args and args[0] == "--help"
            else "CUDA0\tTesla T4\nCUDA1\tTesla T4\n"
        ),
    )
    monkeypatch.setattr(bk, "physical_card_name", lambda ordinal: (None, None))
    split = _m().SdCppCardSplit(
        primary = 0,
        secondary = 1,
        vae_on_secondary = True,
        reason = "r",
        layer = True,
        budgets_mib = (13_876, 13_876),
    )
    flags = bk._offload_with_device_pin_impl([], "/sd/sd-server", 0, split)
    assert flags == [
        "--backend",
        "diffusion=CUDA0&CUDA1,te=CUDA1,vae=CUDA1",
        "--auto-fit",
        "off",
        "--max-vram",
        "CUDA0=13.55,CUDA1=13.55",
    ]
    from core.inference.sd_cpp_args import without_device_backend_flags

    # the CPU restart and status drop the whole pin; a one-card pin keeps any other --max-vram
    assert without_device_backend_flags(["--diffusion-fa", *flags]) == ["--diffusion-fa"]
    assert without_device_backend_flags(
        ["--max-vram", "-1", "--backend", "diffusion=CUDA0,te=CUDA0,vae=CUDA0"]
    ) == ["--max-vram", "-1"]
    monkeypatch.setattr(
        bk, "_sd_cpp_probe_output", lambda _b, *args: "CUDA0\tTesla T4\nCUDA1\tTesla T4\n"
    )
    assert bk._offload_with_device_pin_impl([], "/sd/sd-server", 0, split) == [
        "--backend",
        "diffusion=CUDA0,te=CUDA1,vae=CUDA1",
    ]


_OLD_BOOL_AUTO_FIT_HELP = (
    "  --max-vram <string>    maximum VRAM budget in GiB for graph-cut segmented execution.\n"
    "  --auto-fit             pick the diffusion/te/vae device placements automatically from the model size\n"
)
_ON_OFF_AUTO_FIT_HELP = (
    "  --max-vram <string>    maximum VRAM budget in GiB for graph-cut segmented execution.\n"
    "  --auto-fit             on|off (default: on). Use one GPU for diffusion/te/vae computation and place\n"
)


def test_a_boolean_auto_fit_build_keeps_the_one_card_pin(monkeypatch):
    devices = "CUDA0\tTesla T4\nCUDA1\tTesla T4\n"
    split = _m().SdCppCardSplit(
        primary = 0,
        secondary = 1,
        vae_on_secondary = True,
        reason = "r",
        layer = True,
        budgets_mib = (13_876, 13_876),
    )
    monkeypatch.setattr(bk, "physical_card_name", lambda ordinal: (None, None))
    for help_text, layer in ((_OLD_BOOL_AUTO_FIT_HELP, False), (_ON_OFF_AUTO_FIT_HELP, True)):
        monkeypatch.setattr(
            bk,
            "_sd_cpp_probe_output",
            lambda _b, *args, h = help_text: h if args and args[0] == "--help" else devices,
        )
        assert bk._layer_split_supported("/sd/sd-server") is layer
        flags = bk._offload_with_device_pin_impl([], "/sd/sd-server", 0, split)
        if layer:
            assert flags[flags.index("--auto-fit") + 1] == "off"
        else:
            assert flags == ["--backend", "diffusion=CUDA0,te=CUDA1,vae=CUDA1"]


def test_a_two_card_load_cuts_the_model_across_both_cards_on_a_capable_build(monkeypatch, tmp_path):
    monkeypatch.setattr(bk, "_layer_split_supported", lambda binary: True)
    monkeypatch.setattr("utils.hardware.get_parent_visible_gpu_ids", lambda: [0, 1])
    b, servers = _server_load(monkeypatch, tmp_path, gpu_ordinals = [0, 1])
    started = list(servers[-1].started.get("offload"))
    assert _backend_arg(started) == "diffusion=CUDA0&CUDA1,te=CUDA1,vae=CUDA1"
    assert started[started.index("--auto-fit") + 1] == "off"
    assert "--max-vram" in started
    status = b.status()
    assert status["multi_gpu"]["chosen"] == "layer_split"
    assert status["offload_policy"] == "none" and status["cpu_offload"] is False
    assert b._state.physical_gpu_id is None and b._state.placed_ordinal == 0


def test_a_model_sd_cpp_would_stream_on_one_card_is_cut_across_both(monkeypatch, tmp_path):
    # 2x T4 under Automatic: nvidia-smi shows 15,360 MiB free, the sd-server's CUDA context leaves 14,806 and auto-fit
    # budgets 14,294, 1 MiB short of the 12,247 MiB model plus its 2 GiB reserve: priced as one card, it streams.
    monkeypatch.setattr(bk, "_layer_split_supported", lambda binary: True)
    monkeypatch.setattr("utils.hardware.get_parent_visible_gpu_ids", lambda: [0, 1])
    b, servers = _server_load(
        monkeypatch,
        tmp_path,
        gpu_ordinals = [0, 1],
        limits = {0: (15_360, 15_360), 1: (15_360, 15_360)},
    )
    assert (
        _backend_arg(servers[-1].started.get("offload"))
        == "diffusion=CUDA0&CUDA1,te=CUDA1,vae=CUDA1"
    )
    assert b.status()["multi_gpu"]["chosen"] == "layer_split"
