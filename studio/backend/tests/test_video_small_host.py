# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Video small-host staging: routing, kill switch, bit-identical staged entry."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")
pytest.importorskip("safetensors")

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.inference import diffusion_cast_cache as CC  # noqa: E402
from core.inference import diffusion_prequant_cache as C  # noqa: E402
from core.inference import video_small_host as VS  # noqa: E402

FP16_DEFAULT = {"vae": torch.float32, "default": torch.float16}
# Wan2.1-T2V-1.3B on a Colab T4: UMT5 13.4 GB loaded (wo kept fp32), DiT 2.8 GB; 12.7 GB host, ~9.9 GB available
WAN13 = {"text_encoder": 12748, "transformer": 2709}


@pytest.fixture(autouse = True)
def _clean_env(tmp_path, monkeypatch):
    monkeypatch.setenv(C.ENV_DIR, str(tmp_path / "cache"))
    monkeypatch.delenv(CC.CAST_CACHE_ENV, raising = False)
    monkeypatch.delenv(C.ENV_MODE, raising = False)
    monkeypatch.delenv(VS.SMALL_HOST_ENV, raising = False)
    monkeypatch.setattr(C, "_hub_source_stale", lambda shards, logger = None: False)
    with C._lock:
        C._REJECTED.clear()


def test_big_host_is_unchanged():
    plan = VS.plan_stage(WAN13, available_mib = 48000, reserve_mib = 5300, device_free_mib = 22000)
    assert not plan.engaged and plan.stage == ()
    assert "fits" in plan.reason


def test_small_t4_host_stages_the_encoder_only():
    plan = VS.plan_stage(WAN13, available_mib = 10100, reserve_mib = 3072, device_free_mib = 14600)
    assert plan.engaged
    # the DiT's dense conversion (2.7 GB) fits once the encoder is off the host
    assert plan.stage == ("text_encoder",)
    assert plan.dense_mib == sum(WAN13.values())


def test_stages_largest_first_until_the_rest_fits():
    misses = {"text_encoder": 9000, "transformer": 6000, "transformer_2": 6000}
    plan = VS.plan_stage(misses, available_mib = 12000, reserve_mib = 3000, device_free_mib = 40000)
    assert plan.stage == ("text_encoder", "transformer")


def test_card_too_small_for_the_encoder_stages_it_split():
    plan = VS.plan_stage(WAN13, available_mib = 10100, reserve_mib = 3072, device_free_mib = 11000)
    # the encoder does not fit the card: the card takes what fits, host RAM the 2.5 GB rest, and the DiT then
    # converts densely beside it
    assert plan.stage == ("text_encoder",)
    assert plan.split == {"text_encoder": 11000 - VS._DEVICE_MARGIN_MIB}


def test_a_split_whose_host_share_does_not_fit_keeps_the_dense_load():
    plan = VS.plan_stage(WAN13, available_mib = 10100, reserve_mib = 3072, device_free_mib = 4000)
    # 9.5 GB of the encoder would stay in host RAM beside the DiT: no split; the DiT alone is staged, logged
    assert "text_encoder" in plan.skipped
    assert plan.stage == ("transformer",) and plan.split == {}


def test_kill_switch_and_force(monkeypatch):
    monkeypatch.setenv(VS.SMALL_HOST_ENV, "0")
    assert VS.stage_disabled()
    plan = VS.plan_stage(
        WAN13,
        available_mib = 100,
        reserve_mib = 3072,
        device_free_mib = 14600,
        disabled = VS.stage_disabled(),
    )
    assert not plan.engaged
    monkeypatch.setenv(VS.SMALL_HOST_ENV, "1")
    plan = VS.plan_stage(
        WAN13,
        available_mib = 48000,
        reserve_mib = 5300,
        device_free_mib = 22000,
        forced = VS.stage_forced(),
    )
    assert plan.stage == ("text_encoder", "transformer")


def test_unreadable_host_ram_changes_nothing():
    plan = VS.plan_stage(WAN13, available_mib = None, reserve_mib = 3072, device_free_mib = 14600)
    assert not plan.engaged


def test_small_candidates_are_never_staged():
    plan = VS.plan_stage({"vae": 200}, available_mib = 100, reserve_mib = 3072, device_free_mib = 14600)
    assert not plan.engaged and plan.skipped == {"vae": "small"}


def _tiny_umt5():
    torch.manual_seed(1)
    config = transformers.UMT5Config(
        vocab_size = 64,
        d_model = 32,
        d_kv = 8,
        d_ff = 64,
        num_layers = 2,
        num_heads = 4,
        relative_attention_num_buckets = 8,
    )
    return transformers.UMT5EncoderModel(config).to(torch.float32)


def _tiny_wan():
    torch.manual_seed(0)
    return diffusers.WanTransformer3DModel(
        patch_size = (1, 2, 2),
        num_attention_heads = 2,
        attention_head_dim = 16,
        in_channels = 4,
        out_channels = 4,
        text_dim = 32,
        freq_dim = 16,
        ffn_dim = 64,
        num_layers = 2,
        cross_attn_norm = True,
        qk_norm = "rms_norm_across_heads",
        eps = 1e-6,
        rope_max_seq_len = 32,
    ).to(torch.float32)


@pytest.fixture
def pipe_dir(tmp_path):
    root = tmp_path / "Wan2.1-T2V-tiny-Diffusers"
    _tiny_wan().save_pretrained(root / "transformer", max_shard_size = "60KB")
    _tiny_umt5().save_pretrained(root / "text_encoder")
    (root / "vae").mkdir()
    (root / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "WanPipeline",
                "transformer": ["diffusers", "WanTransformer3DModel"],
                "text_encoder": ["transformers", "UMT5EncoderModel"],
                "vae": ["diffusers", "AutoencoderKLWan"],
                "tokenizer": ["transformers", "T5TokenizerFast"],
                "scheduler": ["diffusers", "UniPCMultistepScheduler"],
            }
        )
    )
    return root


def _cands(pipe_dir):
    return {c.component: c for c in CC.candidates(pipe_dir, torch_dtype = FP16_DEFAULT)}


def _dense(cand):
    kwargs = CC._load_kwargs(cand)
    kwargs.pop("output_loading_info")
    return cand.cls.from_pretrained(str(Path(cand.source) / cand.component), **kwargs)


def _identical(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys()
    for k in sa:
        assert sa[k].dtype == sb[k].dtype, k
        assert sa[k].device.type == "cpu" and sb[k].device.type == "cpu", k
        assert torch.equal(
            sa[k].reshape(-1).view(torch.uint8), sb[k].reshape(-1).view(torch.uint8)
        ), k


def test_loaded_mib_keeps_wo_wide(pipe_dir):
    cand = _cands(pipe_dir)["text_encoder"]
    assert cand.parts["keep_in_fp32"] == ["wo"]
    dense = _dense(cand)
    want = sum(t.numel() * t.element_size() for t in dense.state_dict().values())
    # tied embed_tokens is stored once; the estimate counts stored tensors at their loaded width
    got_bytes = 0
    for shard in cand.shards:
        for name, e in CC._shard_header(shard).items():
            b, end = e["data_offsets"]
            got_bytes += (end - b) if ".wo." in f".{name}" else (end - b) // 2
    assert VS.loaded_mib(cand) == -(-got_bytes // VS._MIB)
    assert got_bytes <= want


@pytest.mark.parametrize("component", ["text_encoder", "transformer"])
def test_staged_entry_reads_bit_identical_to_the_dense_load(pipe_dir, component):
    cand = _cands(pipe_dir)[component]
    assert VS.stage_on_device(cand, "cpu")
    hit = CC.read(_cands(pipe_dir)[component])
    assert hit is not None
    _identical(hit, _dense(cand))
    if component == "text_encoder":
        assert hit.encoder.block[0].layer[1].DenseReluDense.wo.weight.dtype == torch.float32
        assert hit.shared.weight.dtype == torch.float16


def _wire(
    monkeypatch,
    available,
    total = 13000,
    free = 14600,
):
    from core.inference import diffusion_small_host as SH

    calls = []
    real = VS.stage_on_device

    def fake_stage(cand, device, **kw):
        calls.append((cand.component, str(device)))
        return real(cand, "cpu", **kw)

    monkeypatch.setattr(VS, "stage_on_device", fake_stage)
    monkeypatch.setattr(VS, "_device_free_mib", lambda device: free)
    monkeypatch.setattr(VS, "_STAGE_MIN_MIB", 0)
    monkeypatch.setattr(SH, "host_ram_mib", lambda: (total, available))
    return calls


def test_video_reader_stages_then_serves_the_encoder_on_a_small_host(pipe_dir, monkeypatch):
    from core.inference import video as V

    sizes = {"text_encoder": 9000, "transformer": 2700}
    monkeypatch.setattr(VS, "loaded_mib", lambda cand: sizes[cand.component])
    calls = _wire(monkeypatch, available = 10000)
    pipe_kwargs = {"torch_dtype": dict(FP16_DEFAULT), "cache_dir": None, "local_files_only": True}
    missed = V._read_cast_cache(
        pipe_kwargs, source = str(pipe_dir), revision = None, base = None, device = "cuda"
    )
    assert calls == [("text_encoder", "cuda")]
    assert [c.component for c in missed] == ["transformer"]
    _identical(pipe_kwargs["text_encoder"], _dense(_cands(pipe_dir)["text_encoder"]))


def test_video_reader_on_a_big_host_never_stages(pipe_dir, monkeypatch):
    from core.inference import video as V

    calls = _wire(monkeypatch, available = 48000, total = 52000)
    pipe_kwargs = {"torch_dtype": dict(FP16_DEFAULT), "cache_dir": None, "local_files_only": True}
    missed = V._read_cast_cache(
        pipe_kwargs, source = str(pipe_dir), revision = None, base = None, device = "cuda"
    )
    assert calls == []
    assert sorted(c.component for c in missed) == ["text_encoder", "transformer"]
    assert "text_encoder" not in pipe_kwargs


def test_video_reader_kill_switch(pipe_dir, monkeypatch):
    from core.inference import video as V

    monkeypatch.setenv(VS.SMALL_HOST_ENV, "0")
    calls = _wire(monkeypatch, available = 10)
    pipe_kwargs = {"torch_dtype": dict(FP16_DEFAULT), "cache_dir": None, "local_files_only": True}
    V._read_cast_cache(pipe_kwargs, source = str(pipe_dir), revision = None, base = None, device = "cuda")
    assert calls == []


def test_non_cuda_device_never_stages(pipe_dir, monkeypatch):
    calls = _wire(monkeypatch, available = 10)
    assert VS.stage_misses(list(_cands(pipe_dir).values()), "cpu") == []
    assert calls == []


# HunyuanVideo-1.5 on a Kaggle T4 (31 GB host, 15 GB card): the fp32 DiT converts to fp16 (18.4 GiB, a cast-cache
# miss) while the bf16 Qwen2.5-VL-7B converts to fp16 beside it (15.6 GiB, not a cast candidate). Pricing the DiT
# alone let every load through and the host ran out of RAM (rc -9).
HV15_T4 = {"transformer": 18827}
HV15_TE_MIB = 15600


def test_the_encoder_converted_beside_the_dit_is_priced():
    old = VS.plan_stage(HV15_T4, available_mib = 29238, reserve_mib = 3174, device_free_mib = 14600)
    assert not old.engaged  # the DiT alone "fits"
    plan = VS.plan_stage(
        HV15_T4,
        available_mib = 29238,
        reserve_mib = 3174,
        device_free_mib = 14600,
        others_mib = HV15_TE_MIB,
    )
    assert plan.engaged and plan.stage == ("transformer",)
    # 15 GB card: split, 13.8 GB on the card and 4.9 GB in host RAM beside the encoder's 15.6
    assert plan.split == {"transformer": 14600 - VS._DEVICE_MARGIN_MIB}
    assert plan.others_mib == HV15_TE_MIB and "other converted components" in plan.reason


def test_others_that_fit_change_nothing():
    plan = VS.plan_stage(
        HV15_T4,
        available_mib = 60000,
        reserve_mib = 6000,
        device_free_mib = 14600,
        others_mib = HV15_TE_MIB,
    )
    assert not plan.engaged and "fits" in plan.reason


def test_converted_others_counts_only_a_half_dtype_change(pipe_dir):
    import safetensors.torch as st

    te = pipe_dir / "text_encoder"
    for f in te.glob("*.safetensors"):
        f.unlink()
    st.save_file({"w": torch.zeros(1024, 512, dtype = torch.bfloat16)}, str(te / "model.safetensors"))
    cands = [c for c in CC.candidates(pipe_dir, torch_dtype = FP16_DEFAULT)]
    assert {c.component for c in cands} == {
        "transformer"
    }  # the bf16 encoder is not a cast candidate
    fp16 = VS.converted_others_mib(pipe_dir, cands, torch_dtype = FP16_DEFAULT)
    assert fp16 == 1  # 1 MiB of bf16 converted to fp16 in host RAM
    bf16 = VS.converted_others_mib(
        pipe_dir, cands, torch_dtype = {"default": torch.bfloat16, "vae": torch.float32}
    )
    assert bf16 == 0  # same dtype: memory-mapped, nothing converted
    assert (
        VS.converted_others_mib(pipe_dir, cands, torch_dtype = FP16_DEFAULT, skip = ("text_encoder",))
        == 0
    )


def test_device_load_kwargs_put_the_whole_component_on_the_card():
    import types
    from unittest import mock

    cand = types.SimpleNamespace()
    with mock.patch.object(
        CC, "_load_kwargs", lambda c: {"torch_dtype": torch.float16, "output_loading_info": True}
    ):
        whole = VS._device_load_kwargs(cand, "cuda:0")
    assert whole["device_map"] == {"": "cuda:0"} and "max_memory" not in whole
    assert "output_loading_info" not in whole


def test_a_dit_neither_the_card_nor_a_split_takes_is_cast_from_its_shards(pipe_dir, monkeypatch):
    """The HunyuanVideo-1.5 T4 case: the fp16 DiT is bigger than the card and its host share does not fit, so its cast
    is written from the source shards one tensor at a time and the load maps it (no dense conversion in host RAM)."""
    from core.inference import video as V

    sizes = {"text_encoder": 100, "transformer": 20000}
    monkeypatch.setattr(VS, "loaded_mib", lambda cand: sizes[cand.component])
    calls = _wire(monkeypatch, available = 3000, total = 13000)
    pipe_kwargs = {"torch_dtype": dict(FP16_DEFAULT), "cache_dir": None, "local_files_only": True}
    written = []
    real = CC.save_from_source
    monkeypatch.setattr(
        CC,
        "save_from_source",
        lambda cand, **kw: written.append(cand.component) or real(cand, **kw),
    )
    missed = V._read_cast_cache(
        pipe_kwargs, source = str(pipe_dir), revision = None, base = None, device = "cuda"
    )
    assert written == ["transformer"] and ("transformer", "cuda") not in calls
    assert "transformer" in pipe_kwargs and "transformer" not in [c.component for c in missed]
    _identical(pipe_kwargs["transformer"], _dense(_cands(pipe_dir)["transformer"]))


def test_source_cast_respects_the_kill_switch(pipe_dir, monkeypatch):
    from core.inference import video as V

    monkeypatch.setenv(VS.SMALL_HOST_ENV, "0")
    sizes = {"text_encoder": 100, "transformer": 20000}
    monkeypatch.setattr(VS, "loaded_mib", lambda cand: sizes[cand.component])
    _wire(monkeypatch, available = 3000, total = 13000)
    pipe_kwargs = {"torch_dtype": dict(FP16_DEFAULT), "cache_dir": None, "local_files_only": True}
    V._read_cast_cache(pipe_kwargs, source = str(pipe_dir), revision = None, base = None, device = "cuda")
    assert "transformer" not in pipe_kwargs


def test_a_split_plan_casts_from_the_source_shards_instead(pipe_dir, monkeypatch):
    """A DiT a little over the card was split card / host through accelerate's cpu offload, whose host share loads
    as meta tensors (nothing to write, on the Kaggle T4 the dense load then ran out of host RAM): it is cast from
    its source shards instead, without touching the card."""
    from core.inference import video as V

    sizes = {"text_encoder": 100, "transformer": 16000}
    monkeypatch.setattr(VS, "loaded_mib", lambda cand: sizes[cand.component])
    calls = _wire(monkeypatch, available = 18000, total = 20000)
    written = []
    real = CC.save_from_source
    monkeypatch.setattr(
        CC,
        "save_from_source",
        lambda cand, **kw: written.append(cand.component) or real(cand, **kw),
    )
    pipe_kwargs = {"torch_dtype": dict(FP16_DEFAULT), "cache_dir": None, "local_files_only": True}
    V._read_cast_cache(pipe_kwargs, source = str(pipe_dir), revision = None, base = None, device = "cuda")
    assert written == ["transformer"] and ("transformer", "cuda") not in calls
    _identical(pipe_kwargs["transformer"], _dense(_cands(pipe_dir)["transformer"]))


def test_a_split_plan_whose_source_cast_fails_is_left_to_the_normal_load(pipe_dir, monkeypatch):
    """No accelerate card / host split as a fallback (its host share comes back as meta tensors): a split-planned
    component whose source cast fails is not staged at all, so the pipeline loads it as on main."""
    from core.inference import video as V

    sizes = {"text_encoder": 100, "transformer": 16000}
    monkeypatch.setattr(VS, "loaded_mib", lambda cand: sizes[cand.component])
    calls = _wire(monkeypatch, available = 18000, total = 20000)
    monkeypatch.setattr(CC, "save_from_source", lambda cand, **kw: None)
    pipe_kwargs = {"torch_dtype": dict(FP16_DEFAULT), "cache_dir": None, "local_files_only": True}
    V._read_cast_cache(pipe_kwargs, source = str(pipe_dir), revision = None, base = None, device = "cuda")
    assert ("transformer", "cuda") not in calls and "transformer" not in pipe_kwargs


def test_a_lone_dit_too_big_for_the_card_is_still_cast_from_its_shards(pipe_dir, monkeypatch):
    """Nothing else staged: the plan is not engaged, but a skipped component too big for the card and the host still
    takes the source cast, so the load maps it instead of converting it densely."""
    from core.inference import video as V

    sizes = {"text_encoder": 100, "transformer": 20000}
    monkeypatch.setattr(VS, "loaded_mib", lambda cand: sizes[cand.component])
    _wire(monkeypatch, available = 3000, total = 13000)
    monkeypatch.setattr(VS, "_STAGE_MIN_MIB", 512)
    written = []
    real = CC.save_from_source
    monkeypatch.setattr(
        CC,
        "save_from_source",
        lambda cand, **kw: written.append(cand.component) or real(cand, **kw),
    )
    pipe_kwargs = {"torch_dtype": dict(FP16_DEFAULT), "cache_dir": None, "local_files_only": True}
    V._read_cast_cache(pipe_kwargs, source = str(pipe_dir), revision = None, base = None, device = "cuda")
    assert written == ["transformer"] and "transformer" in pipe_kwargs


def test_a_transformers_encoder_over_the_card_is_not_split_planned():
    """Its cast cannot be written from the source shards (diffusers components only), so it is skipped and the
    planner goes on to stage the DiT, as before the split path existed."""
    plan = VS.plan_stage(
        WAN13,
        available_mib = 16000,
        reserve_mib = 3000,
        device_free_mib = 11000,
        source_castable = {"transformer"},
    )
    assert "text_encoder" in plan.skipped and plan.split == {}
    assert plan.stage == ("transformer",)
