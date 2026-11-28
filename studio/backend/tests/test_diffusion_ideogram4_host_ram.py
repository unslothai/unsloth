# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Ideogram 4 fp8 on a small host: the vendor DiTs are float8 on disk, so a widened load puts both ~9.3B DiTs and
the encoder (~53 GB) into anonymous host memory, and a 53 GB Colab L4 was OOM-killed while finalizing. The loader now
reads lazily into meta-built modules, and where the widened pipeline does not fit host RAM it keeps the DiTs' fp8
Linears stored (widened per call to the same values), or refuses up front. CPU only."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
st = pytest.importorskip("safetensors.torch")

from core.inference import diffusion_ideogram4 as ide  # noqa: E402

_GB = 1000.0**3 / (1024.0 * 1024.0)
# diffusion_auto_policy's ideogram-4 bf16 table, MiB
_DIT, _TE, _VAE = int(37.2 * _GB), int(16.3 * _GB), int(0.2 * _GB)

_TINY = dict(
    adaln_dim = 16,
    attention_head_dim = 8,
    in_channels = 4,
    intermediate_size = 32,
    llm_features_dim = 24,
    mrope_section = [2, 1, 1],
    norm_eps = 1e-5,
    num_attention_heads = 2,
    num_layers = 2,
    rope_theta = 10000,
)


def _plan(total_gib, available_gib, **kw):
    return ide.ideogram4_host_plan(
        dit_bf16_mib = _DIT,
        text_encoder_mib = kw.pop("te", _TE),
        vae_mib = _VAE,
        host_total_mib = int(total_gib * 1024),
        host_available_mib = int(available_gib * 1024),
        **kw,
    )


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("UNSLOTH_DIFFUSION_IDEOGRAM4_FP8_STORAGE", raising = False)
    monkeypatch.delenv("UNSLOTH_DIFFUSION_HOST_RAM_CHECK", raising = False)


def test_colab_l4_host_keeps_the_dits_stored_instead_of_widening():
    # Colab L4: 53 GB total, ~50 GiB available after the download. The widened load (~50.4 GiB + reserve) is what
    # the memcg OOM killer ended at anon-rss 53,058,212 kB.
    storage, refuse, reason = _plan(50.6, 49.5)
    assert storage is True and refuse is None, reason


def test_float32_target_sizes_the_stored_codes_at_one_byte():
    # A float32 target (the CPU default) widens the encoder and the DiTs' norms to 4 bytes, but the stored fp8 codes
    # stay 1 byte: ~49 GiB before reserve, which fits a 64 GiB host with 62 GiB free. Scaling the whole stored DiT
    # by the compute width priced it at ~67 GiB and refused.
    storage, refuse, reason = _plan(64.0, 62.0, te = 2 * _TE, compute_width = 2.0)
    assert storage is True and refuse is None, reason


@pytest.mark.parametrize("total_gib, available_gib", [(83.5, 80.0), (167.0, 160.0), (177.0, 170.0)])
def test_hosts_that_load_today_keep_the_widened_load(total_gib, available_gib):
    # A100-40 (83 GB), A100-80 / H100 hosts, G4 (177 GB): unchanged.
    storage, refuse, _ = _plan(total_gib, available_gib)
    assert storage is False and refuse is None


def test_a_host_too_small_even_for_stored_dits_is_refused_up_front():
    storage, refuse, _ = _plan(31.0, 28.0)
    assert refuse is not None and "Not enough system RAM" in refuse


def test_the_hosted_precast_encoder_is_not_counted_twice():
    # The hosted fp8 encoder is already built (and in the host reading) when the plan runs.
    storage, refuse, _ = _plan(31.0, 26.0, te = 0)
    assert storage is True and refuse is None


def test_lora_needs_widened_linears_so_a_small_host_refuses_instead_of_oom():
    storage, refuse, _ = _plan(50.6, 49.5, lora_active = True)
    assert storage is False and refuse is not None


def test_env_overrides(monkeypatch):
    monkeypatch.setenv(ide.FP8_STORAGE_ENV, "1")
    assert _plan(177.0, 170.0)[0] is True
    monkeypatch.setenv(ide.FP8_STORAGE_ENV, "0")
    assert _plan(50.6, 49.5) == (False, None, f"{ide.FP8_STORAGE_ENV}=0")
    monkeypatch.delenv(ide.FP8_STORAGE_ENV)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_HOST_RAM_CHECK", "0")
    assert _plan(31.0, 28.0)[1] is None


def test_unreadable_host_ram_keeps_todays_widened_load():
    storage, refuse, _ = ide.ideogram4_host_plan(
        dit_bf16_mib = _DIT,
        text_encoder_mib = _TE,
        vae_mib = _VAE,
        host_total_mib = None,
        host_available_mib = None,
    )
    assert storage is False and refuse is None


# ---------------------------------------------------------------- loader: stored codes widen to the dense values


def _vendor_fp8_repo(tmp_path):
    """A tiny Ideogram 4 checkpoint in the vendor layout: fused qkv, ``o``, float8 Linears with row scales."""
    diffusers = pytest.importorskip("diffusers")
    if not hasattr(diffusers, "Ideogram4Transformer2DModel") or not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("diffusers without Ideogram4 / torch without float8")
    import json

    torch.manual_seed(0)
    model = diffusers.Ideogram4Transformer2DModel(**_TINY)
    dense = model.state_dict()
    out = {}
    linear_keys = {
        k
        for k in dense
        if k.endswith(".weight")
        and isinstance(model.get_submodule(k[: -len(".weight")]), torch.nn.Linear)
    }

    def quant(w):
        w = w.float()
        scale = w.abs().amax(dim = 1).clamp(min = 1e-12) / 448.0
        return (w / scale[:, None]).to(torch.float8_e4m3fn), scale.to(torch.bfloat16)

    for i in range(_TINY["num_layers"]):
        p = f"layers.{i}.attention."
        fused = torch.cat([dense.pop(p + n + ".weight") for n in ("to_q", "to_k", "to_v")])
        out[p + "qkv.weight"], out[p + "qkv.weight_scale"] = quant(fused)
        out[p + "o.weight"], out[p + "o.weight_scale"] = quant(dense.pop(p + "to_out.0.weight"))
    for k, v in dense.items():
        if k in linear_keys:
            out[k], out[k + "_scale"] = quant(v)
        else:
            out[k] = v.to(torch.bfloat16).contiguous()
    sub = tmp_path / "transformer"
    sub.mkdir(parents = True)
    st.save_file(out, str(sub / "diffusion_pytorch_model.safetensors"))
    (sub / "config.json").write_text(
        json.dumps({"_class_name": "Ideogram4Transformer2DModel", **_TINY}), encoding = "utf-8"
    )
    return str(tmp_path)


def test_stored_dit_widens_to_exactly_the_dense_load(tmp_path):
    from core.inference.diffusion_comfy_quant import fp8_stored_layers
    from core.inference.diffusion_native_quant import is_native_linear

    repo = _vendor_fp8_repo(tmp_path)
    dense = ide.load_ideogram4_transformer(repo, "transformer", torch.bfloat16)
    stored = ide.load_ideogram4_transformer(repo, "transformer", torch.bfloat16, fp8_storage = True)
    assert fp8_stored_layers(dense) == 0
    native = {n: m for n, m in stored.named_modules() if is_native_linear(m)}
    # every Linear of the dense load (all are fp8 in the vendor checkpoint) is stored
    linears = {n for n, m in dense.named_modules() if type(m) is torch.nn.Linear}
    assert set(native) == linears and len(native) == fp8_stored_layers(stored) > 0
    # host bytes of the stored Linears are 1 per weight, not 2
    for name, layer in native.items():
        assert layer.weight_q.element_size() == 1
        ref = dense.get_submodule(name)
        assert type(ref) is torch.nn.Linear
        assert torch.equal(layer.dequantized_weight(torch.bfloat16), ref.weight)
        x = torch.randn(3, ref.in_features, dtype = torch.bfloat16)
        assert torch.equal(layer(x), ref(x))
    dense_rest = {k: v for k, v in dense.state_dict().items() if k.rsplit(".", 1)[0] not in native}
    stored_sd = stored.state_dict()
    for key, value in dense_rest.items():
        assert torch.equal(stored_sd[key], value), key
    assert not any(p.device.type == "meta" for p in stored.parameters())
    assert not any(p.device.type == "meta" for p in dense.parameters())
    assert getattr(stored, "_unsloth_source_precision", None) == "fp8"


def test_widened_load_still_matches_a_plain_dequantization(tmp_path):
    # The lazy, meta-built loader holds the same weights as dequantising the shard in memory (the old path).
    from safetensors.torch import load_file

    repo = _vendor_fp8_repo(tmp_path)
    raw = load_file(str(tmp_path / "transformer" / "diffusion_pytorch_model.safetensors"))
    hidden = _TINY["attention_head_dim"] * _TINY["num_attention_heads"]
    expected = ide._convert_fp8_state_dict(raw, hidden, torch.bfloat16)
    model = ide.load_ideogram4_transformer(repo, "transformer", torch.bfloat16)
    got = model.state_dict()
    for key, value in expected.items():
        assert torch.equal(got[key], value), key
    assert all(p.dtype == torch.bfloat16 for p in model.parameters())


def test_loader_never_holds_a_materialised_scaffold(tmp_path, monkeypatch):
    # Meta-built: no parameter tensor of the built module is a real allocation before the checkpoint is assigned.
    repo = _vendor_fp8_repo(tmp_path)  # skips when diffusers (or its Ideogram 4 model) is missing
    import diffusers

    seen = []
    original = diffusers.Ideogram4Transformer2DModel.from_config

    def spy(config, *a, **k):
        model = original(config, *a, **k)
        seen.append({p.device.type for p in model.parameters()})
        return model

    monkeypatch.setattr(diffusers.Ideogram4Transformer2DModel, "from_config", spy)
    ide.load_ideogram4_transformer(repo, "transformer", torch.bfloat16)
    assert seen == [{"meta"}]


def test_pipeline_threads_storage_to_both_dits(monkeypatch):
    import sys
    import types

    calls = []
    monkeypatch.setattr(ide, "_patch_create_causal_mask", lambda: None)
    monkeypatch.setattr(ide, "load_ideogram4_text_encoder", lambda *a, **k: "te")
    monkeypatch.setattr(ide, "load_krea2_tokenizer", lambda *a, **k: "tok")
    monkeypatch.setattr(
        ide, "load_ideogram4_transformer", lambda repo, sub, *a, **k: calls.append((sub, k)) or sub
    )
    fake = types.SimpleNamespace(
        Ideogram4Pipeline = lambda **kw: types.SimpleNamespace(**kw),
        AutoencoderKLFlux2 = types.SimpleNamespace(from_pretrained = lambda *a, **k: "vae"),
        FlowMatchEulerDiscreteScheduler = types.SimpleNamespace(
            from_pretrained = lambda *a, **k: "sched"
        ),
    )
    monkeypatch.setitem(sys.modules, "diffusers", fake)
    ide.load_ideogram4_pipeline("unsloth/ideogram-4-fp8", torch.bfloat16, fp8_storage = True)
    assert [(s, k.get("fp8_storage")) for s, k in calls] == [
        ("transformer", True),
        ("unconditional_transformer", True),
    ]
    calls.clear()
    ide.load_ideogram4_pipeline("unsloth/ideogram-4-fp8", torch.bfloat16)
    assert all("fp8_storage" not in k for _s, k in calls)


def test_backend_plans_storage_from_the_host_reading(monkeypatch):
    """The backend wiring: a 53 GB host reading turns storage on, a large one leaves the widened load."""
    from core.inference import diffusion as mod
    from core.inference.diffusion_families import detect_family

    fam = detect_family("unsloth/ideogram-4-fp8")
    monkeypatch.setattr(mod, "ideogram4_repo_is_fp8", lambda *a, **k: True)
    backend = mod.DiffusionBackend.__new__(mod.DiffusionBackend)
    monkeypatch.setattr(mod, "host_ram_mib", lambda: (int(50.6 * 1024), int(49.5 * 1024)))
    assert backend._ideogram4_fp8_storage(
        fam, "x", torch.bfloat16, hf_token = None, text_encoder_ready = False, lora_active = False
    )
    monkeypatch.setattr(mod, "host_ram_mib", lambda: (int(83.5 * 1024), int(80 * 1024)))
    assert not backend._ideogram4_fp8_storage(
        fam, "x", torch.bfloat16, hf_token = None, text_encoder_ready = False, lora_active = False
    )
    monkeypatch.setattr(mod, "host_ram_mib", lambda: (int(31 * 1024), int(28 * 1024)))
    with pytest.raises(RuntimeError, match = "Not enough system RAM"):
        backend._ideogram4_fp8_storage(
            fam, "x", torch.bfloat16, hf_token = None, text_encoder_ready = False, lora_active = False
        )
