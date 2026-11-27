# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Load-time LoRA bake for the video backend (VideoLoadRequest.loras). CPU only: tiny torch modules carry the
module paths the Studio DiT trainer adapts on LTX-2 and Wan2.2-TI2V-5B, and the adapter files use the key layout
their pipelines' save_lora_weights writes (``transformer.<module>.lora_A.weight``)."""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError
from safetensors.torch import save_file

from core.inference import diffusion_lora, video_lora
from core.inference.video_families import detect_video_family

RANK = 4


class _Attn(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.to_q = torch.nn.Linear(dim, dim)
        self.to_k = torch.nn.Linear(dim, dim)
        self.to_v = torch.nn.Linear(dim, dim)
        self.to_out = torch.nn.ModuleList([torch.nn.Linear(dim, dim), torch.nn.Dropout(0.0)])


class _Proj(torch.nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(dim, hidden)


class _FFN(torch.nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.net = torch.nn.ModuleList(
            [_Proj(dim, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, dim)]
        )


class _Block(torch.nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.attn1 = _Attn(dim)
        self.attn2 = _Attn(dim)
        self.ffn = _FFN(dim, hidden)


class _WanLike(torch.nn.Module):
    def __init__(
        self,
        dim: int = 16,
        hidden: int = 32,
        layers: int = 2,
    ) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([_Block(dim, hidden) for _ in range(layers)])


class _LtxLike(torch.nn.Module):
    def __init__(
        self,
        dim: int = 16,
        hidden: int = 32,
        layers: int = 2,
    ) -> None:
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([_Block(dim, hidden) for _ in range(layers)])


# The trainer's _WAN_TARGETS / _LTX2_TARGETS (core/training/diffusion_dit_trainer.py), as peft expands them.
_WAN_LEAVES = (
    "attn1.to_q",
    "attn1.to_k",
    "attn1.to_v",
    "attn1.to_out.0",
    "attn2.to_q",
    "attn2.to_k",
    "attn2.to_v",
    "attn2.to_out.0",
    "ffn.net.0.proj",
    "ffn.net.2",
)
_LTX_LEAVES = (
    "attn1.to_q",
    "attn1.to_k",
    "attn1.to_v",
    "attn1.to_out.0",
    "attn2.to_q",
    "attn2.to_k",
    "attn2.to_v",
    "attn2.to_out.0",
)


def _write_adapter(
    path: Path, module: torch.nn.Module, prefix: str, leaves, *, alpha: float, metadata: bool
):
    gen = torch.Generator().manual_seed(0)
    params = dict(module.named_parameters())
    state, factors = {}, {}
    for n in range(len(getattr(module, prefix))):
        for leaf in leaves:
            name = f"{prefix}.{n}.{leaf}"
            out_f, in_f = params[f"{name}.weight"].shape
            a = torch.randn(RANK, in_f, generator = gen) * 0.1
            b = torch.randn(out_f, RANK, generator = gen) * 0.1
            state[f"transformer.{name}.lora_A.weight"] = a.to(torch.bfloat16)
            state[f"transformer.{name}.lora_B.weight"] = b.to(torch.bfloat16)
            factors[name] = (a.to(torch.bfloat16).float(), b.to(torch.bfloat16).float())
    meta = {"format": "pt"}
    if metadata:
        meta["lora_adapter_metadata"] = json.dumps(
            {
                "transformer.r": RANK,
                "transformer.lora_alpha": alpha,
                "transformer.use_rslora": False,
            }
        )
    path.parent.mkdir(parents = True, exist_ok = True)
    save_file(state, str(path), metadata = meta)
    return factors


def _assert_baked(module, base, factors, scale):
    for name, p in module.named_parameters():
        mod = name[: -len(".weight")]
        if name.endswith(".weight") and mod in factors:
            a, b = factors[mod]
            want = (base[name].float() + scale * (b @ a)).to(torch.bfloat16)
            assert torch.allclose(p.float(), want.float(), atol = 1e-2), name
            assert not torch.equal(p, base[name]), name
        else:
            assert torch.equal(p, base[name]), name


def test_video_load_request_takes_loras_and_defaults_to_none():
    from models.inference import VideoLoadRequest

    plain = VideoLoadRequest(model_path = "Lightricks/LTX-2")
    assert plain.loras is None
    req = VideoLoadRequest(model_path = "Lightricks/LTX-2", loras = [{"id": "my-ltx", "weight": 1}])
    assert [(s.id, s.weight) for s in req.loras] == [("my-ltx", 1.0)]
    with pytest.raises(ValidationError):
        VideoLoadRequest(
            model_path = "Lightricks/LTX-2", loras = [{"id": "a"}, {"id": "a", "weight": 0.5}]
        )
    with pytest.raises(ValidationError):
        VideoLoadRequest(model_path = "Lightricks/LTX-2", loras = [{"id": "a", "weight": 3}])


def test_status_and_gallery_models_report_baked_loras():
    from models.inference import GalleryVideo, VideoStatusResponse

    assert "baked_loras" in VideoStatusResponse.model_fields
    assert "baked_lora_specs" in VideoStatusResponse.model_fields
    assert VideoStatusResponse.model_fields["lora_refusal"].default is None
    assert GalleryVideo.model_fields["baked_loras"].default_factory() == []


def test_wan_adapter_bakes_into_the_transformer(tmp_path):
    torch.manual_seed(0)
    module = _WanLike().to(torch.bfloat16)
    base = {k: v.clone() for k, v in module.state_dict().items()}
    path = tmp_path / "wan.safetensors"
    # Wan's saver passes the adapter metadata, so alpha / r comes from the file itself.
    factors = _write_adapter(path, module, "blocks", _WAN_LEAVES, alpha = 8, metadata = True)
    merged = video_lora.bake_into_module_(module, [(str(path), 0.5)])
    assert merged == len(factors) == 2 * len(_WAN_LEAVES)
    _assert_baked(module, base, factors, scale = 0.5 * 8 / RANK)


def test_ltx2_adapter_takes_alpha_from_the_studio_sidecar(tmp_path):
    torch.manual_seed(0)
    module = _LtxLike().to(torch.bfloat16)
    base = {k: v.clone() for k, v in module.state_dict().items()}
    path = tmp_path / "ltx.safetensors"
    # LTX2Pipeline.save_lora_weights stores no adapter metadata; the catalog sidecar holds rank and alpha.
    factors = _write_adapter(
        path, module, "transformer_blocks", _LTX_LEAVES, alpha = 8, metadata = False
    )
    path.with_suffix(".json").write_text(
        json.dumps({"lora_rank": RANK, "lora_alpha": 8}), encoding = "utf-8"
    )
    params_before = {n: id(p) for n, p in module.named_parameters()}
    video_lora.bake_into_module_(module, [(str(path), 1.0)])
    _assert_baked(module, base, factors, scale = 8 / RANK)
    # In place: compile / offload hooks installed later see the same Parameter objects.
    assert {n: id(p) for n, p in module.named_parameters()} == params_before


def test_an_adapter_for_another_model_changes_nothing(tmp_path):
    torch.manual_seed(0)
    wan = _WanLike().to(torch.bfloat16)
    ltx = _LtxLike().to(torch.bfloat16)
    path = tmp_path / "wan.safetensors"
    _write_adapter(path, wan, "blocks", _WAN_LEAVES, alpha = RANK, metadata = True)
    base = {k: v.clone() for k, v in ltx.state_dict().items()}
    with pytest.raises(ValueError, match = "does not have"):
        video_lora.bake_into_module_(ltx, [(str(path), 1.0)])
    assert all(torch.equal(v, base[k]) for k, v in ltx.state_dict().items())


def test_a_quantized_layer_is_refused_not_half_baked(tmp_path):
    module = _WanLike().to(torch.bfloat16)
    path = tmp_path / "wan.safetensors"
    _write_adapter(path, module, "blocks", _WAN_LEAVES, alpha = RANK, metadata = True)
    module.blocks[1].ffn.net[2].weight.data = (
        module.blocks[1].ffn.net[2].weight.data.to(torch.float8_e4m3fn)
    )
    base = {k: v.clone() for k, v in module.state_dict().items()}
    with pytest.raises(ValueError, match = "dense float"):
        video_lora.bake_into_module_(module, [(str(path), 1.0)])
    assert all(torch.equal(v.float(), base[k].float()) for k, v in module.state_dict().items())


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    root = tmp_path / "loras" / "diffusion"
    root.mkdir(parents = True)
    monkeypatch.setattr(diffusion_lora, "loras_dir", lambda: root)

    def add(stem: str, family: str) -> Path:
        path = root / f"{stem}.safetensors"
        save_file(
            {"transformer.blocks.0.attn1.to_q.lora_A.weight": torch.zeros(RANK, 16)}, str(path)
        )
        path.with_suffix(".json").write_text(
            json.dumps(
                {"families": [family], "source": "studio-trained", "kind": "diffusion-lora"}
            ),
            encoding = "utf-8",
        )
        return path

    return add


def test_a_lora_for_another_family_is_refused(catalog):
    catalog("my-wan", "wan2.2-ti2v-5b")
    with pytest.raises(diffusion_lora.LoraSelectionError, match = "not the loaded 'ltx-2'"):
        video_lora.check_families([("my-wan", 1.0)], "ltx-2")
    with pytest.raises(diffusion_lora.LoraSelectionError, match = "not the loaded 'ltx-2'"):
        video_lora.resolve([("my-wan", 1.0)], family = "ltx-2", hf_token = None)
    with pytest.raises(diffusion_lora.LoraSelectionError, match = "unknown LoRA"):
        video_lora.check_families([("no-such-adapter", 1.0)], "ltx-2")


def test_a_lora_for_the_loaded_family_resolves_through_the_catalog(catalog):
    path = catalog("my-wan", "wan2.2-ti2v-5b")
    video_lora.check_families([("my-wan", 1.0)], "wan2.2-ti2v-5b")
    resolved = video_lora.resolve(
        [("my-wan", 0.75), ("ignored", 0.0)], family = "wan2.2-ti2v-5b", hf_token = None
    )
    assert resolved == (("my-wan", str(path), 0.75, None),)


def test_load_refusal_by_family_and_path(monkeypatch):
    ltx2 = detect_video_family("", override = "ltx-2")
    wan = detect_video_family("", override = "wan2.2-ti2v-5b")
    h3 = detect_video_family("", override = "minimax-h3")
    assert ltx2 is not None and wan is not None and h3 is not None
    for fam in (ltx2, wan):
        assert video_lora.load_refusal(fam, "pipeline", None) is None
        assert video_lora.load_refusal(fam, "pipeline", "int8") is None
        assert "GGUF" in video_lora.load_refusal(fam, "gguf", None)
        assert "single-file" in video_lora.load_refusal(fam, "single_file", None)
        assert video_lora.dense_transformer_quant(fam, None, [("x", 1.0)]) is None
    # MiniMax-H3 keeps LoRAs as an unmerged side branch: the hosted quantized denoiser stays a valid pick.
    assert video_lora.load_refusal(h3, "pipeline", None) is None
    assert video_lora.load_refusal(h3, "pipeline", "none") is None
    assert video_lora.load_refusal(h3, "pipeline", "int8") is None
    assert video_lora.dense_transformer_quant(h3, None, [("x", 1.0)]) is None
    # With the side branch off the LoRA bakes into bf16, so an unset precision pins the released denoiser.
    monkeypatch.setenv(video_lora.SIDE_BRANCH_ENV, "0")
    assert "bfloat16" in video_lora.load_refusal(h3, "pipeline", "int8")
    assert video_lora.dense_transformer_quant(h3, None, [("x", 1.0)]) == "none"
    monkeypatch.delenv(video_lora.SIDE_BRANCH_ENV)
    assert video_lora.dense_transformer_quant(h3, None, None) is None
    assert video_lora.dense_transformer_quant(h3, None, [("x", 0.0)]) is None
    moe = types.SimpleNamespace(name = "wan2.2-t2v-a14b", label = "Wan2.2 A14B", is_moe = True)
    # Structural status refusal: none, a two-expert model takes catalog pairs; a single adapter is refused.
    assert video_lora.load_refusal(moe, "pipeline", None) is None
    assert "two-expert" in video_lora.load_refusal(moe, "pipeline", None, [("a", 1.0)])


def test_backend_bake_records_the_adapters_for_status(tmp_path):
    from core.inference.video import VideoBackend

    module = _WanLike().to(torch.bfloat16)
    path = tmp_path / "wan.safetensors"
    _write_adapter(path, module, "blocks", _WAN_LEAVES, alpha = RANK, metadata = True)
    pipe = types.SimpleNamespace(transformer = module)
    fam = detect_video_family("", override = "wan2.2-ti2v-5b")
    merged = VideoBackend._bake_video_loras(
        pipe,
        (("my-wan", str(path), 1.0),),
        [{"id": "my-wan", "weight": 1.0}],
        fam = fam,
        target = types.SimpleNamespace(device = "cpu"),
    )
    assert merged == 2 * len(_WAN_LEAVES)
    assert video_lora.baked_lora_names(pipe) == ["my-wan"]
    assert video_lora.baked_lora_specs(pipe) == [{"id": "my-wan", "weight": 1.0}]
    assert video_lora.baked_lora_names(types.SimpleNamespace(transformer = module)) == []
    assert video_lora.baked_lora_specs(None) == []


def test_unloaded_status_reports_no_baked_loras():
    from core.inference.video import VideoBackend

    status = VideoBackend().status()
    assert status["baked_loras"] == [] and status["baked_lora_specs"] == []
    assert status["lora_refusal"] is None


def test_download_plan_refuses_a_lora_pick_the_load_would_refuse(monkeypatch):
    from core.inference import video as video_mod

    moe = types.SimpleNamespace(name = "wan2.2-t2v-a14b", label = "Wan2.2 A14B", is_moe = True)
    monkeypatch.setattr(video_mod, "_detect_load_family", lambda *a, **k: moe)
    monkeypatch.setattr(video_mod, "_assert_pick_is_not_speech", lambda *a, **k: None)
    with pytest.raises(ValueError, match = "two-expert"):
        video_mod.VideoBackend().download_plan(
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            model_kind = "pipeline",
            loras = [{"id": "a", "weight": 1}],
        )


def test_a_lora_bake_never_writes_the_base_models_quant_cache():
    """The merged denoiser quantized for a LoRA load must not be saved under the bare base model's cache key, on any
    of the cache-write branches (native, pinned precision, resident auto), immediate or deferred."""
    import ast
    import inspect

    from core.inference import video as video_mod

    tree = ast.parse(inspect.getsource(video_mod))
    writers = {"_save_runtime_quant_to_cache", "_save_native_quant_to_cache", "_NativeCacheWrites"}
    guarded = 0
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "baked_loras"
        ):
            continue
        called = {
            n.func.id
            for n in ast.walk(ast.Module(body = node.orelse, type_ignores = []))
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }
        if writers <= called:
            assert not any(isinstance(n, ast.Call) for s in node.body for n in ast.walk(s))
            guarded += 1
    assert guarded == 1
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in writers
    ]
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "baked_loras"
        ):
            inside = {id(n) for s in node.orelse for n in ast.walk(s)}
            if writers <= {n.func.id for n in calls if id(n) in inside}:
                assert all(id(n) in inside for n in calls)


def test_a_cache_only_load_resolves_hub_adapters_without_the_network(monkeypatch, tmp_path):
    """local_files_only reaches the adapter download and the repo file listing, so a cache-only video load never
    downloads an adapter or asks the Hub which file a bare repo id holds."""
    import huggingface_hub

    from core.inference import diffusion_lora

    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    (snapshot / "my_lora.safetensors").write_bytes(b"x")
    (snapshot / "README.md").write_text("")
    downloads, snapshots = [], []

    def fake_download(repo_id, filename, token, **kwargs):
        downloads.append((repo_id, filename, kwargs.get("local_files_only")))
        return str(snapshot / filename)

    def fake_snapshot(repo_id, **kwargs):
        snapshots.append((repo_id, kwargs.get("local_files_only")))
        return str(snapshot)

    def no_listing(*a, **k):
        raise AssertionError("the Hub file listing must not run for a cache-only load")

    monkeypatch.setattr(diffusion_lora, "hf_hub_download_with_xet_fallback", fake_download)
    monkeypatch.setattr(diffusion_lora, "_catalog_by_id", lambda: {})
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    monkeypatch.setattr(huggingface_hub.HfApi, "list_repo_files", no_listing)
    out = video_lora.resolve(
        [("someone/wan-lora", 1.0)], family = "wan2.2-ti2v-5b", hf_token = None, local_files_only = True
    )
    assert out == (("wan-lora", str(snapshot / "my_lora.safetensors"), 1.0, None),)
    assert snapshots == [("someone/wan-lora", True)]
    assert downloads == [("someone/wan-lora", "my_lora.safetensors", True)]


@pytest.mark.parametrize("local_files_only", [False, True])
def test_a_hub_adapter_s_companion_config_sets_its_scale(monkeypatch, tmp_path, local_files_only):
    """A Hub PEFT adapter whose lora_alpha lives only in adapter_config.json: the resolver fetches that config beside
    the weights (same revision, cache-only when asked), so rank 4 / alpha 8 bakes at scale 2, not 1."""
    import huggingface_hub

    from core.inference import diffusion_lora

    torch.manual_seed(0)
    module = _WanLike().to(torch.bfloat16)
    snap = tmp_path / "hub" / "models--someone--wan-lora" / "snapshots" / "abc123"
    weights = snap / "sub" / "w.safetensors"
    factors = _write_adapter(weights, module, "blocks", _WAN_LEAVES, alpha = 8, metadata = False)
    fetched = []

    def fake_download(repo_id, filename, **kwargs):
        fetched.append((repo_id, filename, kwargs.get("revision"), kwargs.get("local_files_only")))
        target = snap / filename
        target.write_text(json.dumps({"r": RANK, "lora_alpha": 8, "use_rslora": False}))
        return str(target)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.setattr(
        diffusion_lora,
        "resolve_specs",
        lambda specs, **k: [
            diffusion_lora.ResolvedLora(
                "someone/wan-lora", "wan-lora", str(weights), "safetensors", 1.0
            )
        ],
    )
    out = video_lora.resolve(
        [("someone/wan-lora", 1.0)],
        family = "wan2.2-ti2v-5b",
        hf_token = None,
        local_files_only = local_files_only,
    )
    assert fetched == [("someone/wan-lora", "sub/adapter_config.json", "abc123", local_files_only)]
    base = {k: v.clone() for k, v in module.state_dict().items()}
    video_lora.bake_into_module_(module, [(out[0][1], 1.0)])
    _assert_baked(module, base, factors, scale = 8 / RANK)


@pytest.mark.parametrize("how", ["config", "magnitude_key"])
def test_a_dora_adapter_is_refused_before_anything_is_baked(monkeypatch, tmp_path, how):
    """DoRA's per-weight magnitude cannot be reproduced by a plain delta bake: refuse it at resolution (a 400 before the
    resident model is touched) instead of reporting a wrong merge as baked."""
    from safetensors.torch import save_file

    from core.inference import diffusion_lora

    path = tmp_path / "dora" / "w.safetensors"
    path.parent.mkdir()
    state = {
        "transformer.blocks.0.attn1.to_q.lora_A.weight": torch.zeros(4, 8),
        "transformer.blocks.0.attn1.to_q.lora_B.weight": torch.zeros(8, 4),
    }
    if how == "magnitude_key":
        state["transformer.blocks.0.attn1.to_q.lora_magnitude_vector"] = torch.ones(8)
    else:
        (path.parent / "adapter_config.json").write_text(
            json.dumps({"r": 4, "lora_alpha": 4, "use_dora": True})
        )
    save_file(state, str(path))
    monkeypatch.setattr(
        diffusion_lora,
        "resolve_specs",
        lambda specs, **k: [
            diffusion_lora.ResolvedLora("me/dora", "dora", str(path), "safetensors", 1.0)
        ],
    )
    with pytest.raises(diffusion_lora.LoraSelectionError, match = "DoRA"):
        video_lora.resolve([("me/dora", 1.0)], family = "wan2.2-ti2v-5b", hf_token = None)


@pytest.mark.parametrize("how", ["config", "bias_key"])
def test_a_lora_bias_adapter_is_refused_before_anything_is_baked(monkeypatch, tmp_path, how):
    """A trained lora_B bias is not part of a weight bake: refuse it at resolution instead of dropping it silently."""
    from safetensors.torch import save_file

    from core.inference import diffusion_lora

    path = tmp_path / "biased" / "w.safetensors"
    path.parent.mkdir()
    state = {
        "transformer.blocks.0.attn1.to_q.lora_A.weight": torch.zeros(4, 8),
        "transformer.blocks.0.attn1.to_q.lora_B.weight": torch.zeros(8, 4),
    }
    if how == "bias_key":
        state["transformer.blocks.0.attn1.to_q.lora_B.bias"] = torch.ones(8)
    else:
        (path.parent / "adapter_config.json").write_text(
            json.dumps({"r": 4, "lora_alpha": 4, "lora_bias": True})
        )
    save_file(state, str(path))
    monkeypatch.setattr(
        diffusion_lora,
        "resolve_specs",
        lambda specs, **k: [
            diffusion_lora.ResolvedLora("me/biased", "biased", str(path), "safetensors", 1.0)
        ],
    )
    with pytest.raises(diffusion_lora.LoraSelectionError, match = "LoRA-bias"):
        video_lora.resolve([("me/biased", 1.0)], family = "wan2.2-ti2v-5b", hf_token = None)


@pytest.mark.parametrize(
    "extra, config, label",
    [
        ({"transformer.norm_out.linear.weight": torch.zeros(8, 8)}, None, "full-weight"),
        ({"transformer.blocks.0.norm1.diff_b": torch.zeros(8)}, None, "full-weight"),
        ({}, {"r": 4, "lora_alpha": 4, "fan_in_fan_out": True}, "fan_in_fan_out"),
    ],
)
def test_adapters_with_parts_a_weight_bake_would_skip_are_refused(
    monkeypatch, tmp_path, extra, config, label
):
    """modules_to_save copies, diff tensors and transposed factors would be dropped or merged wrong by the bake."""
    from safetensors.torch import save_file

    from core.inference import diffusion_lora

    path = tmp_path / "odd" / "w.safetensors"
    path.parent.mkdir()
    state = {
        "transformer.blocks.0.attn1.to_q.lora_A.weight": torch.zeros(4, 8),
        "transformer.blocks.0.attn1.to_q.lora_B.weight": torch.zeros(8, 4),
        **extra,
    }
    if config is not None:
        (path.parent / "adapter_config.json").write_text(json.dumps(config))
    save_file(state, str(path))
    monkeypatch.setattr(
        diffusion_lora,
        "resolve_specs",
        lambda specs, **k: [
            diffusion_lora.ResolvedLora("me/odd", "odd", str(path), "safetensors", 1.0)
        ],
    )
    with pytest.raises(diffusion_lora.LoraSelectionError, match = label):
        video_lora.resolve([("me/odd", 1.0)], family = "wan2.2-ti2v-5b", hf_token = None)
