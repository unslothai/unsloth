# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hosted QUANTIZED conditioner route for MiniMax-H3's Diffusers path.

Mirrors tests/test_video_prequant.py, which covers the denoiser half of the same idea: the
resolver, the download-plan hooks that decide whether the base repo's dense ``text_encoder/``
shards are staged, the ConvRot INT8 arithmetic, and the memory floor recomputed from what the
load actually holds.

Network-free and (except for the two arithmetic tests) torch-free. The 27 GB artifact itself is
exercised end to end on a GPU, not here."""

from __future__ import annotations

import types

import pytest

from core.inference.video import VideoBackend
from core.inference.video_families import detect_video_family
from core.inference.video_minimax_h3 import (
    H3_DIFFUSERS_VRAM_BASE_GB,
    H3_TEXT_ENCODER_BF16_GB,
    H3_TRANSFORMER_BF16_GB,
    estimate_h3_diffusers_vram_gb,
    h3_diffusers_vram_base_gb,
    h3_transformer_resident_gb,
)
from core.inference.video_minimax_h3_te import (
    H3_TE_CONVROT_GROUP,
    H3_TE_QUANT_FILES,
    H3_TE_QUANT_REPO,
    H3_TE_READ_LAYER,
    h3_te_quant_filename,
    h3_te_quant_scheme,
    h3_te_remap_key,
    h3_te_resident_gb,
)

torch = pytest.importorskip("torch", reason = "the ConvRot arithmetic tests need torch")


H3_BASE = "MiniMaxAI/MiniMax-H3"


def _fam(
    modular_workflow = "fl2va",
    name = "minimax-h3",
    base_repo = H3_BASE,
):
    return types.SimpleNamespace(name = name, modular_workflow = modular_workflow, base_repo = base_repo)


# Device targets are passed EXPLICITLY below: the auto default reads the real device when none is
# given, and a test whose answer depends on the runner's GPU is not a test.
def _cuda_target():
    return types.SimpleNamespace(
        device = "cuda", dtype = torch.bfloat16, supports_default_torch_compile = True
    )


def _cpu_target():
    return types.SimpleNamespace(
        device = "cpu", dtype = torch.float32, supports_default_torch_compile = False
    )


def _mps_target():
    return types.SimpleNamespace(
        device = "mps", dtype = torch.bfloat16, supports_default_torch_compile = False
    )


# ── the resolver ─────────────────────────────────────────────────────────────────
def test_only_int8_has_a_hosted_conditioner():
    assert h3_te_quant_scheme("int8") == "int8"
    assert h3_te_quant_scheme("INT8") == "int8"
    # Valid text_encoder_quant modes with no hosted H3 artifact resolve to nothing rather than
    # raising: the request is well formed, this family just cannot serve it.
    for mode in ("fp8", "fp8_dynamic", "nvfp4", "fp8-dynamic", "", None):
        assert h3_te_quant_scheme(mode) is None


def test_the_hosted_filename_is_the_comfy_component_repo():
    # H3_TE_QUANT_REPO must stay the repo the Diffusers path already pulls its VAEs from, or this
    # adds a second component dependency nobody staged.
    from core.inference.video_minimax_h3 import H3_COMPONENT_REPO

    assert H3_TE_QUANT_REPO == H3_COMPONENT_REPO
    assert (
        h3_te_quant_filename("int8")
        == "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
    )
    assert h3_te_quant_filename("fp8") is None
    assert h3_te_quant_filename(None) is None


def test_every_hosted_scheme_is_a_valid_text_encoder_quant_request():
    # A scheme this module offers but normalize_te_quant refuses could never be requested.
    from core.inference.diffusion_precision import normalize_te_quant
    for scheme in H3_TE_QUANT_FILES:
        assert normalize_te_quant(scheme) == scheme


# ── the name mapping ─────────────────────────────────────────────────────────────
def test_remap_puts_comfy_names_into_the_transformers_tree():
    # ComfyUI flattens the Qwen3-VL tree; transformers nests both halves under model.
    assert (
        h3_te_remap_key("model.layers.0.self_attn.q_proj.weight")
        == "model.language_model.layers.0.self_attn.q_proj.weight"
    )
    assert (
        h3_te_remap_key("model.embed_tokens.weight") == "model.language_model.embed_tokens.weight"
    )
    assert h3_te_remap_key("visual.blocks.3.attn.qkv.bias") == "model.visual.blocks.3.attn.qkv.bias"
    # NOT idempotent, on purpose: re-mapping an already-transformers name produces a key no module
    # owns, so an artifact re-uploaded in transformers naming fails the strict load loudly instead
    # of half-matching.
    assert h3_te_remap_key("model.language_model.layers.0.mlp.up_proj.weight") == (
        "model.language_model.language_model.layers.0.mlp.up_proj.weight"
    )


# ── the ConvRot arithmetic ───────────────────────────────────────────────────────
def test_the_convrot_hadamard_is_symmetric_and_its_own_inverse():
    from core.inference.video_minimax_h3_te import build_convrot_hadamard

    h = build_convrot_hadamard(H3_TE_CONVROT_GROUP)
    assert h.shape == (H3_TE_CONVROT_GROUP, H3_TE_CONVROT_GROUP)
    # Exactly, not approximately: the entries are +-1/16 and the products are sums of 256 of them.
    assert torch.equal(h, h.T)
    assert (h @ h - torch.eye(H3_TE_CONVROT_GROUP)).abs().max().item() == 0.0
    # Regular Hadamard: a power of 4 only. 128 is a power of 2 and must still be refused.
    for bad in (128, 100, 2):
        with pytest.raises(ValueError):
            build_convrot_hadamard(bad)


def test_rotating_the_activation_undoes_the_rotation_baked_into_the_weight():
    """The load-bearing identity: x_rot @ W_rot.T == x @ W.T.

    If this were false the INT8 conditioner would decode to noise, and a load that only checked
    file sizes would never notice. Dequantizing WITHOUT the rotation is the control."""
    from core.inference.video_minimax_h3_te import (
        _int8_convrot_linear_class,
        build_convrot_hadamard,
        rotate_convrot_activation,
    )

    torch.manual_seed(0)
    group = H3_TE_CONVROT_GROUP
    out_features, in_features = 64, group * 3
    weight = torch.randn(out_features, in_features) * 0.02
    x = torch.randn(5, in_features)

    h = build_convrot_hadamard(group)
    # The offline half, exactly as the artifact was baked: W_rot = W @ H.T blockwise.
    grouped = weight.reshape(out_features, in_features // group, group)
    weight_rot = grouped.matmul(h.T).reshape(out_features, in_features)
    # Round trip through INT8 per output channel, as the hosted checkpoint stores it.
    scale = (weight_rot.abs().amax(dim = -1, keepdim = True) / 127.0).clamp(min = 1e-30)
    qdata = torch.round(weight_rot / scale).clamp(-127, 127).to(torch.int8)

    module = _int8_convrot_linear_class()(in_features, out_features, bias = False, group_size = group)
    module.load_state_dict({"weight": qdata, "weight_scale": scale}, strict = True, assign = True)
    with torch.no_grad():
        got = module(x)
    reference = torch.nn.functional.linear(x, weight)
    # Everything but the INT8 rounding is exact, so this is a quantization-error bound, not a
    # numerical-slop one.
    assert (got - reference).norm() / reference.norm() < 0.02

    # Control: skipping the activation rotation is not "slightly worse", it is unrelated.
    unrotated = torch.nn.functional.linear(x, qdata.float() * scale)
    assert (unrotated - reference).norm() / reference.norm() > 0.5
    # And the rotation itself is an involution.
    assert torch.allclose(
        rotate_convrot_activation(rotate_convrot_activation(x, h, group), h, group), x, atol = 1e-5
    )


def test_the_int8_module_keeps_its_weight_quantized():
    """The resident footprint IS the point, so a cached dense view would defeat the whole path."""
    from core.inference.video_minimax_h3_te import _int8_convrot_linear_class

    group = H3_TE_CONVROT_GROUP
    module = _int8_convrot_linear_class()(group, 8, bias = False, group_size = group)
    module.load_state_dict(
        {
            "weight": torch.zeros(8, group, dtype = torch.int8),
            "weight_scale": torch.ones(8, 1),
        },
        strict = True,
        assign = True,
    )
    with torch.no_grad():
        module(torch.randn(2, group))
    assert module.weight.dtype is torch.int8
    assert sum(t.numel() * t.element_size() for t in module.buffers()) == 8 * group + 8 * 4


# ── the read layer ───────────────────────────────────────────────────────────────
def test_the_read_layer_matches_the_diffusers_pipeline():
    """MiniMax-H3 conditions on hidden_states[50]. If diffusers ever moves it, the 50-layer
    artifact stops being lossless and this must fail rather than ship a silent approximation."""
    pytest.importorskip("diffusers", reason = "reads the installed pipeline's own constant")
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import MiniMaxH3ModularPipeline

    assert MiniMaxH3ModularPipeline.text_encoder_layer.fget(None) == H3_TE_READ_LAYER


# ── the memory floor ─────────────────────────────────────────────────────────────
def test_the_default_floor_is_unchanged():
    """Backwards compatibility: nothing that does not pass a size sees a different number."""
    assert h3_diffusers_vram_base_gb() == pytest.approx(H3_DIFFUSERS_VRAM_BASE_GB, abs = 0.001)
    assert estimate_h3_diffusers_vram_gb(960, 544, 124) == pytest.approx(73.68, abs = 0.02)
    assert estimate_h3_diffusers_vram_gb(1344, 768, 345) == pytest.approx(96.98, abs = 0.02)


def test_an_unpinned_floor_is_the_largest_component_not_the_sum():
    # Every component runs under CPU offload, so quantizing ONE of the two 66 GB components buys
    # nothing while both stay in the rotation.
    assert h3_diffusers_vram_base_gb(transformer_gb = 20.3) == pytest.approx(68.5, abs = 0.001)
    assert h3_diffusers_vram_base_gb(text_encoder_gb = 27.2) == pytest.approx(68.1, abs = 0.001)


def test_a_pinned_denoiser_makes_the_floor_additive():
    """The pre-quantized denoiser is pinned out of the offload rotation, so it is resident
    alongside whatever else is running. Treating that as a max under-states the floor by the whole
    denoiser, which is the direction that lets a doomed generation start."""
    pinned = h3_diffusers_vram_base_gb(transformer_gb = 20.3, transformer_pinned = True)
    assert pinned == pytest.approx(20.3 + 66.7 + 2.6, abs = 0.001)
    assert pinned > h3_diffusers_vram_base_gb(transformer_gb = 20.3)
    # A tiny conditioner cannot drag the floor under the VAEs, which still rotate through.
    assert h3_diffusers_vram_base_gb(
        text_encoder_gb = 1.0, transformer_gb = 20.3, transformer_pinned = True
    ) == pytest.approx(20.3 + 11.1 + 2.6, abs = 0.001)


def test_the_floor_matches_the_two_measured_runs():
    """Calibration, against torch.cuda.max_memory_allocated over a real 960x544x124 generation with
    the int8 denoiser pinned. Both must be covered, and neither by an absurd margin."""
    for text_encoder_gb, measured in ((H3_TEXT_ENCODER_BF16_GB, 94.62), (27.2, 55.20)):
        predicted = estimate_h3_diffusers_vram_gb(
            960,
            544,
            124,
            text_encoder_gb = text_encoder_gb,
            transformer_gb = 20.3,
            transformer_pinned = True,
        )
        assert measured <= predicted <= measured + 1.0, (text_encoder_gb, predicted, measured)


def test_quantizing_the_conditioner_is_what_finally_moves_the_floor():
    dense = estimate_h3_diffusers_vram_gb(
        960, 544, 124, transformer_gb = 20.3, transformer_pinned = True
    )
    quantized = estimate_h3_diffusers_vram_gb(
        960, 544, 124, text_encoder_gb = 27.2, transformer_gb = 20.3, transformer_pinned = True
    )
    # The saving is the conditioner delta and nothing else.
    assert dense - quantized == pytest.approx(H3_TEXT_ENCODER_BF16_GB - 27.2, abs = 0.001)
    assert quantized < H3_DIFFUSERS_VRAM_BASE_GB


def test_resident_sizes_track_the_engaged_scheme_only():
    assert h3_te_resident_gb("int8", bf16_gb = H3_TEXT_ENCODER_BF16_GB) == 27.2
    # A scheme with no hosted artifact, and a declined request recorded as None, both keep the
    # dense budget. Under-stating the floor is the expensive direction.
    for mode in (None, "fp8", "nvfp4"):
        assert h3_te_resident_gb(mode, bf16_gb = H3_TEXT_ENCODER_BF16_GB) == H3_TEXT_ENCODER_BF16_GB
    assert h3_transformer_resident_gb("int8") == 20.3
    assert h3_transformer_resident_gb(None) == H3_TRANSFORMER_BF16_GB


# ── the download plan ────────────────────────────────────────────────────────────
def test_only_a_modular_family_drops_its_dense_encoder():
    assert VideoBackend._h3_te_quant_scheme(_fam(), "int8", H3_BASE) == "int8"
    # A conventional family casts its own dense encoder in place and still needs those shards.
    assert VideoBackend._h3_te_quant_scheme(_fam(modular_workflow = None), "int8", H3_BASE) is None
    assert VideoBackend._h3_te_quant_scheme(_fam(), "fp8", H3_BASE) is None


# ── the tri-state: unset is the fast default, "none" is the escape hatch ─────────
def test_an_unset_request_takes_the_hosted_conditioner():
    """Unset is what the video page sends, so this is the whole point: it must resolve to the
    hosted 27.1 GB / 50-layer conditioner, not to the released 66.7 GB / 64-layer one."""
    assert VideoBackend._h3_te_quant_scheme(_fam(), None, H3_BASE, _cuda_target()) == "int8"
    assert VideoBackend._h3_te_quant_scheme(_fam(), "auto", H3_BASE, _cuda_target()) == "int8"
    assert VideoBackend._h3_te_quant_scheme(_fam(), "", H3_BASE, _cuda_target()) == "int8"


def test_none_still_pins_the_released_encoder():
    """The escape hatch has to exist and has to be distinguishable from unset, or a bit-exact
    comparison against the released components becomes unexpressible."""
    for pinned in ("none", "None", "off", " OFF "):
        assert VideoBackend._h3_te_quant_scheme(_fam(), pinned, H3_BASE, _cuda_target()) is None


def test_the_auto_default_is_cuda_only():
    """MPS and CPU keep the components they load today. The ConvRot forward is plain torch and
    would likely run there, but nobody has measured it, and the modular loader does not reach a Mac
    at all (ComponentsManager.enable_auto_cpu_offload needs mem_get_info, which torch.mps lacks).
    An EXPLICIT request is unaffected by this gate."""
    for target in (_cpu_target(), _mps_target()):
        assert VideoBackend._h3_te_quant_scheme(_fam(), None, H3_BASE, target) is None
        assert VideoBackend._h3_te_quant_scheme(_fam(), "int8", H3_BASE, target) == "int8"
    # A CUDA target whose compute dtype is not bf16 (a pre-Ampere card promoted to fp32) is not a
    # device this was measured on either.
    fp32_cuda = types.SimpleNamespace(device = "cuda", dtype = torch.float32)
    assert VideoBackend._h3_te_quant_scheme(_fam(), None, H3_BASE, fp32_cuda) is None


def test_an_unset_request_on_a_derivative_still_keeps_its_own_encoder():
    """The auto default must not loosen the base gate: substituting someone else's conditioner is
    exactly as wrong when the backend chose it as when the user asked for it."""
    assert (
        VideoBackend._h3_te_quant_scheme(_fam(), None, "someone/MiniMax-H3-anime", _cuda_target())
        is None
    )
    assert VideoBackend._h3_te_quant_scheme(_fam(), None, None, _cuda_target()) is None


def test_only_the_base_the_artifact_was_cut_from_gets_the_hosted_conditioner():
    """A derivative can keep the Qwen3-VL architecture and change the conditioner weights. The
    strict load cannot tell -- every name and shape still matches -- so gate on the base instead,
    exactly as the pre-quantized denoiser does through its baked base_model_id."""
    # A local snapshot of the same base still qualifies (same repo tail).
    assert VideoBackend._h3_te_quant_scheme(_fam(), "int8", "/models/MiniMax-H3") == "int8"
    # A derivative, however it was selected (detection or family_override), keeps its own encoder.
    for other in ("someone/MiniMax-H3-anime", "someone/MiniMax-H3-v2", "MiniMaxAI/MiniMax-H2"):
        assert VideoBackend._h3_te_quant_scheme(_fam(), "int8", other) is None
    # And an unknown base is not a licence to substitute one either.
    assert VideoBackend._h3_te_quant_scheme(_fam(), "int8", None) is None
    assert VideoBackend._h3_te_quant_scheme(_fam(base_repo = None), "int8", H3_BASE) is None


def test_an_unsupported_request_never_breaks_the_plan():
    # The plan runs before validate_load_request has had the last word on some paths, and a raise
    # here would cost the whole download plan rather than one optimisation.
    assert VideoBackend._h3_te_quant_scheme(_fam(), "not-a-scheme", H3_BASE) is None
    assert VideoBackend._h3_te_quant_scheme(object(), "int8", H3_BASE) is None


def test_the_real_family_resolves_the_hosted_conditioner():
    fam = detect_video_family(H3_BASE)
    assert fam is not None and fam.modular_workflow
    assert VideoBackend._h3_te_quant_scheme(fam, "int8", H3_BASE) == "int8"


def test_an_unresolvable_artifact_keeps_the_dense_shards():
    """Only an artifact that really exists on the Hub earns the right to drop 62 GB of encoder."""

    class _Boom:
        def model_info(self, *_args, **_kwargs):
            raise RuntimeError("hub down")

    assert VideoBackend._h3_te_quant_hub_files("int8", _Boom()) == (None, [])
    assert VideoBackend._h3_te_quant_hub_files(None, _Boom()) == (None, [])

    class _Empty:
        def model_info(self, *_args, **_kwargs):
            return types.SimpleNamespace(siblings = [])

    assert VideoBackend._h3_te_quant_hub_files("int8", _Empty()) == (None, [])


def test_a_resolvable_artifact_is_staged_in_place_of_the_dense_shards():
    wanted = h3_te_quant_filename("int8")

    class _Api:
        def model_info(self, repo_id, **_kwargs):
            assert repo_id == H3_TE_QUANT_REPO
            return types.SimpleNamespace(
                siblings = [
                    types.SimpleNamespace(rfilename = wanted, size = 27_141_342_152),
                    types.SimpleNamespace(rfilename = "vae/other.safetensors", size = 5),
                ]
            )

    repo, files = VideoBackend._h3_te_quant_hub_files("int8", _Api())
    assert repo == H3_TE_QUANT_REPO
    # Exactly the one artifact, at its real size: the disk preflight is sized off this.
    assert files == [(wanted, 27_141_342_152)]


def test_the_conditioner_repo_is_protected_while_the_load_is_in_flight(monkeypatch):
    """The artifact comes from a THIRD repo. Without it in ``asset_repos`` the delete-cached guard
    would let it go while the fetch (or the base pull that no longer carries a dense encoder) is
    still running."""
    from core.inference import video as video_mod

    backend = VideoBackend()
    backend._load_token = 7
    backend._loading = video_mod._VideoLoadingState(repo_id = H3_BASE, base_repo = H3_BASE)

    seen: dict[str, tuple[str, ...]] = {}
    # The verified resolver reads the base's modular index; this test is about the delete guard,
    # not the index, and the suite blocks the network.
    monkeypatch.setattr(backend, "_h3_te_base_index_source", lambda *a, **k: H3_BASE)
    monkeypatch.setattr(backend, "_estimate_download_bytes", lambda *a, **k: 1)
    monkeypatch.setattr(backend, "_fetch_te_prequant", lambda *a, **k: ())
    monkeypatch.setattr(backend, "_predownload_base", lambda *a, **k: None)
    monkeypatch.setattr(backend, "load_pipeline", lambda **k: None)

    def _fetch(scheme, _token, **_kwargs):
        seen["scheme"] = scheme
        seen["ids"] = backend.loading_repo_ids()
        return ("text_encoder",)

    monkeypatch.setattr(backend, "_fetch_h3_te_quant", _fetch)
    backend._run_load(
        repo_id = H3_BASE,
        model_kind = "pipeline",
        text_encoder_quant = "int8",
        _load_token = 7,
    )
    assert seen["scheme"] == "int8"
    assert H3_TE_QUANT_REPO in seen["ids"]


def test_the_conditioner_repo_is_not_claimed_by_a_load_that_does_not_want_it(monkeypatch):
    from core.inference import video as video_mod

    backend = VideoBackend()
    backend._load_token = 7
    backend._loading = video_mod._VideoLoadingState(repo_id = H3_BASE, base_repo = H3_BASE)
    monkeypatch.setattr(backend, "_estimate_download_bytes", lambda *a, **k: 1)
    monkeypatch.setattr(backend, "_fetch_te_prequant", lambda *a, **k: ())
    monkeypatch.setattr(backend, "_fetch_h3_te_quant", lambda *a, **k: ())
    monkeypatch.setattr(backend, "_predownload_base", lambda *a, **k: None)
    monkeypatch.setattr(backend, "load_pipeline", lambda **k: None)
    backend._run_load(repo_id = H3_BASE, model_kind = "pipeline", _load_token = 7)
    assert backend._loading is None or H3_TE_QUANT_REPO not in backend.loading_repo_ids()


def test_the_encoder_config_is_read_from_the_pinned_cache_not_the_default_one(monkeypatch):
    """Unsloth runs on a configured cache root. An AutoConfig call that ignores it resolves against
    huggingface_hub's import-time default, which re-downloads into a root Unsloth does not read and
    simply fails on an offline host that has already staged the model."""
    import sys

    import core.inference.video_minimax_h3_te as te_mod

    captured: dict[str, object] = {}

    class _AutoConfig:
        @staticmethod
        def from_pretrained(name, **kwargs):
            captured["name"] = name
            captured["kwargs"] = kwargs
            # The load is best-effort by contract, so raising here returns None and exercises
            # exactly the one call this test is about.
            raise RuntimeError("only the call shape is under test")

    # Every import the loader makes before the config read is stubbed, so this asserts the call
    # shape on any host. Without accelerate / safetensors it would otherwise return None from the
    # import line and never reach AutoConfig, which is a pass for the wrong reason (and a KeyError
    # on the assertions below) on a CPU runner that has torch but not the rest.
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoConfig = _AutoConfig))
    monkeypatch.setitem(
        sys.modules,
        "accelerate",
        types.SimpleNamespace(init_empty_weights = lambda **_k: None),
    )
    monkeypatch.setitem(
        sys.modules, "safetensors", types.SimpleNamespace(safe_open = lambda *a, **k: None)
    )
    monkeypatch.setitem(
        sys.modules,
        "utils.hf_xet_fallback",
        types.SimpleNamespace(hf_hub_download_with_xet_fallback = lambda *a, **k: "/nope"),
    )

    assert (
        te_mod.load_h3_quantized_text_encoder(
            H3_BASE, "int8", dtype = None, cache_dir = "/tmp/studio-hub"
        )
        is None
    )
    assert captured["name"] == H3_BASE
    assert captured["kwargs"]["cache_dir"] == "/tmp/studio-hub"
    assert captured["kwargs"]["subfolder"] == "text_encoder"

    # A staged snapshot is preferred over the hub id: its config.json is already on disk, so the
    # resolution cannot go to the network at all.
    captured.clear()
    te_mod.load_h3_quantized_text_encoder(
        H3_BASE, "int8", dtype = None, cache_dir = "/tmp/studio-hub", local_base = "/snap/h3"
    )
    assert captured["name"] == "/snap/h3"


def test_the_dense_encoder_is_dropped_from_the_base_pull_but_its_config_is_kept():
    """The loader meta-inits from <base>/text_encoder/config.json, so dropping that would break
    the very load that made the skip safe."""
    from core.inference.diffusion_te_prequant import is_prequant_covered_weight

    covered = ("text_encoder",)
    assert is_prequant_covered_weight("text_encoder/model-00001-of-00014.safetensors", covered)
    assert not is_prequant_covered_weight("text_encoder/config.json", covered)
    assert not is_prequant_covered_weight("text_encoder/tokenizer.json", covered)
    # Other components are untouched.
    assert not is_prequant_covered_weight("vae/diffusion_pytorch_model.safetensors", covered)
    assert not is_prequant_covered_weight(
        "transformer/diffusion_pytorch_model.safetensors", covered
    )


# ── the resolved record ──────────────────────────────────────────────────────────
def test_a_declined_request_reads_as_a_fallback_not_as_never_asked():
    """The record has to keep the REQUEST on the left. Erasing it to None makes a refused fp8
    request indistinguishable from a load nobody asked to quantize."""
    from core.inference.diffusion_auto_policy import build_resolved_record

    record = build_resolved_record(
        {
            "text_encoder_quant": (
                "fp8",
                "off",
                "no hosted quantized fp8 conditioner for minimax-h3; "
                "loaded the released bfloat16 encoder instead",
            )
        }
    )
    entry = record["text_encoder_quant"]
    assert entry["requested"] == "fp8"
    assert entry["value"] == "off"
    assert entry["status"] != "as_requested"


# ── the identity gate the base NAME cannot make ──────────────────────────────────
def test_the_index_names_the_conditioner_a_pipeline_would_have_loaded():
    """A repo-id comparison cannot tell a derivative stored as .../MiniMax-H3 from the real one.
    The pipeline's own modular index can: it records where each component comes from, and a
    derivative that retrained the conditioner ships it under its own id."""

    def _pipe(source):
        spec = types.SimpleNamespace(pretrained_model_name_or_path = source)
        return types.SimpleNamespace(get_component_spec = lambda name: spec)

    assert VideoBackend._h3_te_index_source(_pipe(H3_BASE)) == H3_BASE
    assert VideoBackend._h3_te_index_source(_pipe("someone/MiniMax-H3")) == "someone/MiniMax-H3"
    # Unanswerable is None, and the caller reads None as a refusal.
    assert VideoBackend._h3_te_index_source(_pipe(None)) is None
    assert VideoBackend._h3_te_index_source(_pipe(["a", "b"])) is None
    assert VideoBackend._h3_te_index_source(_pipe("")) is None
    assert VideoBackend._h3_te_index_source(object()) is None

    # The deprecated spelling, for a spec built by hand rather than parsed from the index.
    legacy = types.SimpleNamespace(repo = H3_BASE, pretrained_model_name_or_path = None)
    assert (
        VideoBackend._h3_te_index_source(
            types.SimpleNamespace(get_component_spec = lambda name: legacy)
        )
        == H3_BASE
    )


def test_the_real_index_records_where_the_conditioner_comes_from():
    """Pinned against the shape MiniMaxAI/MiniMax-H3 actually ships, so a schema change is caught
    here rather than by silently declining every quantized conditioner."""
    entry = [
        "transformers",
        "Qwen3VLForConditionalGeneration",
        {
            "type_hint": ["transformers", "Qwen3VLForConditionalGeneration"],
            "pretrained_model_name_or_path": H3_BASE,
            "subfolder": "text_encoder",
            "variant": None,
            "revision": None,
        },
    ]
    from core.inference.diffusion_te_prequant import te_base_equivalent

    source = entry[2]["pretrained_model_name_or_path"]
    assert te_base_equivalent("MiniMaxAI/MiniMax-H3", source)
    # A derivative that retrained its conditioner names itself here and is refused, even though
    # its own repo id would pass the tail-segment comparison.
    assert not te_base_equivalent("MiniMaxAI/MiniMax-H3", "someone/MiniMax-H3-anime")


def test_the_index_compare_has_no_tail_name_tolerance():
    """The plan-side gate uses te_base_equivalent, which accepts a matching final path segment.
    That tolerance is the hole this gate closes, so it must not be reused here: someone/MiniMax-H3
    is a different repo with different weights."""
    from core.inference.diffusion_te_prequant import te_base_equivalent
    from core.inference.video import _h3_te_canonical

    # What the tolerant helper accepts and this one must not.
    for other in ("someone/MiniMax-H3", "/models/MiniMax-H3", "someone/minimax-h3"):
        assert te_base_equivalent(H3_BASE, other), "precondition: the tolerant helper accepts it"
        assert _h3_te_canonical(other) != _h3_te_canonical(H3_BASE)
    # The canonical id, in any casing or with stray whitespace, still matches.
    for same in (H3_BASE, H3_BASE.lower(), f"  {H3_BASE} "):
        assert _h3_te_canonical(same) == _h3_te_canonical(H3_BASE)
    assert _h3_te_canonical(None) == ""


def test_a_known_mirror_is_still_the_same_conditioner():
    """canonical_base folds a mirror onto the id it copies, so tightening the compare must not
    refuse one. Skipped when no mirror is registered rather than asserting a table entry."""
    from core.inference.diffusion_families import _MIRROR_UPSTREAM
    from core.inference.video import _h3_te_canonical

    for mirror, upstream in _MIRROR_UPSTREAM.items():
        assert _h3_te_canonical(mirror) == _h3_te_canonical(upstream)


# ── the precision contract ───────────────────────────────────────────────────────
def test_a_hosted_conditioner_is_not_judged_by_the_generic_precision_gate(monkeypatch):
    """The gate rewrites int8 -> fp8 (H3 has no keep-bf16 schedule) and then asks for fp8 tensor
    cores. This loader uses neither: INT8 storage, a Hadamard rotation, an ordinary F.linear. Left
    to the generic path, a CPU H3 int8 load comes back as a 409 for hardware nothing needs."""
    from core.inference.video import assert_video_precision_available

    fam = detect_video_family(H3_BASE)
    monkeypatch.setattr(
        "core.inference.video.precision_fallback_allowed", lambda: False, raising = False
    )
    monkeypatch.setattr(
        "core.inference.video.te_quant_supported", lambda *_a, **_k: False, raising = False
    )
    # The hosted scheme is exempt.
    assert_video_precision_available(fam, model_kind = "pipeline", text_encoder_quant = "int8")
    # A scheme with no hosted artifact is still judged by the generic gate.
    with pytest.raises(RuntimeError):
        assert_video_precision_available(fam, model_kind = "pipeline", text_encoder_quant = "fp8")


def test_an_explicit_encoder_request_that_engages_nothing_is_refused(monkeypatch):
    """The conventional path already raises here: a render that succeeds at an unrequested
    precision quietly invalidates whatever it was measuring. The modular path has to match, with
    the same documented escape hatch."""
    import core.inference.video as video_mod

    seen: dict = {}

    class _Pipe:
        def get_component_spec(self, _name):
            return types.SimpleNamespace(pretrained_model_name_or_path = "someone/MiniMax-H3")

        def update_components(self, **kwargs):
            seen["seeded"] = kwargs

        def load_components(self, **kwargs):
            seen["loaded"] = kwargs

    fam = detect_video_family(H3_BASE)
    backend = video_mod.VideoBackend()
    fake_diffusers = types.SimpleNamespace(
        ComponentsManager = lambda: object(),
        ModularPipeline = types.SimpleNamespace(from_pretrained = lambda *a, **k: _Pipe()),
    )
    monkeypatch.setattr(video_mod, "precision_fallback_allowed", lambda: False)

    with pytest.raises(RuntimeError) as exc:
        backend._load_h3_modular_pipeline(
            fam = fam,
            repo_id = H3_BASE,
            base = H3_BASE,
            kind = "pipeline",
            dtype = None,
            device = "cpu",
            hf_token = None,
            memory_mode = None,
            text_encoder_quant = "int8",
            # This test is about the ENCODER refusal, so pin the denoiser dense: unset would
            # resolve to the hosted int8 checkpoint and the fake diffusers module has no
            # transformer class to build it. CPU target so the answer cannot depend on the
            # runner's GPU.
            transformer_quant = "none",
            target = _cpu_target(),
            diffusers = fake_diffusers,
            torch = None,
        )
    assert "text_encoder_quant" in str(exc.value)
    # Refused BEFORE anything is built, so the refusal costs nothing.
    assert "loaded" not in seen and "seeded" not in seen


def test_the_staging_skip_reads_the_same_index_the_seed_will(tmp_path):
    """The plan compares repo NAMES, which is right for a plan and wrong for dropping a
    derivative's dense encoder: the seed would then decline and leave load_components to fetch
    62 GB inline. The staging resolver reads the base's own index first."""
    import json

    backend = VideoBackend()
    fam = detect_video_family(H3_BASE)

    def _base_dir(source):
        root = tmp_path / source.replace("/", "_")
        root.mkdir(parents = True, exist_ok = True)
        entry = [
            "transformers",
            "Qwen3VLForConditionalGeneration",
            {"pretrained_model_name_or_path": source, "subfolder": "text_encoder"},
        ]
        with open(root / "modular_model_index.json", "w", encoding = "utf-8") as handle:
            json.dump({"text_encoder": entry}, handle)
        return str(root)

    canonical = _base_dir(H3_BASE)
    derivative = _base_dir("someone/MiniMax-H3")
    # Both pass the NAME comparison: the directories are called MiniMaxAI_MiniMax-H3 and
    # someone_MiniMax-H3, so this is the index doing the work, not the path.
    assert backend._h3_te_base_index_source(canonical, None) == H3_BASE
    assert backend._h3_te_base_index_source(derivative, None) == "someone/MiniMax-H3"
    assert backend._h3_te_quant_scheme_verified(fam, "int8", derivative, None) is None
    # An unreadable index keeps the dense shards rather than guessing.
    assert backend._h3_te_base_index_source(str(tmp_path / "nope"), None) is None
    assert backend._h3_te_quant_scheme_verified(fam, "int8", str(tmp_path / "nope"), None) is None


def test_a_projection_left_dense_is_refused_not_budgeted():
    """strict=True proves the artifact and the skeleton name the same tensors, not that they are
    quantized. A projection re-uploaded as a plain dense weight drops out of the swap, loads
    cleanly, and would be recorded as engaged int8 while the resident encoder crept back toward
    51 GB -- and the VRAM preflight sizes the floor from the ENGAGED scheme."""
    from torch import nn

    from core.inference.video_minimax_h3_te import _int8_convrot_linear_class

    quantized_cls = _int8_convrot_linear_class()

    class _Layer(nn.Module):
        def __init__(self, dense_projection):
            super().__init__()
            self.self_attn = nn.Module()
            self.self_attn.q_proj = (
                nn.Linear(8, 8) if dense_projection else quantized_cls(8, 8, False, 256)
            )

    def _dense_names(stack):
        return [n for n, m in stack.named_modules() if isinstance(m, nn.Linear)]

    # The check the loader makes, on a stack where every projection was swapped.
    assert _dense_names(nn.ModuleList([_Layer(False), _Layer(False)])) == []
    # And on one where a single projection came back dense.
    assert _dense_names(nn.ModuleList([_Layer(False), _Layer(True)])) == ["1.self_attn.q_proj"]
    # The stand-in is not an nn.Linear, so it cannot be mistaken for one.
    assert not isinstance(quantized_cls(8, 8, False, 256), nn.Linear)
