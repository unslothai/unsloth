# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hosted pre-quantized text encoders shared across image families.

Equivalence groups let byte-identical encoders share one hosted file; FLUX.2 klein 9B resolves its own (Qwen3-8B)
files rather than the family base's Qwen3-4B ones; HiDream's T5 loads FLUX.1's file under another component name; the
int8 ConvRot loader takes text-only Qwen3 and T5 encoders; a missing hosted file falls back to the dense encoder.
"""

from __future__ import annotations

import json
import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("accelerate")

import core.inference.diffusion_te_prequant as tpq  # noqa: E402
import core.inference.prequant_safetensors as ps  # noqa: E402
from core.inference.diffusion_families import _FAMILIES, detect_family  # noqa: E402
from core.inference.diffusion_prequant import ALLOW_LOCAL_PREQUANT_PATH_ENV  # noqa: E402

CUDA_BF16 = types.SimpleNamespace(device = "cuda", dtype = torch.bfloat16)
GROUP = 256


def _fam(name):
    return next(f for f in _FAMILIES if f.name == name)


@pytest.mark.parametrize(
    "name, base, component, repo, filename",
    [
        # every Qwen2.5-VL-7B release shares Qwen-Image's file
        ("qwen-image-edit", "Qwen/Qwen-Image-Edit", "text_encoder", "unsloth/Qwen-Image-FP8", None),
        ("qwen-image-edit", "Qwen/Qwen-Image-Edit-2509", "text_encoder", "unsloth/Qwen-Image-FP8", None),
        ("qwen-image-edit", "Qwen/Qwen-Image-Edit-2511", "text_encoder", "unsloth/Qwen-Image-FP8", None),
        ("qwen-image-layered", "Qwen/Qwen-Image-Layered", "text_encoder", "unsloth/Qwen-Image-FP8", None),
        # every FLUX.1 release shares schnell's T5
        ("flux.1-kontext", "black-forest-labs/FLUX.1-Kontext-dev", "text_encoder_2", "unsloth/FLUX.1-schnell-FP8", None),
        # HiDream's text_encoder_3 is the same T5, under another slot, for every distillation
        (
            "hidream-i1",
            "HiDream-ai/HiDream-I1-Dev",
            "text_encoder_3",
            "unsloth/FLUX.1-schnell-FP8",
            "FLUX.1-schnell-text_encoder_2-FP8.safetensors",
        ),
        (
            "hidream-i1",
            "HiDream-ai/HiDream-I1-Fast",
            "text_encoder_3",
            "unsloth/FLUX.1-schnell-FP8",
            "FLUX.1-schnell-text_encoder_2-FP8.safetensors",
        ),
        # klein: the base and distilled repos of each size share their encoder; the sizes never share
        (
            "flux.2-klein",
            "black-forest-labs/FLUX.2-klein-base-4B",
            "text_encoder",
            "unsloth/FLUX.2-klein-4B-FP8",
            "FLUX.2-klein-4B-text_encoder-INT8-ConvRot.safetensors",
        ),
        (
            "flux.2-klein",
            "black-forest-labs/FLUX.2-klein-9B",
            "text_encoder",
            "unsloth/FLUX.2-klein-9B-FP8",
            "FLUX.2-klein-9B-text_encoder-INT8-ConvRot.safetensors",
        ),
        (
            "flux.2-klein",
            "black-forest-labs/FLUX.2-klein-base-9B",
            "text_encoder",
            "unsloth/FLUX.2-klein-9B-FP8",
            "FLUX.2-klein-9B-text_encoder-INT8-ConvRot.safetensors",
        ),
        # the ungated mirror of a gated base is the same encoder
        (
            "flux.2-klein",
            "unsloth/FLUX.2-klein-9B",
            "text_encoder",
            "unsloth/FLUX.2-klein-9B-FP8",
            "FLUX.2-klein-9B-text_encoder-INT8-ConvRot.safetensors",
        ),
    ],
)
def test_identical_encoders_resolve_the_shared_file_for_their_base(name, base, component, repo, filename):
    fam = _fam(name)
    sources = tpq.te_prequant_sources_for_base(
        fam, base, te_quant_mode = "int8" if filename and "INT8" in filename else "fp8", target = CUDA_BF16
    )
    assert component in sources, sources
    assert sources[component].location == repo
    if filename is not None:
        assert sources[component].filename == filename


def test_klein_9b_int8_falls_back_to_its_own_fp8_file_never_the_4b_one():
    fam = _fam("flux.2-klein")
    nine = tpq.te_prequant_sources_for_base(
        fam, "black-forest-labs/FLUX.2-klein-9B", te_quant_mode = "int8", target = CUDA_BF16
    )["text_encoder"]
    assert nine.fallback_filenames[0] == "FLUX.2-klein-9B-text_encoder-FP8.safetensors"
    four = tpq.te_prequant_sources_for_base(
        fam, "black-forest-labs/FLUX.2-klein-4B", te_quant_mode = "int8", target = CUDA_BF16
    )["text_encoder"]
    assert four.location == "unsloth/FLUX.2-klein-4B-FP8"
    assert four.fallback_filenames[0] == "FLUX.2-klein-4B-text_encoder-FP8.safetensors"
    # explicit fp8 on the 9B: its own file
    fp8 = tpq.te_prequant_sources_for_base(
        fam, "black-forest-labs/FLUX.2-klein-base-9B", te_quant_mode = "fp8", target = CUDA_BF16
    )["text_encoder"]
    assert (fp8.location, fp8.filename) == (
        "unsloth/FLUX.2-klein-9B-FP8",
        "FLUX.2-klein-9B-text_encoder-FP8.safetensors",
    )


@pytest.mark.parametrize(
    "name, base",
    [
        # Z-Image's Qwen3-4B is its own variant, not klein's stock one
        ("flux.2-klein", "Tongyi-MAI/Z-Image-Turbo"),
        ("z-image", "black-forest-labs/FLUX.2-klein-4B"),
        # Qwen-Image-2.1 is Qwen3-VL-8B
        ("qwen-image-edit", "Qwen/Qwen-Image-2.1"),
        # an unknown finetune keeps its own dense encoder
        ("flux.2-klein", "someone/FLUX.2-klein-9B-finetune"),
        ("qwen-image-edit", "someone/Qwen-Image-Edit-finetune"),
    ],
)
def test_a_different_encoder_never_takes_the_hosted_file(name, base):
    fam = _fam(name)
    for scheme in ("fp8", "int8"):
        assert tpq.te_prequant_sources_for_base(fam, base, te_quant_mode = scheme, target = CUDA_BF16) == {}


def test_equivalence_groups_hold_the_byte_identical_encoders():
    eq = tpq.te_base_equivalent
    for edit in ("Qwen/Qwen-Image-Edit", "Qwen/Qwen-Image-Edit-2509", "Qwen/Qwen-Image-Edit-2511"):
        assert eq("Qwen/Qwen-Image", edit)
    assert eq("Qwen/Qwen-Image", "Qwen/Qwen-Image-Layered")
    assert eq("black-forest-labs/FLUX.1-schnell", "black-forest-labs/FLUX.1-Kontext-dev")
    assert eq("black-forest-labs/FLUX.1-schnell", "HiDream-ai/HiDream-I1-Dev")
    assert eq("black-forest-labs/FLUX.2-klein-4B", "black-forest-labs/FLUX.2-klein-base-4B")
    assert eq("black-forest-labs/FLUX.2-klein-9B", "black-forest-labs/FLUX.2-klein-base-9B")
    assert not eq("black-forest-labs/FLUX.2-klein-4B", "black-forest-labs/FLUX.2-klein-9B")
    assert not eq("black-forest-labs/FLUX.2-klein-4B", "Tongyi-MAI/Z-Image-Turbo")
    assert not eq("Qwen/Qwen-Image-Edit-2511", "Qwen/Qwen-Image-2.1")


def test_a_missing_hosted_file_falls_back_to_the_dense_encoder(monkeypatch):
    """A default nobody asked for must never fail the load: no file -> no injected encoder, and the caller assembles
    the dense one (dense weights were kept because nothing resolved on the Hub)."""
    calls = []

    def fake_load(base, component, source, **kwargs):
        calls.append((component, source.filename, kwargs["scheme"]))
        return None

    monkeypatch.setattr(tpq, "load_prequant_text_encoder", fake_load)
    monkeypatch.setattr(tpq, "_held_locally", lambda *a, **k: False)
    for name, base in (
        ("qwen-image-edit", "Qwen/Qwen-Image-Edit-2511"),
        ("flux.2-klein", "black-forest-labs/FLUX.2-klein-9B"),
        ("flux.1-kontext", "black-forest-labs/FLUX.1-Kontext-dev"),
    ):
        fam = _fam(name)
        calls.clear()
        out = tpq.te_prequant_pipe_kwargs(
            fam, base, te_quant_mode = "int8", target = CUDA_BF16, dtype = None
        )
        assert {k: v for k, v in out.items() if k.startswith("text_encoder")} == {}
        assert calls, name


def test_klein_reads_hidden_states_only_so_its_qwen3_head_is_trimmed():
    from core.inference.diffusion_text_encoder_trim import class_trims_lm_head, family_trims_lm_head

    assert family_trims_lm_head("flux.2-klein")
    assert class_trims_lm_head("Qwen3ForCausalLM")
    # SDXL's CLIPs have no head and the family is not opted in
    assert not family_trims_lm_head("sdxl")


# ── loading: HiDream's T5 from FLUX.1's text_encoder_2 file ──────────────────


def test_hosted_component_name_is_what_the_checkpoint_is_validated_against():
    src = tpq.resolve_te_prequant_source(_fam("hidream-i1"), "text_encoder_3", "fp8")
    assert src.component == "text_encoder_2"
    ckpt = {
        "format": tpq.TE_PREQUANT_FORMAT,
        "state_dict": {},
        "metadata": {
            "scheme": "fp8",
            "component": "text_encoder_2",
            "base_model_id": "black-forest-labs/FLUX.1-schnell",
        },
    }
    assert tpq._validate_checkpoint(ckpt, "fp8", src.component, "HiDream-ai/HiDream-I1-Full", None)
    # without the row's component the same file is refused for text_encoder_3
    assert not tpq._validate_checkpoint(ckpt, "fp8", "text_encoder_3", "HiDream-ai/HiDream-I1-Full", None)
    # and the T5 row never leaks into a slot it does not name
    assert tpq.resolve_te_prequant_source(_fam("hidream-i1"), "text_encoder_2", "fp8") is None
    # FLUX.1's own row is unchanged (no hosted-component override)
    assert tpq.resolve_te_prequant_source(_fam("flux.1"), "text_encoder_2", "fp8").component is None


def test_a_four_element_row_is_read_everywhere_a_row_is_unpacked():
    from core.inference.diffusion_families import (
        prequant_only_repo_ids,
        sd_cpp_companion_only_repo_ids,
        prequant_repo_role,
    )

    assert "unsloth/flux.2-klein-9b-fp8" in prequant_only_repo_ids()
    assert "unsloth/flux.2-klein-9b-fp8" not in sd_cpp_companion_only_repo_ids()
    bases, _schemes, te = prequant_repo_role(_fam("flux.2-klein"), "unsloth/FLUX.2-klein-9B-FP8")
    assert te == ("fp8",)
    assert bases == ("black-forest-labs/FLUX.2-klein-9B",)


# ── loading: a text-only Qwen3 int8 ConvRot file (klein) ──────────────────────



def _qwen3_config(tied):
    from transformers import Qwen3Config

    config = Qwen3Config(
        hidden_size = 256,
        intermediate_size = 512,
        num_hidden_layers = 2,
        num_attention_heads = 2,
        num_key_value_heads = 1,
        head_dim = 128,
        vocab_size = 64,
        tie_word_embeddings = tied,
    )
    config._attn_implementation = "sdpa"
    return config


def _int8_file(tmp_path, encoder, config):
    """scripts/build_te_int8_convrot_checkpoint.py's layout, built with its own regex."""
    import importlib.util
    from pathlib import Path

    from safetensors.torch import save_file

    script = Path(__file__).resolve().parents[3] / "scripts" / "build_te_int8_convrot_checkpoint.py"
    spec = importlib.util.spec_from_file_location("build_te_int8", script)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    blob = torch.tensor(
        list(json.dumps({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": GROUP}).encode()),
        dtype = torch.uint8,
    )
    state = {}
    for k, v in encoder.state_dict().items():
        if k == "lm_head.weight":
            continue
        if builder.DECODER_LINEAR.match(k):
            q, s = tpq.quantize_int8_convrot_weight(v, GROUP)
            state[k], state[k[:-7] + ".weight_scale"] = q, s
            state[k[:-7] + ".comfy_quant"] = blob.clone()
        else:
            state[k] = v.detach().clone()
    assert sum(k.endswith(".weight_scale") for k in state) == 7 * config.num_hidden_layers
    meta = {
        ps.UNSLOTH_FORMAT_KEY: tpq.TE_PREQUANT_FORMAT_INT8_CONVROT,
        ps.UNSLOTH_METADATA_KEY: json.dumps(
            {
                "scheme": "int8",
                "component": "text_encoder",
                "base_model_id": "black-forest-labs/FLUX.2-klein-4B",
                "te_class": "Qwen3ForCausalLM",
            }
        ),
        "tensor_names": json.dumps(sorted(state)),
        **{n: json.dumps({"_type": "Tensor"}) for n in state},
    }
    path = tmp_path / "klein-int8.safetensors"
    save_file({k: v.contiguous() for k, v in state.items()}, str(path), metadata = meta)
    return path


@pytest.mark.parametrize("tied", [True, False], ids = ["qwen3-4b-tied", "qwen3-8b-untied"])
def test_text_only_qwen3_int8_convrot_file_loads_and_tracks_the_dense_hidden_states(
    tmp_path, monkeypatch, tied
):
    import transformers
    from transformers import Qwen3ForCausalLM

    from core.inference.diffusion_text_encoder_trim import is_trimmed

    config = _qwen3_config(tied)
    torch.manual_seed(3)
    dense = Qwen3ForCausalLM(config).eval()
    path = _int8_file(tmp_path, dense, config)
    monkeypatch.setattr(ps, "_torchao_helpers", lambda: None)
    monkeypatch.setenv(ALLOW_LOCAL_PREQUANT_PATH_ENV, str(tmp_path))
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *_a, **_k: _qwen3_config(tied))
    failures: list = []
    encoder = tpq.load_prequant_text_encoder(
        "black-forest-labs/FLUX.2-klein-4B",
        "text_encoder",
        tpq.TePrequantSource(kind = "path", location = str(path)),
        dtype = torch.float32,
        scheme = "int8",
        local_files_only = True,
        trim_lm_head = True,
        failures = failures,
    )
    assert encoder is not None, failures
    assert getattr(encoder, tpq.TE_PREQUANT_SCHEME_ATTR) == "int8"
    assert is_trimmed(encoder)
    assert not any(isinstance(m, torch.nn.Linear) for m in encoder.model.layers.modules())
    ids = torch.tensor([[1, 5, 9, 13, 17, 21]])
    with torch.no_grad():
        ref = dense(input_ids = ids, output_hidden_states = True).hidden_states
        got = encoder(input_ids = ids, output_hidden_states = True).hidden_states
    for a, b in zip(ref[1:], got[1:]):
        cos = torch.nn.functional.cosine_similarity(a.flatten(1), b.flatten(1)).min()
        assert cos > 0.99


def test_klein_int8_file_has_the_fp8_file_behind_it():
    fam = detect_family("black-forest-labs/FLUX.2-klein-4B")
    source = tpq.resolve_te_prequant_source(fam, "text_encoder", "int8")
    assert source.filename == "FLUX.2-klein-4B-text_encoder-INT8-ConvRot.safetensors"
    assert source.fallback_filenames[0] == "FLUX.2-klein-4B-text_encoder-FP8.safetensors"


# ── loading: a T5 int8 ConvRot file (FLUX.1, HiDream text_encoder_3) ──────────


def test_t5_int8_convrot_file_loads_with_its_tied_embedding_and_tracks_dense(tmp_path, monkeypatch):
    """T5EncoderModel has no head and ties encoder.embed_tokens to shared; every projection, wo included, is int8."""
    import transformers
    from safetensors.torch import save_file
    from transformers import T5Config, T5EncoderModel

    def config():
        c = T5Config(d_model = 256, d_ff = 512, d_kv = 64, num_heads = 4, num_layers = 2, vocab_size = 64,
                     feed_forward_proj = "gated-gelu")
        c._attn_implementation = "eager"
        return c

    torch.manual_seed(1)
    dense = T5EncoderModel(config()).eval()
    blob = torch.tensor(
        list(json.dumps({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": GROUP}).encode()),
        dtype = torch.uint8,
    )
    linears = {n for n, m in dense.named_modules() if isinstance(m, torch.nn.Linear) and tpq.TE_INT8_CONVROT_LINEAR.search(n)}
    assert len(linears) == 2 * 7  # q, k, v, o, wi_0, wi_1, wo per block
    state = {}
    for k, v in dense.state_dict().items():
        prefix = k[:-7] if k.endswith(".weight") else None
        if prefix in linears:
            q, s = tpq.quantize_int8_convrot_weight(v, GROUP)
            state[k], state[prefix + ".weight_scale"], state[prefix + ".comfy_quant"] = q, s, blob.clone()
        else:
            state[k] = v.detach().clone()
    meta = {
        ps.UNSLOTH_FORMAT_KEY: tpq.TE_PREQUANT_FORMAT_INT8_CONVROT,
        ps.UNSLOTH_METADATA_KEY: json.dumps(
            {"scheme": "int8", "component": "text_encoder_2", "base_model_id": "black-forest-labs/FLUX.1-schnell",
             "te_class": "T5EncoderModel"}
        ),
        "tensor_names": json.dumps(sorted(state)),
        **{n: json.dumps({"_type": "Tensor"}) for n in state},
    }
    path = tmp_path / "t5-int8.safetensors"
    save_file({k: v.contiguous() for k, v in state.items()}, str(path), metadata = meta)
    monkeypatch.setattr(ps, "_torchao_helpers", lambda: None)
    monkeypatch.setenv(ALLOW_LOCAL_PREQUANT_PATH_ENV, str(tmp_path))
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *_a, **_k: config())
    failures: list = []
    # Loaded into HiDream's text_encoder_3 slot from a file built as FLUX.1's text_encoder_2.
    encoder = tpq.load_prequant_text_encoder(
        "HiDream-ai/HiDream-I1-Full",
        "text_encoder_3",
        tpq.TePrequantSource(kind = "path", location = str(path), component = "text_encoder_2"),
        dtype = torch.float32,
        scheme = "int8",
        local_files_only = True,
        failures = failures,
    )
    assert encoder is not None, failures
    assert encoder.encoder.embed_tokens.weight is encoder.shared.weight
    ids = torch.tensor([[3, 9, 27, 5, 1]])
    with torch.no_grad():
        ref = dense(input_ids = ids).last_hidden_state
        got = encoder(input_ids = ids).last_hidden_state
    assert torch.nn.functional.cosine_similarity(ref.flatten(1), got.flatten(1)).min() > 0.99


def test_int8_leaves_the_small_companion_encoders_bf16(monkeypatch):
    """FLUX.1's CLIP-L next to its hosted int8 T5: never cast in place (a runtime-only conversion), reported applied."""
    import core.inference.diffusion_precision as prec

    monkeypatch.setattr(prec, "te_quant_supported", lambda *a, **k: True)
    monkeypatch.setattr(prec, "_cast_fp8", lambda *a, **k: pytest.fail("CLIP cast in place"))
    t5 = torch.nn.Linear(4, 4)
    setattr(t5, tpq.TE_PREQUANT_SCHEME_ATTR, "int8")
    pipe = types.SimpleNamespace(text_encoder = torch.nn.Linear(4, 4), text_encoder_2 = t5)
    out = prec.quantize_text_encoders(pipe, CUDA_BF16, mode = "int8", family = "flux.1")
    assert (out.mode, out.status) == ("int8", prec.RESOLVED_APPLIED)
    assert "text_encoder kept bf16" in out.reason
