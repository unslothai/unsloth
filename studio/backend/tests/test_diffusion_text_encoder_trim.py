# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Qwen-Image-2.1's Qwen3-VL text encoder drops its unused lm_head without changing a hidden state."""

from __future__ import annotations

import json
import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers.models.qwen3_vl")

from core.inference import diffusion_text_encoder_trim as trim_mod  # noqa: E402
from core.inference.diffusion_text_encoder_trim import (  # noqa: E402
    KEEP_LM_HEAD_ENV,
    LM_HEAD_KEY,
    is_trimmed,
    trim_text_encoder,
)

VOCAB, HIDDEN = 96, 32


def _config(tie: bool = False):
    from transformers.models.qwen3_vl.configuration_qwen3_vl import (
        Qwen3VLConfig,
        Qwen3VLTextConfig,
        Qwen3VLVisionConfig,
    )

    config = Qwen3VLConfig(
        vision_config = Qwen3VLVisionConfig(
            depth = 1,
            hidden_size = 32,
            intermediate_size = 64,
            num_heads = 4,
            out_hidden_size = HIDDEN,
            deepstack_visual_indexes = [],
        ).to_dict(),
        text_config = Qwen3VLTextConfig(
            hidden_size = HIDDEN,
            intermediate_size = 64,
            num_hidden_layers = 2,
            num_attention_heads = 4,
            num_key_value_heads = 2,
            head_dim = 8,
            vocab_size = VOCAB,
            tie_word_embeddings = tie,
            # transformers 4.57 reads mrope_section from rope_scaling; 5.x maps it onto rope_parameters.
            rope_scaling = {
                "rope_type": "default",
                "mrope_section": [2, 1, 1],
                "mrope_interleaved": True,
            },
        ).to_dict(),
        tie_word_embeddings = tie,
    )
    config._attn_implementation = "sdpa"
    return config


def _encoder(tie: bool = False, seed: int = 0):
    from transformers import Qwen3VLForConditionalGeneration
    torch.manual_seed(seed)
    return Qwen3VLForConditionalGeneration(_config(tie)).eval()


def _pipeline_hidden_states(encoder, input_ids):
    """What QwenImage21Pipeline._get_qwen_prompt_embeds reads: every hidden state with the final norm bypassed."""
    text_model = getattr(encoder.model, "language_model", encoder.model)
    handle = text_model.norm.register_forward_hook(lambda module, args, output: args[0])
    try:
        with torch.no_grad():
            out = encoder(
                input_ids = input_ids,
                attention_mask = torch.ones_like(input_ids),
                output_hidden_states = True,
            )
    finally:
        handle.remove()
    return out


def test_trim_keeps_every_hidden_state_bit_identical_and_drops_the_head():
    encoder = _encoder()
    ids = torch.randint(0, VOCAB, (2, 11))
    before = _pipeline_hidden_states(encoder, ids)
    params_before = sum(p.numel() for p in encoder.parameters())
    record = trim_text_encoder(encoder, family = "qwen-image-2.1")
    after = _pipeline_hidden_states(encoder, ids)

    assert record == {"lm_head": "dropped", "params": VOCAB * HIDDEN}
    assert is_trimmed(encoder)
    assert sum(p.numel() for p in encoder.parameters()) == params_before - VOCAB * HIDDEN
    assert not any("lm_head" in name for name, _ in encoder.named_parameters())
    assert len(before.hidden_states) == len(after.hidden_states)
    for a, b in zip(before.hidden_states, after.hidden_states):
        assert torch.equal(a, b)
    assert after.logits.shape == (2, 11, 0)
    # The vision tower stays: the same pipeline serves image-conditioned requests.
    assert sum(p.numel() for p in encoder.model.visual.parameters()) > 0


def test_trim_is_idempotent_and_gated():
    encoder = _encoder()
    assert trim_text_encoder(encoder, family = "qwen-image")["lm_head"] == "kept"
    assert not is_trimmed(encoder)
    assert trim_text_encoder(encoder, family = "qwen-image-2.1")["lm_head"] == "dropped"
    assert trim_text_encoder(encoder, family = "qwen-image-2.1") == {
        "lm_head": "dropped",
        "params": 0,
        "already": True,
    }
    other = torch.nn.Module()
    other.lm_head = torch.nn.Linear(4, 8, bias = False)
    assert trim_text_encoder(other, family = "qwen-image-2.1")["lm_head"] == "kept"
    assert isinstance(other.lm_head, torch.nn.Linear)
    assert trim_text_encoder(None)["lm_head"] == "kept"


def test_kill_switch_keeps_the_head(monkeypatch):
    monkeypatch.setenv(KEEP_LM_HEAD_ENV, "1")
    encoder = _encoder()
    assert trim_text_encoder(encoder, family = "qwen-image-2.1")["lm_head"] == "kept"
    assert isinstance(encoder.lm_head, torch.nn.Linear)


def test_a_tied_head_is_kept():
    encoder = _encoder(tie = True)
    assert trim_text_encoder(encoder, family = "qwen-image-2.1") == {
        "lm_head": "kept",
        "reason": "tied to embed_tokens",
    }


def test_layerwise_fp8_cast_accepts_a_trimmed_encoder():
    pytest.importorskip("diffusers.hooks")
    from core.inference.diffusion_precision import _cast_fp8

    encoder = _encoder()
    trim_text_encoder(encoder, family = "qwen-image-2.1")
    _cast_fp8(encoder, types.SimpleNamespace(dtype = torch.float32))
    assert getattr(encoder, "_unsloth_te_cast_complete", False)
    assert is_trimmed(encoder)


def _write_precast(tmp_path, encoder, base):
    from safetensors.torch import save_file

    import core.inference.prequant_safetensors as ps
    from core.inference.diffusion_te_prequant import TE_PREQUANT_FORMAT

    state = {k: v.detach().clone().contiguous() for k, v in encoder.state_dict().items()}
    metadata = {
        ps.UNSLOTH_FORMAT_KEY: TE_PREQUANT_FORMAT,
        ps.UNSLOTH_METADATA_KEY: json.dumps(
            {
                "scheme": "fp8",
                "component": "text_encoder",
                "base_model_id": base,
                "te_class": "Qwen3VLForConditionalGeneration",
            }
        ),
        "tensor_names": json.dumps(sorted(state)),
    }
    metadata.update({name: json.dumps({"_type": "Tensor"}) for name in state})
    path = tmp_path / "Qwen-Image-2.1-text_encoder-FP8.safetensors"
    save_file(state, str(path), metadata = metadata)
    return path


@pytest.mark.parametrize("trim", [False, True])
def test_precast_loader_skips_the_head_before_reading_it(monkeypatch, tmp_path, trim):
    pytest.importorskip("diffusers.hooks")
    pytest.importorskip("accelerate")
    import transformers

    import core.inference.diffusion_te_prequant as tpq
    import core.inference.prequant_safetensors as ps
    from core.inference.diffusion_prequant import ALLOW_LOCAL_PREQUANT_PATH_ENV

    base = "Qwen/Qwen-Image-2.1"
    reference = _encoder(seed = 3)
    path = _write_precast(tmp_path, reference, base)
    monkeypatch.setenv(ALLOW_LOCAL_PREQUANT_PATH_ENV, str(tmp_path))
    monkeypatch.setattr(ps, "_torchao_helpers", lambda: None)
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *_a, **_k: _config())
    read: list = []
    real_reader = ps.load_plain_prequant_safetensors

    def spy(p, **kwargs):
        ckpt = real_reader(p, **kwargs)
        read.extend(ckpt["state_dict"])
        return ckpt

    monkeypatch.setattr(ps, "load_plain_prequant_safetensors", spy)
    encoder = tpq.load_prequant_text_encoder(
        base,
        "text_encoder",
        tpq.TePrequantSource(kind = "path", location = str(path)),
        dtype = torch.float32,
        local_files_only = True,
        trim_lm_head = trim,
    )
    assert encoder is not None
    assert is_trimmed(encoder) is trim
    assert (LM_HEAD_KEY in read) is (not trim)
    ids = torch.randint(0, VOCAB, (1, 9))
    expected = _pipeline_hidden_states(_reference_through_same_cast(reference), ids).hidden_states
    got = _pipeline_hidden_states(encoder, ids).hidden_states
    for a, b in zip(expected, got):
        assert torch.equal(a, b)


def _reference_through_same_cast(encoder):
    from core.inference.diffusion_precision import _cast_fp8
    _cast_fp8(encoder, types.SimpleNamespace(dtype = torch.float32))
    return encoder


def test_pipe_kwargs_asks_the_loader_to_trim_only_for_hidden_state_families(monkeypatch):
    import core.inference.diffusion_te_prequant as tpq

    seen: list = []
    monkeypatch.setattr(
        tpq,
        "te_prequant_sources_for_base",
        lambda *a, **k: {"text_encoder": tpq.TePrequantSource(kind = "repo", location = "x/y")},
    )
    monkeypatch.setattr(
        tpq,
        "load_prequant_text_encoder",
        lambda *a, **k: seen.append(k.get("trim_lm_head")) or object(),
    )
    for name in ("qwen-image-2.1", "qwen-image"):
        tpq.te_prequant_pipe_kwargs(
            types.SimpleNamespace(name = name),
            "base",
            te_quant_mode = "fp8",
            target = types.SimpleNamespace(device = "cuda", dtype = None),
            dtype = None,
        )
    assert seen == [True, False]


def test_plain_reader_skip_names_never_reads_the_tensor(tmp_path, monkeypatch):
    from safetensors.torch import save_file

    import core.inference.prequant_safetensors as ps

    monkeypatch.setattr(ps, "_torchao_helpers", lambda: None)
    names = ["a.weight", "lm_head.weight"]
    metadata = {
        ps.UNSLOTH_FORMAT_KEY: "unsloth_prequant_text_encoder_state_dict_v1",
        ps.UNSLOTH_METADATA_KEY: json.dumps({"scheme": "fp8"}),
        "tensor_names": json.dumps(names),
        **{n: json.dumps({"_type": "Tensor"}) for n in names},
    }
    path = str(tmp_path / "e.safetensors")
    save_file({"a.weight": torch.ones(2), "lm_head.weight": torch.ones(3)}, path, metadata = metadata)
    assert set(ps.load_plain_prequant_safetensors(path)["state_dict"]) == set(names)
    assert set(
        ps.load_plain_prequant_safetensors(path, skip_names = ["lm_head.weight"])["state_dict"]
    ) == {"a.weight"}
    assert trim_mod.LM_HEAD_KEY == "lm_head.weight"
