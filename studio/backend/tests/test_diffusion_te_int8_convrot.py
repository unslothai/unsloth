# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Qwen-Image-2.1 defaults to the hosted int8 ConvRot weight-only text encoder and falls back to the fp8 one."""

from __future__ import annotations

import json
import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers.models.qwen3_vl")
pytest.importorskip("accelerate")

import core.inference.diffusion_te_prequant as tpq  # noqa: E402
import core.inference.prequant_safetensors as ps  # noqa: E402
from core.inference.diffusion_families import detect_family  # noqa: E402
from core.inference.diffusion_precision import (  # noqa: E402
    RESOLVED_APPLIED,
    RESOLVED_FELL_BACK,
    effective_te_quant,
    quantize_text_encoders,
    resolve_te_quant_request,
    te_quant_needs_resident_weights,
)
from core.inference.diffusion_prequant import ALLOW_LOCAL_PREQUANT_PATH_ENV  # noqa: E402
from core.inference.diffusion_text_encoder_trim import is_trimmed  # noqa: E402

GROUP = 256
HIDDEN, VOCAB = 256, 64
BASE = "Qwen/Qwen-Image-2.1"
REPO, INT8_NAME = (
    "unsloth/Qwen-Image-2.1-FP8",
    "Qwen-Image-2.1-text_encoder-INT8-ConvRot.safetensors",
)
FP8_NAME = "Qwen-Image-2.1-text_encoder-FP8.safetensors"
CUDA_BF16 = types.SimpleNamespace(device = "cuda", dtype = torch.bfloat16)


def get_family(name):
    return detect_family("", override = name)


def _config():
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
            intermediate_size = 512,
            num_hidden_layers = 2,
            num_attention_heads = 4,
            num_key_value_heads = 2,
            head_dim = 64,
            vocab_size = VOCAB,
            tie_word_embeddings = False,
            rope_scaling = {
                "rope_type": "default",
                "mrope_section": [16, 8, 8],
                "mrope_interleaved": True,
            },
        ).to_dict(),
        tie_word_embeddings = False,
    )
    config._attn_implementation = "sdpa"
    return config


def _encoder(seed = 0):
    from transformers import Qwen3VLForConditionalGeneration
    torch.manual_seed(seed)
    return Qwen3VLForConditionalGeneration(_config()).eval()


def _hidden(encoder, ids):
    lm = encoder.model.language_model
    handle = lm.norm.register_forward_hook(lambda m, a, o: a[0])
    try:
        with torch.no_grad():
            out = encoder(
                input_ids = ids, attention_mask = torch.ones_like(ids), output_hidden_states = True
            )
    finally:
        handle.remove()
    return out.hidden_states


def _container(tmp_path, name, state, fmt, scheme):
    from safetensors.torch import save_file

    meta = {
        ps.UNSLOTH_FORMAT_KEY: fmt,
        ps.UNSLOTH_METADATA_KEY: json.dumps(
            {
                "scheme": scheme,
                "component": "text_encoder",
                "base_model_id": BASE,
                "te_class": "Qwen3VLForConditionalGeneration",
            }
        ),
        "tensor_names": json.dumps(sorted(state)),
        **{n: json.dumps({"_type": "Tensor"}) for n in state},
    }
    path = tmp_path / name
    save_file({k: v.contiguous() for k, v in state.items()}, str(path), metadata = meta)
    return path


def _int8_state(encoder):
    """The builder's layout: decoder projections int8 + scale + comfy_quant, lm_head dropped, the rest dense."""
    import re

    lin = re.compile(
        r"^model\.language_model\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)\.weight$"
    )
    blob = torch.tensor(
        list(
            json.dumps(
                {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": GROUP}
            ).encode()
        ),
        dtype = torch.uint8,
    )
    state = {}
    for k, v in encoder.state_dict().items():
        if k == "lm_head.weight":
            continue
        if lin.match(k):
            q, s = tpq.quantize_int8_convrot_weight(v, GROUP)
            state[k] = q
            state[k[:-7] + ".weight_scale"] = s
            state[k[:-7] + ".comfy_quant"] = blob.clone()
        else:
            state[k] = v.detach().clone()
    return state


def _dequantized_reference(encoder, state):
    """Dense encoder whose projections are the int8 weights mapped back: W = (q * s) @ blockdiag(H)."""
    from core.inference.diffusion_convrot import build_convrot_hadamard

    h = build_convrot_hadamard(GROUP, dtype = torch.float32)
    sd = encoder.state_dict()
    for k in list(sd):
        if k + "_scale" in state:
            q, s = state[k].float(), state[k + "_scale"]
            o, i = q.shape
            sd[k] = ((q * s).reshape(o, i // GROUP, GROUP) @ h).reshape(o, i)
    encoder.load_state_dict(sd)
    return encoder


@pytest.fixture
def int8_file(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, "_torchao_helpers", lambda: None)
    monkeypatch.setenv(ALLOW_LOCAL_PREQUANT_PATH_ENV, str(tmp_path))
    import transformers

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *_a, **_k: _config())
    encoder = _encoder(seed = 5)
    state = _int8_state(encoder)
    path = _container(tmp_path, INT8_NAME, state, tpq.TE_PREQUANT_FORMAT_INT8_CONVROT, "int8")
    return encoder, state, path


def _load(
    path,
    scheme = "int8",
    trim = True,
):
    return tpq.load_prequant_text_encoder(
        BASE,
        "text_encoder",
        tpq.TePrequantSource(kind = "path", location = str(path)),
        dtype = torch.float32,
        scheme = scheme,
        local_files_only = True,
        trim_lm_head = trim,
    )


def test_family_default_is_hosted_int8_and_needs_no_torchao():
    fam = get_family("qwen-image-2.1")
    mode, auto = resolve_te_quant_request(None, fam.te_quant_auto)
    assert (mode, auto) == ("int8", True)
    # No keep-bf16 schedule, so pre-load gates see the fp8 storage cast: no torchao, offload allowed.
    assert effective_te_quant(mode, fam.name) == "fp8"
    assert not te_quant_needs_resident_weights(effective_te_quant(mode, fam.name))


def test_int8_source_prefers_the_convrot_file_then_the_fp8_one():
    fam = get_family("qwen-image-2.1")
    sources = tpq.te_prequant_sources_for_base(fam, BASE, te_quant_mode = "int8", target = CUDA_BF16)
    src = sources["text_encoder"]
    assert (src.kind, src.location) == ("repo", REPO)
    assert tpq.te_candidate_filenames(src) == (
        INT8_NAME,
        FP8_NAME,
        "Qwen-Image-2.1-text_encoder-FP8.pt",
    )
    fp8 = tpq.te_prequant_sources(fam, te_quant_mode = "fp8", target = CUDA_BF16)["text_encoder"]
    assert INT8_NAME not in tpq.te_candidate_filenames(fp8)
    assert (
        tpq.te_prequant_sources(get_family("qwen-image"), te_quant_mode = "int8", target = CUDA_BF16)
        == {}
    )


def test_keeping_the_lm_head_resolves_the_fp8_file(monkeypatch):
    from core.inference.diffusion_text_encoder_trim import KEEP_LM_HEAD_ENV

    monkeypatch.setenv(KEEP_LM_HEAD_ENV, "1")
    src = tpq.resolve_te_prequant_source(get_family("qwen-image-2.1"), "text_encoder", "int8")
    assert src.filename == FP8_NAME


def test_int8_file_loads_as_convrot_linears_and_matches_its_dequantized_weights(int8_file):
    from core.inference.video_minimax_h3_te import _int8_convrot_linear_class

    reference, state, path = int8_file
    encoder = _load(path)
    assert encoder is not None
    assert getattr(encoder, tpq.TE_PREQUANT_SCHEME_ATTR) == "int8"
    assert is_trimmed(encoder)
    cls = _int8_convrot_linear_class()
    layers = encoder.model.language_model.layers
    assert sum(isinstance(m, cls) for m in layers.modules()) == 2 * 7
    assert not any(isinstance(m, torch.nn.Linear) for m in layers.modules())
    assert all(m.weight.dtype == torch.int8 for m in layers.modules() if isinstance(m, cls))
    assert any(isinstance(m, torch.nn.Linear) for m in encoder.model.visual.modules())
    # Plain tensors only, so Module.to() and group offloading move it like any module.
    assert all(
        type(t) in (torch.Tensor, torch.nn.Parameter)
        for t in [*encoder.parameters(), *encoder.buffers()]
    )
    ids = torch.randint(0, VOCAB, (1, 9))
    expected = _hidden(_dequantized_reference(_encoder(seed = 5), state), ids)
    got = _hidden(encoder, ids)
    for a, b in zip(expected, got):
        torch.testing.assert_close(b, a, rtol = 1e-4, atol = 1e-4)
    dense = _hidden(reference, ids)[-1]
    cos = torch.nn.functional.cosine_similarity(dense.flatten(), got[-1].flatten(), dim = 0)
    assert cos > 0.99


def test_int8_file_is_refused_for_an_fp8_request_and_without_the_trim(int8_file):
    _, _, path = int8_file
    assert _load(path, scheme = "fp8") is None
    assert _load(path, trim = False) is None


def test_int8_request_falls_back_to_the_fp8_file_when_the_hub_404s(tmp_path, monkeypatch):
    from huggingface_hub.errors import EntryNotFoundError

    monkeypatch.setattr(ps, "_torchao_helpers", lambda: None)
    monkeypatch.delenv(tpq.TE_PREQUANT_MIRROR_ENV, raising = False)
    fp8 = tmp_path / FP8_NAME
    fp8.write_bytes(b"x")
    asked: list = []

    def fake_download(repo_id, filename, **_k):
        asked.append(filename)
        if filename == INT8_NAME:
            raise EntryNotFoundError(f"404 {filename}")
        return str(fp8)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    src = tpq.resolve_te_prequant_source(get_family("qwen-image-2.1"), "text_encoder", "int8")
    assert tpq._resolve_checkpoint_path(src, None, cache_dir = str(tmp_path)) == str(fp8)
    assert asked == [INT8_NAME, FP8_NAME]


def test_mirror_serves_the_int8_file_before_the_hub(tmp_path, monkeypatch):
    mirror = tmp_path / "mirror"
    target = mirror / REPO / INT8_NAME
    target.parent.mkdir(parents = True)
    target.write_bytes(b"x" * 7)
    monkeypatch.setenv(tpq.TE_PREQUANT_MIRROR_ENV, str(mirror))
    monkeypatch.setattr(ps, "_torchao_helpers", lambda: None)
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download", lambda *a, **k: pytest.fail("the Hub was asked")
    )
    src = tpq.resolve_te_prequant_source(get_family("qwen-image-2.1"), "text_encoder", "int8")
    assert tpq._resolve_checkpoint_path(src, None, cache_dir = str(tmp_path)) == str(target.resolve())

    class NoApi:
        def model_info(self, *a, **k):
            raise AssertionError("planning asked the Hub")

    assert tpq.te_prequant_hub_files({"text_encoder": src}, NoApi()) == {
        "text_encoder": [(INT8_NAME, 7)]
    }
    assert tpq.te_prequant_mirror_path(REPO, "../x") is None


def test_hub_plan_counts_the_file_the_repo_holds(monkeypatch):
    monkeypatch.delenv(tpq.TE_PREQUANT_MIRROR_ENV, raising = False)
    src = tpq.resolve_te_prequant_source(get_family("qwen-image-2.1"), "text_encoder", "int8")

    def api(names):
        sib = [types.SimpleNamespace(rfilename = n, size = s) for n, s in names]
        return types.SimpleNamespace(model_info = lambda *a, **k: types.SimpleNamespace(siblings = sib))

    assert tpq.te_prequant_hub_files({"te": src}, api([(FP8_NAME, 9)])) == {"te": [(FP8_NAME, 9)]}
    assert tpq.te_prequant_hub_files({"te": src}, api([(FP8_NAME, 9), (INT8_NAME, 8)])) == {
        "te": [(INT8_NAME, 8)]
    }


def _pipe(encoder):
    return types.SimpleNamespace(text_encoder = encoder)


def test_hosted_int8_encoder_is_reported_and_never_recast(int8_file, monkeypatch):
    import core.inference.diffusion_precision as prec

    _, _, path = int8_file
    encoder = _load(path)
    monkeypatch.setattr(
        prec, "_cast_fp8", lambda *a, **k: pytest.fail("int8 encoder re-cast to fp8")
    )
    for offload in (False, True):
        out = quantize_text_encoders(
            _pipe(encoder), CUDA_BF16, mode = "int8", family = "qwen-image-2.1", offload_active = offload
        )
        assert (out.mode, out.status) == ("int8", RESOLVED_APPLIED)


def test_fp8_fallback_encoder_reports_why_int8_did_not_engage(monkeypatch):
    import core.inference.diffusion_precision as prec

    encoder = torch.nn.Linear(2, 2)
    setattr(encoder, tpq.TE_PREQUANT_SCHEME_ATTR, "fp8")
    cast: list = []
    monkeypatch.setattr(prec, "_cast_fp8", lambda enc, tgt: cast.append(enc))
    monkeypatch.setattr(prec, "te_quant_supported", lambda *a: True)
    out = quantize_text_encoders(_pipe(encoder), CUDA_BF16, mode = "int8", family = "qwen-image-2.1")
    assert (out.mode, out.status) == ("fp8", RESOLVED_FELL_BACK)
    assert "int8 ConvRot" in out.reason
    assert cast == [encoder]


def test_pipe_kwargs_loads_with_the_requested_scheme(monkeypatch):
    seen: list = []
    monkeypatch.setattr(
        tpq,
        "te_prequant_sources_for_base",
        lambda *a, **k: {"text_encoder": tpq.TePrequantSource(kind = "repo", location = REPO)},
    )
    monkeypatch.setattr(
        tpq, "load_prequant_text_encoder", lambda *a, **k: seen.append(k["scheme"]) or object()
    )
    for mode in ("int8", "fp8"):
        tpq.te_prequant_pipe_kwargs(
            get_family("qwen-image-2.1"), BASE, te_quant_mode = mode, target = CUDA_BF16, dtype = None
        )
    assert seen == ["int8", "fp8"]


def test_an_unreachable_hub_still_takes_the_cached_fp8_file(tmp_path, monkeypatch):
    """Online, int8 never cached, Hub down: the cached fp8 file loads instead of a dense pull."""
    from huggingface_hub.errors import LocalEntryNotFoundError

    monkeypatch.setattr(ps, "_torchao_helpers", lambda: None)
    monkeypatch.delenv(tpq.TE_PREQUANT_MIRROR_ENV, raising = False)
    fp8 = tmp_path / FP8_NAME
    fp8.write_bytes(b"x")
    asked: list = []

    def fake_download(
        repo_id,
        filename,
        local_files_only = False,
        **_k,
    ):
        asked.append((filename, local_files_only))
        if filename == FP8_NAME and local_files_only:
            return str(fp8)
        raise LocalEntryNotFoundError("connection error")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    src = tpq.resolve_te_prequant_source(get_family("qwen-image-2.1"), "text_encoder", "int8")
    assert tpq._resolve_checkpoint_path(src, None, cache_dir = str(tmp_path)) == str(fp8)
    assert asked == [(INT8_NAME, False), (FP8_NAME, True)]
    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda *a, **k: (_ for _ in ()).throw(LocalEntryNotFoundError("connection error")),
    )
    with pytest.raises(LocalEntryNotFoundError):
        tpq._resolve_checkpoint_path(src, None, cache_dir = str(tmp_path))


def test_an_unreachable_hub_takes_the_fp8_file_from_the_other_cache_root(tmp_path, monkeypatch):
    """Outage with the fp8 file cached only under the default root (a moved cache setting)."""
    from huggingface_hub.errors import LocalEntryNotFoundError

    monkeypatch.setattr(ps, "_torchao_helpers", lambda: None)
    monkeypatch.delenv(tpq.TE_PREQUANT_MIRROR_ENV, raising = False)
    fp8 = tmp_path / "other" / FP8_NAME
    fp8.parent.mkdir()
    fp8.write_bytes(b"x")
    import core.inference.diffusion_prequant as dpq

    monkeypatch.setattr(
        dpq,
        "_cached_in_root",
        lambda _src, root, name = None: str(fp8) if root is None and name == FP8_NAME else None,
    )

    def fake_download(
        repo_id,
        filename,
        cache_dir = None,
        local_files_only = False,
        **_k,
    ):
        if filename == FP8_NAME and local_files_only and cache_dir is None:
            return str(fp8)
        raise LocalEntryNotFoundError("connection error")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    src = tpq.resolve_te_prequant_source(get_family("qwen-image-2.1"), "text_encoder", "int8")
    assert tpq._resolve_checkpoint_path(src, None, cache_dir = str(tmp_path / "live")) == str(fp8)


def test_an_int8_file_that_will_not_build_falls_back_to_the_fp8_names(monkeypatch):
    calls: list = []
    sentinel = object()
    src = tpq.resolve_te_prequant_source(get_family("qwen-image-2.1"), "text_encoder", "int8")
    monkeypatch.setattr(tpq, "te_prequant_sources_for_base", lambda *a, **k: {"text_encoder": src})

    def fake_load(base, component, source, **k):
        calls.append((source.filename, k["scheme"]))
        return None if source.filename == INT8_NAME else sentinel

    monkeypatch.setattr(tpq, "load_prequant_text_encoder", fake_load)
    for held, expected in ((True, sentinel), (False, None)):
        calls.clear()
        monkeypatch.setattr(tpq, "_held_locally", lambda *a, held = held: held)
        out = tpq.te_prequant_pipe_kwargs(
            get_family("qwen-image-2.1"), BASE, te_quant_mode = "int8", target = CUDA_BF16, dtype = None
        )
        assert out.get("text_encoder") is expected
        # Only a present-but-unbuildable int8 file earns an fp8 retry; a 404 already fell through.
        assert calls == (
            [(INT8_NAME, "int8"), (FP8_NAME, "fp8")] if held else [(INT8_NAME, "int8")]
        )


def test_keep_lm_head_reason_names_the_switch(monkeypatch):
    import core.inference.diffusion_precision as prec
    from core.inference.diffusion_text_encoder_trim import KEEP_LM_HEAD_ENV

    monkeypatch.setenv(KEEP_LM_HEAD_ENV, "1")
    monkeypatch.setattr(prec, "_cast_fp8", lambda enc, tgt: None)
    monkeypatch.setattr(prec, "te_quant_supported", lambda *a: True)
    out = quantize_text_encoders(
        _pipe(torch.nn.Linear(2, 2)), CUDA_BF16, mode = "int8", family = "qwen-image-2.1"
    )
    assert out.mode == "fp8" and KEEP_LM_HEAD_ENV in out.reason
