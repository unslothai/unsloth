# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FLUX.3 Klein's hosted pre-quantized Qwen3-VL-4B text encoders: fp8 layerwise and int8 ConvRot, both variants."""

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
from core.inference.diffusion_text_encoder_trim import family_trims_lm_head, is_trimmed  # noqa: E402

GROUP = 256
HIDDEN, VOCAB = 256, 64
CUDA_BF16 = types.SimpleNamespace(device = "cuda", dtype = torch.bfloat16)
MPS_BF16 = types.SimpleNamespace(device = "mps", dtype = torch.bfloat16)
DISTILLED, BASE = (
    "black-forest-labs/flux-3-image-klein",
    "black-forest-labs/flux-3-image-klein-base",
)
VARIANTS = {
    "flux.3-klein": (DISTILLED, "unsloth/flux-3-image-klein-FP8", "flux-3-image-klein"),
    "flux.3-klein-base": (BASE, "unsloth/flux-3-image-klein-base-FP8", "flux-3-image-klein-base"),
}


def _names(stem):
    return (
        f"{stem}-text_encoder-INT8-ConvRot.safetensors",
        f"{stem}-text_encoder-FP8.safetensors",
        f"{stem}-text_encoder-FP8.pt",
    )


def _config():
    """A tiny Qwen3-VL with the real 4B's TIED embedding (tie_word_embeddings: true in text_encoder/config.json)."""
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
            tie_word_embeddings = True,
            rope_scaling = {
                "rope_type": "default",
                "mrope_section": [16, 8, 8],
                "mrope_interleaved": True,
            },
        ).to_dict(),
        tie_word_embeddings = True,
    )
    config._attn_implementation = "sdpa"
    return config


def _encoder(seed = 0):
    from transformers import Qwen3VLForConditionalGeneration
    torch.manual_seed(seed)
    return Qwen3VLForConditionalGeneration(_config()).eval()


def _hidden(encoder, ids):
    with torch.no_grad():
        out = encoder(input_ids = ids, attention_mask = torch.ones_like(ids), output_hidden_states = True)
    return out.hidden_states


def _write(
    path,
    state,
    fmt,
    scheme,
    base = DISTILLED,
):
    from safetensors.torch import save_file

    meta = {
        ps.UNSLOTH_FORMAT_KEY: fmt,
        ps.UNSLOTH_METADATA_KEY: json.dumps(
            {
                "scheme": scheme,
                "component": "text_encoder",
                "base_model_id": base,
                "te_class": "Qwen3VLForConditionalGeneration",
            }
        ),
        "tensor_names": json.dumps(sorted(state)),
        **{n: json.dumps({"_type": "Tensor"}) for n in state},
    }
    path.parent.mkdir(parents = True, exist_ok = True)
    save_file(
        {k: v.detach().clone().contiguous() for k, v in state.items()}, str(path), metadata = meta
    )
    return path


def _int8_state(encoder):
    """scripts/build_te_int8_convrot_checkpoint.py's layout: projections int8 + scale + comfy_quant, no lm_head."""
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
            state[k], state[k[:-7] + ".weight_scale"], state[k[:-7] + ".comfy_quant"] = (
                q,
                s,
                blob.clone(),
            )
        else:
            state[k] = v
    return state


def _fp8_state(encoder):
    # _cast_fp8 is diffusers' layerwise casting; the fp8 file it writes is what the loader is checked against.
    pytest.importorskip("diffusers.hooks")
    from core.inference.diffusion_precision import _cast_fp8

    _cast_fp8(encoder, types.SimpleNamespace(dtype = torch.float32))
    return dict(encoder.state_dict())


@pytest.fixture
def mirror(tmp_path, monkeypatch):
    """Both variants' files in a local mirror (one encoder, so the same bytes under each repo's names)."""
    monkeypatch.setattr(ps, "_torchao_helpers", lambda: None)
    import transformers

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *_a, **_k: _config())
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download", lambda *a, **k: pytest.fail("the Hub was asked")
    )
    root = tmp_path / "mirror"
    int8 = _int8_state(_encoder(seed = 3))
    fp8 = _fp8_state(_encoder(seed = 3))
    for _base, repo, stem in VARIANTS.values():
        int8_name, fp8_name, _pt = _names(stem)
        _write(root / repo / int8_name, int8, tpq.TE_PREQUANT_FORMAT_INT8_CONVROT, "int8")
        _write(root / repo / fp8_name, fp8, tpq.TE_PREQUANT_FORMAT, "fp8")
    monkeypatch.setenv(tpq.TE_PREQUANT_MIRROR_ENV, str(root))
    return root


class _Log:
    def __init__(self):
        self.lines = []

    def __getattr__(self, _name):
        return lambda msg, *a, **_k: self.lines.append(str(msg) % a if a else str(msg))


def _pipe_kwargs(
    family,
    mode,
    target = CUDA_BF16,
    logger = None,
):
    fam = detect_family("", override = family)
    return tpq.te_prequant_pipe_kwargs(
        fam,
        fam.base_repo,
        te_quant_mode = mode,
        target = target,
        dtype = torch.float32,
        local_files_only = True,
        logger = logger,
    )


@pytest.mark.parametrize("family", sorted(VARIANTS))
def test_both_variants_host_fp8_and_int8_convrot_encoders(family):
    _base, repo, stem = VARIANTS[family]
    fam = detect_family("", override = family)
    assert fam.te_prequant_repos == (("fp8", "text_encoder", repo),)
    assert tpq.family_te_int8_convrot(fam, "text_encoder") == (repo, _names(stem)[0])
    # The int8 file has no lm_head, so the family must read hidden states only or the resolver drops to fp8.
    assert family_trims_lm_head(family)


@pytest.mark.parametrize("family", sorted(VARIANTS))
def test_filenames_follow_the_qwen_image_2_1_pattern_per_variant(family):
    base, repo, stem = VARIANTS[family]
    fam = detect_family("", override = family)
    int8 = tpq.te_prequant_sources_for_base(fam, base, te_quant_mode = "int8", target = CUDA_BF16)
    assert (int8["text_encoder"].kind, int8["text_encoder"].location) == ("repo", repo)
    assert tpq.te_candidate_filenames(int8["text_encoder"]) == _names(stem)
    fp8 = tpq.te_prequant_sources_for_base(fam, base, te_quant_mode = "fp8", target = CUDA_BF16)
    assert tpq.te_candidate_filenames(fp8["text_encoder"]) == _names(stem)[1:]
    # The ungated unsloth mirror of the base resolves the same artifact.
    mirror_id = base.replace("black-forest-labs/", "unsloth/")
    assert (
        tpq.te_prequant_sources_for_base(fam, mirror_id, te_quant_mode = "int8", target = CUDA_BF16)
        == int8
    )
    # MPS takes the int8 file alone and budgets it at its own measured share.
    mps = tpq.te_prequant_sources_for_base(fam, base, te_quant_mode = "int8", target = MPS_BF16)
    assert tpq.te_candidate_filenames(mps["text_encoder"]) == _names(stem)[:1]
    assert tpq.te_int8_convrot_budget_scale(fam) >= 5_247_076_320 / 8_875_719_344


def test_one_encoder_serves_both_variants():
    # The distilled and base text_encoder/ shards are byte-identical, so either base id validates the other's file.
    assert tpq.te_base_equivalent(DISTILLED, BASE)
    assert tpq.te_base_equivalent("unsloth/flux-3-image-klein-FP8", BASE)
    assert not tpq.te_base_equivalent(DISTILLED, "Qwen/Qwen-Image-2.1")


@pytest.mark.parametrize("family", sorted(VARIANTS))
def test_int8_convrot_encoder_loads_with_the_int8_format(mirror, family):
    from core.inference.video_minimax_h3_te import _int8_convrot_linear_class

    log = _Log()
    encoder = _pipe_kwargs(family, "int8", logger = log)["text_encoder"]
    assert getattr(encoder, tpq.TE_PREQUANT_SCHEME_ATTR) == "int8"
    assert any("int8 ConvRot" in line for line in log.lines)
    # The tied head has no key in the file: dropped, never loaded random or left on meta.
    assert is_trimmed(encoder)
    cls = _int8_convrot_linear_class()
    layers = encoder.model.language_model.layers
    assert sum(isinstance(m, cls) for m in layers.modules()) == 2 * 7
    assert not any(isinstance(m, torch.nn.Linear) for m in layers.modules())
    # Every hidden state FLUX.3 stacks stays close to the dense encoder's.
    ids = torch.randint(0, VOCAB, (1, 9))
    dense, got = _hidden(_encoder(seed = 3), ids), _hidden(encoder, ids)
    assert len(dense) == len(got)
    for a, b in zip(dense[1:], got[1:]):
        assert torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim = 0) > 0.99


@pytest.mark.parametrize("family", sorted(VARIANTS))
def test_fp8_request_loads_the_fp8_file(mirror, family):
    encoder = _pipe_kwargs(family, "fp8")["text_encoder"]
    assert getattr(encoder, tpq.TE_PREQUANT_SCHEME_ATTR) == "fp8"
    assert encoder.lm_head.weight is encoder.model.language_model.embed_tokens.weight


def test_fp8_request_refuses_an_int8_file_and_int8_never_reports_an_fp8_file(mirror, monkeypatch):
    _base, repo, stem = VARIANTS["flux.3-klein"]
    int8_name, fp8_name, _pt = _names(stem)
    # Swap the two files: the int8 bytes under the fp8 name, the fp8 bytes under the int8 name.
    a, b = mirror / repo / int8_name, mirror / repo / fp8_name
    tmp = a.with_suffix(".swap")
    a.rename(tmp)
    b.rename(a)
    tmp.rename(b)
    log = _Log()
    assert _pipe_kwargs("flux.3-klein", "fp8", logger = log) == {}
    assert any("checkpoint scheme 'int8' != 'fp8'" in line for line in log.lines)
    # An int8 request reads the int8-named file by its format tag: fp8 bytes load as fp8 and are labelled fp8.
    encoder = _pipe_kwargs("flux.3-klein", "int8")["text_encoder"]
    assert getattr(encoder, tpq.TE_PREQUANT_SCHEME_ATTR) == "fp8"


@pytest.mark.parametrize("family", sorted(VARIANTS))
def test_an_invisible_repo_falls_back_to_the_dense_encoder(tmp_path, monkeypatch, family):
    from huggingface_hub.errors import RepositoryNotFoundError

    monkeypatch.delenv(tpq.TE_PREQUANT_MIRROR_ENV, raising = False)
    _base, repo, stem = VARIANTS[family]
    asked: list = []

    def invisible(repo_id, filename, **_k):
        asked.append((repo_id, filename))
        raise RepositoryNotFoundError(f"404 {repo_id}")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", invisible)
    # The load asked the hosted repo for the int8 file, then handed back nothing, so assembly loads the dense bf16
    # encoder (an unknown repo ends the name chain: no fp8 name can be there either).
    assert _pipe_kwargs(family, "int8") == {}
    assert asked and asked[0] == (repo, _names(stem)[0])
    assert {r for r, _n in asked} == {repo}

    class Invisible:
        def model_info(self, *a, **k):
            raise RepositoryNotFoundError("404")

    fam = detect_family("", override = family)
    sources = tpq.te_prequant_sources_for_base(
        fam, fam.base_repo, te_quant_mode = "int8", target = CUDA_BF16
    )
    assert sources
    # Nothing resolves on the Hub, so the plan keeps the dense text_encoder/ shards.
    assert tpq.te_prequant_hub_files(sources, Invisible()) == {}


def test_both_flux3_families_default_to_the_hosted_int8_convrot_encoder():
    # BFL's ComfyUI templates load qwen3vl_4b_int8_convrot, so an unset request gets the hosted int8 file; "none"
    # still reaches the released bf16 encoder.
    from core.inference.diffusion_families import _FAMILIES as FAMILIES
    from core.inference.diffusion_precision import resolve_te_quant_request
    for name in ("flux.3-klein", "flux.3-klein-base"):
        fam = next(f for f in FAMILIES if f.name == name)
        assert fam.te_quant_auto == "int8", name
        assert resolve_te_quant_request(None, fam.te_quant_auto) == ("int8", True)
        assert resolve_te_quant_request("none", fam.te_quant_auto)[0] is None


def _staged_only_config(tmp_path, monkeypatch):
    """A staged companion dir (as a cache-only load resolves under a fallback cache root) holding the encoder config
    but no dense shards; AutoConfig finds the config only there, never under the canonical base id."""
    import transformers

    staged = tmp_path / "fallback_root" / "snapshot"
    (staged / "text_encoder").mkdir(parents = True)
    (staged / "text_encoder" / "config.json").write_text("{}")
    asked = []

    def from_pretrained(source, *a, **k):
        asked.append(str(source))
        if str(source) != str(staged):
            raise OSError(f"{source} is not in the active cache")
        return _config()

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", from_pretrained)
    return staged, asked


@pytest.mark.parametrize("family", sorted(VARIANTS))
def test_the_prequant_encoder_reads_its_config_from_the_staged_base(
    mirror, tmp_path, monkeypatch, family
):
    staged, asked = _staged_only_config(tmp_path, monkeypatch)
    fam = detect_family("", override = family)
    kwargs = dict(
        te_quant_mode = "int8", target = CUDA_BF16, dtype = torch.float32, local_files_only = True
    )
    # without a config source the canonical id misses the config, and the dense fallback is refused
    with pytest.raises(RuntimeError, match = "could not be loaded"):
        tpq.te_prequant_pipe_kwargs(fam, fam.base_repo, dense_source = str(staged), **kwargs)
    asked.clear()
    encoder = tpq.te_prequant_pipe_kwargs(
        fam, fam.base_repo, dense_source = str(staged), config_source = str(staged), **kwargs
    )["text_encoder"]
    assert getattr(encoder, tpq.TE_PREQUANT_SCHEME_ATTR) == "int8"
    assert asked == [str(staged)]


def test_flux3_assembly_hands_the_staged_dir_to_the_prequant_encoder(tmp_path, monkeypatch):
    import core.inference.diffusion as d

    seen = {}

    def te_kwargs(fam, base, **k):
        seen["base"], seen["config_source"] = base, k.get("config_source")
        return {"text_encoder": "enc"}

    class _Pipe:
        def to(self, device):
            return self

    monkeypatch.setattr(d, "te_prequant_pipe_kwargs", te_kwargs)
    monkeypatch.setattr(
        d,
        "load_flux3_pipeline",
        lambda root, *a, **k: seen.update(root = root, te = k["text_encoder"]) or _Pipe(),
    )
    staged = str(tmp_path / "staged")
    d.DiffusionBackend._assemble_pipe(
        None,
        DISTILLED,
        object(),
        torch.float32,
        None,
        "cpu",
        base_local_dir = staged,
        fam = detect_family("", override = "flux.3-klein"),
        te_quant_mode = "int8",
        target = CUDA_BF16,
        fetch_base = "unsloth/flux-3-image-klein-FP8",
        local_files_only = True,
    )
    # the canonical fetch id for checkpoint compatibility, the staged dir for the config, as the pipeline itself
    assert seen == {
        "base": "unsloth/flux-3-image-klein-FP8",
        "config_source": staged,
        "root": staged,
        "te": "enc",
    }
