# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FLUX.3 and Krea-2 load their Qwen3-VL text encoder without its unread vision tower (dense, int8 ConvRot and fp8
files alike), with bit-identical text embeddings; families that run the tower keep it."""

from __future__ import annotations

import json
import shutil
import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers.models.qwen3_vl")
pytest.importorskip("accelerate")

import core.inference.diffusion_te_prequant as tpq  # noqa: E402
import core.inference.diffusion_te_vision as tv  # noqa: E402
import core.inference.prequant_safetensors as ps  # noqa: E402
from core.inference.diffusion_families import detect_family  # noqa: E402

GROUP = 256
HIDDEN, VOCAB = 256, 64
CUDA_BF16 = types.SimpleNamespace(device = "cuda", dtype = torch.bfloat16)
FLUX3 = {
    "flux.3-klein": ("unsloth/flux-3-image-klein-FP8", "flux-3-image-klein"),
    "flux.3-klein-base": ("unsloth/flux-3-image-klein-base-FP8", "flux-3-image-klein-base"),
}
KREA = ("unsloth/Krea-2-Turbo-FP8", "Krea-2-Turbo")


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


def _encoder(cls_name = "Qwen3VLForConditionalGeneration", seed = 0):
    import transformers
    torch.manual_seed(seed)
    return getattr(transformers, cls_name)(_config()).eval()


def _hidden(encoder, ids):
    with torch.no_grad():
        out = encoder(
            input_ids = ids,
            attention_mask = torch.ones_like(ids),
            output_hidden_states = True,
            use_cache = False,
        )
    return out.hidden_states


def _same_hidden(a, b, ids):
    ha, hb = _hidden(a, ids), _hidden(b, ids)
    return len(ha) == len(hb) and all(torch.equal(x, y) for x, y in zip(ha, hb))


def _vision_params(module):
    return [n for n, _ in module.named_parameters() if "visual" in n.split(".")]


def _ids():
    return torch.randint(0, VOCAB, (1, 11), generator = torch.Generator().manual_seed(1))


# ── the helper ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("cls_name", ["Qwen3VLModel", "Qwen3VLForConditionalGeneration"])
def test_drop_frees_the_tower_of_either_qwen3_vl_class(cls_name):
    enc = _encoder(cls_name)
    assert tv.has_vision_tower(enc) and _vision_params(enc)
    assert tv.drop_vision_tower(enc) is True
    assert not _vision_params(enc) and not tv.has_vision_tower(enc)
    assert tv.drop_vision_tower(enc) is False


def test_a_stub_torch_in_sys_modules_keeps_the_stand_in_a_module_and_the_drop_working(monkeypatch):
    # Backend tests stub ``torch`` / ``torch.nn`` in sys.modules; a scope or a drop that runs then must still see the
    # real nn.Module, or the stand-in is set as a plain attribute and the drop reports no tower.
    import sys

    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    real_module = torch.nn.Module
    enc = _encoder("Qwen3VLModel")
    stub = types.ModuleType("torch.nn")
    stub.Module = type("Module", (), {})
    monkeypatch.setitem(sys.modules, "torch.nn", stub)
    monkeypatch.setattr(torch, "nn", stub)
    assert issubclass(tv._stand_in_class(Qwen3VLVisionModel), real_module)
    assert tv.has_vision_tower(enc) and tv.drop_vision_tower(enc) is True
    assert enc.visual is None


def test_only_text_only_families_drop_and_the_kill_switch_keeps_it(monkeypatch):
    for family in ("flux.3-klein", "flux.3-klein-base", "krea-2"):
        assert tv.family_drops_vision_tower(family)
    # Their edit / reference modes run the tower.
    for family in (
        "qwen-image-edit",
        "qwen-image-2.1",
        "minimax-h3-image",
        "ideogram-4",
        "moss-image",
        None,
    ):
        assert not tv.family_drops_vision_tower(family)
    monkeypatch.setenv(tv.DROP_VISION_ENV, "0")
    assert not tv.family_drops_vision_tower("flux.3-klein")


def test_moss_keeps_its_drop():
    moss = pytest.importorskip("core.inference.diffusion_moss_image")
    assert moss.drop_vision_tower is tv.drop_vision_tower
    enc = types.SimpleNamespace()
    enc.visual = None
    assert moss.drop_vision_tower(torch.nn.Module()) is False
    holder = torch.nn.Module()
    holder.visual = torch.nn.Linear(2, 2)
    assert moss.drop_vision_tower(holder) is True and holder.visual is None


# ── dense loads: the tower is never built nor read ───────────────────────────
@pytest.mark.parametrize("cls_name", ["Qwen3VLModel", "Qwen3VLForConditionalGeneration"])
def test_from_pretrained_in_scope_skips_the_tower_bit_identically(tmp_path, cls_name):
    import transformers

    _encoder(cls_name, seed = 5).save_pretrained(tmp_path)
    cls = getattr(transformers, cls_name)
    full = cls.from_pretrained(tmp_path, dtype = torch.float32).eval()
    with tv.vision_tower_skipped():
        lean = cls.from_pretrained(tmp_path, dtype = torch.float32).eval()
    assert tv.drop_vision_tower(lean)
    assert not _vision_params(lean) and _vision_params(full)
    assert _same_hidden(full, lean, _ids())
    # Out of scope (another family on any thread) the tower loads as before.
    again = cls.from_pretrained(tmp_path, dtype = torch.float32)
    assert _vision_params(again)


def _tokenizer_files(dst):
    """A tiny Qwen3-VL processor (the tokenizer FLUX.3 reads from text_encoder/)."""
    from transformers import AutoProcessor

    try:
        proc = AutoProcessor.from_pretrained(
            "trl-internal-testing/tiny-Qwen3VLForConditionalGeneration"
        )
    except Exception as exc:  # noqa: BLE001 - offline host without the tiny repo cached
        pytest.skip(f"tiny Qwen3-VL processor unavailable: {exc}")
    proc.save_pretrained(dst)


def test_flux3_dense_load_has_no_vision_tower(tmp_path, monkeypatch):
    pytest.importorskip("diffusers")
    import diffusers
    from transformers import Qwen3VLForConditionalGeneration

    from core.inference import diffusion_flux3 as f3

    if not f3.flux3_available():
        pytest.skip("no FLUX.3 modular blocks in this diffusers")
    root = tmp_path / "klein"
    _encoder(seed = 7).save_pretrained(root / "text_encoder")
    _tokenizer_files(root / "text_encoder")
    diffusers.AutoencoderKLFlux2(
        block_out_channels = (16, 16, 16, 16), layers_per_block = 1, norm_num_groups = 8
    ).save_pretrained(root / "vae")
    spec = lambda folder, lib, cls: [
        lib,
        cls,
        {"pretrained_model_name_or_path": str(root), "subfolder": folder},
    ]
    (root / f3.FLUX3_INDEX).write_text(
        json.dumps(
            {
                "_blocks_class_name": "Flux3DistilledAutoBlocks",
                "_class_name": "Flux3DistilledModularPipeline",
                **{name: spec(*row) for name, row in f3.FLUX3_COMPONENT_FOLDERS.items()},
            }
        )
    )
    pipe = f3.load_flux3_pipeline(
        str(root),
        torch.float32,
        local_files_only = True,
        with_transformer = False,
        family_name = f3.FLUX3_KLEIN_FAMILY_NAME,
    )
    assert pipe.text_encoder.model.visual is None and not _vision_params(pipe.text_encoder)
    full = Qwen3VLForConditionalGeneration.from_pretrained(
        root / "text_encoder", dtype = torch.float32
    ).eval()
    assert _same_hidden(full, pipe.text_encoder, _ids())

    monkeypatch.setenv(tv.DROP_VISION_ENV, "0")
    kept = f3.load_flux3_pipeline(
        str(root),
        torch.float32,
        local_files_only = True,
        with_transformer = False,
        family_name = f3.FLUX3_KLEIN_FAMILY_NAME,
    )
    assert _vision_params(kept.text_encoder)


def test_krea2_dense_load_has_no_vision_tower(tmp_path, monkeypatch):
    import sys

    from transformers import Qwen3VLModel

    from core.inference import diffusion_krea2 as krea2

    root = tmp_path / "krea"
    _encoder("Qwen3VLModel", seed = 9).save_pretrained(root / "text_encoder")
    fake = types.SimpleNamespace(
        Krea2Pipeline = lambda **kw: types.SimpleNamespace(**kw),
        FlowMatchEulerDiscreteScheduler = types.SimpleNamespace(
            from_pretrained = lambda *a, **k: object()
        ),
        AutoencoderKLQwenImage = types.SimpleNamespace(from_pretrained = lambda *a, **k: object()),
        Krea2Transformer2DModel = types.SimpleNamespace(from_pretrained = lambda *a, **k: object()),
    )
    monkeypatch.setitem(sys.modules, "diffusers", fake)
    monkeypatch.setattr(krea2, "_load_model_index", lambda *a, **k: {})
    monkeypatch.setattr(krea2, "load_krea2_tokenizer", lambda *a, **k: object())
    pipe = krea2.load_krea2_pipeline(str(root), torch.float32, local_files_only = True)
    assert pipe.text_encoder.visual is None and not _vision_params(pipe.text_encoder)
    full = Qwen3VLModel.from_pretrained(root / "text_encoder", dtype = torch.float32).eval()
    assert _same_hidden(full, pipe.text_encoder, _ids())
    # A supplied encoder (pre-cast or single-file) is freed too.
    handed = Qwen3VLModel.from_pretrained(root / "text_encoder", dtype = torch.float32)
    pipe = krea2.load_krea2_pipeline(
        str(root), torch.float32, text_encoder = handed, local_files_only = True
    )
    assert pipe.text_encoder.visual is None
    # Ideogram 4 reuses the Krea-2 encoder loader outside the pipeline: unchanged.
    assert _vision_params(
        krea2.load_krea2_text_encoder(str(root), torch.float32, local_files_only = True)
    )


# ── pre-cast files: the tower's tensors are never read ───────────────────────
def _write(path, state, fmt, scheme, te_class, base):
    from safetensors.torch import save_file

    meta = {
        ps.UNSLOTH_FORMAT_KEY: fmt,
        ps.UNSLOTH_METADATA_KEY: json.dumps(
            {
                "scheme": scheme,
                "component": "text_encoder",
                "base_model_id": base,
                "te_class": te_class,
            }
        ),
        "tensor_names": json.dumps(sorted(state)),
        **{n: json.dumps({"_type": "Tensor"}) for n in state},
    }
    path.parent.mkdir(parents = True, exist_ok = True)
    save_file(
        {k: v.detach().clone().contiguous() for k, v in state.items()}, str(path), metadata = meta
    )


def _int8_state(encoder):
    import re

    lin = re.compile(
        r"^(model\.)?language_model\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)\.weight$"
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
    pytest.importorskip("diffusers.hooks")
    from core.inference.diffusion_precision import _cast_fp8

    _cast_fp8(encoder, types.SimpleNamespace(dtype = torch.float32))
    return dict(encoder.state_dict())


@pytest.fixture
def mirror(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, "_torchao_helpers", lambda: None)
    import huggingface_hub
    import transformers

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *_a, **_k: _config())
    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download", lambda *a, **k: pytest.fail("the Hub was asked")
    )
    root = tmp_path / "mirror"
    rows = [
        (fam, repo, stem, "Qwen3VLForConditionalGeneration") for fam, (repo, stem) in FLUX3.items()
    ]
    rows.append(("krea-2", *KREA, "Qwen3VLModel"))
    for fam, repo, stem, cls_name in rows:
        base = detect_family("", override = fam).base_repo
        _write(
            root / repo / f"{stem}-text_encoder-INT8-ConvRot.safetensors",
            _int8_state(_encoder(cls_name, seed = 3)),
            tpq.TE_PREQUANT_FORMAT_INT8_CONVROT,
            "int8",
            cls_name,
            base,
        )
        _write(
            root / repo / f"{stem}-text_encoder-FP8.safetensors",
            _fp8_state(_encoder(cls_name, seed = 3)),
            tpq.TE_PREQUANT_FORMAT,
            "fp8",
            cls_name,
            base,
        )
    monkeypatch.setenv(tpq.TE_PREQUANT_MIRROR_ENV, str(root))
    return root


def _pipe_kwargs(family, mode):
    fam = detect_family("", override = family)
    return tpq.te_prequant_pipe_kwargs(
        fam,
        fam.base_repo,
        te_quant_mode = mode,
        target = CUDA_BF16,
        dtype = torch.float32,
        local_files_only = True,
    )


@pytest.fixture
def reads(monkeypatch):
    """Every tensor name a pre-cast load reads from its safetensors file."""
    import safetensors

    seen: list = []
    real = safetensors.safe_open

    class _Spy:
        def __init__(self, *a, **k):
            self._h = real(*a, **k)

        def __enter__(self):
            self._inner = self._h.__enter__()
            return self

        def __exit__(self, *exc):
            return self._h.__exit__(*exc)

        def get_tensor(self, name):
            seen.append(name)
            return self._inner.get_tensor(name)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    monkeypatch.setattr(safetensors, "safe_open", _Spy)
    return seen


@pytest.mark.parametrize("mode", ["int8", "fp8"])
@pytest.mark.parametrize("family", [*sorted(FLUX3), "krea-2"])
def test_precast_encoder_loads_without_reading_the_tower(mirror, reads, monkeypatch, family, mode):
    lean = _pipe_kwargs(family, mode)["text_encoder"]
    assert reads and not [n for n in reads if tv.is_vision_tower_key(n)]
    assert getattr(lean, tpq.TE_PREQUANT_SCHEME_ATTR) == mode
    assert not tv.has_vision_tower(lean) and not _vision_params(lean)
    reads.clear()
    monkeypatch.setenv(tv.DROP_VISION_ENV, "0")
    full = _pipe_kwargs(family, mode)["text_encoder"]
    assert [n for n in reads if tv.is_vision_tower_key(n)] and _vision_params(full)
    assert _same_hidden(full, lean, _ids())


def test_a_family_that_runs_the_tower_still_reads_it(tmp_path, monkeypatch):
    """The skip is per call: the same file loaded for a family not in the list keeps every tower tensor."""
    monkeypatch.setattr(ps, "_torchao_helpers", lambda: None)
    path = tmp_path / "te.safetensors"
    state = _int8_state(_encoder(seed = 3))
    _write(
        path,
        state,
        tpq.TE_PREQUANT_FORMAT_INT8_CONVROT,
        "int8",
        "Qwen3VLForConditionalGeneration",
        "x",
    )
    kept = ps.load_plain_prequant_safetensors(str(path))["state_dict"]
    skipped = ps.load_plain_prequant_safetensors(str(path), skip_prefixes = tv.VISION_TOWER_PREFIXES)[
        "state_dict"
    ]
    vis = {k for k in state if tv.is_vision_tower_key(k)}
    assert vis and vis <= set(kept) and not vis & set(skipped)
    assert set(kept) - set(skipped) == vis
    shutil.rmtree(tmp_path, ignore_errors = True)
