# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The bf16 MiniMax-H3 / H3-Image conditioner loads only the 50 decoder layers ``hidden_states[50]`` reads (the hosted
int8 layout), and its ``hidden_states[50]`` is bit-identical to the whole released encoder's."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from core.inference import video_minimax_h3_te as te  # noqa: E402

IMAGE, START, END = 120, 121, 122
LAYERS = 56  # deeper than the read layer, like the 64-layer release


def _config(layers = LAYERS):
    cfg = transformers.Qwen3VLConfig(
        text_config = dict(
            hidden_size = 32,
            intermediate_size = 64,
            num_hidden_layers = layers,
            num_attention_heads = 4,
            num_key_value_heads = 2,
            head_dim = 8,
            vocab_size = 128,
            rope_scaling = {"rope_type": "default", "mrope_section": [1, 1, 2]},
        ),
        vision_config = dict(
            depth = 2,
            hidden_size = 32,
            intermediate_size = 64,
            num_heads = 2,
            out_hidden_size = 32,
            patch_size = 16,
            spatial_merge_size = 2,
            temporal_patch_size = 2,
            num_position_embeddings = 64,
            deepstack_visual_indexes = [0, 1],
        ),
        image_token_id = IMAGE,
        vision_start_token_id = START,
        vision_end_token_id = END,
        tie_word_embeddings = False,
    )
    return cfg


@pytest.fixture(scope = "module")
def release(tmp_path_factory):
    """A tiny random bf16 release folder with a ``text_encoder/`` deeper than the read layer."""
    root = tmp_path_factory.mktemp("h3_te_release")
    torch.manual_seed(0)
    model = transformers.Qwen3VLForConditionalGeneration(_config()).eval()
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator = g) * 0.2)
    model.to(torch.bfloat16).save_pretrained(str(root / "text_encoder"))
    return root


def _full(root):
    return transformers.Qwen3VLForConditionalGeneration.from_pretrained(
        str(root / "text_encoder"), torch_dtype = torch.bfloat16
    ).eval()


def _trimmed(root):
    return (
        te.h3_read_layers_text_encoder_class()
        .from_pretrained(str(root / "text_encoder"), torch_dtype = torch.bfloat16)
        .eval()
    )


def _inputs(with_image):
    ids = [5, 6, 7]
    kwargs = {}
    if with_image:
        grid = torch.tensor([[1, 4, 6]], dtype = torch.long)
        g = torch.Generator().manual_seed(2)
        kwargs["pixel_values"] = torch.rand(24, 3 * 2 * 16 * 16, generator = g) * 2 - 1
        kwargs["image_grid_thw"] = grid
        ids += [START] + [IMAGE] * 6 + [END]
    ids += [8, 9]
    return ids, kwargs


def _hidden_50(model, ids, vision):
    input_ids = torch.tensor([ids])
    kwargs = dict(
        input_ids = input_ids,
        attention_mask = torch.ones_like(input_ids),
        use_cache = False,
        output_hidden_states = True,
    )
    if "mm_token_type_ids" in inspect.signature(model.model.forward).parameters:
        kwargs["mm_token_type_ids"] = (input_ids == IMAGE).long()
    with torch.no_grad():
        return model.model(**kwargs, **vision).hidden_states[te.H3_TE_READ_LAYER]


def test_trimmed_layout_matches_the_hosted_int8_one(release):
    model = _trimmed(release)
    lm = model.model.language_model
    assert len(lm.layers) == te.H3_TE_READ_LAYER + 1
    assert model.config.text_config.num_hidden_layers == te.H3_TE_READ_LAYER + 1
    assert type(lm.layers[te.H3_TE_READ_LAYER]).__name__ == "H3TextEncoderTerminatorLayer"
    assert isinstance(lm.norm, torch.nn.Identity)
    assert isinstance(model.lm_head, torch.nn.Identity)
    assert isinstance(model, transformers.Qwen3VLForConditionalGeneration)
    assert not [n for n, p in model.named_parameters() if p.is_meta]


def test_trimmed_drops_exactly_the_unread_tensors(release):
    full = dict(_full(release).named_parameters())
    trimmed = dict(_trimmed(release).named_parameters())
    assert set(trimmed) < set(full)
    dropped = set(full) - set(trimmed)
    unread = {
        n
        for n in full
        if n.startswith("lm_head.")
        or n.startswith("model.language_model.norm.")
        or any(
            n.startswith(f"model.language_model.layers.{i}.")
            for i in range(te.H3_TE_READ_LAYER, LAYERS)
        )
    }
    assert dropped == unread
    for name, p in trimmed.items():
        assert torch.equal(p, full[name]), name


@pytest.mark.parametrize("with_image", [False, True])
def test_hidden_states_50_is_bit_identical(release, with_image):
    ids, vision = _inputs(with_image)
    want = _hidden_50(_full(release), ids, vision)
    got = _hidden_50(_trimmed(release), ids, vision)
    assert got.dtype == want.dtype and got.shape == want.shape
    assert torch.equal(got, want)


def test_h3_image_read_paths_are_bit_identical(release):
    from core.inference import diffusion_minimax_h3_image as h3i

    ids, vision = _inputs(True)
    processor = SimpleNamespace(
        create_mm_token_type_ids = lambda rows: [[int(t == IMAGE) for t in row] for row in rows]
    )
    full, trimmed = _full(release), _trimmed(release)
    with torch.no_grad():
        want = h3i.qwen3vl_hidden_state(
            full, processor, ids, dict(vision), layer = 50, device = torch.device("cpu")
        )
        got = h3i.qwen3vl_hidden_state(
            trimmed, processor, ids, dict(vision), layer = 50, device = torch.device("cpu")
        )
    assert torch.equal(got, want)


def test_too_shallow_conditioner_is_refused(tmp_path):
    with pytest.raises(ValueError, match = "hidden_states\\[50\\]"):
        te.h3_read_layers_text_encoder_class()(_config(layers = te.H3_TE_READ_LAYER))


def test_h3_image_bf16_loader_builds_the_read_layers(release):
    from core.inference import diffusion_minimax_h3_image as h3i

    encoder, engaged = h3i.load_h3_image_text_encoder(
        None,
        dtype = torch.bfloat16,
        text_encoder_repo = str(release),
        hf_token = None,
        cache_dir = None,
        local_files_only = True,
    )
    assert engaged is None
    assert len(encoder.model.language_model.layers) == te.H3_TE_READ_LAYER + 1
    assert isinstance(encoder.lm_head, torch.nn.Identity)


def test_video_spec_swap_loads_the_read_layers_and_restores(release):
    modular = pytest.importorskip("diffusers.modular_pipelines")
    ComponentSpec = modular.ComponentSpec

    spec = ComponentSpec(
        name = "text_encoder",
        type_hint = transformers.Qwen3VLForConditionalGeneration,
        pretrained_model_name_or_path = str(release),
        subfolder = "text_encoder",
    )
    pipe = SimpleNamespace(
        _component_specs = {"text_encoder": spec}, text_encoder_layer = 50, text_encoder = None
    )
    with te.h3_read_layers_spec(pipe) as engaged:
        assert engaged
        encoder = spec.load(torch_dtype = torch.bfloat16)
    assert spec.type_hint is transformers.Qwen3VLForConditionalGeneration
    assert len(encoder.model.language_model.layers) == te.H3_TE_READ_LAYER + 1
    ids, vision = _inputs(True)
    assert torch.equal(
        _hidden_50(encoder.eval(), ids, vision), _hidden_50(_full(release), ids, vision)
    )


@pytest.mark.parametrize(
    "pipe_kwargs",
    [
        {"text_encoder_layer": 49},  # another read layer: the trim would be wrong
        {"text_encoder": object()},  # already seeded (hosted int8)
    ],
)
def test_video_spec_swap_stays_off(pipe_kwargs):
    spec = SimpleNamespace(type_hint = transformers.Qwen3VLForConditionalGeneration)
    kwargs = {"text_encoder_layer": 50, "text_encoder": None, **pipe_kwargs}
    pipe = SimpleNamespace(_component_specs = {"text_encoder": spec}, **kwargs)
    with te.h3_read_layers_spec(pipe) as engaged:
        assert not engaged
        assert spec.type_hint is transformers.Qwen3VLForConditionalGeneration


def test_video_h3_load_components_runs_under_the_read_layer_spec():
    """The H3 modular loader's load_components_strict call sits inside ``h3_read_layers_spec``."""
    src = Path(__file__).resolve().parents[1] / "core" / "inference" / "video.py"
    tree = ast.parse(src.read_text(encoding = "utf-8"))
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_load_h3_modular_pipeline"
    )
    wrapped = [
        call
        for w in ast.walk(fn)
        if isinstance(w, ast.With)
        and any(
            isinstance(i.context_expr, ast.Call)
            and getattr(i.context_expr.func, "id", None) == "h3_read_layers_spec"
            for i in w.items
        )
        for call in ast.walk(w)
        if isinstance(call, ast.Call) and getattr(call.func, "id", None) == "load_components_strict"
    ]
    assert len(wrapped) == 1


def test_real_config_saving():
    """Bytes the trim never allocates at the released MiniMax-H3 conditioner's sizes (bf16)."""
    hidden, inter, heads, kv, head_dim, vocab, layers = 5120, 25600, 64, 8, 128, 151936, 64
    per_layer = (
        hidden * heads * head_dim * 2  # q, o
        + hidden * kv * head_dim * 2  # k, v
        + 3 * hidden * inter  # gate, up, down
        + 2 * hidden  # input / post-attention norms
        + 2 * head_dim  # q / k norms
    )
    saved = 2 * ((layers - te.H3_TE_READ_LAYER) * per_layer + vocab * hidden + hidden)
    assert 15.0e9 < saved < 15.3e9
    assert te.h3_te_unread_bytes() == saved
    assert te.H3_TE_UNREAD_BF16_GB == saved / 1e9


def test_release_shape_matches_the_tiny_count(release):
    """The byte formula counts exactly what the read-layer load drops, checked on a real (tiny) model."""
    cfg = _config().text_config
    shape = {
        "num_hidden_layers": LAYERS,
        "hidden_size": cfg.hidden_size,
        "intermediate_size": cfg.intermediate_size,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads,
        "head_dim": cfg.head_dim,
        "vocab_size": cfg.vocab_size,
    }
    full = sum(p.numel() * p.element_size() for p in _full(release).parameters())
    trimmed = sum(p.numel() * p.element_size() for p in _trimmed(release).parameters())
    assert full - trimmed == te.h3_te_unread_bytes(shape)


# ---- memory planner budgets ---------------------------------------------------------------------------------------

STOCK_GB = 66.7


def _spec_pipe(
    type_hint,
    text_encoder = None,
    layer = 50,
):
    spec = SimpleNamespace(type_hint = type_hint)
    return SimpleNamespace(
        _component_specs = {"text_encoder": spec}, text_encoder_layer = layer, text_encoder = text_encoder
    )


def test_bf16_budget_is_the_read_layer_size_only_where_that_class_loads(release):
    read_gb = te.h3_te_read_layers_gb(STOCK_GB)
    assert abs(read_gb - (STOCK_GB - 15.2086)) < 1e-3
    stock_cls = transformers.Qwen3VLForConditionalGeneration
    # Spec swapped at load, or a loaded read-layer encoder: the read-layer size.
    assert te.h3_bf16_te_gb(STOCK_GB, pipe = _spec_pipe(stock_cls)) == read_gb
    assert te.h3_bf16_te_gb(STOCK_GB, module = _trimmed(release)) == read_gb
    assert (
        te.h3_bf16_te_gb(STOCK_GB, pipe = _spec_pipe(stock_cls, text_encoder = _trimmed(release)))
        == read_gb
    )


@pytest.mark.parametrize(
    "case",
    ["another_class", "other_read_layer", "stock_module_loaded", "unknown"],
)
def test_bf16_budget_keeps_the_whole_size_where_the_stock_class_can_load(release, case):
    stock_cls = transformers.Qwen3VLForConditionalGeneration

    class OtherEncoder(stock_cls):
        pass

    kwargs = {
        "another_class": {"pipe": _spec_pipe(OtherEncoder)},
        "other_read_layer": {"pipe": _spec_pipe(stock_cls, layer = 49)},
        "stock_module_loaded": {"module": _full(release)},
        "unknown": {},
    }[case]
    assert te.h3_bf16_te_gb(STOCK_GB, **kwargs) == STOCK_GB


def test_h3_image_plan_refused_at_the_whole_encoder_is_admitted_at_the_read_layers():
    from core.inference import diffusion_minimax_h3_image as h3i

    assert h3i.te_gb_for(None) == te.h3_te_read_layers_gb(h3i.H3_IMAGE_TE_BF16_GB)
    assert h3i.te_gb_for("int8") == h3i.H3_IMAGE_TE_INT8_GB
    args = dict(vram_gb = 80.0, dit_gb = 40.3)
    # 60 GB host: the whole 66.7 GB encoder is refused, the read-layer one is kept on the host.
    with pytest.raises(RuntimeError, match = "free system memory"):
        h3i.plan_placement(host_gb = 60.0, te_gb = h3i.H3_IMAGE_TE_BF16_GB, **args)
    plan = h3i.plan_placement(host_gb = 60.0, te_gb = h3i.te_gb_for(None), **args)
    assert plan.te_kept_on_host
    # 70 GB host: the whole encoder would be re-read every prompt, the read-layer one stays.
    assert not h3i.plan_placement(
        host_gb = 70.0, te_gb = h3i.H3_IMAGE_TE_BF16_GB, **args
    ).te_kept_on_host
    assert h3i.plan_placement(host_gb = 70.0, te_gb = h3i.te_gb_for(None), **args).te_kept_on_host


def test_h3_image_status_text_names_the_read_layer_size():
    from core.inference import diffusion_minimax_h3_image as h3i

    src = Path(h3i.__file__).read_text(encoding = "utf-8")
    assert "66.7 GB bf16" not in src
    assert src.count("{H3_IMAGE_TE_BF16_READ_GB:.1f} GB bf16") == 2


def _video_dense_mib(monkeypatch, pipe):
    import core.inference.video as video_mod
    from core.inference.video import VideoBackend
    from core.inference.video_families import detect_video_family

    fam = detect_video_family("MiniMaxAI/MiniMax-H3")
    seen = []

    def _capture(*, model_dense_mib = None, **kwargs):
        seen.append(int(model_dense_mib or 0))
        return SimpleNamespace(offload_policy = "none", estimates = {}, device_memory = None)

    monkeypatch.setattr(video_mod, "plan_diffusion_memory", _capture)
    monkeypatch.setattr(video_mod, "settled_snapshot_device_memory", lambda t: None)
    monkeypatch.setattr(video_mod, "raise_on_unified_memory_shortfall", lambda *a, **k: None)
    VideoBackend._raise_on_modular_unified_shortfall(
        fam,
        target = None,
        dtype = torch.bfloat16,
        device = "mps",
        memory_mode = None,
        scheme = None,
        pipe = pipe,
    )
    return fam, seen[-1]


def test_h3_video_unified_check_prices_the_read_layer_encoder(monkeypatch):
    stock_cls = transformers.Qwen3VLForConditionalGeneration

    class OtherEncoder(stock_cls):
        pass

    fam, trimmed_mib = _video_dense_mib(monkeypatch, _spec_pipe(stock_cls))
    _fam, stock_mib = _video_dense_mib(monkeypatch, _spec_pipe(OtherEncoder))
    mib_per_gb = 1000.0**3 / (1024.0 * 1024.0)
    transformer_gb, te_gb, vae_gb = fam.bf16_components_gb
    assert stock_mib == int((transformer_gb + te_gb + vae_gb) * mib_per_gb)
    assert trimmed_mib == int(
        (transformer_gb + te.h3_te_read_layers_gb(te_gb) + vae_gb) * mib_per_gb
    )


def test_h3_video_host_ram_refused_at_the_whole_encoder_is_admitted(monkeypatch):
    from core.inference import video_minimax_h3 as vh
    from core.inference.video_minimax_h3_te import h3_te_resident_gb

    stock = h3_te_resident_gb(None, bf16_gb = vh.H3_TEXT_ENCODER_BF16_GB)
    read = h3_te_resident_gb(
        None,
        bf16_gb = te.h3_bf16_te_gb(
            vh.H3_TEXT_ENCODER_BF16_GB,
            pipe = _spec_pipe(transformers.Qwen3VLForConditionalGeneration),
        ),
    )
    need_stock = vh.estimate_h3_diffusers_host_ram_gb(24.0, text_encoder_gb = stock)
    need_read = vh.estimate_h3_diffusers_host_ram_gb(24.0, text_encoder_gb = read)
    assert abs((need_stock - need_read) - te.H3_TE_UNREAD_BF16_GB) < 1e-6
    capacity = (need_read + need_stock) / 2
    monkeypatch.setattr(vh, "h3_host_capacity_bytes", lambda: int(capacity * 1e9))
    assert vh.h3_host_ram_shortfall(24.0, text_encoder_gb = stock) is not None
    assert vh.h3_host_ram_shortfall(24.0, text_encoder_gb = read) is None


def test_h3_video_others_bytes_follow_the_spec():
    from core.inference import video as video_mod
    from core.inference.video_families import detect_video_family

    fam = detect_video_family("MiniMaxAI/MiniMax-H3")
    comps = fam.bf16_components_gb
    stock_cls = transformers.Qwen3VLForConditionalGeneration

    def others(pipe):
        return video_mod._h3_others_bytes(
            fam, comps, te_scheme = None, dtype = torch.bfloat16, rotating = False, pipe = pipe
        )

    assert abs(others(None) - others(_spec_pipe(stock_cls)) - te.h3_te_unread_bytes()) <= 2
    # int8 is priced at its hosted size whatever the spec says.
    assert video_mod._h3_others_bytes(
        fam,
        comps,
        te_scheme = "int8",
        dtype = torch.bfloat16,
        rotating = False,
        pipe = _spec_pipe(stock_cls),
    ) == video_mod._h3_others_bytes(
        fam, comps, te_scheme = "int8", dtype = torch.bfloat16, rotating = False
    )


def test_h3_video_runtime_checks_price_the_loaded_encoder():
    """Every runtime H3 budget in video.py sizes the bf16 encoder from the loaded pipeline, none from the constant."""
    src = (Path(__file__).resolve().parents[1] / "core" / "inference" / "video.py").read_text(
        encoding = "utf-8"
    )
    assert "bf16_gb = H3_TEXT_ENCODER_BF16_GB" not in src
    assert src.count("h3_bf16_te_gb(H3_TEXT_ENCODER_BF16_GB, pipe = state.pipe)") == 3
