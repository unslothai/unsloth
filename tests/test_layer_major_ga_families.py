# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""offload_layers layer-major gradient accumulation decides by the decoder's structure, not its class name: tiny
random Llama / Qwen3 / Olmo / Gemma 3 vision / Qwen3-VL / Gemma 4 / Qwen3.5 (hybrid linear attention) models through
transformers.Trainer on CPU with a stand-in block swap. Where a window runs (auto after a measured stall), the logged
losses and every trained weight are bitwise those of UNSLOTH_OFFLOAD_LAYER_MAJOR = 0; where the structure cannot run
layer by layer (KV sharing, deepstack visual features between layers, a per-layer input feeding trained weights),
the step runs the standard way with one printed reason."""

import inspect

import pytest
import torch

from tests.test_layer_major_ga import _Swap, _windows, ga  # noqa: F401  (ga is the shared fixture)

transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
_IMAGE = 299  # image placeholder token of the tiny vision models
_VOCAB = 300


def _need(*names):
    for name in names:
        if not hasattr(transformers, name):
            pytest.skip(f"this transformers has no {name}")


def _dense(arch):
    _need(f"{arch}ForCausalLM")
    cfg = getattr(transformers, f"{arch}Config")(
        hidden_size = 64,
        intermediate_size = 128,
        num_hidden_layers = 4,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        vocab_size = _VOCAB,
        max_position_embeddings = 256,
        tie_word_embeddings = False,
    )
    return getattr(transformers, f"{arch}ForCausalLM")(cfg)


def _gemma3_vision():
    _need("Gemma3ForConditionalGeneration")
    text = transformers.Gemma3TextConfig(
        hidden_size = 64,
        intermediate_size = 128,
        num_hidden_layers = 4,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 16,
        vocab_size = _VOCAB,
        max_position_embeddings = 256,
        sliding_window = 8,
        layer_types = ["sliding_attention", "full_attention"] * 2,
    )
    vision = transformers.SiglipVisionConfig(
        hidden_size = 32,
        intermediate_size = 64,
        num_hidden_layers = 2,
        num_attention_heads = 2,
        image_size = 32,
        patch_size = 8,
    )
    cfg = transformers.Gemma3Config(
        text_config = text,
        vision_config = vision,
        mm_tokens_per_image = 4,
        image_token_index = _IMAGE,
        boi_token_index = 297,
        eoi_token_index = 298,
    )
    return transformers.Gemma3ForConditionalGeneration(cfg)


def _qwen3_vl():
    _need("Qwen3VLForConditionalGeneration")
    text = dict(
        hidden_size = 64,
        intermediate_size = 128,
        num_hidden_layers = 4,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 16,
        vocab_size = _VOCAB,
        max_position_embeddings = 256,
        rope_scaling = {
            "rope_type": "default",
            "mrope_section": [2, 3, 3],
            "mrope_interleaved": True,
        },
    )
    vision = dict(
        depth = 2,
        hidden_size = 32,
        intermediate_size = 64,
        num_heads = 2,
        out_hidden_size = 64,
        patch_size = 4,
        spatial_merge_size = 2,
        temporal_patch_size = 2,
        num_position_embeddings = 16,
        deepstack_visual_indexes = [0, 1],
    )
    cfg = transformers.Qwen3VLConfig(text_config = text, vision_config = vision, image_token_id = _IMAGE)
    return transformers.Qwen3VLForConditionalGeneration(cfg)


def _gemma4(shared):
    _need("Gemma4ForCausalLM")
    cfg = transformers.Gemma4TextConfig(
        hidden_size = 64,
        intermediate_size = 128,
        num_hidden_layers = 4,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 16,
        global_head_dim = 16,
        vocab_size = _VOCAB,
        vocab_size_per_layer_input = _VOCAB,
        hidden_size_per_layer_input = 8,
        max_position_embeddings = 256,
        sliding_window = 8,
        num_kv_shared_layers = shared,
        layer_types = ["sliding_attention", "full_attention"] * 2,
        tie_word_embeddings = False,
    )
    return transformers.Gemma4ForCausalLM(cfg)


def _qwen3_5():
    _need("Qwen3_5ForCausalLM")
    cfg = transformers.Qwen3_5TextConfig(
        hidden_size = 64,
        intermediate_size = 128,
        num_hidden_layers = 4,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 16,
        vocab_size = _VOCAB,
        max_position_embeddings = 256,
        layer_types = ["linear_attention"] * 3 + ["full_attention"],
        linear_num_key_heads = 2,
        linear_num_value_heads = 4,
        linear_key_head_dim = 16,
        linear_value_head_dim = 16,
        linear_conv_kernel_dim = 4,
        tie_word_embeddings = False,
    )
    return transformers.Qwen3_5ForCausalLM(cfg)


_BUILD = {
    "llama": lambda: _dense("Llama"),
    "qwen3": lambda: _dense("Qwen3"),
    "olmo": lambda: _dense("Olmo"),
    "gemma3_vision": _gemma3_vision,
    "qwen3_vl": _qwen3_vl,
    "gemma4": lambda: _gemma4(0),
    "gemma4_kv_shared": lambda: _gemma4(2),
    "qwen3_5": _qwen3_5,
}


def _decoder_layers(model):
    """The longest ModuleList of decoder layers (the text model's, for a vision model)."""
    best = None
    for module in model.modules():
        if (
            isinstance(module, torch.nn.ModuleList)
            and len(module)
            and type(module[0]).__name__.endswith("DecoderLayer")
            and (best is None or len(module) > len(best))
        ):
            best = module
    return best


def _build(
    family,
    lora = True,
    targets = _TARGETS,
):
    torch.manual_seed(0)
    model = _BUILD[family]()
    model.config.use_cache = False
    model.config._attn_implementation = "eager"
    for sub in getattr(model.config, "sub_configs", {}) or {}:
        sub_cfg = getattr(model.config, sub, None)
        if sub_cfg is not None:
            sub_cfg._attn_implementation = "eager"
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": True})
    if lora:
        model = peft.get_peft_model(
            model,
            peft.LoraConfig(
                r = 4, lora_alpha = 8, lora_dropout = 0.0, init_lora_weights = False, target_modules = targets
            ),
        )
    else:
        # Full fine-tuning of the decoder (embedding and head frozen: a vision model ties them).
        for name, p in model.named_parameters():
            p.requires_grad_(".layers." in name)
    model.enable_input_require_grads()
    return model


class _Text(torch.utils.data.Dataset):
    def __init__(
        self,
        n = 24,
        image = None,
    ):
        g = torch.Generator().manual_seed(1)
        self.rows = [
            torch.randint(0, 290, (8 + int(torch.randint(0, 12, (1,), generator = g)),), generator = g)
            for _ in range(n)
        ]
        self.image = image

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return {"input_ids": self.rows[i], "i": i}


def _pad(rows, prefix = 0):
    n = max(len(r) for r in rows) + prefix
    ids = torch.zeros(len(rows), n, dtype = torch.long)
    mask = torch.zeros(len(rows), n, dtype = torch.long)
    for k, r in enumerate(rows):
        ids[k, prefix : prefix + len(r)] = r
        mask[k, : prefix + len(r)] = 1
    return ids, mask


def _text_collate(rows):
    ids, mask = _pad([r["input_ids"] for r in rows])
    return {"input_ids": ids, "attention_mask": mask, "labels": ids.masked_fill(mask == 0, -100)}


def _gemma3_image_collate(rows):
    """Every row starts with one 32 x 32 image (4 image tokens); odd rows of the dataset carry no image so text-only
    and image micro-batches alternate."""
    if rows[0]["i"] % 2:
        return _text_collate(rows)
    ids, mask = _pad([r["input_ids"] for r in rows], prefix = 4)
    ids[:, :4] = _IMAGE
    g = torch.Generator().manual_seed(int(rows[0]["i"]))
    labels = ids.masked_fill((mask == 0) | (ids == _IMAGE), -100)
    return {
        "input_ids": ids,
        "attention_mask": mask,
        "labels": labels,
        "token_type_ids": (ids == _IMAGE).long(),
        "pixel_values": torch.randn(len(rows), 3, 32, 32, generator = g),
    }


def _qwen3_vl_image_collate(rows):
    """Each row: one 16 x 16 image, 1 x 4 x 4 patches of 4 x 4 pixels, 4 image tokens after the 2 x 2 merge."""
    ids, mask = _pad([r["input_ids"] for r in rows], prefix = 4)
    ids[:, :4] = _IMAGE
    g = torch.Generator().manual_seed(int(rows[0]["i"]))
    patches = 16 * len(rows)
    batch = {
        "input_ids": ids,
        "attention_mask": mask,
        "labels": ids.masked_fill((mask == 0) | (ids == _IMAGE), -100),
        "pixel_values": torch.randn(patches, 3 * 2 * 4 * 4, generator = g),
        "image_grid_thw": torch.tensor([[1, 4, 4]] * len(rows)),
    }
    if (
        "mm_token_type_ids"
        in inspect.signature(transformers.Qwen3VLForConditionalGeneration.forward).parameters
    ):
        batch["mm_token_type_ids"] = (ids == _IMAGE).int()  # transformers 5.x M-RoPE needs it
    return batch


def _run(
    mod,
    tmp_path,
    monkeypatch,
    family,
    mode,
    steps = 4,
    collate = _text_collate,
    stall = 50.0,
    **build,
):
    if mode is None:
        monkeypatch.delenv("UNSLOTH_OFFLOAD_LAYER_MAJOR", raising = False)
    else:
        monkeypatch.setenv("UNSLOTH_OFFLOAD_LAYER_MAJOR", mode)
    model = _build(family, **build)
    sw = _Swap(stall)
    layers = _decoder_layers(model)
    sw.total = len(layers)
    layers._unsloth_block_swap = sw
    args = transformers.TrainingArguments(
        output_dir = str(tmp_path / f"{family}_{mode}"),
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 3,
        max_steps = steps,
        learning_rate = 1e-2,
        logging_steps = 1,
        report_to = [],
        save_strategy = "no",
        use_cpu = True,
        seed = 0,
        dataloader_num_workers = 0,
        optim = "adamw_torch",
        lr_scheduler_type = "constant",
        disable_tqdm = True,
        remove_unused_columns = False,
    )
    T = type("T", (transformers.Trainer,), {})
    trainer = T(model = model, args = args, train_dataset = _Text(), data_collator = collate)
    trainer.train()
    weights = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    return weights, losses


def _bitwise(base, run):
    (w0, l0), (w1, l1) = base, run
    assert l0 == l1 and len(l0) >= 2
    assert w0.keys() == w1.keys() and len(w0)
    for k in w0:
        assert torch.equal(w0[k], w1[k]), k


@pytest.mark.parametrize(
    "family, lora",
    [
        ("llama", True),
        ("qwen3", True),
        ("olmo", True),
        ("qwen3", False),
        ("gemma3_vision", True),
        ("qwen3_vl", True),
        ("gemma4", True),
        ("qwen3_5", True),
        ("qwen3_5", False),
    ],
)
def test_auto_engages_by_structure_bitwise(ga, tmp_path, monkeypatch, capsys, family, lora):
    """No class list: auto runs windows for every decoder whose layers map hidden states to hidden states (after its
    two measured steps: 2 windows in 4 steps), bitwise equal to standard accumulation, LoRA and full fine-tuning."""
    mod, engine = ga
    base = _run(mod, tmp_path, monkeypatch, family, "0", lora = lora)
    engine.layer_major_stats(reset = True)
    run = _run(mod, tmp_path, monkeypatch, family, None, lora = lora)
    out = capsys.readouterr().out
    assert _windows(engine) == 2, out
    assert engine.layer_major_stats()["aborts"] == 0, engine.layer_major_stats()["last_abort"]
    _bitwise(base, run)


def test_gemma3_vision_image_and_text_micro_batches_bitwise(ga, tmp_path, monkeypatch):
    """Gemma 3 with images through the real vision tower (its features enter before the first decoder layer): image
    and text micro-batches both run in windows, bitwise, with LoRA on the vision tower too."""
    mod, engine = ga
    targets = _TARGETS + ["fc1", "fc2", "out_proj"]
    base = _run(
        mod,
        tmp_path,
        monkeypatch,
        "gemma3_vision",
        "0",
        collate = _gemma3_image_collate,
        targets = targets,
    )
    engine.layer_major_stats(reset = True)
    run = _run(
        mod,
        tmp_path,
        monkeypatch,
        "gemma3_vision",
        "1",
        collate = _gemma3_image_collate,
        targets = targets,
    )
    assert _windows(engine) == 4 and engine.layer_major_stats()["aborts"] == 0
    assert any("vision_tower" in k for k in run[0])
    _bitwise(base, run)


def test_qwen3_vl_deepstack_images_keep_the_step_standard(ga, tmp_path, monkeypatch, capsys):
    """Qwen3-VL adds visual features to the hidden states after its first decoder layers (deepstack) when a
    micro-batch has images: those steps run the standard way with one reason (no aborted window), bitwise."""
    mod, engine = ga
    base = _run(mod, tmp_path, monkeypatch, "qwen3_vl", "0", collate = _qwen3_vl_image_collate)
    capsys.readouterr()
    engine.layer_major_stats(reset = True)
    run = _run(mod, tmp_path, monkeypatch, "qwen3_vl", "1", collate = _qwen3_vl_image_collate)
    out = capsys.readouterr().out
    assert _windows(engine) == 0 and engine.layer_major_stats()["aborts"] == 0
    assert (
        out.count("adds visual features to the hidden states between decoder layers (deepstack)")
        == 1
    )
    _bitwise(base, run)


@pytest.mark.parametrize("mode", [None, "1"])
def test_gemma4_kv_sharing_is_refused_with_a_reason(ga, tmp_path, monkeypatch, capsys, mode):
    mod, engine = ga
    base = _run(mod, tmp_path, monkeypatch, "gemma4_kv_shared", "0")
    capsys.readouterr()
    engine.layer_major_stats(reset = True)
    run = _run(mod, tmp_path, monkeypatch, "gemma4_kv_shared", mode)
    out = capsys.readouterr().out
    assert _windows(engine) == 0 and engine.layer_major_stats()["aborts"] == 0
    assert out.count("decoder layers share key / value states (num_kv_shared_layers = 2)") == 1
    _bitwise(base, run)


def test_gemma4_per_layer_input_feeding_trained_weights_is_refused(
    ga, tmp_path, monkeypatch, capsys
):
    """LoRA on Gemma 4's per-layer input projection: that input's gradient now trains a weight, which the window
    cannot give it, so the window stops with that reason and training continues the standard way, bitwise."""
    mod, engine = ga
    targets = _TARGETS + ["per_layer_model_projection"]
    base = _run(mod, tmp_path, monkeypatch, "gemma4", "0", targets = targets)
    capsys.readouterr()
    engine.layer_major_stats(reset = True)
    run = _run(mod, tmp_path, monkeypatch, "gemma4", "1", targets = targets)
    out = capsys.readouterr().out
    assert _windows(engine) == 0
    assert "a decoder layer input other than the hidden states needs gradients" in out
    _bitwise(base, run)


def test_structural_blocker_reasons(ga):
    mod, _ = ga
    model = _build("llama")
    assert mod._structural_blocker(model, _decoder_layers(model)) is None

    class NoHidden(torch.nn.Module):
        def forward(self, *, attention_mask = None):
            return attention_mask

    layers = torch.nn.ModuleList([NoHidden()])
    assert (
        mod._structural_blocker(model, layers)
        == "NoHidden.forward does not take the hidden states first"
    )
    assert (
        mod._structural_blocker(model, torch.nn.ModuleList())
        == "the block swap is not on a decoder layer list"
    )
    if hasattr(transformers, "Gemma4ForCausalLM"):
        shared = _build("gemma4_kv_shared")
        assert "num_kv_shared_layers = 2" in mod._structural_blocker(
            shared, _decoder_layers(shared)
        )


def test_unsloth_compiled_patches_inside_layers_stay_eligible(ga, monkeypatch):
    """A forward compiled onto a class by Unsloth's own temporary patches (Gemma 4's TextMLP fp16 clamp) does not
    refuse windows; a user's torch.compile on a layer's submodule still does."""
    import types

    mod, engine = ga
    stub = types.SimpleNamespace(args = types.SimpleNamespace(optim = "adamw_torch", n_gpu = 1))
    model = _build("llama")
    layers = _decoder_layers(model)
    mlp_cls = type(layers[0].mlp)

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    forward.__module__ = "unsloth_zoo.temporary_patches.llama_test"
    monkeypatch.setattr(mlp_cls, "forward", torch.compile(forward, backend = "eager"))
    assert mod._window_blocker(stub, model, layers) is None
    attn = layers[1].self_attn
    attn.forward = torch.compile(attn.forward, backend = "eager")
    assert "decoder layer" in mod._window_blocker(stub, model, layers)
    del attn.forward
    user = torch.compile(lambda self, x: forward(self, x), backend = "eager")
    monkeypatch.setattr(mlp_cls, "forward", user)
    assert "decoder layer" in mod._window_blocker(stub, model, layers)
