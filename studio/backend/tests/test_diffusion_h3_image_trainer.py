# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3-Image LoRA training on CPU with the tiny random native release: the 6-step objective, the trainer end
to end (loss falls on a fixed picture, adapter saved with the trunk keys only), the adapter merged into the
inference load (keys match, output changes, weight 0 is the base), routing, family facts and the refusals."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
for p in (_BACKEND, Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _stack():
    torch = pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    pytest.importorskip("transformers")
    pytest.importorskip("accelerate")
    pytest.importorskip("peft")
    return torch


@pytest.fixture(scope = "module")
def release(tmp_path_factory):
    _stack()
    import _minimax_h3_image_tiny as tiny

    root = tmp_path_factory.mktemp("h3img_train_release")
    ref = tiny.make_tiny_release(root)
    te = tiny.make_text_encoder(ref["tokenizer"])
    return root, ref, te


@pytest.fixture(scope = "module")
def dataset(tmp_path_factory):
    _stack()
    import numpy as np
    from PIL import Image

    d = tmp_path_factory.mktemp("h3img_data")
    # One picture, one caption: a fixed batch the loss must fit.
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:96, 0:64]
    pic = np.stack([(xx * 4) % 256, (yy * 2) % 256, ((xx + yy) * 3) % 256], -1).astype(np.uint8)
    pic = np.clip(pic.astype(np.int16) + rng.integers(0, 8, pic.shape), 0, 255).astype(np.uint8)
    Image.fromarray(pic).save(d / "a.png")
    (d / "a.txt").write_text("a cat")
    return d


def _train(
    release,
    dataset,
    out,
    monkeypatch,
    steps = 24,
    **extra,
):
    from core.training import diffusion_h3_image_trainer as trainer
    from core.training.diffusion_train_common import DiffusionLoraConfig, get_trainer

    root, ref, te = release
    monkeypatch.setattr(
        trainer,
        "load_text_encoder",
        lambda cfg, r: (te, ref["tokenizer"], ref["processor"]),
    )
    # The tiny release shares the encoder across runs: the trainer frees its storage after encoding.
    monkeypatch.setattr(trainer, "release_text_encoder", lambda m: None)
    monkeypatch.setattr(trainer, "resolve_train_device", lambda: "cpu")
    # the real training adapter is for the 33B DiT, not the tiny one
    monkeypatch.setenv(trainer.TRAINING_ADAPTER_ENV, "off")
    events = []
    cfg = DiffusionLoraConfig(
        base_model = str(root),
        data_dir = str(dataset),
        output_dir = str(out),
        model_family = "minimax-h3-image",
        resolution = 64,
        train_steps = steps,
        learning_rate = extra.pop("learning_rate", 3e-3),
        lora_rank = 4,
        lr_scheduler = "constant",
        seed = 0,
        **extra,
    )
    assert get_trainer("minimax-h3-image") is trainer.run_h3_image_lora_training
    trainer.run_h3_image_lora_training(cfg, on_event = events.append)
    return events


def test_six_step_objective_uses_one_head_per_sampler_step():
    _stack()
    from core.inference import diffusion_minimax_h3_image as h3i
    from core.training.diffusion_h3_image_trainer import training_sigmas

    steps = training_sigmas()
    assert [k for _s, _t, k in steps] == list(range(6))
    table = h3i.table_timesteps()
    for sigma, t, _k in steps:
        # every training timestep is a row of the AdaLN table, so the table form trains exactly
        assert t in table
        assert abs((1.0 - sigma) - t) < 1e-6
    assert steps[0][0] == pytest.approx(1.0)


def test_trainer_end_to_end_loss_falls_and_saves_trunk_keys(
    release, dataset, tmp_path, monkeypatch
):
    torch = _stack()
    from safetensors import safe_open

    catalog = tmp_path / "catalog"
    catalog.mkdir()
    monkeypatch.setattr("core.inference.diffusion_lora.loras_dir", lambda: catalog)
    events = _train(release, dataset, tmp_path / "out", monkeypatch, steps = 30)
    progress = [e for e in events if e["type"] == "progress"]
    assert len(progress) == 30
    losses = [e["loss"] for e in progress]
    assert all(torch.isfinite(torch.tensor(losses)))
    first, last = sum(losses[:6]) / 6, sum(losses[-6:]) / 6
    assert last < first, (first, last)
    done = [e for e in events if e["type"] == "complete"][-1]
    path = Path(done["lora_path"])
    assert path.is_file()
    # published to the Images LoRA catalog, which Deploy, Export and the picker read
    assert done["catalog_path"] and Path(done["catalog_path"]).parent == catalog
    assert Path(done["catalog_path"]).read_bytes() == path.read_bytes()
    with safe_open(str(path), framework = "pt") as f:
        keys = list(f.keys())
        meta = f.metadata()
    assert keys and all(k.startswith("transformer.transformer_blocks.") for k in keys)
    leaves = {k.split(".", 3)[3].rsplit(".lora_", 1)[0] for k in keys}
    assert leaves == {
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
        "attn.to_out.0",
        "ff.net.0.proj",
        "ff.net.2",
    }
    adapter = json.loads(meta["lora_adapter_metadata"])
    assert adapter["transformer.r"] == 4


def test_trained_adapter_merges_into_the_inference_load_and_changes_the_output(
    release, dataset, tmp_path, monkeypatch
):
    torch = _stack()
    from core.inference import diffusion_minimax_h3_image as h3i

    monkeypatch.setenv("UNSLOTH_H3_IMAGE_SOURCE_RESIZE", "0")
    _train(release, dataset, tmp_path / "out", monkeypatch, steps = 20)
    lora = tmp_path / "out" / "pytorch_lora_weights.safetensors"
    root, _ref, te = release

    def render(loras = ()):
        pipe = h3i.load_minimax_h3_image_pipeline(
            str(root), device = "cpu", text_encoder = te, loras = loras
        )
        gen = torch.Generator().manual_seed(7)
        out = pipe("a cat", height = 64, width = 64, generator = gen, output_type = "latent")
        return pipe, out.images[0]

    _base_pipe, base = render()
    pipe, with_lora = render([(str(lora), 1.0)])
    _p0, zero = render([(str(lora), 0.0)])
    # every trained layer found its Linear: 2 blocks x 6 projections
    assert pipe._unsloth_h3_lora_layers == 12
    assert not torch.equal(base, with_lora)
    assert torch.equal(base, zero)
    assert (with_lora - base).abs().max() > 1e-4


def test_merge_matches_peft_on_the_dense_layer():
    torch = _stack()
    from core.inference import diffusion_minimax_h3_image as h3i

    w = torch.randn(8, 6, dtype = torch.bfloat16)
    a, b = torch.randn(2, 6), torch.randn(8, 2)
    state = {"transformer_blocks.0.attn.to_q.weight": w.clone(), "other.weight": torch.zeros(3)}
    merged = h3i.merge_lora_deltas_(state, {"transformer_blocks.0.attn.to_q": [(a, b, 0.5)]})
    assert merged == {"transformer_blocks.0.attn.to_q"}
    want = (w.float() + 0.5 * b @ a).to(torch.bfloat16)
    assert torch.equal(state["transformer_blocks.0.attn.to_q.weight"], want)


def test_lora_reader_scales_by_alpha_over_rank(tmp_path):
    torch = _stack()
    from safetensors.torch import save_file

    from core.inference import diffusion_minimax_h3_image as h3i

    a, b = torch.randn(4, 6), torch.randn(8, 4)
    meta = {"lora_adapter_metadata": json.dumps({"transformer.r": 4, "transformer.lora_alpha": 8})}
    path = tmp_path / "x.safetensors"
    save_file(
        {
            "transformer.transformer_blocks.1.ff.net.2.lora_A.weight": a,
            "transformer.transformer_blocks.1.ff.net.2.lora_B.weight": b,
        },
        str(path),
        metadata = meta,
    )
    deltas = h3i.read_lora_deltas([(str(path), 0.5)])
    ((ra, rb, scale),) = deltas["transformer_blocks.1.ff.net.2"]
    assert torch.equal(ra, a) and torch.equal(rb, b)
    assert scale == pytest.approx(8 / 4 * 0.5)
    # kohya-style names with an alpha key
    path2 = tmp_path / "k.safetensors"
    save_file(
        {
            "transformer_blocks.0.attn.to_k.lora_down.weight": a,
            "transformer_blocks.0.attn.to_k.lora_up.weight": b,
            "transformer_blocks.0.attn.to_k.alpha": torch.tensor(2.0),
        },
        str(path2),
    )
    ((_a, _b, s2),) = h3i.read_lora_deltas([(str(path2), 1.0)])["transformer_blocks.0.attn.to_k"]
    assert s2 == pytest.approx(0.5)


def test_a_lora_for_another_model_is_refused_at_load(release, tmp_path):
    torch = _stack()
    from safetensors.torch import save_file

    from core.inference import diffusion_minimax_h3_image as h3i

    root, _ref, te = release
    path = tmp_path / "other.safetensors"
    save_file(
        {
            "transformer.single_transformer_blocks.0.proj_out.lora_A.weight": torch.zeros(2, 4),
            "transformer.single_transformer_blocks.0.proj_out.lora_B.weight": torch.zeros(4, 2),
        },
        str(path),
    )
    with pytest.raises(ValueError, match = "name no Linear"):
        h3i.load_minimax_h3_image_pipeline(
            str(root), device = "cpu", text_encoder = te, loras = [(str(path), 1.0)]
        )


def test_family_is_trainable_with_its_own_trainer_and_facts():
    _stack()
    from core.inference.diffusion_families import detect_family
    from core.inference.diffusion_transformer_quant import _family_train_denied
    from core.training.diffusion_train_common import (
        DiffusionLoraConfig,
        FAMILY_TRAIN_DEFAULTS,
        train_recipe_overrides,
    )

    fam = detect_family("MiniMaxAI/MiniMax-H3-Image")
    assert fam.name == "minimax-h3-image" and fam.trainable
    assert fam.train_base_repos == ("MiniMaxAI/MiniMax-H3-Image",)
    assert FAMILY_TRAIN_DEFAULTS["minimax-h3-image"]["resolution"] % 32 == 0
    for scheme in ("nf4", "fp8", "mxfp8"):
        assert _family_train_denied("minimax-h3-image", scheme)
    for scheme in ("bf16", "int8"):
        assert not _family_train_denied("minimax-h3-image", scheme)
    cfg = DiffusionLoraConfig(
        base_model = "MiniMaxAI/MiniMax-H3-Image",
        data_dir = "/tmp/d",
        output_dir = "/tmp/o",
        instance_prompt = "p",
    ).normalized()
    assert cfg.resolved_family == "minimax-h3-image"
    assert train_recipe_overrides(cfg)["base_precision"] == "auto"


@pytest.mark.parametrize(
    "field, value, match",
    [
        ("mixed_precision", "fp16", "bf16"),
        ("resolution", 1000, "multiples of 32"),
        ("cfg_dropout", 0.1, "guidance-distilled"),
        ("train_batch_size", 2, "batch size 1"),
        ("weighting_scheme", "bell", "weighting"),
        ("base_precision", "fp8", "fp8"),
    ],
)
def test_refusals(field, value, match):
    _stack()
    from core.training.diffusion_train_common import (
        DiffusionLoraConfig,
        h3_train_unsupported_reason,
    )

    kwargs = dict(
        base_model = "MiniMaxAI/MiniMax-H3-Image",
        data_dir = "/tmp/d",
        output_dir = "/tmp/o",
        instance_prompt = "p",
    )
    kwargs[field] = value
    try:
        cfg = DiffusionLoraConfig(**kwargs).normalized()
    except ValueError as exc:
        assert match.split()[0].lower() in str(exc).lower()
        return
    reason = h3_train_unsupported_reason(cfg)
    assert reason and match.lower() in reason.lower()


def test_auto_precision_follows_free_memory():
    from core.training.diffusion_h3_image_trainer import resolve_h3_image_base_precision as pick

    assert pick("auto", 79.0) == "bf16"
    assert pick("auto", 39.0) == "int8"
    assert pick("auto", None) == "int8"
    assert pick("int8", 200.0) == "int8"
    with pytest.raises(ValueError):
        pick("fp8", 80.0)
    # no functional int8 (ROCm, torchao stub): bf16 when it fits, else a clear refusal, never a dense "int8" base
    assert pick("auto", 79.0, int8_ok = False) == "bf16"
    with pytest.raises(ValueError, match = "torchao"):
        pick("auto", 39.0, int8_ok = False)
    with pytest.raises(ValueError, match = "torchao"):
        pick("int8", 200.0, int8_ok = False)


def test_backend_runs_only_the_load_time_lora_set_for_h3_image():
    _stack()
    import threading
    import types

    from core.inference.diffusion import DiffusionBackend

    baked = (("style", "/x/style.safetensors", 1.0),)
    fam = types.SimpleNamespace(name = "minimax-h3-image")
    pipe = types.SimpleNamespace(_unsloth_loras = baked)
    state = types.SimpleNamespace(pipe = pipe, family = fam, hf_token = None)
    backend = types.SimpleNamespace(
        _resolve_lora_set = lambda specs, **kw: tuple(
            ("style", "/x/style.safetensors", w) for _i, w in specs
        ),
    )
    backend._check_h3_image_loras = types.MethodType(
        DiffusionBackend._check_h3_image_loras, backend
    )
    apply = types.MethodType(DiffusionBackend._apply_loras, backend)
    cancel = threading.Event()
    apply(state, [("style", 1.0)], cancel)  # the baked set: a no-op
    with pytest.raises(ValueError, match = "Load the model again"):
        apply(state, [("style", 0.5)], cancel)
    with pytest.raises(ValueError, match = "Load the model again"):
        apply(state, [], cancel)
    pipe._unsloth_loras = ()
    apply(state, [], cancel)  # nothing baked, nothing asked


def test_sd_cpp_says_h3_image_loras_need_the_diffusers_engine():
    from core.inference.diffusion_lora import lora_unavailable_reason

    msg = lora_unavailable_reason(
        engine = "sd_cpp", family = "minimax-h3-image", model_kind = "gguf", transformer_quant = None
    )
    assert msg and "diffusers engine" in msg
    assert (
        lora_unavailable_reason(
            engine = "diffusers",
            family = "minimax-h3-image",
            model_kind = "pipeline",
            transformer_quant = "int8",
        )
        is None
    )


def test_training_target_is_the_samplers_velocity_sign():
    """An Euler step of the inference scheduler fed the training target (x0 - noise) lands exactly on the next noise
    level of the same (x0, noise) pair, and the x0 estimate is exact; the negated output moves away from x0."""
    torch = _stack()
    from core.inference import diffusion_minimax_h3_image as h3i
    from core.training.diffusion_h3_image_trainer import training_sigmas

    g = torch.Generator().manual_seed(0)
    x0, noise = torch.randn(64, 96, generator = g), torch.randn(64, 96, generator = g)
    sched = h3i.MiniMaxH3ImageScheduler(sample_sigmas = list(h3i.H3_IMAGE_SIGMAS))
    sched.set_timesteps(6, device = "cpu")
    for k, (sigma, t, head) in enumerate(training_sigmas()):
        sched._step_index = k
        x_t = (1 - sigma) * x0 + sigma * noise
        target = x0 - noise
        nxt = float(sched.sigmas[k + 1])
        stepped = sched.step(target, t, x_t)[0]
        assert torch.allclose(stepped, (1 - nxt) * x0 + nxt * noise, atol = 1e-5)
        # x0 estimate the sampler forms (x - out * sigma with out = -model_output) is exact
        assert torch.allclose(x_t + sigma * target, x0, atol = 1e-5)
        sched._step_index = k
        wrong = sched.step(-target, t, x_t)[0]
        if nxt < sigma:
            assert (wrong - x0).norm() > (stepped - x0).norm()


def test_training_sigmas_are_the_samplers_not_the_nominal_grid():
    _stack()
    from core.inference import diffusion_minimax_h3_image as h3i
    from core.training.diffusion_h3_image_trainer import training_sigmas

    used = [s for s, _t, _k in training_sigmas()]
    nominal = list(h3i.H3_IMAGE_SIGMAS[:-1])
    assert [round(float(s), 6) for s in h3i.reference_sigmas(6)[:-1]] == [round(s, 6) for s in used]
    # the sampler's sigmas sit within 1e-3 of the published grid, each nearest its own head
    for k, (u, n) in enumerate(zip(used, nominal)):
        assert abs(u - n) < 1e-3
        assert min(range(6), key = lambda j: abs(nominal[j] - u)) == k


def test_native_ai_toolkit_lora_names_merge_like_the_converted_weights():
    """A LoRA on MiniMax's native modules (fused interleaved qkv, [gate; value] fc1) merged after conversion equals the
    converter applied to the native weight with the LoRA merged first."""
    torch = _stack()
    from core.inference import diffusion_minimax_h3_image as h3i

    heads, hd, hidden, ffn, r = 2, 4, 8, 12, 3
    g = torch.Generator().manual_seed(1)
    w_qkv = torch.randn(3 * heads * hd, hidden, generator = g)
    w_fc1 = torch.randn(2 * ffn, hidden, generator = g)
    lora = {
        "blocks.0.attn.qkv_proj": [
            (torch.randn(r, hidden, generator = g), torch.randn(3 * heads * hd, r, generator = g), 0.7)
        ],
        "blocks.0.mlp.fc1": [
            (torch.randn(r, hidden, generator = g), torch.randn(2 * ffn, r, generator = g), 0.5)
        ],
        "token_refiner.blocks.1.attn.out_proj": [
            (torch.randn(r, heads * hd, generator = g), torch.randn(hidden, r, generator = g), 1.0)
        ],
    }
    native = {"blocks.0.attn.qkv_proj.weight": w_qkv, "blocks.0.mlp.fc1.weight": w_fc1}
    want = {}
    for key, w in native.items():
        a, b, s = lora[key[: -len(".weight")]][0]
        want.update(dict(h3i.convert_native_transformer_tensor(key, w + s * b @ a, heads, hd)))
    state = {}
    for key, w in native.items():
        state.update(dict(h3i.convert_native_transformer_tensor(key, w, heads, hd)))
    deltas = h3i.native_lora_deltas(lora, heads, hd)
    assert "token_refiner.refiner_blocks.1.attn.to_out.0" in deltas
    h3i.merge_lora_deltas_(state, deltas)
    assert set(state) == set(want)
    for key in want:
        assert torch.allclose(state[key], want[key], atol = 1e-5), key


def test_training_adapter_auto_falls_back_and_explicit_values(tmp_path, monkeypatch):
    _stack()
    import huggingface_hub

    from core.training import diffusion_h3_image_trainer as trainer

    seen = []

    def boom(*a, **k):
        seen.append(k.get("revision"))
        raise OSError("offline")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", boom)
    notes = []
    monkeypatch.delenv(trainer.TRAINING_ADAPTER_ENV, raising = False)
    assert trainer.training_adapter_path(None, notes.append) is None and notes
    # the fetch is pinned to one commit of the third-party repo, never its moving main
    assert seen == [trainer.OSTRIS_ADAPTER_REVISION] and len(trainer.OSTRIS_ADAPTER_REVISION) == 40
    monkeypatch.setenv(trainer.TRAINING_ADAPTER_ENV, "ostris")
    with pytest.raises(OSError):
        trainer.training_adapter_path(None)
    monkeypatch.setenv(trainer.TRAINING_ADAPTER_ENV, "off")
    assert trainer.training_adapter_path(None) is None
    f = tmp_path / "a.safetensors"
    f.write_bytes(b"x")
    monkeypatch.setenv(trainer.TRAINING_ADAPTER_ENV, str(f))
    assert trainer.training_adapter_path(None) == str(f)


def test_x0_error_eval_covers_all_six_heads_and_grads_reach_the_adapter(
    release, dataset, tmp_path, monkeypatch
):
    """The diagnostic the GPU sweep reads (UNSLOTH_H3_IMAGE_TRAIN_EVAL=1): the sampler's x0-estimate error from the
    MODEL output per head, before / after; every head is trained, and the adapter gets gradient through checkpointing.
    (The tiny random DiT's gates are ~0.05, so its error barely moves; the real reduction is measured on GPU.)"""
    torch = _stack()
    from core.training import diffusion_h3_image_trainer as trainer

    monkeypatch.setenv(trainer.EVAL_ENV, "1")
    events = _train(release, dataset, tmp_path / "out", monkeypatch, steps = 40)
    ev = {e["tag"]: e["x0_error_by_head"] for e in events if e["type"] == "eval"}
    assert set(ev["before"]) == set(ev["after"]) == {str(k) for k in range(6)}
    assert all(torch.isfinite(torch.tensor(list(v.values()))).all() for v in ev.values())
    progress = [e for e in events if e["type"] == "progress"]
    assert set(progress[-1]["head_losses"]) == {str(k) for k in range(6)}
    assert all(e["grad_norm"] and e["grad_norm"] > 0 for e in progress)
