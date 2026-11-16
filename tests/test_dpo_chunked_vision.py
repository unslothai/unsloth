# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.

"""VLM DPO (TRL < 0.29): a pixel_values batch takes the chunked head and matches TRL's logits path on loss, metrics
and every gradient, vision tower included. CPU only."""

from __future__ import annotations

import copy
import sys
import textwrap
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
trl = pytest.importorskip("trl")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_dpo_chunked_logprobs as base  # noqa: E402

# TRL 0.22's processing path fails on the tiny Gemma 3 processor (no eos_token_id) and tiny Idefics3 takes minutes
# on CPU; their keys (token_type_ids, pixel_attention_mask) are covered by test_vision_keys_do_not_refuse_the_head.
MODELS = {
    "llava": "trl-internal-testing/tiny-LlavaForConditionalGeneration",
}


@pytest.fixture(scope = "module")
def env():
    import trl.import_utils as iu

    iu._mergekit_available = False
    import trl.trainer.dpo_trainer as dpo_mod

    if not hasattr(base._stock(dpo_mod, "DPOTrainer"), "concatenated_forward"):
        pytest.skip(reason = "TRL >= 0.29 has no concatenated_forward")
    ns = base._helpers()
    import inspect

    source = inspect.getsource(base._stock(dpo_mod, "DPOTrainer").concatenated_forward)
    patched = ns["dpo_trainer_chunked_logprobs"]("concatenated_forward", source)
    assert patched != source
    g = dict(vars(base._stock_module(dpo_mod)))
    g.update({k: v for k, v in ns.items() if not k.startswith("__")})
    exec(compile(textwrap.dedent(patched), "<patched concatenated_forward>", "exec"), g)
    return dpo_mod, g["concatenated_forward"]


def _images(n):
    from PIL import Image, ImageDraw

    out = []
    for i in range(n):
        img = Image.new("RGB", (32, 32), (255, 255, 255))
        ImageDraw.Draw(img).rectangle([8, 8, 24, 24], fill = [(220, 40, 40), (40, 80, 220)][i % 2])
        out.append(img)
    return out


def _dataset():
    from datasets import Dataset

    rows = []
    for i, img in enumerate(_images(4)):
        rows.append(
            {
                "images": [img],
                "prompt": [
                    {
                        "role": "user",
                        "content": [{"type": "image"}, {"type": "text", "text": f"Color {i}?"}],
                    }
                ],
                "chosen": [
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "It is red, a warm color."}],
                    }
                ],
                "rejected": [
                    {"role": "assistant", "content": [{"type": "text", "text": "Blue" * (1 + i)}]}
                ],
            }
        )
    return Dataset.from_list(rows)


def _load(name):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    repo = MODELS[name]
    try:
        model = AutoModelForImageTextToText.from_pretrained(repo, torch_dtype = torch.float32)
        proc = AutoProcessor.from_pretrained(repo)
    except Exception as e:  # noqa: BLE001
        pytest.skip(reason = f"{repo} does not load here: {type(e).__name__}: {e}"[:200])
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}
    return model, proc


def _run(dpo_mod, forward, name, chunked, cfg_kwargs):
    torch.manual_seed(0)
    model, proc = _load(name)
    ref_model = copy.deepcopy(model)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.02 * torch.randn_like(p))
    base._honour_hidden_states(model)
    base._honour_hidden_states(ref_model)
    cls = base._stock(dpo_mod, "DPOTrainer")
    if chunked:
        cls = type("PatchedDPOTrainer", (cls,), {"concatenated_forward": forward})
    cfg = dpo_mod.DPOConfig(
        output_dir = str(base.REPO / ".pytest_dpo_chunked_vision"),
        per_device_train_batch_size = 2,
        report_to = "none",
        use_cpu = True,
        max_length = None,
        max_prompt_length = None,
        remove_unused_columns = False,
        dataset_num_proc = None,
        **cfg_kwargs,
    )
    cfg.dataset_num_proc = None
    trainer = cls(
        model = model, ref_model = ref_model, args = cfg, train_dataset = _dataset(), processing_class = proc
    )
    batch = next(iter(trainer.get_train_dataloader()))
    assert "pixel_values" in batch, (trainer.is_vision_model, trainer.train_dataset.column_names)
    loss, metrics = trainer.get_batch_loss_metrics(trainer.model, batch, train_eval = "train")
    loss.backward()
    grads = {n: p.grad.clone() for n, p in trainer.model.named_parameters() if p.grad is not None}
    return trainer, loss.detach(), metrics, grads, sorted(batch)


@pytest.mark.parametrize("name", list(MODELS))
@pytest.mark.parametrize(
    "cfg",
    [{"loss_type": "sigmoid"}, {"loss_type": "sigmoid", "rpo_alpha": 0.5}],
    ids = ["sigmoid", "rpo"],
)
def test_vision_batch_takes_the_chunked_head_and_matches_trl(env, name, cfg):
    dpo_mod, forward = env
    _, base_loss, base_metrics, base_grads, _ = _run(dpo_mod, forward, name, False, cfg)
    trainer, loss, metrics, grads, keys = _run(dpo_mod, forward, name, True, cfg)
    assert getattr(trainer, "_unsloth_dpo_chunked_calls", 0) >= 1, (
        keys,
        trainer.__dict__.get("_unsloth_dpo_chunked_fallbacks"),
    )
    assert torch.allclose(loss, base_loss, atol = 1e-5, rtol = 1e-5), (loss, base_loss)
    assert metrics.keys() == base_metrics.keys()
    for k in base_metrics:
        assert abs(metrics[k] - base_metrics[k]) <= 1e-5 + 1e-5 * abs(base_metrics[k]), (
            k,
            metrics[k],
            base_metrics[k],
        )
    assert grads.keys() == base_grads.keys()
    vision = [
        n for n in grads if any(t in n for t in ("vision", "multi_modal_projector", "connector"))
    ]
    assert vision and any(grads[n].abs().sum() > 0 for n in vision)
    # Key-projection biases have an exactly zero gradient (softmax is shift invariant), so both sides hold
    # rounding noise there: compare against the parameter's own scale plus a floor at 1e-3 of the largest gradient.
    floor = 1e-3 * max(g.abs().max().item() for g in base_grads.values())
    rel = {
        n: (grads[n] - base_grads[n]).abs().max().item()
        / (base_grads[n].abs().max().item() + floor)
        for n in grads
    }
    worst = max(rel.values())
    assert worst < 1e-4, sorted(rel.items(), key = lambda kv: -kv[1])[:4]


@pytest.mark.parametrize(
    "keys",
    [
        ("pixel_values",),
        ("pixel_values", "pixel_attention_mask"),
        ("pixel_values", "image_sizes"),
        ("pixel_values", "token_type_ids"),
        ("pixel_values", "pixel_position_ids", "image_position_ids", "mm_token_type_ids"),
    ],
)
def test_vision_keys_do_not_refuse_the_head(keys):
    ns = base._helpers()
    head = torch.nn.Linear(8, 11, bias = False)

    class _M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = head
            self.config = type("C", (), {"model_type": "llava"})()

        def get_output_embeddings(self):
            return self.lm_head

    class _T:
        accelerator = type("A", (), {"unwrap_model": staticmethod(lambda m: m)})()

        def get_batch_logps(self, *a, **k):
            return None

    trainer = _T()
    # The stock-hook probe compares against TRL's own method; let it pass for this bare object.
    ns["_unsloth_dpo_chunked_head"].__globals__["_unsloth_stock_logps_hook"] = lambda *a, **k: True
    batch = {
        "prompt_input_ids": torch.ones(1, 2, dtype = torch.long),
        **{k: torch.ones(1) for k in keys},
    }
    out = ns["_unsloth_dpo_chunked_head"](trainer, _M(), batch)
    assert out is not None, trainer.__dict__.get("_unsloth_dpo_chunked_fallbacks")
    assert out[1] is head
