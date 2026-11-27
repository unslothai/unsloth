# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Vision preference trainers encode each distinct image once: DPO repeats every image for chosen and rejected."""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

import pytest


RL_PATH = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "rl.py"


def _lift():
    tree = ast.parse(RL_PATH.read_text(encoding = "utf-8"))
    nodes = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name == "_image_dedup_post")
        or (
            isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "_IMAGE_DEDUP_TRAINERS" for t in node.targets
            )
        )
    ]
    assert len(nodes) == 2, f"_image_dedup_post / _IMAGE_DEDUP_TRAINERS missing from {RL_PATH}"
    namespace = {}
    exec(compile(ast.Module(body = nodes, type_ignores = []), str(RL_PATH), "exec"), namespace)
    return namespace


def _run_post(code, model, ref_model):
    trainer = types.SimpleNamespace(model = model, ref_model = ref_model)

    def body(self):
        exec(code, {}, {"self": self})

    body(trainer)


def test_trainers_that_repeat_images_get_dedup():
    ns = _lift()
    post = ns["_image_dedup_post"]
    for trainer_file in ("grpo_trainer", "rloo_trainer", "dpo_trainer", "online_dpo_trainer"):
        assert "install_image_dedup" in post(trainer_file), trainer_file
    # One image per forward: nothing to deduplicate.
    for trainer_file in (
        "kto_trainer",
        "orpo_trainer",
        "cpo_trainer",
        "sft_trainer",
        "bco_trainer",
        "gkd_trainer",
    ):
        assert post(trainer_file) == "", trainer_file
    assert "ref_model" not in post("grpo_trainer")
    for trainer_file in ("rloo_trainer", "dpo_trainer", "online_dpo_trainer"):
        assert "ref_model" in post(trainer_file)
    for trainer_file in ns["_IMAGE_DEDUP_TRAINERS"]:
        compile(post(trainer_file), trainer_file, "exec")


def test_post_installs_on_policy_and_reference(monkeypatch):
    post = _lift()["_image_dedup_post"]
    calls = []
    fake = types.ModuleType("unsloth_zoo.vision_dedup")
    fake.install_image_dedup = lambda m: calls.append(m) or 0
    monkeypatch.setitem(sys.modules, "unsloth_zoo.vision_dedup", fake)
    policy, ref = object(), object()
    _run_post(post("dpo_trainer"), policy, ref)
    assert calls == [policy, ref]
    calls.clear()
    _run_post(post("grpo_trainer"), policy, ref)
    assert calls == [policy]


def test_post_is_a_no_op_without_zoo_support(monkeypatch):
    post = _lift()["_image_dedup_post"]
    # An unsloth_zoo without vision_dedup: the import fails and the trainer still builds.
    monkeypatch.setitem(sys.modules, "unsloth_zoo.vision_dedup", None)
    _run_post(post("dpo_trainer"), object(), None)


# Real TRL batches on tiny VLMs: the installed TRL builds the repeated images, the dedup must undo it exactly.
MODELS = {
    "gemma3": "trl-internal-testing/tiny-Gemma3ForConditionalGeneration",
    "qwen2_5_vl": "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
}
N = 3


def _images():
    from PIL import Image, ImageDraw

    out = []
    for p in range(N):
        img = Image.new("RGB", (64, 64), (255, 40 + 70 * p, 200 - 50 * p))
        ImageDraw.Draw(img).text((4, 20), f"{p + 2}+{p + 5}", fill = (0, 0, 0))
        out.append(img)
    return out


def _dpo_batch(key, proc):
    """(model kwargs, completion mask) for one DPO forward, built by the installed TRL."""
    import torch
    from trl.trainer import dpo_trainer

    tok = getattr(proc, "tokenizer", proc)
    if hasattr(dpo_trainer, "DataCollatorForVisionPreference"):
        examples = [
            {
                "prompt": [
                    {
                        "role": "user",
                        "content": [{"type": "image"}, {"type": "text", "text": f"Sum {p}?"}],
                    }
                ],
                "chosen": [
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": f"It is {2 * p + 7}."}],
                    }
                ],
                "rejected": [
                    {"role": "assistant", "content": [{"type": "text", "text": "No idea at all."}]}
                ],
                "images": [img],
            }
            for p, img in enumerate(_images())
        ]
        batch = dpo_trainer.DataCollatorForVisionPreference(processor = proc)(examples)
        mask = batch.pop("completion_mask")
        return dict(batch), mask
    # TRL < 0.24: process_row keeps one pixel_values entry per prompt, concatenated_inputs repeats it.
    if key != "gemma3":
        pytest.skip(
            "this TRL's vision DPO keeps only the first patch row of grid-layout processors"
        )
    rows = []
    for p, img in enumerate(_images()):
        msgs = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": f"Sum {p}?"}]}
        ]
        prompt = proc.apply_chat_template(msgs, add_generation_prompt = True)
        row = dpo_trainer.DPOTrainer.process_row(
            {
                "prompt": prompt,
                "images": [img],
                "chosen": f"It is {2 * p + 7}.",
                "rejected": "No idea at all.",
            },
            proc,
            add_special_tokens = False,
        )
        rows.append(row)
    collated = dpo_trainer.DataCollatorForPreference(pad_token_id = tok.pad_token_id)(rows)
    cat = dpo_trainer.DPOTrainer.concatenated_inputs(collated, padding_value = tok.pad_token_id)
    input_ids = torch.cat((cat["prompt_input_ids"], cat["completion_input_ids"]), dim = 1)
    attention_mask = torch.cat(
        (cat["prompt_attention_mask"], cat["completion_attention_mask"]), dim = 1
    )
    mask = torch.cat(
        (torch.zeros_like(cat["prompt_attention_mask"]), cat["completion_attention_mask"]), dim = 1
    )
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": cat["pixel_values"],
    }
    if "token_type_ids" in collated:
        kwargs["token_type_ids"] = torch.cat(
            (collated["token_type_ids"], collated["token_type_ids"]), dim = 0
        )
    elif hasattr(proc, "image_token_id") or getattr(proc, "image_token", None):
        image_id = getattr(proc, "image_token_id", None)
        if image_id is None:
            image_id = tok.convert_tokens_to_ids(proc.image_token)
        kwargs["token_type_ids"] = (input_ids == image_id).long()
    return kwargs, mask


def _load(key):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    try:
        proc = AutoProcessor.from_pretrained(MODELS[key])
        model = AutoModelForImageTextToText.from_pretrained(MODELS[key], dtype = torch.float32)
    except Exception as e:  # offline without the cached tiny repo
        pytest.skip(f"{MODELS[key]} unavailable: {e}")
    return proc, model


def _tower(model):
    for name, module in model.named_modules():
        if "language_model" in name:
            continue
        if name.endswith("vision_tower") or name.endswith("visual"):
            return module
    raise AssertionError("no vision tower")


def _logps(model, kwargs, mask):
    import torch

    logits = model(**kwargs, use_cache = False).logits[:, :-1].float()
    target = kwargs["input_ids"][:, 1:]
    lp = torch.log_softmax(logits, -1).gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return (lp * mask[:, 1:]).sum(-1)


def _dpo_step(
    model,
    kwargs,
    mask,
    tower,
    beta = 0.1,
):
    """Policy forward + adapter-off reference forward, as DPOTrainer does on a PEFT model."""
    import torch
    import torch.nn.functional as F

    rows = []
    handle = tower.register_forward_pre_hook(
        lambda m, a, k: rows.append(int((a[0] if a else next(iter(k.values()))).shape[0])),
        with_kwargs = True,
    )
    try:
        model.zero_grad(set_to_none = True)
        logps = _logps(model, kwargs, mask)
        with torch.no_grad(), model.disable_adapter():
            ref = _logps(model, kwargs, mask)
        n = logps.shape[0] // 2
        margin = (logps[:n] - ref[:n]) - (logps[n:] - ref[n:])
        loss = -F.logsigmoid(beta * margin).mean()
        loss.backward()
    finally:
        handle.remove()
    grads = {k: p.grad.detach().clone() for k, p in model.named_parameters() if p.grad is not None}
    return loss.detach(), grads, rows


@pytest.mark.parametrize("key", sorted(MODELS))
def test_dpo_vision_step_encodes_each_image_once(key, monkeypatch):
    import torch

    pytest.importorskip("trl")
    peft = pytest.importorskip("peft")
    vision_dedup = pytest.importorskip("unsloth_zoo.vision_dedup")

    torch.manual_seed(0)
    proc, model = _load(key)
    kwargs, mask = _dpo_batch(key, proc)
    model = peft.get_peft_model(
        model,
        peft.LoraConfig(
            r = 4,
            lora_alpha = 8,
            target_modules = r".*\.(q_proj|v_proj|qkv|fc1)$",
            init_lora_weights = False,
        ),
    )
    model.train()
    tower = _tower(model)
    assert any(
        "lora_" in n for n, _ in tower.named_parameters()
    ), "LoRA must reach the vision tower"

    # What rl.py appends to DPOTrainer.__init__: no separate ref_model on a PEFT policy.
    _run_post(_lift()["_image_dedup_post"]("dpo_trainer"), model, None)

    # First use self-checks the mapping against a full encode; measure after it.
    monkeypatch.setenv("UNSLOTH_GRPO_IMAGE_DEDUP", "1")
    _dpo_step(model, kwargs, mask, tower)
    assert vision_dedup.DEDUP_STATS["fallbacks"] == 0, vision_dedup.DEDUP_STATS["last_fallback"]

    monkeypatch.setenv("UNSLOTH_GRPO_IMAGE_DEDUP", "0")
    loss_off, grads_off, rows_off = _dpo_step(model, kwargs, mask, tower)
    monkeypatch.setenv("UNSLOTH_GRPO_IMAGE_DEDUP", "1")
    before = dict(vision_dedup.DEDUP_STATS)
    loss_on, grads_on, rows_on = _dpo_step(model, kwargs, mask, tower)
    stats = {
        k: vision_dedup.DEDUP_STATS[k] - before[k]
        for k in ("images_in", "images_encoded", "deduped_calls", "fallbacks")
    }

    # Policy + reference forward: off, the tower sees all 2N repeated images each time; on, the N distinct ones.
    assert stats == {
        "images_in": 2 * 2 * N,
        "images_encoded": 2 * N,
        "deduped_calls": 2,
        "fallbacks": 0,
    }, stats
    assert len(rows_off) == len(rows_on) == 2, (rows_off, rows_on)
    assert [2 * r for r in rows_on] == rows_off, (rows_off, rows_on)

    torch.testing.assert_close(loss_on, loss_off, rtol = 1e-5, atol = 1e-6)
    assert grads_on.keys() == grads_off.keys()
    assert any("vision" in k or "visual" in k for k in grads_on)
    for name in grads_off:
        torch.testing.assert_close(grads_on[name], grads_off[name], rtol = 1e-4, atol = 1e-6, msg = name)
