# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""VLM DPO on TRL < 0.29 with a grid-layout processor (Qwen2-VL, Qwen2.5-VL, Qwen3-VL).

These processors return every image's patch rows flattened into one (patches, dim) array plus one
image_grid_thw row per image. TRL's legacy row format keeps pixel_values[0] (a single patch row) and drops
image_grid_thw, so the vision tower fails. Unsloth's rewrites keep every row and carry image_grid_thw through
the collator, concatenated_inputs and the forward. CPU only.
"""

from __future__ import annotations

import ast
import copy
import inspect
import sys
import textwrap
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
trl = pytest.importorskip("trl")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_dpo_chunked_logprobs as base  # noqa: E402

RL_PATH = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "rl_replacements.py"
MODELS = {
    "qwen2_vl": "trl-internal-testing/tiny-Qwen2VLForConditionalGeneration",
    "qwen2_5_vl": "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
}
LIFTED = (
    "dpo_trainer_vision_process_row",
    "dpo_trainer_prepare_dataset",
    "dpo_trainer_vision_signature_columns",
    "dpo_trainer_concatenated_inputs",
    "_dpo_trainer_extend_vision_model_kwargs",
    "dpo_trainer_concatenated_forward",
    "dpo_trainer_data_collator_vision_keys",
)


def _lift():
    src = RL_PATH.read_text(encoding = "utf-8")
    nodes = [
        n
        for n in ast.parse(src).body
        if (isinstance(n, ast.FunctionDef) and n.name in LIFTED)
        or (
            isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "_DPO_VISION_KEYS" for t in n.targets)
        )
    ]
    ns = {"torch": torch}
    exec(compile(ast.Module(body = nodes, type_ignores = []), str(RL_PATH), "exec"), ns)
    return ns


def _legacy_module():
    import trl.import_utils as iu

    iu._mergekit_available = False
    import trl.trainer.dpo_trainer as dpo_mod

    return base._stock_module(dpo_mod)


def _patched_trainer_class(mod):
    """TRL's legacy DPOTrainer with the rewrites rl.py applies, on a private copy of TRL's module."""
    ns = _lift()
    g = dict(vars(mod))
    g.update({k: v for k, v in ns.items() if not k.startswith("__")})
    cls = mod.DPOTrainer
    attrs = {}
    for name in (
        "_prepare_dataset",
        "_set_signature_columns_if_needed",
        "concatenated_inputs",
        "concatenated_forward",
    ):
        source = inspect.getsource(getattr(cls, name))
        for rewrite in (
            "dpo_trainer_prepare_dataset",
            "dpo_trainer_vision_signature_columns",
            "dpo_trainer_concatenated_inputs",
            "dpo_trainer_concatenated_forward",
        ):
            source = ns[rewrite](name, source)
        exec(compile(textwrap.dedent(source), f"<patched {name}>", "exec"), g)
        attrs[name] = g[name]
    collator = ns["dpo_trainer_data_collator_vision_keys"]("data_collator", "")
    collator = collator.replace(
        "from trl.trainer.dpo_trainer import DataCollatorForPreference\n", ""
    )
    exec(compile(collator, "<patched collator>", "exec"), g)
    return type("PatchedDPOTrainer", (cls,), attrs)


def _dataset():
    from datasets import Dataset
    from PIL import Image, ImageDraw

    rows = []
    for i in range(4):
        # Two image sizes: rows carry different patch counts.
        img = Image.new("RGB", (56 + 28 * (i % 2), 56), (255, 255, 255))
        ImageDraw.Draw(img).rectangle([8, 8, 40, 40], fill = [(220, 40, 40), (40, 80, 220)][i % 2])
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
                    {"role": "assistant", "content": [{"type": "text", "text": "It is red."}]}
                ],
                "rejected": [
                    {"role": "assistant", "content": [{"type": "text", "text": "Blue" * (1 + i)}]}
                ],
            }
        )
    return Dataset.from_list(rows)


def _load(repo):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    try:
        model = AutoModelForImageTextToText.from_pretrained(repo, torch_dtype = torch.float32)
        proc = AutoProcessor.from_pretrained(repo)
    except Exception as e:  # noqa: BLE001
        pytest.skip(reason = f"{repo} does not load here: {type(e).__name__}: {e}"[:200])
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}
    return model, proc


def _tower(model):
    for name, module in model.named_modules():
        if name.endswith("visual"):
            return module
    raise AssertionError("no vision tower")


@pytest.mark.parametrize("name", list(MODELS))
def test_grid_rows_reach_the_vision_tower_and_train(name, tmp_path):
    mod = _legacy_module()
    if not hasattr(mod.DPOTrainer, "concatenated_forward"):
        # TRL >= 0.29 ships its own vision collator, which keeps every patch row and image_grid_thw.
        assert "image_grid_thw" in inspect.getsource(mod.DataCollatorForVisionPreference)
        return
    torch.manual_seed(0)
    model, proc = _load(MODELS[name])
    ref_model = copy.deepcopy(model)
    cls = _patched_trainer_class(mod)
    cfg = mod.DPOConfig(
        output_dir = str(tmp_path),
        per_device_train_batch_size = 2,
        max_steps = 1,
        report_to = "none",
        save_strategy = "no",
        use_cpu = True,
        optim = "adamw_torch",
        max_length = None,
        max_prompt_length = None,
        remove_unused_columns = False,
        dataset_num_proc = None,
    )
    cfg.dataset_num_proc = None
    dataset = _dataset()
    trainer = cls(
        model = model, ref_model = ref_model, args = cfg, train_dataset = dataset, processing_class = proc
    )
    assert trainer.is_vision_model
    assert "image_grid_thw" in trainer.train_dataset.column_names

    rows = [trainer.train_dataset[i] for i in range(2)]
    batch = trainer.data_collator(rows)
    grid = batch["image_grid_thw"]
    assert grid.shape == (2, 3)
    assert batch["pixel_values"].shape[0] == int(grid.prod(-1).sum())

    seen = []
    _tower(model).register_forward_pre_hook(
        lambda m, a, k: seen.append(int((a[0] if a else k["hidden_states"]).shape[0])),
        with_kwargs = True,
    )
    model.eval()
    with torch.no_grad():
        out = trainer.concatenated_forward(model, batch)
        # Chosen rows then rejected rows, each with every patch row of its image.
        assert seen == [2 * int(grid.prod(-1).sum())]
        # Oracle: one plain forward per (prompt + completion) on the processor's own output.
        want = []
        for i, row in enumerate(rows):
            enc = proc(images = dataset[i]["images"], text = row["prompt"], return_tensors = "pt")
            for key in ("chosen_input_ids", "rejected_input_ids"):
                completion = torch.tensor([row[key]])
                ids = torch.cat([enc["input_ids"], completion], 1)
                logits = model(
                    input_ids = ids,
                    attention_mask = torch.ones_like(ids),
                    pixel_values = enc["pixel_values"],
                    image_grid_thw = enc["image_grid_thw"],
                ).logits
                logps = torch.log_softmax(logits[0, :-1].float(), -1).gather(-1, ids[0, 1:, None])[
                    :, 0
                ]
                want.append(logps[-completion.shape[1] :].sum())
    got = torch.stack(
        [
            out["chosen_logps"][0],
            out["rejected_logps"][0],
            out["chosen_logps"][1],
            out["rejected_logps"][1],
        ]
    )
    torch.testing.assert_close(got, torch.stack(want), atol = 1e-4, rtol = 1e-5)

    model.train()
    result = trainer.train()
    assert torch.isfinite(torch.tensor(result.training_loss))
    assert trainer.state.global_step == 1
