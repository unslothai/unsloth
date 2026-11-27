# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""One gloo rank for test_ddp_tower_sync.py: full finetune of a tiny VLM whose language layers are checkpointed
reentrantly (as vision.py does for Gemma 4 / Gemma 3N under DDP), on batches where a rank, or every rank, has
no image. Final weights are compared with a single-process run over the same global batches."""

import copy
import importlib.util
import os
import sys
from types import SimpleNamespace

import torch
import torch.distributed as dist
import transformers
from torch.utils.checkpoint import checkpoint

HELPER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "unsloth",
    "models",
    "_ddp_defaults.py",
)
# Global batch 4 = 2 rows per rank. Step 1: rank 0 has an image; step 2: no image anywhere; step 3: both.
IMAGE_ROWS = {0, 8, 10}
STEPS, PER_RANK, LR = 3, 2, 0.5


class TinyVLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type = "gemma4", vision_config = SimpleNamespace())
        self.vision_tower = torch.nn.Linear(8, 16)
        self.embed_vision = torch.nn.Linear(16, 16)
        self.embed_tokens = torch.nn.Embedding(32, 16)
        self.layers = torch.nn.ModuleList(torch.nn.Linear(16, 16) for _ in range(2))
        self.lm_head = torch.nn.Linear(16, 32)

    def forward(
        self,
        input_ids,
        labels,
        pixel_values = None,
        image_mask = None,
    ):
        h = self.embed_tokens(input_ids)
        if pixel_values is not None:
            image = self.embed_vision(self.vision_tower(pixel_values))
            h = h + image[:, None, :] * image_mask[:, None, None]
        for layer in self.layers:
            h = checkpoint(lambda x, layer = layer: torch.tanh(layer(x)), h, use_reentrant = True)
        logits = self.lm_head(h)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten())
        return {"loss": loss, "logits": logits}


def rows():
    g = torch.Generator().manual_seed(0)
    out = []
    for k in range(STEPS * PER_RANK * 2):
        ids = torch.randint(0, 32, (6,), generator = g)
        out.append({"idx": k, "input_ids": ids, "pixel": torch.randn(8, generator = g)})
    return out


def collate(batch):
    out = {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels": torch.stack([b["input_ids"] for b in batch]),
    }
    mask = torch.tensor([float(b["idx"] in IMAGE_ROWS) for b in batch])
    if mask.any():
        out["pixel_values"] = torch.stack([b["pixel"] for b in batch])
        out["image_mask"] = mask
    return out


def reference(model, data):
    # Equal-length rows: the mean over the global batch equals the mean of the per-rank, per-micro-batch means.
    model = copy.deepcopy(model)
    opt = torch.optim.SGD(model.parameters(), lr = LR)
    for step in range(STEPS):
        opt.zero_grad(set_to_none = True)
        model(**collate(data[step * 4 : step * 4 + 4]))["loss"].backward()
        opt.step()
    return model


def run(module, out_dir, ga):
    torch.manual_seed(0)
    model = TinyVLM()
    initial = copy.deepcopy(model)
    data = rows()
    args = transformers.TrainingArguments(
        output_dir = out_dir,
        per_device_train_batch_size = PER_RANK // ga,
        gradient_accumulation_steps = ga,
        max_steps = STEPS,
        learning_rate = LR,
        lr_scheduler_type = "constant",
        optim = "sgd",
        max_grad_norm = 0,
        ddp_find_unused_parameters = False,
        report_to = "none",
        save_strategy = "no",
        use_cpu = True,
        dataloader_num_workers = 0,
        remove_unused_columns = False,
    )

    if dist.is_initialized() and dist.get_rank() == 1:
        with torch.no_grad():  # a rank-dependent tower init: rank 0's weights must win, as DDP does
            model.vision_tower.weight.add_(1.0)

    class SeqTrainer(transformers.Trainer):
        def _get_train_sampler(self, *a, **k):
            return torch.utils.data.SequentialSampler(self.train_dataset)

    trainer = SeqTrainer(model = model, args = args, train_dataset = data, data_collator = collate)
    module.install_unsloth_ddp_comm_defaults(trainer)
    syncs = getattr(module, "_TOWER_SYNC_COUNT", [0])[0]
    trainer.train()
    ref = reference(initial, data)
    diff = max(
        (p - q).abs().max().item() for p, q in zip(trainer.model.parameters(), ref.parameters())
    )
    moved = (trainer.model.vision_tower.weight - initial.vision_tower.weight).abs().max().item()
    syncs = getattr(module, "_TOWER_SYNC_COUNT", [0])[0] - syncs
    print(
        f"\nRANK_OK rank={dist.get_rank()} ga={ga} syncs={syncs} diff={diff:.3g} moved={moved:.3g}",
        flush = True,
    )


def main(out_dir):
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location("_unsloth_ddp_defaults_towers", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for ga in (1, 2):
        run(module, os.path.join(out_dir, str(ga)), ga)


if __name__ == "__main__":
    main(sys.argv[1])
