# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""One gloo rank of the on-policy distillation collective-count check (tests/test_on_policy_distillation.py).

Each forward of the stub model all-reduces, like an FSDP2 / ZeRO-3 gather or a tensor parallel block, and the
ranks score rollouts of different lengths, so their teacher chunk counts differ. Usage:
python _opd_collective_worker.py RANK WORLD INIT_FILE OPD_PATH SCENARIO"""

import collections
import contextlib
import datetime
import importlib.util
import json
import sys
import types

import torch
import torch.distributed as dist

VOCAB = 1 << 17  # with the default row budget: 3 rows per chunk at L = 600, 1 row at L = 1500


class CollectiveLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.embed = torch.nn.Embedding(VOCAB, 4)
        self.lm_head = torch.nn.Linear(4, VOCAB, bias = False)
        with torch.no_grad():
            self.embed.weight.copy_(torch.randn(VOCAB, 4, generator = g))
            self.lm_head.weight.copy_(torch.randn(VOCAB, 4, generator = g))
        self.config = types.SimpleNamespace(vocab_size = VOCAB)
        # Marks the forward as collective for unsloth_zoo's grpo_forwards_are_collective.
        self._unsloth_reshard_after_forward = True
        self.collective = True
        self.calls = 0

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
        self,
        input_ids,
        attention_mask = None,
        use_cache = None,
        logits_to_keep = 0,
    ):
        self.calls += 1
        if self.collective:
            dist.all_reduce(torch.ones(1))  # the per-forward gather of a sharded model
        h = self.embed(input_ids).cumsum(dim = 1) / torch.arange(1, input_ids.shape[1] + 1).view(
            1, -1, 1
        )
        if logits_to_keep:
            h = h[:, -logits_to_keep:]
        return types.SimpleNamespace(logits = self.lm_head(h))


class FakeLora(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.base_model = types.SimpleNamespace(model = inner)
        self.inner = inner
        self.peft_config = {"default": object()}
        self.enabled = True

    @contextlib.contextmanager
    def disable_adapter(self):
        self.enabled = False
        try:
            yield
        finally:
            self.enabled = True

    def get_input_embeddings(self):
        return self.inner.embed

    def forward(self, input_ids, **kwargs):
        out = self.inner(input_ids, **kwargs)
        if self.enabled:
            out = types.SimpleNamespace(logits = out.logits + 0.01 * torch.arange(VOCAB))
        return out


def _batch(
    rank,
    rows = 4,
    completion = 8,
):
    prompt = 592 if rank == 0 else 1492  # L = 600 vs 1500
    g = torch.Generator().manual_seed(rank)
    prompt_ids = torch.randint(1, VOCAB, (rows, prompt), generator = g)
    prompt_mask = torch.ones(rows, prompt, dtype = torch.long)
    prompt_mask[0, :5] = 0
    prompt_ids = prompt_ids * prompt_mask
    completion_ids = torch.randint(1, VOCAB, (rows, completion), generator = g)
    completion_mask = torch.ones(rows, completion, dtype = torch.long)
    completion_mask[1, 6:] = 0
    completion_ids = completion_ids * completion_mask
    return prompt_ids, prompt_mask, completion_ids, completion_mask


def main():
    rank, world, init_file, opd_path, scenario = sys.argv[1:6]
    rank, world = int(rank), int(world)
    dist.init_process_group(
        "gloo",
        init_method = init_file if init_file.startswith("file:") else f"file://{init_file}",
        rank = rank,
        world_size = world,
        timeout = datetime.timedelta(seconds = 20),
    )
    spec = importlib.util.spec_from_file_location("_opd_worker_under_test", opd_path)
    opd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(opd)
    opd._require_zoo = lambda flag, option: None

    inner = CollectiveLM()
    student = FakeLora(inner)

    def reduce(t, reduction = "sum"):
        dist.all_reduce(t)
        return t

    trainer = types.SimpleNamespace(
        args = types.SimpleNamespace(unsloth_distillation_coef = 1.0, temperature = 1.0),
        model = student,
        accelerator = types.SimpleNamespace(
            num_processes = world, unwrap_model = lambda m: m, reduce = reduce
        ),
        _metrics = {"train": collections.defaultdict(list)},
    )
    opd.grpo_setup_distillation(trainer)
    prompt_ids, prompt_mask, completion_ids, completion_mask = _batch(rank)
    output = dict(
        prompt_ids = prompt_ids,
        prompt_mask = prompt_mask,
        completion_ids = completion_ids,
        completion_mask = completion_mask,
    )
    if scenario == "one_rank_sampled" and rank == 0:
        # Rank 0 has the sampler's logprobs, rank 1 rescores the student.
        output["sampling_per_token_logps"] = -torch.rand(completion_ids.shape)
    opd.grpo_add_distillation_advantages(trainer, output)
    calls = inner.calls
    dist.monitored_barrier(timeout = datetime.timedelta(seconds = 20))
    counts = [None] * world
    dist.all_gather_object(counts, calls)

    # Unsharded reference: the same scoring with the collective off, one process.
    inner.collective = False
    with student.disable_adapter():
        teacher = opd.score_completion_logps(
            student, prompt_ids, prompt_mask, completion_ids, completion_mask
        )
    if "sampling_per_token_logps" in output:
        sampled = output["sampling_per_token_logps"]
    else:
        sampled = opd.score_completion_logps(
            student, prompt_ids, prompt_mask, completion_ids, completion_mask
        )
    want = opd.on_policy_distillation_advantages(sampled, teacher, completion_mask, 1.0)
    equal = bool(torch.equal(output["unsloth_distillation_advantages"], want))
    print(json.dumps({"rank": rank, "calls": calls, "counts": counts, "equal": equal}), flush = True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
