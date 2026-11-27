# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""GRPO context parallelism: CP peers generate their own whole groups and all-gather them (CPU, gloo).

The gathered batch must equal the one the group root generates alone (the previous path), with TRL's
in-_generate metric gathers seeing every completion once, and anything that cannot be merged must fall
back to the root path with the metrics of the discarded attempt removed."""

import collections
import os
import tempfile
import types

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import unsloth  # noqa: F401
from unsloth import context_parallel as cp

GENERATIONS = 4


def _completion(prompt, index):
    # Per (prompt, index in its group) seed: same tokens whichever process samples the row.
    generator = torch.Generator().manual_seed(hash((prompt, index)) % (2**31))
    length = int(torch.randint(1, 6, (1,), generator = generator))
    return torch.randint(3, 50, (length,), generator = generator).tolist()


class _Accelerator:
    device = torch.device("cpu")

    def gather(self, tensor):
        # As accelerate: equal shapes on every process, concatenated over the world.
        out = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(out, tensor)
        return torch.cat(out)


def _trainer_class(
    layout,
    calls,
    dp = 0,
):
    """TRL-shaped _generate: 7 items (0.26 - 1.8), 8 (1.9 - 1.18), 9 (>= 1.19) or 5 with a bad leaf."""

    class Trainer:
        def __init__(self, args):
            self.args = args
            self.model = torch.nn.Linear(1, 1)
            self.accelerator = _Accelerator()
            self.state = types.SimpleNamespace(num_input_tokens_seen = 0)
            self._metrics = {
                "train": collections.defaultdict(list),
                "eval": collections.defaultdict(list),
            }

        def _generate(self, prompts):
            calls.append(len(prompts))
            prompt_ids = [[len(p)] for p in prompts]
            completion_ids = []
            for i, prompt in enumerate(prompts):
                completion_ids.append(_completion(prompt, i % GENERATIONS))
            lengths = torch.tensor([len(c) for c in completion_ids])
            gathered = self.accelerator.gather(lengths)
            self.state.num_input_tokens_seen += int(gathered.sum())
            self._metrics["train"]["completions/mean_length"].append(gathered.float().mean().item())
            self._metrics["train"]["num_tokens"] = [self.state.num_input_tokens_seen]
            total = gathered.sum()
            logprobs = [[-0.5 * t for t in c] for c in completion_ids]
            completions = [f"text{c}" for c in completion_ids]
            if layout == "seven":
                return prompt_ids, completion_ids, None, completions, total, logprobs, {}
            if layout == "nine":
                extra = {"score": [float(len(c)) for c in completion_ids]}
                return (
                    prompt_ids,
                    completion_ids,
                    None,
                    completions,
                    logprobs,
                    None,
                    extra,
                    None,
                    [[] for _ in prompts],
                )
            if layout == "eight":  # TRL 1.9 - 1.18: tool_images = [] without tools
                return prompt_ids, completion_ids, None, completions, logprobs, {}, None, []
            if layout == "mixed" and dp == 1:  # mergeable here, unmergeable in the other dp group
                return prompt_ids, completion_ids, total, logprobs, {}
            # A leaf that is not per sample: must fall back to the root generating everything.
            return prompt_ids, completion_ids, total, logprobs, {"pixel_values": torch.ones(3, 2)}

        def _get_train_sampler(self, *a):
            return types.SimpleNamespace(mini_repeat_count = GENERATIONS, batch_size = 8)

        def train(self):
            return None

        def evaluate(self, eval_dataset = None):
            return None

        def evaluation_loop(self):
            return None

        def training_step(self, model, inputs):
            return 0.0

        def prediction_step(self, model, inputs, prediction_loss_only):
            return None

        def compute_loss(self, model, inputs):
            return 0.0

        def _prepare_inputs(self, inputs):
            return inputs

        def _get_per_token_logps_and_entropies(self, *a, **k):
            return None

    return Trainer


def _args():
    return types.SimpleNamespace(
        context_parallel_size = 2,
        context_parallel_backend = "ulysses",
        per_device_train_batch_size = 8,
        per_device_eval_batch_size = 8,
        gradient_accumulation_steps = 1,
        steps_per_generation = 1,
        num_generations = GENERATIONS,
    )


def _worker(rank, world, init_file, queue):
    dist.init_process_group("gloo", init_method = f"file://{init_file}", rank = rank, world_size = world)
    try:
        cp_size = 2
        groups = [dist.new_group(list(range(s, s + cp_size))) for s in range(0, world, cp_size)]
        group = groups[rank // cp_size]
        manager = types.SimpleNamespace(
            size = cp_size,
            backend = "ulysses",
            fsdp2 = False,
            _clear_real_tokens = lambda: None,
            mesh = types.SimpleNamespace(
                get_local_rank = lambda: rank % cp_size, get_group = lambda: group
            ),
        )
        manager.group = group
        cp._check_cp_runtime = lambda *a: None
        cp._check_cp_trainer = lambda *a: None
        cp._install_accelerator_state = lambda *a: None
        cp.ContextParallelManager = lambda size, backend: manager
        cp.attach_rl_hooks = lambda *a: None
        cp._decoders = lambda model: []
        dp = rank // cp_size
        # Two data parallel slices, each 3 prompts x GENERATIONS completions... and 2 groups per peer.
        prompts = [f"dp{dp} prompt {i // GENERATIONS}" for i in range(4 * GENERATIONS)]

        results = {}
        for layout in ("seven", "eight", "nine", "bad", "mixed"):
            for split in ("1", "0"):
                os.environ["UNSLOTH_CP_SPLIT_GENERATION"] = split
                calls = []
                Trainer = _trainer_class(layout, calls, dp)
                cp.patch_rl_trainer(Trainer, "grpo_trainer")
                trainer = Trainer(_args())
                assert trainer._context_parallel_manager is manager
                out = trainer._generate(list(prompts))
                results[layout, split] = (
                    out,
                    calls,
                    dict(trainer._metrics["train"]),
                    trainer.state.num_input_tokens_seen,
                )

        for layout in ("seven", "eight", "nine"):
            split, split_calls, split_metrics, split_seen = results[layout, "1"]
            root, root_calls, root_metrics, root_seen = results[layout, "0"]
            assert split_calls == [len(prompts) // cp_size], split_calls
            assert root_calls == [len(prompts)], root_calls
            # Bitwise the batch the group root generates alone, in prompt order, groups intact.
            assert len(split) == len(root)
            for a, b in zip(split, root):
                if isinstance(a, torch.Tensor):
                    assert torch.equal(a, b), (a, b)
                else:
                    assert a == b, (layout, a, b)
            assert split[1] == [_completion(p, i % GENERATIONS) for i, p in enumerate(prompts)]
            # The metric gathers inside _generate now see each completion once (root path: cp copies).
            assert (
                split_metrics["completions/mean_length"] == root_metrics["completions/mean_length"]
            )
            assert split_seen * cp_size == root_seen

        # Fallback: the share is discarded, its metrics removed, and the root path runs once. With "mixed"
        # only one data parallel group cannot merge, and every group must still fall back together.
        for layout in ("bad", "mixed"):
            bad, bad_calls, bad_metrics, bad_seen = results[layout, "1"]
            root, _, root_metrics, root_seen = results[layout, "0"]
            assert bad_calls == [len(prompts) // cp_size, len(prompts)], (layout, bad_calls)
            assert bad[1] == root[1] and torch.equal(bad[2], root[2])
            assert bad_metrics == root_metrics and bad_seen == root_seen
        queue.put((rank, "ok"))
    except BaseException:  # pragma: no cover - reported to the parent
        import traceback
        queue.put((rank, traceback.format_exc()))
    finally:
        dist.destroy_process_group()


def test_split_generation_matches_root_generation():
    if "fork" not in mp.get_all_start_methods():
        pytest.skip("needs fork")
    ctx = mp.get_context("fork")
    queue = ctx.Queue()
    world = 4
    with tempfile.TemporaryDirectory() as tmp:
        init_file = os.path.join(tmp, "init")
        procs = [
            ctx.Process(target = _worker, args = (r, world, init_file, queue)) for r in range(world)
        ]
        for p in procs:
            p.start()
        results = dict(queue.get(timeout = 120) for _ in procs)
        for p in procs:
            p.join(timeout = 60)
    assert all(v == "ok" for v in results.values()), results


def _share_trainer(**attrs):
    trainer = types.SimpleNamespace(
        args = _args(), model = torch.nn.Linear(1, 1), accelerator = None, rollout_func = None
    )
    for k, v in attrs.items():
        setattr(trainer, k, v)
    return trainer


@pytest.mark.parametrize(
    "attrs, prompts, args, expect",
    [
        ({}, ["p"] * 16, (), slice(8, 16)),
        ({}, ["p"] * 4, (), None),  # one group cannot be split over two peers
        ({}, ["p"] * 8, (), slice(4, 8)),  # one group per peer
        ({}, ["p"] * 24, (), slice(12, 24)),  # 3 groups per peer
        ({}, ["p"] * 14, (), None),  # not whole groups
        ({}, ["p"] * 16, (["img"] * 16,), None),  # TRL 0.24 images
        ({}, ["p"] * 16, (None,), slice(8, 16)),
        ({"rollout_func": lambda *a: None}, ["p"] * 16, (), None),
        ({"environment_factories": [object]}, ["p"] * 16, (), None),
        ({"is_deepspeed_enabled": True}, ["p"] * 16, (), None),
        (
            {
                "accelerator": types.SimpleNamespace(
                    distributed_type = types.SimpleNamespace(name = "FSDP")
                )
            },
            ["p"] * 16,
            (),
            None,
        ),
        ({"_unsloth_cp_split_generation_off": True}, ["p"] * 16, (), None),
    ],
)
def test_generation_share_gates(attrs, prompts, args, expect):
    manager = types.SimpleNamespace(
        size = 2, fsdp2 = False, mesh = types.SimpleNamespace(get_local_rank = lambda: 1)
    )
    assert cp._generation_share(_share_trainer(**attrs), prompts, args, {}, manager) == expect
    collective = types.SimpleNamespace(size = 2, fsdp2 = True, mesh = manager.mesh)
    assert cp._generation_share(_share_trainer(), prompts, (), {}, collective) is None


def test_generation_share_kill_switch(monkeypatch):
    manager = types.SimpleNamespace(
        size = 2, fsdp2 = False, mesh = types.SimpleNamespace(get_local_rank = lambda: 0)
    )
    monkeypatch.setenv("UNSLOTH_CP_SPLIT_GENERATION", "0")
    assert cp._generation_share(_share_trainer(), ["p"] * 16, (), {}, manager) is None
