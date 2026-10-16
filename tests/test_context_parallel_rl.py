# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Context parallelism for DPO / GRPO: decoder sharding + hidden-state gather, peer sync, refusals (CPU, gloo)."""

import contextlib
import os
import tempfile
import types

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers.modeling_outputs import BaseModelOutputWithPast

import unsloth  # noqa: F401
from unsloth import context_parallel as cp


class _Decoder(torch.nn.Module):
    """Causal toy decoder: position-dependent, mixes earlier tokens, returns an HF-style output."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.embed_tokens = torch.nn.Embedding(50, 8)
        self.layers = torch.nn.ModuleList([torch.nn.Linear(8, 8)])

    def forward(self, input_ids = None, attention_mask = None, position_ids = None, output_hidden_states = False, **kwargs):
        h = self.embed_tokens(input_ids) * (1 + 0.01 * position_ids[..., None].float())
        first = h
        # Sum over earlier tokens of the FULL sequence (stands in for Ulysses attention).
        manager = cp.get_cp_manager()
        if manager is not None:
            rank = dist.get_rank()
            full = cp.gather_sequence(h, manager)
            prefix = full.cumsum(1)[:, rank * h.shape[1] : (rank + 1) * h.shape[1]]
        else:
            prefix = h.cumsum(1)
        h = torch.tanh(self.layers[0](prefix))
        return BaseModelOutputWithPast(
            last_hidden_state = h, hidden_states = (first, h) if output_hidden_states else None
        )


def _manager(rank, size = 2):
    manager = object.__new__(cp.ContextParallelManager)
    manager.size = size
    manager.backend = "ulysses"
    manager.mesh = types.SimpleNamespace(get_local_rank = lambda: rank, get_group = lambda: dist.group.WORLD)
    return manager


def _worker(rank, world, init_file, queue):
    dist.init_process_group("gloo", init_method = f"file://{init_file}", rank = rank, world_size = world)
    try:
        torch.manual_seed(1)
        ids = torch.randint(1, 50, (2, 7))
        mask = torch.ones(2, 7, dtype = torch.long)
        mask[1, 5:] = 0
        weight = torch.randn(2, 7, 8)

        reference = _Decoder()
        full = reference(input_ids = ids, attention_mask = mask, position_ids = torch.arange(7).expand(2, 7))
        (full.last_hidden_state * weight).sum().backward()

        model = _Decoder()
        manager = _manager(rank, world)
        cp.attach_rl_hooks(manager, torch.nn.ModuleDict({"model": model}))
        with cp._manager_scope(manager):
            out = model(input_ids = ids, attention_mask = mask, output_hidden_states = True)
            # Every peer computes the same loss on the gathered (unpadded) sequence.
            (out.last_hidden_state * weight).sum().backward()
        assert out.last_hidden_state.shape == (2, 7, 8)
        assert out.hidden_states[-1] is out.last_hidden_state
        assert out.hidden_states[0].shape == (2, 7, 8)
        torch.testing.assert_close(out.last_hidden_state, full.last_hidden_state, rtol = 1e-5, atol = 1e-6)
        # DDP's average over the CP group, as the trainer gets it.
        for p in model.parameters():
            dist.all_reduce(p.grad)
            p.grad /= world
        for p, q in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(p.grad, q.grad, rtol = 1e-5, atol = 1e-6)

        # Outside the scope the hooks pass through untouched.
        plain = model(input_ids = ids, attention_mask = mask, position_ids = torch.arange(7).expand(2, 7))
        torch.testing.assert_close(plain.last_hidden_state, full.last_hidden_state)
        assert model._unsloth_cp_rl_stack == []

        # Peers adopt rank 0's object; rank 1's own differs.
        mine = {"ids": torch.full((2,), rank), "text": [f"r{rank}"], "pair": (rank, None)}
        got = cp._broadcast_from_group_root(mine, manager, torch.device("cpu"))
        assert got["text"] == ["r0"] and got["pair"] == (0, None)
        assert torch.equal(got["ids"], torch.zeros(2, dtype = torch.long))

        cp._check_same_prompts(["same"], manager)
        with pytest.raises(RuntimeError, match = "different prompts"):
            cp._check_same_prompts([f"prompt {rank}"], manager)
        queue.put((rank, "ok"))
    except BaseException as error:  # pragma: no cover - reported to the parent
        import traceback

        queue.put((rank, traceback.format_exc()))
    finally:
        dist.destroy_process_group()


def test_decoder_shard_gather_matches_full_sequence_with_grads():
    if "fork" not in mp.get_all_start_methods():
        pytest.skip("needs fork")
    ctx = mp.get_context("fork")
    queue = ctx.Queue()
    with tempfile.TemporaryDirectory() as tmp:
        init_file = os.path.join(tmp, "init")
        procs = [ctx.Process(target = _worker, args = (r, 2, init_file, queue)) for r in range(2)]
        for p in procs:
            p.start()
        results = dict(queue.get(timeout = 120) for _ in procs)
        for p in procs:
            p.join(30)
    assert results == {0: "ok", 1: "ok"}, results


def test_decoder_hook_refuses_cache_and_prefix_grouper():
    model = _Decoder()
    manager = _manager(0)
    cp.attach_rl_hooks(manager, torch.nn.ModuleDict({"model": model}))
    ids = torch.ones(1, 4, dtype = torch.long)
    with cp._manager_scope(manager):
        with pytest.raises(NotImplementedError, match = "KV cache"):
            model(input_ids = ids, past_key_values = ((torch.zeros(1),),))
        with pytest.raises(NotImplementedError, match = "prefix_seg_info"):
            model(input_ids = ids, prefix_seg_info = object())


def _args(**kw):
    base = dict(
        context_parallel_size = 2,
        context_parallel_backend = "ulysses",
        per_device_train_batch_size = 4,
        per_device_eval_batch_size = 4,
        gradient_accumulation_steps = 1,
        steps_per_generation = 1,
        num_generations = 4,
    )
    base.update(kw)
    return types.SimpleNamespace(**base)


@pytest.mark.parametrize(
    "kind, attrs, message",
    [
        ("dpo_trainer", {"padding_free": True}, "padding_free"),
        ("dpo_trainer", {"precompute_ref_log_probs": True}, "precompute_ref_log_probs"),
        ("dpo_trainer", {"generate_during_eval": True}, "generate_during_eval"),
        ("grpo_trainer", {"use_vllm": True}, "vLLM"),
        ("grpo_trainer", {"per_device_train_batch_size": 3}, "multiples of num_generations"),
        ("grpo_trainer", {"scale_rewards": "batch"}, "per-group reward scaling"),
        ("grpo_trainer", {"multi_objective_aggregation": "normalize_then_sum"}, "per-group"),
    ],
)
def test_unsupported_rl_configs_are_refused(kind, attrs, message):
    trainer = types.SimpleNamespace(args = _args(**attrs), _generate = None, eval_dataset = None)
    with pytest.raises(NotImplementedError, match = message):
        cp._check_rl_config(trainer, kind)


def test_grpo_without_generate_hook_is_refused():
    trainer = types.SimpleNamespace(args = _args(), eval_dataset = None)
    with pytest.raises(NotImplementedError, match = "TRL >= 0.24"):
        cp._check_rl_config(trainer, "grpo_trainer")


def _fake_trainer_class(log):
    class Sampler:
        mini_repeat_count = 4
        batch_size = 8

    class Trainer:
        def __init__(self, args):
            self.args = args
            self.model = torch.nn.Linear(1, 1)
            self.accelerator = types.SimpleNamespace(device = torch.device("cpu"))

        def training_step(self, model, inputs):
            self._prepare_inputs(inputs)
            log.append(("compute_loss_returned", self.compute_loss(model, inputs)))
            log.append(("backward", cp.get_cp_manager()))
            return 0.0

        def prediction_step(self, model, inputs, prediction_loss_only):
            log.append(("predict", cp.get_cp_manager()))

        def _prepare_inputs(self, inputs):
            self._generate(["p"])
            self._get_per_token_logps_and_entropies()
            return {"rows": [cp.get_cp_manager() is None]}

        def _generate(self, prompts):
            log.append(("generate", cp.get_cp_manager()))
            return (prompts,)

        def _get_per_token_logps_and_entropies(self):
            log.append(("logps", cp.get_cp_manager()))

        def compute_loss(self, model, inputs):
            log.append(("loss", cp.get_cp_manager()))
            return 1.0

        def _get_train_sampler(self, *a):
            return Sampler()

        def train(self):
            return "trained"

        def evaluate(self):
            return "evaluated"

        def evaluation_loop(self):
            return "looped"

    return Trainer


@pytest.mark.parametrize("kind", ["dpo_trainer", "grpo_trainer"])
def test_rl_scopes_keep_generation_unsharded(monkeypatch, kind):
    log = []
    manager = types.SimpleNamespace(size = 2, backend = "ulysses")
    monkeypatch.setattr(cp, "_check_cp_runtime", lambda *a: None)
    monkeypatch.setattr(cp, "_check_cp_trainer", lambda *a: None)
    monkeypatch.setattr(cp, "_install_accelerator_state", lambda *a: None)
    monkeypatch.setattr(cp, "ContextParallelManager", lambda size, backend: manager)
    monkeypatch.setattr(cp, "attach_rl_hooks", lambda m, model: log.append(("hooked", m)))
    monkeypatch.setattr(cp, "_decoders", lambda model: [])
    monkeypatch.setattr(cp, "_check_same_prompts", lambda prompts, m: log.append(("same", prompts)))
    monkeypatch.setattr(cp, "_broadcast_from_group_root", lambda obj, m, device: obj)
    Trainer = _fake_trainer_class(log)
    cp.patch_rl_trainer(Trainer, kind)
    trainer = Trainer(_args())
    assert trainer._context_parallel_manager is manager
    assert cp.get_cp_manager() is None
    trainer.training_step(trainer.model, {})
    assert cp.get_cp_manager() is None
    seen = {name: value for name, value in log if name in ("generate", "logps", "loss", "backward")}
    assert seen["generate"] is (None if kind == "grpo_trainer" else manager)
    assert seen["logps"] is manager and seen["loss"] is manager and seen["backward"] is manager
    trainer.prediction_step(trainer.model, {}, True)
    assert dict(log)["predict"] is (manager if kind == "dpo_trainer" else None)
    if kind == "grpo_trainer":
        assert trainer._get_train_sampler().batch_size == 4
        assert ("same", ["p"]) in log


def test_rl_trainers_refuse_ring(monkeypatch):
    Trainer = _fake_trainer_class([])
    cp.patch_rl_trainer(Trainer, "dpo_trainer")
    with pytest.raises(NotImplementedError, match = "ulysses"):
        Trainer(_args(context_parallel_backend = "ring"))


def test_rl_trainers_without_cp_are_untouched(monkeypatch):
    monkeypatch.setattr(cp, "_install_accelerator_state", lambda *a: None)
    log = []
    Trainer = _fake_trainer_class(log)
    originals = {name: Trainer.__dict__[name] for name in ("training_step", "compute_loss", "_generate")}
    cp.patch_rl_trainer(Trainer, "grpo_trainer")
    trainer = Trainer(_args(context_parallel_size = 1))
    assert trainer._context_parallel_manager is None
    # Without CP only __init__ is wrapped: the step methods stay the original functions.
    assert all(Trainer.__dict__[name] is fn for name, fn in originals.items())
    trainer.training_step(trainer.model, {})
    assert all(value is None for name, value in log if name != "compute_loss_returned")
    assert trainer._get_train_sampler().batch_size == 8
