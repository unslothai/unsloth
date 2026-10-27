# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Context parallelism for DPO / GRPO: decoder sharding + hidden-state gather, peer sync, refusals (CPU, gloo)."""

import contextlib
import dataclasses as _dc
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

    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        position_ids = None,
        output_hidden_states = False,
        **kwargs,
    ):
        if cp.get_cp_manager() is not None:
            # Per-shard masks differ across peers (only the last holds padding): the hook must drop it.
            assert attention_mask is None
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
    manager.mesh = types.SimpleNamespace(
        get_local_rank = lambda: rank, get_group = lambda: dist.group.WORLD
    )
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
        full = reference(
            input_ids = ids, attention_mask = mask, position_ids = torch.arange(7).expand(2, 7)
        )
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
        # Trimmed from the padded gather as fresh tensors: compiled consumers must not guard on a view base.
        assert out.last_hidden_state._base is None and out.hidden_states[0]._base is None
        torch.testing.assert_close(
            out.last_hidden_state, full.last_hidden_state, rtol = 1e-5, atol = 1e-6
        )
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
        # The root's inference-mode flag reaches every peer (compiled forwards guard on it).
        with torch.inference_mode():
            scored = {"ids": torch.full((2,), rank)}
        got = cp._broadcast_from_group_root(scored, manager, torch.device("cpu"))
        assert got["ids"].is_inference() and torch.equal(
            got["ids"], torch.zeros(2, dtype = torch.long)
        )
        got = cp._broadcast_from_group_root({"ids": torch.ones(1)}, manager, torch.device("cpu"))
        assert not got["ids"].is_inference()

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

        def evaluate(self, eval_dataset = None):
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


def _accelerate_cp_accelerator(monkeypatch, cp_size):
    # What `accelerate launch` with parallelism_config_cp_size hands the Accelerator: a ParallelismConfig
    # built from the environment, read through the real Accelerator.parallelism_config property.
    from accelerate import Accelerator
    from accelerate.parallelism_config import ParallelismConfig

    monkeypatch.setenv("ACCELERATE_USE_PARALLELISM_CONFIG", "true")
    monkeypatch.setenv("PARALLELISM_CONFIG_CP_SIZE", str(cp_size))

    class _Accelerator:
        parallelism_config = Accelerator.parallelism_config

        def __init__(self):
            self.state = types.SimpleNamespace(parallelism_config = ParallelismConfig())
            self.device = torch.device("cpu")

    return _Accelerator()


@pytest.mark.parametrize("kind", ["dpo_trainer", "grpo_trainer"])
@pytest.mark.parametrize("route", ["accelerate config", "parallelism_config"])
@pytest.mark.parametrize("cp_size", [1, 2])
@pytest.mark.parametrize("unsloth_size", [1, 2])
def test_rl_trainers_refuse_transformers_cp(monkeypatch, kind, route, cp_size, unsloth_size):
    monkeypatch.setattr(cp, "_install_accelerator_state", lambda *a: None)
    monkeypatch.setattr(cp, "_check_cp_runtime", lambda *a: None)
    monkeypatch.setattr(cp, "_check_cp_trainer", lambda *a: None)
    monkeypatch.setattr(cp, "attach_rl_hooks", lambda *a: None)
    monkeypatch.setattr(cp, "ContextParallelManager", lambda size, backend: object())
    accelerator = _accelerate_cp_accelerator(monkeypatch, cp_size)
    # transformers forwards TrainingArguments.parallelism_config to the Accelerator.
    args_pc = accelerator.state.parallelism_config if route == "parallelism_config" else None

    class Trainer(_fake_trainer_class([])):
        def __init__(self, args):
            super().__init__(args)
            self.accelerator = accelerator
            self._generate = lambda prompts: (prompts,)

    cp.patch_rl_trainer(Trainer, kind)
    args = _args(context_parallel_size = unsloth_size, parallelism_config = args_pc)
    if cp_size == 1:
        Trainer(args)
        return
    config = "DPOConfig" if kind == "dpo_trainer" else "GRPOConfig"
    with pytest.raises(NotImplementedError, match = rf"{config}\(context_parallel_size = 2.*{route}"):
        Trainer(args)


def test_rl_trainers_without_cp_are_untouched(monkeypatch):
    monkeypatch.setattr(cp, "_install_accelerator_state", lambda *a: None)
    log = []
    Trainer = _fake_trainer_class(log)
    originals = {
        name: Trainer.__dict__[name] for name in ("training_step", "compute_loss", "_generate")
    }
    cp.patch_rl_trainer(Trainer, "grpo_trainer")
    trainer = Trainer(_args(context_parallel_size = 1))
    assert trainer._context_parallel_manager is None
    # Without CP only __init__ is wrapped: the step methods stay the original functions.
    assert all(Trainer.__dict__[name] is fn for name, fn in originals.items())
    trainer.training_step(trainer.model, {})
    assert all(value is None for name, value in log if name != "compute_loss_returned")
    assert trainer._get_train_sampler().batch_size == 8


# Objective parity: a Llama-shaped decoder (RoPE on global positions, GQA, real Ulysses all-to-all) under the
# RL hooks, scored with TRL's DPO sigmoid loss and GRPO's default dapo loss, against the same model without CP.
_H, _NH, _NKV, _D, _V = 16, 4, 2, 4, 40


def _rope(x, pos):
    inv = 1.0 / (10000 ** (torch.arange(0, _D, 2, dtype = torch.float64) / _D))
    ang = (pos[..., None].double() * inv)[:, None].to(x.dtype)
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack(
        [x1 * ang.cos() - x2 * ang.sin(), x1 * ang.sin() + x2 * ang.cos()], -1
    ).flatten(-2)


def _causal_attention(*, config, context, Q, K, V):
    K = K.repeat_interleave(config.n_groups, 1)
    V = V.repeat_interleave(config.n_groups, 1)
    return torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal = True).transpose(1, 2)


@_dc.dataclass
class _UlyssesConfig:
    n_kv_heads: int
    n_groups: int


@_dc.dataclass
class _UlyssesContext:
    bsz: int
    q_len: int
    kv_seq_len: int
    n_heads: int
    head_dim: int
    attention_mask: object = None
    causal_mask: object = None
    prefix_seg_info: object = None


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = torch.nn.Linear(_H, _NH * _D, bias = False)
        self.kv = torch.nn.Linear(_H, 2 * _NKV * _D, bias = False)
        self.o = torch.nn.Linear(_NH * _D, _H, bias = False)
        self.up = torch.nn.Linear(_H, 2 * _H, bias = False)
        self.down = torch.nn.Linear(2 * _H, _H, bias = False)

    def forward(self, x, pos):
        from unsloth.utils import ulysses

        B, T, _ = x.shape
        q = _rope(self.q(x).view(B, T, _NH, _D).transpose(1, 2), pos)
        k, v = self.kv(x).view(B, T, 2, _NKV, _D).unbind(2)
        k, v = _rope(k.transpose(1, 2), pos), v.transpose(1, 2)
        config = _UlyssesConfig(_NKV, _NH // _NKV)
        manager = cp.get_cp_manager()
        if manager is None:
            a = _causal_attention(config = config, context = None, Q = q, K = k, V = v)
        else:
            a = ulysses.ulysses_attention(
                _causal_attention, config, _UlyssesContext(B, T, T, _NH, _D), q, k, v, manager.group
            )
        x = x + self.o(a.reshape(B, T, -1))
        return x + self.down(torch.nn.functional.silu(self.up(x)))


class _LlamaLike(torch.nn.Module):
    def __init__(self, seed, dtype):
        super().__init__()
        torch.manual_seed(seed)
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(_V, _H)
        self.model.layers = torch.nn.ModuleList([_Block(), _Block()])
        self.model.forward = self._decode
        self.lm_head = torch.nn.Linear(_H, _V, bias = False)
        with torch.no_grad():
            for p in self.parameters():
                p.normal_(0, 0.4)
        self.to(dtype)

    def _decode(
        self,
        input_ids = None,
        attention_mask = None,
        position_ids = None,
        **kwargs,
    ):
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1]).expand_as(input_ids)
        x = self.model.embed_tokens(input_ids)
        for block in self.model.layers:
            x = block(x, position_ids)
        return BaseModelOutputWithPast(last_hidden_state = torch.nn.functional.rms_norm(x, (_H,)))

    def token_logps(self, ids, mask):
        h = self.model(input_ids = ids, attention_mask = mask).last_hidden_state
        return (
            torch.log_softmax(self.lm_head(h)[:, :-1], -1).gather(-1, ids[:, 1:, None]).squeeze(-1)
        )


def _right_pad(rows):
    ids = torch.zeros(len(rows), max(map(len, rows)), dtype = torch.long)
    mask = torch.zeros_like(ids)
    for i, row in enumerate(rows):
        ids[i, : len(row)], mask[i, : len(row)] = torch.tensor(row), 1
    return ids, mask


def _dpo_sigmoid(
    policy,
    ref,
    batch,
    world,
    beta = 0.1,
):
    # trl DPOTrainer._compute_loss: completion sums, rows [chosen, rejected], -logsigmoid(beta * delta).mean()
    ids, mask, completion = batch
    logps = (policy.token_logps(ids, mask) * completion).sum(1)
    with torch.no_grad():
        ref_logps = (ref.token_logps(ids, mask) * completion).sum(1)
    (c, r), (rc, rr) = logps.chunk(2), ref_logps.chunk(2)
    return -torch.nn.functional.logsigmoid(beta * ((c - rc) - (r - rr))).mean()


def _grpo_dapo(
    policy,
    ref,
    batch,
    world,
    beta = 0.04,
    eps = 0.2,
):
    # trl GRPOTrainer (loss_type "dapo", num_iterations 1): token sum over gather(mask.sum()).sum() / num_processes
    ids, mask, completion, rewards = batch
    adv = ((rewards - rewards.mean()) / (rewards.std() + 1e-4)).to(policy.lm_head.weight.dtype)[
        :, None
    ]
    completion = completion.to(adv.dtype)
    counts = [completion.sum().float()]
    if world > 1:  # accelerator.gather over every process: CP peers each count the full rows
        counts = [torch.zeros(()) for _ in range(world)]
        dist.all_gather(counts, completion.sum().float())
    normalizer = torch.stack(counts).sum().clamp(min = 1.0) / world
    logps = policy.token_logps(ids, mask)
    with torch.no_grad():
        ref_logps = ref.token_logps(ids, mask)
    ratio = torch.exp(logps - logps.detach())
    loss = -torch.min(ratio * adv, ratio.clamp(1 - eps, 1 + eps) * adv)
    loss = loss + beta * (torch.exp(ref_logps - logps) - (ref_logps - logps) - 1)
    return (loss * completion).sum() / normalizer


def _objective_worker(rank, world, init_file, queue):
    dist.init_process_group("gloo", init_method = f"file://{init_file}", rank = rank, world_size = world)
    try:
        torch.set_num_threads(1)
        g = torch.Generator().manual_seed(3)
        rand = lambda n: torch.randint(1, _V, (n,), generator = g).tolist()
        prompts = [rand(7), rand(12)]
        chosen, rejected = (
            [p + rand(n) for p, n in zip(prompts, (9, 5))],
            [p + rand(n) for p, n in zip(prompts, (4, 14))],
        )
        ids, mask = _right_pad(
            chosen + rejected
        )  # 26 tokens: odd row lengths, the pad tail crosses the shard
        completion = mask.clone()
        for i, n in enumerate([7, 12, 7, 12]):
            completion[i, :n] = 0
        dpo = (ids, mask, completion[:, 1:])
        prompt = rand(9)
        g_ids, g_mask = _right_pad([prompt + rand(n) for n in (11, 3, 17, 8)])
        g_completion = g_mask.clone()
        g_completion[:, :9] = 0
        grpo = (g_ids, g_mask, g_completion[:, 1:], torch.tensor([0.3, 1.0, 0.1, 0.65]))
        tolerances = {torch.float64: 1e-12, torch.float32: 1e-6}
        for dtype, tol in tolerances.items():
            for objective, batch in ((_dpo_sigmoid, dpo), (_grpo_dapo, grpo)):
                results = []
                for use_cp in (False, True):
                    policy, ref = _LlamaLike(0, dtype), _LlamaLike(1, dtype).requires_grad_(False)
                    scope = contextlib.nullcontext()
                    if use_cp:
                        manager = _manager(rank, world)
                        cp.attach_rl_hooks(manager, policy)
                        cp.attach_rl_hooks(manager, ref)
                        scope = cp._manager_scope(manager)
                    with scope:
                        loss = objective(policy, ref, batch, world if use_cp else 1)
                        loss.backward()
                    grads = torch.cat([p.grad.flatten() for p in policy.parameters()])
                    if use_cp:  # DDP's average over the CP group
                        dist.all_reduce(grads)
                        grads /= world
                    results.append((loss.detach(), grads))
                (loss_ref, grad_ref), (loss_cp, grad_cp) = results
                name = f"{objective.__name__} {dtype}"
                assert abs(float(loss_cp - loss_ref)) <= tol * abs(float(loss_ref)), name
                grad_rel = float((grad_cp - grad_ref).norm() / grad_ref.norm())
                assert grad_rel <= tol, f"{name}: grad rel {grad_rel:.3e}"
        queue.put((rank, "ok"))
    except BaseException:  # pragma: no cover - reported to the parent
        import traceback
        queue.put((rank, traceback.format_exc()))
    finally:
        dist.destroy_process_group()


def test_dpo_and_grpo_objectives_match_without_cp_in_fp64_and_fp32():
    if "fork" not in mp.get_all_start_methods():
        pytest.skip("needs fork")
    ctx = mp.get_context("fork")
    queue = ctx.Queue()
    with tempfile.TemporaryDirectory() as tmp:
        init_file = os.path.join(tmp, "init")
        procs = [
            ctx.Process(target = _objective_worker, args = (r, 2, init_file, queue)) for r in range(2)
        ]
        for p in procs:
            p.start()
        results = dict(queue.get(timeout = 180) for _ in procs)
        for p in procs:
            p.join(30)
    assert results == {0: "ok", 1: "ok"}, results


# Review follow-ups: evaluation entry checks, pre-CP trainers, split batches, the GRPO DAPO token count.
def _cp_fake_trainer(
    monkeypatch,
    kind,
    log,
    installs = None,
):
    manager = types.SimpleNamespace(size = 2, backend = "ulysses")
    monkeypatch.setattr(cp, "_check_cp_runtime", lambda *a: None)
    monkeypatch.setattr(cp, "_check_cp_trainer", lambda *a: None)
    monkeypatch.setattr(
        cp,
        "_install_accelerator_state",
        lambda trainer: (installs if installs is not None else []).append(trainer),
    )
    monkeypatch.setattr(cp, "ContextParallelManager", lambda size, backend: manager)
    monkeypatch.setattr(cp, "attach_rl_hooks", lambda m, model: None)
    monkeypatch.setattr(cp, "_decoders", lambda model: [])
    Trainer = _fake_trainer_class(log)
    Trainer.predict = lambda self, test_dataset: "predicted"
    cp.patch_rl_trainer(Trainer, kind)
    return Trainer


def test_grpo_evaluate_checks_generation_groups_of_a_late_eval_dataset(monkeypatch):
    # No eval_dataset at construction, so __init__ cannot check the eval batch; evaluate() must.
    Trainer = _cp_fake_trainer(monkeypatch, "grpo_trainer", [])
    trainer = Trainer(_args(per_device_eval_batch_size = 3))
    trainer.eval_dataset = None
    with pytest.raises(NotImplementedError, match = "multiples of num_generations"):
        trainer.evaluate(eval_dataset = [{"prompt": "p"}] * 3)
    ok = Trainer(_args(per_device_eval_batch_size = 8))
    ok.eval_dataset = None
    assert ok.evaluate(eval_dataset = [{"prompt": "p"}] * 8) == "evaluated"


@pytest.mark.parametrize("kind", ["dpo_trainer", "grpo_trainer"])
def test_rl_evaluate_and_predict_refuse_iterable_datasets(monkeypatch, kind):
    class Stream(torch.utils.data.IterableDataset):
        def __iter__(self):
            return iter(range(4))

    Trainer = _cp_fake_trainer(monkeypatch, kind, [])
    trainer = Trainer(_args())
    trainer.eval_dataset = None
    with pytest.raises(NotImplementedError, match = "iterable"):
        trainer.evaluate(eval_dataset = Stream())
    trainer.eval_dataset = {"held_out": Stream()}
    with pytest.raises(NotImplementedError, match = "iterable"):
        trainer.evaluate(eval_dataset = "held_out")
    with pytest.raises(NotImplementedError, match = "iterable"):
        trainer.predict(Stream())
    trainer.eval_dataset = None
    assert trainer.evaluate(eval_dataset = [1, 2, 3, 4]) == "evaluated"
    assert trainer.predict([1, 2, 3, 4]) == "predicted"


def test_trainers_built_before_cp_restore_the_shared_accelerator_state(monkeypatch):
    # The generated RL trainers bind self.train on the instance in __init__ (rl.py: MethodType(...)):
    # a trainer built before CP was enabled keeps that plain method, so it must still drop the CP mesh
    # a later CP trainer installed on the shared AcceleratorState.
    installs = []
    Trainer = _cp_fake_trainer(monkeypatch, "dpo_trainer", [], installs)

    def bind_train(trainer):
        trainer.train = types.MethodType(type(trainer).train, trainer)

    plain = Trainer(_args(context_parallel_size = 1))
    bind_train(plain)
    cp_trainer = Trainer(_args())
    bind_train(cp_trainer)
    installs.clear()
    assert plain.train() == "trained"
    assert plain in installs, "the pre-CP trainer's bound train() skipped the accelerator restore"
    installs.clear()
    assert cp_trainer.train() == "trained" and cp_trainer in installs


def test_grpo_refuses_split_batches():
    # split_batches hands each data parallel slice per_device_train_batch_size / D rows, so a slice holds part
    # of a generation group and TRL's gather would group peer copies together.
    trainer = types.SimpleNamespace(
        args = _args(),
        _generate = None,
        eval_dataset = None,
        accelerator = types.SimpleNamespace(split_batches = True),
    )
    with pytest.raises(NotImplementedError, match = "split_batches"):
        cp._check_rl_config(trainer, "grpo_trainer")
    trainer.accelerator.split_batches = False
    cp._check_rl_config(trainer, "grpo_trainer")


def _dapo_count_worker(rank, world, init_file, queue):
    dist.init_process_group("gloo", init_method = f"file://{init_file}", rank = rank, world_size = world)
    try:
        manager = _manager(rank, world)
        # Forked and discarded: patch the module directly (no monkeypatch fixture in the child).
        cp._check_cp_runtime = cp._check_cp_trainer = cp._install_accelerator_state = (
            lambda *a: None
        )
        cp.ContextParallelManager = lambda size, backend: manager
        cp.attach_rl_hooks = lambda m, model: None

        def gather(t):
            parts = [torch.empty_like(t) for _ in range(world)]
            dist.all_gather(parts, t)
            return torch.cat(parts)

        # Each rank samples its own completions (rank 0: 2 tokens each, rank 1: 4); peers then adopt rank 0's.
        completion_ids = [[1] * (2 + 2 * rank) for _ in range(4)]
        tool_mask = [[1] * (1 + rank) + [0] * (1 + rank) for _ in range(4)]
        lengths = torch.tensor([len(ids) for ids in completion_ids])
        tool_lengths = torch.tensor([sum(m) for m in tool_mask])
        layouts = {
            # TRL 0.24 / 0.25: prompt_ids, completion_ids, total_completion_tokens, logprobs, extra
            5: (lambda: (["p"], completion_ids, gather(lengths).sum(), None, {}), 2, 8 * world),
            # TRL 0.27 - 1.0: prompt_ids, completion_ids, tool_mask, completions, count, logprobs, extra
            7: (
                lambda: (
                    ["p"],
                    completion_ids,
                    tool_mask,
                    ["c"],
                    gather(tool_lengths).sum(),
                    None,
                    {},
                ),
                4,
                4 * world,
            ),
            # TRL 1.7 - 1.11: the 7 above plus images, tool_images
            9: (
                lambda: (
                    ["p"],
                    completion_ids,
                    None,
                    ["c"],
                    gather(lengths).sum(),
                    None,
                    {},
                    None,
                    None,
                ),
                4,
                8 * world,
            ),
            # TRL >= 1.9 counts after _generate returns: nothing to fix
            8: (lambda: (["p"], completion_ids, None, ["c"], None, {}, None, None), None, None),
        }
        for size, (generate, index, expected) in layouts.items():
            Trainer = _fake_trainer_class([])
            Trainer._generate = lambda self, prompts, generate = generate: generate()
            cp.patch_rl_trainer(Trainer, "grpo_trainer")
            trainer = Trainer(_args())
            trainer.accelerator = types.SimpleNamespace(device = torch.device("cpu"), gather = gather)
            out = trainer._generate(["p"])
            assert type(out) is tuple and len(out) == size
            assert out[1] == [[1] * 2] * 4  # rank 0's completions on every peer
            if index is not None:
                # TRL's count is gather(lengths).sum() over every process, of the completions actually scored.
                assert int(out[index]) == expected, (size, int(out[index]), expected)
        queue.put((rank, "ok"))
    except BaseException:  # pragma: no cover - reported to the parent
        import traceback
        queue.put((rank, traceback.format_exc()))
    finally:
        dist.destroy_process_group()


def test_grpo_dapo_token_count_follows_the_adopted_completions():
    if "fork" not in mp.get_all_start_methods():
        pytest.skip("needs fork")
    ctx = mp.get_context("fork")
    queue = ctx.Queue()
    with tempfile.TemporaryDirectory() as tmp:
        init_file = os.path.join(tmp, "init")
        procs = [
            ctx.Process(target = _dapo_count_worker, args = (r, 2, init_file, queue)) for r in range(2)
        ]
        for p in procs:
            p.start()
        results = dict(queue.get(timeout = 120) for _ in procs)
        for p in procs:
            p.join(30)
    assert results == {0: "ok", 1: "ok"}, results


def _gather_both_paths(x, weight, group, rank, size):
    results = {}
    for fused in (False, True):
        calls = []
        original = cp._fused_gather_ok
        cp._fused_gather_ok = lambda group, fused = fused: fused
        rs, ag = cp._reduce_scatter, cp._all_gather_into

        def _count(name, fn):
            def wrapper(*args, **kwargs):
                calls.append(name)
                return fn(*args, **kwargs)

            return wrapper

        cp._reduce_scatter = _count("reduce_scatter", rs)
        cp._all_gather_into = _count("all_gather_into_tensor", ag)
        try:
            leaf = x.clone().requires_grad_(True)
            full = cp._GatherSequence.apply(leaf, group, rank, size)
            # Rank-dependent upstream grad: the reduction must really sum the peers.
            (full.float() * (weight + rank)).sum().backward()
        finally:
            cp._fused_gather_ok, cp._reduce_scatter = original, rs
            cp._all_gather_into = ag
        results[fused] = (full.detach(), leaf.grad, calls)
    return results


def _gather_worker(rank, world, init_file, queue):
    dist.init_process_group("gloo", init_method = f"file://{init_file}", rank = rank, world_size = world)
    try:
        group = dist.group.WORLD
        assert cp._fused_gather_ok(group)
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for shape in ((1, 3, 4), (3, 2, 5), (2, 4, 2, 3)):
                torch.manual_seed(100 + rank)
                x = torch.randn(shape).to(dtype)
                full_shape = (shape[0], shape[1] * world, *shape[2:])
                torch.manual_seed(7)
                weight = torch.randn(full_shape)
                got = _gather_both_paths(x, weight, group, rank, world)
                (old, old_grad, old_calls), (new, new_grad, new_calls) = got[False], got[True]
                assert old_calls == [] and new_calls == ["all_gather_into_tensor", "reduce_scatter"]
                assert new.shape == full_shape and new.dtype == dtype and new.is_contiguous()
                assert new._base is None
                assert torch.equal(old, new), (dtype, shape)
                assert new_grad.shape == shape and new_grad.dtype == dtype
                assert torch.equal(old_grad, new_grad), (dtype, shape)
                # And against a closed form: this rank's slice of sum_r (weight + r).
                local = shape[1]
                expect = sum((weight + r).to(dtype) for r in range(world)).narrow(
                    1, rank * local, local
                )
                torch.testing.assert_close(new_grad, expect)
        # A backend without the single-tensor collectives (MPI) keeps the list path.
        dist_get_backend = dist.get_backend
        try:
            dist.get_backend = lambda group = None: "mpi"
            assert not cp._fused_gather_ok(group)
            dist.get_backend = lambda group = None: "cpu:gloo,cuda:nccl"
            assert cp._fused_gather_ok(group)
        finally:
            dist.get_backend = dist_get_backend
        x = torch.randn(2, 3, 4)
        leaf = x.clone().requires_grad_(True)
        calls = []
        ag = cp._all_gather_into
        cp._all_gather_into = lambda *a, **k: calls.append(1)
        original = cp._fused_gather_ok
        cp._fused_gather_ok = lambda group: False
        try:
            full = cp._GatherSequence.apply(leaf, group, rank, world)
            full.sum().backward()
        finally:
            cp._all_gather_into, cp._fused_gather_ok = ag, original
        assert calls == [] and full.shape == (2, 6, 4)
        assert torch.equal(leaf.grad, torch.full_like(x, float(world)))
        queue.put((rank, "ok"))
    except BaseException:  # pragma: no cover - reported to the parent
        import traceback
        queue.put((rank, traceback.format_exc()))
    finally:
        dist.destroy_process_group()


def test_gather_sequence_fused_collectives_match_list_path_bitwise():
    if "fork" not in mp.get_all_start_methods():
        pytest.skip("needs fork")
    ctx = mp.get_context("fork")
    queue = ctx.Queue()
    with tempfile.TemporaryDirectory() as tmp:
        init_file = os.path.join(tmp, "init")
        procs = [
            ctx.Process(target = _gather_worker, args = (r, 2, init_file, queue)) for r in range(2)
        ]
        for p in procs:
            p.start()
        results = dict(queue.get(timeout = 120) for _ in procs)
        for p in procs:
            p.join(30)
    assert results == {0: "ok", 1: "ok"}, results
