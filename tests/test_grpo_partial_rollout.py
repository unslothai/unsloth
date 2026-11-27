# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Opt-in GRPO partial rollout (UNSLOTH_GRPO_PARTIAL_ROLLOUT): a fake vLLM engine whose next token is a function of
the whole sequence so far, so a row cut and resumed under new weights must come out token for token as if it had run
in one piece, while each token keeps the logprob of the policy version that sampled it. Also checks group integrity,
the per-step trained group counts, the max-age bound, the refusals, and in float64 that the Unsloth GRPO loss of a
carried batch equals the same tokens scored as a fresh rollout with the stored logprobs."""

import copy
import enum
import types
from collections import defaultdict

import pytest
import torch

from unsloth.models import _partial_rollout as pr  # noqa: E402

EOS = 2
MAX_TOKENS = 24


class FinishReason(enum.Enum):
    ABORT = "abort"


class Params:
    def __init__(
        self,
        max_tokens = MAX_TOKENS,
        **kw,
    ):
        self.max_tokens, self.n, self.min_tokens = max_tokens, 1, 0
        self.stop, self.presence_penalty, self.frequency_penalty = None, 0.0, 0.0
        self.output_kind = None
        for k, v in kw.items():
            setattr(self, k, v)

    def clone(self):
        return copy.copy(self)


def next_token(
    seq,
    prompt_len,
    version,
    draw = 0,
):
    """A fresh row's first token comes from `draw` (the sampler's randomness); every later token, the row's length and
    every logprob are a function of the sequence so far, so they do not depend on how generation was split."""
    generated = len(seq) - prompt_len
    if generated == 0:
        token = 10 + draw % 50
    else:
        length = 3 + (seq[prompt_len] * 7) % 18
        h = (sum((i + 1) * t for i, t in enumerate(seq)) * 2654435761) % 1000003
        token = EOS if generated + 1 >= length else 10 + h % 50
    h = (sum((i + 1) * t for i, t in enumerate(seq + [token])) * 40503) % 1000003
    return token, -((h % 97) / 100.0) - 0.125 * version


class Completion:
    def __init__(self, token_ids, logprobs, finish_reason):
        self.token_ids, self.logprobs, self.finish_reason, self.stop_reason = (
            token_ids,
            logprobs,
            finish_reason,
            None,
        )


class Output:
    def __init__(self, request_id, prompt_token_ids, completion, finished):
        self.request_id, self.prompt_token_ids, self.outputs, self.finished = (
            request_id, prompt_token_ids, [completion], finished,
        )  # fmt: skip


class FakeState:
    def __init__(self, req):
        self.req = req

    def make_request_output(self, new_token_ids, pooling_output, finish_reason, stop_reason):
        r = self.req
        return Output(
            r["rid"], r["prompt"], Completion(list(r["toks"]), list(r["lps"]), finish_reason), True
        )


class FakeEngine:
    """One token per step for at most `slots` running requests (the rest wait), like a scheduler with max_num_seqs."""

    def __init__(self, slots = 6):
        self.slots = slots
        self.version = 0
        self.reqs = {}
        self.order = []
        self.aborted = []
        self.added = []  # (prompt ids, max_tokens)
        self.output_processor = types.SimpleNamespace(external_req_ids = {}, request_states = {})
        self.draws = 0

    def add_request(
        self,
        request_id,
        prompt,
        params,
        lora_request = None,
        priority = 0,
    ):
        ids = list(prompt["prompt_token_ids"])
        self.added.append((ids, params.max_tokens))
        req = dict(rid = request_id, prompt = ids, toks = [], lps = [], max = params.max_tokens, done = False)
        self.reqs[request_id] = req
        self.order.append(request_id)
        self.output_processor.external_req_ids[request_id] = [request_id + "-int"]
        self.output_processor.request_states[request_id + "-int"] = FakeState(req)
        return request_id

    def has_unfinished_requests(self):
        return any(not r["done"] for r in self.reqs.values())

    def abort_request(self, ids):
        self.aborted.extend(ids)
        for rid in ids:
            self.reqs[rid]["done"] = True
            self.output_processor.request_states.pop(rid + "-int", None)

    def step(self):
        outs = []
        live = [rid for rid in self.order if not self.reqs[rid]["done"]][: self.slots]
        for rid in live:
            r = self.reqs[rid]
            seq = r["prompt"] + r["toks"]
            self.draws += 1
            token, lp = next_token(
                seq, self._prompt_len(r), self.version, (self.draws * 7919) % 1009
            )
            r["toks"].append(token)
            r["lps"].append(lp)
            finished = token == EOS or len(r["toks"]) >= r["max"]
            if finished:
                r["done"] = True
                self.output_processor.request_states.pop(rid + "-int", None)
                outs.append(
                    Output(
                        rid, r["prompt"], Completion(list(r["toks"]), list(r["lps"]), "stop"), True
                    )
                )
        return outs

    def _prompt_len(self, r):
        # the original prompt is the leading run of ids >= 100 (generated ids are < 100)
        n = 0
        for t in r["prompt"]:
            if t < 100:
                break
            n += 1
        return n


def reference(prompt_ids, first):
    """The row generated in one piece from its first token (logprobs at version 0)."""
    seq, toks, lps = list(prompt_ids), [], []
    while True:
        token, lp = next_token(seq, len(prompt_ids), 0, first - 10)
        toks.append(token)
        lps.append(lp)
        seq.append(token)
        if token == EOS or len(toks) >= MAX_TOKENS:
            return toks, lps


class FakeTrainer:
    def __init__(
        self,
        engine,
        groups = 4,
        G = 4,
        steps_per_generation = 2,
        is_on = True,
    ):
        self.use_vllm = True
        self.vllm_mode = "colocate"
        self.num_generations = G
        self.vllm_importance_sampling_correction = is_on
        self.tools = None
        self.rollout_func = None
        self.environment_factories = None
        self.model = types.SimpleNamespace(training = True)
        self.accelerator = types.SimpleNamespace(num_processes = 1)
        self.args = types.SimpleNamespace(
            steps_per_generation = steps_per_generation, vllm_tensor_parallel_size = 1
        )
        self.state = types.SimpleNamespace(global_step = 0)
        self.vllm_generation = types.SimpleNamespace(llm = types.SimpleNamespace(llm_engine = engine))
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self.engine = engine
        self.counter = 0
        self.groups = groups
        self.calls = []
        self.params_kw = {}

    # TRL 1.18 shapes
    def _tokenize_prompts(self, prompts):
        return [[100 + p, 101, 102 + p] for p in prompts], None, {}

    def _generate_single_turn(self, prompt_ids, images, multimodal_fields, num_generations):
        params = Params(**self.params_kw)
        engine = self.engine
        for ids in prompt_ids:
            engine.add_request(str(self.counter), {"prompt_token_ids": ids}, params)
            self.counter += 1
        done = []
        while engine.has_unfinished_requests():
            done += [o for o in engine.step() if o.finished]
        done.sort(key = lambda o: int(o.request_id))
        assert len(done) == len(prompt_ids)
        return (
            [list(o.outputs[0].token_ids) for o in done],
            [list(o.outputs[0].logprobs) for o in done],
            None,
        )


def original_gas(trainer, inputs):
    """What TRL's _generate_and_score_completions needs from generation: prompts in, completions back, in order."""
    prompt_ids, _, _ = trainer._tokenize_prompts([x["prompt"] for x in inputs])
    completion_ids, logprobs, _ = trainer._generate_single_turn(
        prompt_ids, None, {}, trainer.num_generations
    )
    trainer.calls.append(len(inputs))
    return dict(
        inputs = list(inputs), prompt_ids = prompt_ids, completion_ids = completion_ids, logprobs = logprobs
    )


def batch(
    step,
    groups = 4,
    G = 4,
):
    return [dict(prompt = step * groups + g, answer = g) for g in range(groups) for _ in range(G)]


@pytest.fixture
def on(monkeypatch):
    for k in (
        "UNSLOTH_ROLLOUT_TAIL_CUT",
        "UNSLOTH_GRPO_TRUNCATED_SUPPORT",
        "UNSLOTH_GRPO_ROUTING_REPLAY",
    ):
        monkeypatch.delenv(k, raising = False)
    monkeypatch.setenv("UNSLOTH_GRPO_PARTIAL_ROLLOUT", "0.25")
    monkeypatch.setenv("UNSLOTH_GRPO_PARTIAL_ROLLOUT_MAX_AGE", "2")
    monkeypatch.setattr(pr, "_abort_finish_reason", lambda: FinishReason.ABORT)
    pr._WARNED.clear()
    yield


def run_steps(trainer, steps):
    outs = []
    for step in range(steps):
        trainer.engine.version = trainer.state.global_step
        outs.append(
            pr.partial_rollout_generate_and_score(
                trainer, original_gas, batch(step, trainer.groups)
            )
        )
        trainer.state.global_step += 1
    return outs


def test_settings(monkeypatch):
    monkeypatch.delenv("UNSLOTH_GRPO_PARTIAL_ROLLOUT", raising = False)
    assert pr.partial_rollout_settings() is None
    monkeypatch.setenv("UNSLOTH_GRPO_PARTIAL_ROLLOUT", "1")
    assert pr.partial_rollout_settings() == (0.25, 2)
    monkeypatch.setenv("UNSLOTH_GRPO_PARTIAL_ROLLOUT", "0.5")
    monkeypatch.setenv("UNSLOTH_GRPO_PARTIAL_ROLLOUT_MAX_AGE", "1")
    assert pr.partial_rollout_settings() == (0.5, 1)
    for bad in ("1.5", "x", "-0.1"):
        monkeypatch.setenv("UNSLOTH_GRPO_PARTIAL_ROLLOUT", bad)
        assert pr.partial_rollout_settings() is None
    assert pr._unit(8, 8) == 1 and pr._unit(4, 8) == 2 and pr._unit(2, 8) == 4
    assert (
        pr._spare_groups(0.25, 8, 1) == 2
        and pr._spare_groups(0.25, 8, 2) == 2
        and pr._spare_groups(0.9, 4, 2) == 2
    )


def test_off_is_passthrough(monkeypatch):
    monkeypatch.delenv("UNSLOTH_GRPO_PARTIAL_ROLLOUT", raising = False)
    engine = FakeEngine()
    trainer = FakeTrainer(engine)
    inputs = batch(0)
    out = pr.partial_rollout_generate_and_score(trainer, original_gas, inputs)
    assert out["inputs"] == inputs and not engine.aborted
    assert "add_request" not in engine.__dict__ and not hasattr(trainer, "_unsloth_partial_rollout")


def test_carry_resume_is_token_exact(on):
    engine = FakeEngine(slots = 6)
    trainer = FakeTrainer(engine, groups = 4, G = 4, steps_per_generation = 2)
    outs = run_steps(trainer, 6)
    # first step trains B - E = 3 groups, every later step B = 4; E = 1 group stays carried
    assert [len(o["inputs"]) // 4 for o in outs] == [3, 4, 4, 4, 4, 4]
    assert len(trainer._unsloth_partial_rollout.carry) == 1
    assert engine.aborted, "the rollout never stopped early"
    resumed = [(ids, mx) for ids, mx in engine.added if len(ids) > 3]
    assert resumed, "no carried row was resumed"
    for ids, mx in resumed:
        assert mx == MAX_TOKENS - (len(ids) - 3)
    seen_stale = 0
    for step, out in enumerate(outs):
        rows = out["inputs"]
        for g in range(0, len(rows), 4):  # whole groups of one prompt
            assert len({r["prompt"] for r in rows[g : g + 4]}) == 1
        for pids, toks, lps in zip(out["prompt_ids"], out["completion_ids"], out["logprobs"]):
            want, base_lps = reference(pids, toks[0])
            assert toks == want, "a carried row did not continue exactly where it stopped"
            assert toks[-1] == EOS or len(toks) == MAX_TOKENS  # only finished rows train
            # each token keeps the logprob of the version that sampled it: base - 0.125 * version
            offsets = [round((b - l) / 0.125) for b, l in zip(base_lps, lps)]
            assert all(0 <= o <= step for o in offsets) and offsets == sorted(offsets)
            seen_stale += any(o != step for o in offsets)
    assert seen_stale > 0
    last = trainer._unsloth_partial_rollout_last
    assert all(v == sorted(v) for v in last["versions"])
    assert trainer._metrics["train"]["partial_rollout/trained_groups"] == [
        3.0,
        4.0,
        4.0,
        4.0,
        4.0,
        4.0,
    ]
    assert max(trainer._metrics["train"]["partial_rollout/token_staleness_mean"]) > 0


def test_max_age_bounds_staleness(on, monkeypatch):
    monkeypatch.setenv("UNSLOTH_GRPO_PARTIAL_ROLLOUT_MAX_AGE", "1")
    engine = FakeEngine(slots = 4)
    trainer = FakeTrainer(engine, groups = 4, G = 4, steps_per_generation = 2)
    for step in range(6):
        engine.version = trainer.state.global_step
        out = pr.partial_rollout_generate_and_score(trainer, original_gas, batch(step))
        for v in trainer._unsloth_partial_rollout_last["versions"]:
            assert not v or step - min(v) <= 1
        assert all(g.age <= 1 for g in trainer._unsloth_partial_rollout.carry)
        trainer.state.global_step += 1


@pytest.mark.parametrize(
    "setup, reason",
    [
        (lambda t, m: setattr(t, "vllm_importance_sampling_correction", False), "importance"),
        (lambda t, m: m.setenv("UNSLOTH_ROLLOUT_TAIL_CUT", "6"), "TAIL_CUT"),
        (lambda t, m: setattr(t, "tools", [print]), "tool"),
        (
            lambda t, m: setattr(t, "accelerator", types.SimpleNamespace(num_processes = 2)),
            "single process",
        ),
    ],
)
def test_refusals_fall_back(on, monkeypatch, setup, reason):
    engine = FakeEngine()
    trainer = FakeTrainer(engine)
    setup(trainer, monkeypatch)
    out = pr.partial_rollout_generate_and_score(trainer, original_gas, batch(0))
    assert len(out["inputs"]) == 16 and not engine.aborted
    assert reason in trainer._unsloth_partial_rollout_refused


def test_uncuttable_params_train_everything(on):
    engine = FakeEngine()
    trainer = FakeTrainer(engine)
    trainer.params_kw = dict(stop = ["\n\n"])
    out = pr.partial_rollout_generate_and_score(trainer, original_gas, batch(0))
    assert len(out["inputs"]) == 16 and not engine.aborted  # identical to a normal rollout
    assert "stop" in trainer._unsloth_partial_rollout_refused
    out = pr.partial_rollout_generate_and_score(trainer, original_gas, batch(1))
    assert len(out["inputs"]) == 16 and not engine.aborted


def test_wrapper_keeps_source_reachable():
    import inspect

    class T:
        def _generate_and_score_completions(self, inputs):
            return "_unsloth_grpo_is_truncated(self, completion_ids_list"

    pr.wrap_grpo_partial_rollout(T)
    pr.wrap_grpo_partial_rollout(T)  # idempotent
    fn = T._generate_and_score_completions
    assert (
        fn._unsloth_partial_rollout_wrapped
        and fn.__wrapped__.__name__ == "_generate_and_score_completions"
    )
    assert "_unsloth_grpo_is_truncated(self, completion_ids_list" in inspect.getsource(fn)


# --------------------------------------------------------------------------------------------- float64 loss check
def _pad(rows, value, dtype):
    width = max(len(r) for r in rows)
    out = torch.full((len(rows), width), value, dtype = dtype)
    mask = torch.zeros((len(rows), width), dtype = torch.long)
    for i, r in enumerate(rows):
        out[i, : len(r)] = torch.tensor(r, dtype = dtype)
        mask[i, : len(r)] = 1
    return out, mask


def _policy_logps(completion_ids, seed):
    """A float64 stand-in for the trainer's current-policy per-token logprobs (a function of the token ids)."""
    gen = torch.Generator().manual_seed(seed)
    table = -torch.rand(200, dtype = torch.float64, generator = gen) * 2
    pos = torch.arange(completion_ids.shape[1], dtype = torch.float64) * 1e-3
    return (table[completion_ids] - pos).requires_grad_(True)


def _zoo_loss(new, sampling, completion_ids, mask, advantages):
    rr = pytest.importorskip("unsloth_zoo.rl_replacements")
    old = new.detach()  # num_iterations = 1: the old policy is the current one
    kwargs = dict(
        loss_type = "dapo", epsilon_low = 0.2, epsilon_high = 0.2, max_completion_length = MAX_TOKENS,
        current_gradient_accumulation_steps = 1, num_items_in_batch = mask.sum(), steps_per_generation = 1,
        num_processes = 1, importance_sampling_level = "token", use_vllm = True,
        vllm_importance_sampling_correction = True, vllm_importance_sampling_mode = "token_truncate",
        vllm_importance_sampling_clip_max = 3.0,
    )  # fmt: skip
    loss, *_ = rr.grpo_compute_loss(
        None, new, old, sampling, completion_ids, mask, 0.0, advantages, **kwargs
    )
    (grad,) = torch.autograd.grad(loss, [new])
    return loss.detach(), grad


def _reference_loss(new, sampling, mask, advantages):
    """DAPO, beta = 0, num_iterations = 1, token-level truncated IS (cap 3) in float64."""
    ratio = torch.exp(new - new.detach())
    w = torch.exp(new.detach() - sampling).clamp(max = 3.0)
    per_token = (
        -torch.min(ratio * advantages[:, None], ratio.clamp(0.8, 1.2) * advantages[:, None]) * w
    )
    loss = (per_token * mask).sum() / mask.sum()
    (grad,) = torch.autograd.grad(loss, [new])
    return loss.detach(), grad


def test_float64_loss_of_carried_rows_equals_fresh_rollout(on):
    engine = FakeEngine(slots = 6)
    trainer = FakeTrainer(engine, groups = 4, G = 4, steps_per_generation = 2)
    outs = run_steps(trainer, 4)
    out = outs[-1]
    versions = trainer._unsloth_partial_rollout_last["versions"]
    assert any(v and min(v) < 3 for v in versions), "the checked batch holds no carried token"
    completion_ids, mask = _pad(out["completion_ids"], 0, torch.long)
    carried_lps, _ = _pad(out["logprobs"], 0.0, torch.float64)
    # The same tokens as a fresh rollout: one generate call per row, logprobs taken from the engine at the version
    # each token was sampled under (what the stored logprobs are supposed to be).
    fresh_lps = []
    for pids, toks, v in zip(out["prompt_ids"], out["completion_ids"], versions):
        seq, row = list(pids), []
        for t, ver in zip(toks, v):
            token, lp = next_token(seq, len(pids), ver, toks[0] - 10)
            assert token == t
            row.append(lp)
            seq.append(t)
        fresh_lps.append(row)
    fresh_lps, _ = _pad(fresh_lps, 0.0, torch.float64)
    assert torch.equal(carried_lps, fresh_lps)
    rewards = torch.tensor([float(sum(t) % 5) for t in out["completion_ids"]], dtype = torch.float64)
    grouped = rewards.view(-1, 4)
    advantages = (
        (grouped - grouped.mean(1, keepdim = True)) / (grouped.std(1, keepdim = True) + 1e-4)
    ).view(-1)
    new = _policy_logps(completion_ids, 3407)
    got = _zoo_loss(new, carried_lps, completion_ids, mask, advantages)
    want = _zoo_loss(
        _policy_logps(completion_ids, 3407), fresh_lps, completion_ids, mask, advantages
    )
    assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])
    ref = _reference_loss(_policy_logps(completion_ids, 3407), fresh_lps, mask, advantages)
    torch.testing.assert_close(got[0], ref[0], rtol = 1e-12, atol = 1e-12)
    torch.testing.assert_close(got[1], ref[1], rtol = 1e-12, atol = 1e-12)
    # Negative control: rescoring the carried tokens at the current version (losing the stored logprobs) changes
    # the IS weights and the loss, so the equality above is not vacuous.
    current = []
    for pids, toks in zip(out["prompt_ids"], out["completion_ids"]):
        seq, row = list(pids), []
        for t in toks:
            row.append(next_token(seq, len(pids), 3, toks[0] - 10)[1])
            seq.append(t)
        current.append(row)
    current, _ = _pad(current, 0.0, torch.float64)
    wrong = _zoo_loss(
        _policy_logps(completion_ids, 3407), current, completion_ids, mask, advantages
    )
    assert not torch.allclose(wrong[0], got[0])


# ------------------------------------------------------------------------------------------ review regressions
def test_two_value_single_turn_keeps_its_arity(on):
    """TRL 1.0-1.16 return (completion_ids, logprobs); the replay must return the same shape."""
    engine = FakeEngine(slots = 6)
    trainer = FakeTrainer(engine)
    three = trainer._generate_single_turn
    trainer._generate_single_turn = lambda *a, **k: three(*a, **k)[:2]

    def gas(trainer, inputs):
        prompt_ids, _, _ = trainer._tokenize_prompts([x["prompt"] for x in inputs])
        completion_ids, logprobs = trainer._generate_single_turn(prompt_ids, None, {}, 4)
        return dict(inputs = list(inputs), completion_ids = completion_ids)

    for step in range(3):
        engine.version = step
        out = pr.partial_rollout_generate_and_score(trainer, gas, batch(step))
        trainer.state.global_step += 1
    assert len(out["inputs"]) == 16 and engine.aborted


@pytest.mark.parametrize(
    "field, value",
    [("seed", 42), ("truncate_prompt_tokens", 3), ("repetition_detection", object())],
)
def test_history_or_rng_dependent_params_never_cut(on, field, value):
    engine = FakeEngine()
    trainer = FakeTrainer(engine)
    trainer.params_kw = {field: value}
    out = pr.partial_rollout_generate_and_score(trainer, original_gas, batch(0))
    assert len(out["inputs"]) == 16 and not engine.aborted
    assert field in trainer._unsloth_partial_rollout_refused


def test_context_derived_budget_survives_resume(on):
    engine = FakeEngine(slots = 6)
    trainer = FakeTrainer(engine)
    trainer.params_kw = dict(max_tokens = None)
    engine_add = engine.add_request

    def add(request_id, prompt, params, **kw):
        if params.max_tokens is None:  # what vLLM does: budget from max_model_len - prompt
            params = params.clone()
            params.max_tokens = MAX_TOKENS + 3 - len(prompt["prompt_token_ids"])
        return engine_add(request_id, prompt, params, **kw)

    engine.add_request = add
    run_steps(trainer, 3)
    resumed = [(ids, mx) for ids, mx in engine.added if len(ids) > 3]
    assert resumed and all(mx == MAX_TOKENS - (len(ids) - 3) for ids, mx in resumed)


def test_sampling_mask_backend_is_refused(on):
    engine = FakeEngine()
    trainer = FakeTrainer(engine)
    trainer.vllm_generation.return_sampling_mask = True
    out = pr.partial_rollout_generate_and_score(trainer, original_gas, batch(0))
    assert len(out["inputs"]) == 16 and not engine.aborted
    assert "sampling-support" in trainer._unsloth_partial_rollout_refused


def test_failed_scoring_keeps_the_previous_carry(on):
    engine = FakeEngine(slots = 6)
    trainer = FakeTrainer(engine)
    run_steps(trainer, 2)
    before = [
        (g.serial, [list(r.tokens) for r in g.rows]) for g in trainer._unsloth_partial_rollout.carry
    ]

    def boom(trainer, inputs):
        original_gas(trainer, inputs)
        raise RuntimeError("reward server down")

    with pytest.raises(RuntimeError, match = "reward server down"):
        pr.partial_rollout_generate_and_score(trainer, boom, batch(2))
    after = [
        (g.serial, [list(r.tokens) for r in g.rows]) for g in trainer._unsloth_partial_rollout.carry
    ]
    assert after == before
    out = pr.partial_rollout_generate_and_score(trainer, original_gas, batch(2))  # the retry
    serials = [g.serial for g in trainer._unsloth_partial_rollout.carry]
    assert len(out["inputs"]) == 16 and len(set(serials)) == len(serials)


def test_logprob_convention_is_set_before_generating(on, monkeypatch):
    vu = pytest.importorskip("unsloth_zoo.vllm_utils")
    calls = []
    monkeypatch.setattr(
        vu,
        "set_vllm_untruncated_logprobs",
        lambda enabled, llm: calls.append(("set", len(engine.added))) or True,
        raising = False,
    )
    engine = FakeEngine()
    trainer = FakeTrainer(engine)
    pr.partial_rollout_generate_and_score(trainer, original_gas, batch(0))
    assert calls and calls[0] == ("set", 0) and trainer._unsloth_vllm_untruncated_logprobs is True


def test_new_training_run_drops_the_carry(on):
    engine = FakeEngine(slots = 6)
    trainer = FakeTrainer(engine)
    run_steps(trainer, 2)
    assert trainer._unsloth_partial_rollout.carry
    trainer.state = types.SimpleNamespace(global_step = 0)  # train() / resume_from_checkpoint
    out = pr.partial_rollout_generate_and_score(trainer, original_gas, batch(9))
    assert {r["prompt"] for r in out["inputs"]} <= {
        9 * 4 + g for g in range(4)
    }  # nothing from the old run
    assert len(out["inputs"]) == 12  # a fresh pool: B - E


@pytest.mark.parametrize(
    "field, value",
    [("trace_decode_token_ids", [1, 2]), ("prompt_logprobs", 0), ("watermarking", True)],
)
def test_more_history_dependent_params_never_cut(on, field, value):
    engine = FakeEngine()
    trainer = FakeTrainer(engine)
    trainer.params_kw = {field: value}
    out = pr.partial_rollout_generate_and_score(trainer, original_gas, batch(0))
    assert len(out["inputs"]) == 16 and not engine.aborted
