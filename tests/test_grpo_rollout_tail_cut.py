# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Opt-in GRPO rollout tail cut (UNSLOTH_ROLLOUT_TAIL_CUT): a fake vLLM engine checks which rows are cut, that cut
rows come back as partial completions and are reported as truncated, and that nothing changes when it is off."""

import types

import pytest

try:
    from vllm.sampling_params import RequestOutputKind, SamplingParams
except (
    Exception
):  # no vLLM (or one that needs another transformers): the two names the tail cut uses
    import copy
    import enum

    class RequestOutputKind(enum.Enum):
        CUMULATIVE = 0
        DELTA = 1
        FINAL_ONLY = 2

    class SamplingParams:
        def __init__(
            self,
            max_tokens = 16,
            n = 1,
        ):
            self.max_tokens, self.n, self.output_kind = max_tokens, n, RequestOutputKind.CUMULATIVE

        def clone(self):
            return copy.copy(self)


VLLM_TYPES = (RequestOutputKind, SamplingParams)

from unsloth.models import _rollout_tail_cut as tc  # noqa: E402
from unsloth.models import rl_replacements  # noqa: E402

EOS, PAD = 2, 0


class FakeEngine:
    """Decodes one token per step per running request; request i stops (EOS) after lengths[i] tokens."""

    def __init__(self, lengths, prompts):
        self.lengths = lengths
        self.prompts = prompts
        self.reqs = {}
        self.order = []
        self.aborted = []
        self.kinds = []
        self.steps = 0

    def add_request(
        self,
        request_id,
        prompt,
        params,
        lora_request = None,
        priority = 0,
    ):
        self.kinds.append(params.output_kind)
        i = len(self.order)
        self.order.append(request_id)
        self.reqs[request_id] = dict(i = i, toks = [], kind = params.output_kind, done = False)
        return request_id

    def has_unfinished_requests(self):
        return any(not r["done"] for r in self.reqs.values())

    def abort_request(self, ids):
        self.aborted.extend(ids)
        for rid in ids:
            self.reqs[rid]["done"] = True

    def step(self):
        self.steps += 1
        outs = []
        for rid in self.order:
            r = self.reqs[rid]
            if r["done"]:
                continue
            n = len(r["toks"]) + 1
            finished = n >= self.lengths[r["i"]]
            r["toks"].append(EOS if finished else 100 + r["i"])
            r["done"] = finished
            if finished or r["kind"] == RequestOutputKind.CUMULATIVE:
                completion = types.SimpleNamespace(
                    token_ids = r["toks"], logprobs = None, finish_reason = "stop" if finished else None,
                    stop_reason = None,
                )  # fmt: skip
                outs.append(types.SimpleNamespace(
                    request_id = rid, prompt_token_ids = self.prompts[r["i"]], outputs = [completion],
                    finished = finished,
                ))  # fmt: skip
        return outs


def run_engine(engine, n, params):
    """What vLLM's LLM.generate does: add every request, step until done, keep finished outputs in order."""
    params.output_kind = RequestOutputKind.FINAL_ONLY  # LLM._add_request sets this
    for i in range(n):
        engine.add_request(str(i), {"prompt_token_ids": engine.prompts[i]}, params)
    done = []
    while engine.has_unfinished_requests():
        done += [o for o in engine.step() if o.finished]
    return sorted(done, key = lambda o: int(o.request_id))


def _wrap(engine, G, k_min, frac, min_tokens):
    state = tc.TailCutEngine(engine, G, k_min, frac, min_tokens, VLLM_TYPES)
    add, step = engine.add_request, engine.step
    engine.add_request = lambda *a, **k: state.add_request(add, *a, **k)
    engine.step = lambda: state.step(step)
    return state


def _batch(
    G = 4,
    groups = 4,
    long_row = 3,
    long_len = 400,
    short = 150,
):
    lengths, prompts = [], []
    for g in range(groups):
        for j in range(G):
            lengths.append(long_len if (g == 1 and j == long_row) else short + 10 * j)
            prompts.append([1, 50 + g])
    return lengths, prompts


def test_cut_only_when_every_running_row_is_expendable():
    lengths, prompts = _batch()
    params = SamplingParams(max_tokens = 500)
    base = run_engine(FakeEngine(lengths, prompts), len(lengths), params.clone())
    engine = FakeEngine(lengths, prompts)
    state = _wrap(engine, G = 4, k_min = 3, frac = 0.1, min_tokens = 128)
    out = run_engine(engine, len(lengths), params.clone())
    # Cumulative outputs were requested, so a partial completion exists to return.
    assert set(engine.kinds) == {RequestOutputKind.CUMULATIVE}
    assert [o.request_id for o in out] == [o.request_id for o in base]
    # Row 7 (group 1, the 400-token straggler) is cut once it is the only row left (step 180).
    assert engine.aborted == ["7"]
    assert len(out[7].outputs[0].token_ids) == 180
    assert out[7].outputs[0].token_ids[-1] != EOS and out[7].outputs[0].finish_reason == "abort"
    assert engine.steps == 180 < 400
    # Every other row is untouched.
    for a, b in zip(base, out):
        if a.request_id != "7":
            assert list(a.outputs[0].token_ids) == list(b.outputs[0].token_ids)
    assert state.cut_rows == [tuple(out[7].outputs[0].token_ids)] and state.num_cut == 1


@pytest.mark.parametrize(
    "kw",
    [
        dict(k_min = 4, frac = 0.1, min_tokens = 128),  # needs every sibling: never
        dict(k_min = 3, frac = 0.05, min_tokens = 128),  # budget int(0.05 * 16) = 0
        dict(k_min = 3, frac = 0.1, min_tokens = 1000),  # too few tokens generated
    ],
)
def test_no_cut_outside_the_limits(kw):
    lengths, prompts = _batch()
    engine = FakeEngine(lengths, prompts)
    _wrap(engine, G = 4, **kw)
    out = run_engine(engine, len(lengths), SamplingParams(max_tokens = 500))
    assert engine.aborted == [] and engine.steps == 400
    assert all(o.outputs[0].token_ids[-1] == EOS for o in out)


def test_no_cut_when_groups_do_not_line_up():
    lengths, prompts = _batch()
    prompts[5] = [
        1,
        99,
    ]  # group 1 holds two prompts: the arrival-order grouping is wrong, never cut
    engine = FakeEngine(lengths, prompts)
    _wrap(engine, G = 4, k_min = 3, frac = 0.1, min_tokens = 128)
    run_engine(engine, len(lengths), SamplingParams(max_tokens = 500))
    assert engine.aborted == []
    # A partial batch (not a multiple of G) is never cut either.
    lengths, prompts = _batch()
    engine = FakeEngine(lengths[:15], prompts[:15])
    _wrap(engine, G = 4, k_min = 3, frac = 0.2, min_tokens = 128)
    run_engine(engine, 15, SamplingParams(max_tokens = 500))
    assert engine.aborted == []


def _trainer(cut_rows = None, limit = 500):
    t = types.SimpleNamespace(max_completion_length = limit, use_vllm = False)
    if cut_rows is not None:
        t._unsloth_tail_cut_rows = cut_rows
    return t


def test_cut_rows_count_as_truncated_and_off_is_unchanged():
    is_trunc = rl_replacements._unsloth_grpo_is_truncated
    rows = [[5, 6, EOS], [7, 8, 9], [7, 8, 9], [3] * 500]
    eos_and_pad = [EOS, PAD]
    off = is_trunc(_trainer(), rows, None, eos_and_pad)
    assert off == [False, False, False, True]  # [7, 8, 9] ends on a stop string, not truncated
    assert is_trunc(_trainer([]), rows, None, eos_and_pad) == off
    assert is_trunc(_trainer(None), rows, None, eos_and_pad) == off
    # One cut row with the same tokens as an uncut stop-string row: only one of the two is flagged.
    assert is_trunc(_trainer([(7, 8, 9)]), rows, None, eos_and_pad) == [False, True, False, True]
    # Called twice per rollout in TRL (metrics, then the mask): the record is not consumed.
    t = _trainer([(7, 8, 9)])
    assert is_trunc(t, rows, None, eos_and_pad) == is_trunc(t, rows, None, eos_and_pad)


def test_settings_and_scope_off(monkeypatch):
    monkeypatch.delenv("UNSLOTH_ROLLOUT_TAIL_CUT", raising = False)
    assert tc.tail_cut_settings() is None
    engine = FakeEngine(*_batch())
    llm = types.SimpleNamespace(llm_engine = engine)
    trainer = types.SimpleNamespace(llm = llm, use_vllm = True, _unsloth_tail_cut_rows = [(1,)])
    scope = rl_replacements._unsloth_tail_cut_scope(trainer)
    with scope:
        assert "add_request" not in engine.__dict__ and "step" not in engine.__dict__
    assert trainer._unsloth_tail_cut_rows is None
    monkeypatch.setenv("UNSLOTH_ROLLOUT_TAIL_CUT", "6")
    assert tc.tail_cut_settings() == (6, 0.125, 128)
    monkeypatch.setenv("UNSLOTH_ROLLOUT_TAIL_CUT_MAX_FRAC", "2")
    assert tc.tail_cut_settings() is None


def test_scope_needs_mask_truncated_completions(monkeypatch):
    monkeypatch.setenv("UNSLOTH_ROLLOUT_TAIL_CUT", "3")
    engine = FakeEngine(*_batch())
    trainer = types.SimpleNamespace(
        llm = types.SimpleNamespace(llm_engine = engine), use_vllm = True, vllm_mode = "colocate",
        mask_truncated_completions = False, tools = None, rollout_func = None, num_generations = 4,
        model = types.SimpleNamespace(training = True),
    )  # fmt: skip
    with tc.rollout_tail_cut_scope(trainer):
        assert "step" not in engine.__dict__
    trainer.mask_truncated_completions = True
    # SimpleNamespace has no TRL _generate_and_score_completions, so the masking-site gate keeps it off.
    with tc.rollout_tail_cut_scope(trainer):
        assert "step" not in engine.__dict__


def test_scope_installs_and_restores(monkeypatch):
    real = tc.TailCutEngine
    monkeypatch.setattr(tc, "TailCutEngine", lambda *a: real(*a, VLLM_TYPES))
    monkeypatch.setenv("UNSLOTH_ROLLOUT_TAIL_CUT", "3")
    monkeypatch.setenv("UNSLOTH_ROLLOUT_TAIL_CUT_MAX_FRAC", "0.1")
    monkeypatch.setattr(tc, "_masking_site_patched", lambda cls: True)
    lengths, prompts = _batch()
    engine = FakeEngine(lengths, prompts)
    metrics = {"train": {"completions/tail_cut_ratio": []}}
    trainer = types.SimpleNamespace(
        llm = types.SimpleNamespace(llm_engine = engine), use_vllm = True, vllm_mode = "colocate",
        mask_truncated_completions = True, tools = None, rollout_func = None, num_generations = 4,
        model = types.SimpleNamespace(training = True), _metrics = metrics,
    )  # fmt: skip
    with tc.rollout_tail_cut_scope(trainer):
        out = run_engine(engine, len(lengths), SamplingParams(max_tokens = 500))
        # Visible inside the scope: TRL's _generate runs its truncation check (clipped_ratio) before returning.
        assert trainer._unsloth_tail_cut_rows == [tuple(out[7].outputs[0].token_ids)]
    assert "step" not in engine.__dict__ and "add_request" not in engine.__dict__
    assert engine.aborted == ["7"] and len(out) == 16
    assert trainer._unsloth_tail_cut_rows == [tuple(out[7].outputs[0].token_ids)]
    assert metrics["train"]["completions/tail_cut_ratio"] == [1 / 16]
    # Eval rollouts are never cut.
    engine = FakeEngine(lengths, prompts)
    trainer.llm = types.SimpleNamespace(llm_engine = engine)
    trainer.model.training = False
    with tc.rollout_tail_cut_scope(trainer):
        run_engine(engine, len(lengths), SamplingParams(max_tokens = 500))
    assert engine.aborted == []


def test_masking_site_gate_sees_through_method_wrappers():
    def _generate_and_score_completions(self):
        return _unsloth_grpo_is_truncated(self, completion_ids_list, prompt_ids_list, eos_and_pad)  # noqa: F821

    def wrap(original):  # the shape of rl.py's _wrap_grpo_generate_and_score
        def wrapped(self, *a, **k):
            return original(self, *a, **k)

        return wrapped

    class Patched:
        pass

    Patched._generate_and_score_completions = wrap(_generate_and_score_completions)
    assert tc._masking_site_patched(Patched)

    class Stock:
        def _generate_and_score_completions(self):
            return [ids[-1] not in eos_and_pad for ids in completion_ids_list]  # noqa: F821

    assert not tc._masking_site_patched(Stock)
