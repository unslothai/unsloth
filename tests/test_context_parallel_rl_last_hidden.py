# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Context parallelism for GRPO's hidden-state forward: only the last hidden state is gathered, and only
where nothing else can see the others (CPU, gloo, 2 ranks, through the installed GRPO wrapper)."""

import os
import tempfile
import threading
import types
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

import unsloth  # noqa: F401
from unsloth import context_parallel as cp
from unsloth.models import rl as rl_module

WORLD = 2


class _Decoder(torch.nn.Module):
    """Causal toy decoder with three hidden states (embeddings, two layers); one gather stands in for attention."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.embed_tokens = torch.nn.Embedding(50, 8)
        self.layers = torch.nn.ModuleList([torch.nn.Linear(8, 8), torch.nn.Linear(8, 8)])

    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        position_ids = None,
        output_hidden_states = False,
        **kwargs,
    ):
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1]).expand_as(input_ids)
        h = self.embed_tokens(input_ids) * (1 + 0.01 * position_ids[..., None].float())
        states = [h]
        manager = cp.get_cp_manager()
        if manager is not None:
            rank = dist.get_rank()
            full = cp.gather_sequence(h, manager)
            prefix = full.cumsum(1)[:, rank * h.shape[1] : (rank + 1) * h.shape[1]]
        else:
            prefix = h.cumsum(1)
        h = torch.tanh(self.layers[0](prefix))
        states.append(h)
        h = torch.tanh(self.layers[1](h))
        states.append(h)
        return BaseModelOutputWithPast(
            last_hidden_state = h, hidden_states = tuple(states) if output_hidden_states else None
        )


class _LM(torch.nn.Module):
    """HF-style CausalLM: reads the decoder's last state, hands its hidden_states on (identity head)."""

    def __init__(self, decoder):
        super().__init__()
        self.model = decoder
        self.before = None

    def forward(
        self,
        input_ids = None,
        output_hidden_states = None,
        return_dict = None,
        logits_to_keep = 0,
        **kwargs,
    ):
        if self.before is not None:
            self.before()
        outputs = self.model(
            input_ids = input_ids, output_hidden_states = output_hidden_states, **kwargs
        )
        hidden_states = outputs.last_hidden_state
        logits = hidden_states[:, -logits_to_keep:, :]
        return CausalLMOutputWithPast(logits = logits, hidden_states = outputs.hidden_states)


class _ConsumingLM(torch.nn.Module):
    """Reads an intermediate state inside its own forward; optionally refuses the one-position limiter."""

    def __init__(
        self,
        decoder,
        refuse_limiter = False,
    ):
        super().__init__()
        self.model = decoder
        self.refuse_limiter = refuse_limiter
        self.probe = None

    def forward(
        self,
        input_ids = None,
        output_hidden_states = None,
        return_dict = None,
        logits_to_keep = 0,
        **kwargs,
    ):
        if self.refuse_limiter and logits_to_keep == 1:
            raise TypeError("forward() got an unsupported logits_to_keep")
        outputs = self.model(
            input_ids = input_ids, output_hidden_states = output_hidden_states, **kwargs
        )
        if outputs.hidden_states is not None:
            self.probe = outputs.hidden_states[1]
        logits = outputs.last_hidden_state[:, -logits_to_keep:, :]
        return CausalLMOutputWithPast(logits = logits, hidden_states = outputs.hidden_states)


class _DroppingLM(torch.nn.Module):
    """Accepts the limiter but never returns hidden states: the wrapper re-runs it for real logits."""

    def __init__(self, decoder):
        super().__init__()
        self.model = decoder

    def forward(
        self,
        input_ids = None,
        output_hidden_states = None,
        return_dict = None,
        logits_to_keep = 0,
        **kwargs,
    ):
        outputs = self.model(
            input_ids = input_ids, output_hidden_states = output_hidden_states, **kwargs
        )
        logits = outputs.last_hidden_state[:, -logits_to_keep:, :]
        return CausalLMOutputWithPast(logits = logits, hidden_states = None)


class _MappingLM(torch.nn.Module):
    """No logits limiter, returns a plain dict: getattr finds no hidden states, so the caller gets the mapping."""

    def __init__(self, decoder):
        super().__init__()
        self.model = decoder

    def forward(
        self,
        input_ids = None,
        output_hidden_states = None,
        return_dict = None,
    ):
        outputs = self.model(input_ids = input_ids, output_hidden_states = output_hidden_states)
        return {"logits": outputs.last_hidden_state, "hidden_states": outputs.hidden_states}

    # Claims passthrough (wrongly for the mapping): the wrapper itself must still not hand back a cut tuple.
    forward._unsloth_cp_hidden_states_passthrough = True


def _manager(rank):
    manager = object.__new__(cp.ContextParallelManager)
    manager.size = WORLD
    manager.backend = "ulysses"
    manager.mesh = types.SimpleNamespace(
        get_local_rank = lambda: rank, get_group = lambda: dist.group.WORLD
    )
    return manager


class _Env:
    def __init__(self, rank):
        torch.manual_seed(1)
        self.rank = rank
        self.ids = torch.randint(1, 50, (2, 7))
        self.full = _Decoder()(input_ids = self.ids, output_hidden_states = True)
        self.manager = _manager(rank)
        self.calls = []

    def attach(self, lm_cls, **kw):
        decoder = _Decoder()
        lm = lm_cls(decoder, **kw)
        cp.attach_rl_hooks(self.manager, lm)
        assert rl_module._install_grpo_hidden_states_forward_wrapper(lm)
        return lm, decoder

    def plain_decoder(self):
        decoder = _Decoder()
        cp.attach_rl_hooks(self.manager, torch.nn.ModuleDict({"model": decoder}))
        return decoder

    def assert_states(self, states):
        assert len(states) == len(self.full.hidden_states), len(states)
        for got, want in zip(states, self.full.hidden_states):
            torch.testing.assert_close(got, want, rtol = 1e-5, atol = 1e-6)


def _run(case):
    def worker(rank, init_file):
        dist.init_process_group(
            "gloo", store = dist.FileStore(init_file, WORLD), rank = rank, world_size = WORLD,
            timeout = timedelta(seconds = 60),
        )  # fmt: skip
        os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"
        env = _Env(rank)
        gather = cp.gather_sequence

        def counting(x, m):
            env.calls.append(tuple(x.shape))
            return gather(x, m)

        cp.gather_sequence = counting
        try:
            with cp._manager_scope(env.manager):
                case(env)
        finally:
            cp.gather_sequence = gather
            dist.destroy_process_group()

    if "fork" not in mp.get_all_start_methods():
        pytest.skip("needs fork")
    with tempfile.TemporaryDirectory() as tmp:
        mp.start_processes(
            worker, args = (os.path.join(tmp, "init"),), nprocs = WORLD, start_method = "fork"
        )


def _last_only(env):
    lm, decoder = env.attach(_LM)
    # The stand-in attention gather plus the last state: 1 hidden-state gather instead of 3.
    out = lm(input_ids = env.ids)
    assert len(env.calls) == 2, env.calls
    torch.testing.assert_close(out.logits, env.full.last_hidden_state, rtol = 1e-5, atol = 1e-6)
    assert out.hidden_states is None
    # An ordinary forward afterwards gathers every state again.
    del env.calls[:]
    every = decoder(input_ids = env.ids, output_hidden_states = True)
    assert len(env.calls) == 4, env.calls
    env.assert_states(every.hidden_states)
    # A caller that asks for the hidden states itself is not scoped.
    del env.calls[:]
    lm(input_ids = env.ids, output_hidden_states = True)
    assert len(env.calls) == 4, env.calls


def test_grpo_wrapper_gathers_only_the_last_state():
    _run(_last_only)


def _other_thread(env):
    # Finding 1: another thread's decoder forward, overlapping the GRPO forward, keeps every state.
    lm, _ = env.attach(_LM)
    other = env.plain_decoder()
    seen = []

    def side():
        seen.append(other(input_ids = env.ids, output_hidden_states = True).hidden_states)

    def before():
        thread = threading.Thread(target = side)
        thread.start()
        thread.join(60)

    lm.before = before
    out = lm(input_ids = env.ids)
    env.assert_states(seen[0])
    torch.testing.assert_close(out.logits, env.full.last_hidden_state, rtol = 1e-5, atol = 1e-6)


def test_scope_does_not_leak_to_another_thread():
    _run(_other_thread)


def _nested_decoder(env):
    # Finding 2: a different decoder called on the same thread inside the GRPO forward (a nested policy or
    # reference forward) keeps every state.
    lm, _ = env.attach(_LM)
    other = env.plain_decoder()
    seen = []
    lm.before = lambda: seen.append(
        other(input_ids = env.ids, output_hidden_states = True).hidden_states
    )
    out = lm(input_ids = env.ids)
    env.assert_states(seen[0])
    torch.testing.assert_close(out.logits, env.full.last_hidden_state, rtol = 1e-5, atol = 1e-6)


def test_scope_does_not_reach_a_nested_decoder():
    _run(_nested_decoder)


def _later_hook(env, ranks):
    # Finding 4: a forward hook registered after the CP hook reads states by index; a hook on one rank only
    # must not make the peers enter different gathers.
    lm, decoder = env.attach(_LM)
    seen = []
    if env.rank in ranks:
        decoder.register_forward_hook(
            lambda module, args, output: seen.append(output["hidden_states"])
        )
    out = lm(input_ids = env.ids)
    if env.rank in ranks:
        env.assert_states(seen[0])
    assert len(env.calls) == 4, env.calls
    torch.testing.assert_close(out.logits, env.full.last_hidden_state, rtol = 1e-5, atol = 1e-6)


def test_later_decoder_hook_sees_every_state():
    _run(lambda env: _later_hook(env, (0, 1)))


def test_later_decoder_hook_on_one_rank_keeps_ranks_in_step():
    _run(lambda env: _later_hook(env, (0,)))


def _consuming(env, refuse_limiter):
    # Findings 5 and 6: a forward that reads an intermediate state before returning, on the first attempt and
    # on the retry without the limiter.
    lm, _ = env.attach(_ConsumingLM, refuse_limiter = refuse_limiter)
    out = lm(input_ids = env.ids)
    torch.testing.assert_close(lm.probe, env.full.hidden_states[1], rtol = 1e-5, atol = 1e-6)
    torch.testing.assert_close(out.logits, env.full.last_hidden_state, rtol = 1e-5, atol = 1e-6)


def test_forward_reading_an_intermediate_state_gets_it():
    _run(lambda env: _consuming(env, False))


def test_limiter_retry_reading_an_intermediate_state_gets_it():
    _run(lambda env: _consuming(env, True))


def _logits_width(env):
    # Finding 3: the re-run for real logits uses the caller's arguments, so it gathers no hidden states.
    lm, _ = env.attach(_DroppingLM)
    out = lm(input_ids = env.ids)
    # First attempt: attention + last state; re-run without output_hidden_states: attention + last state.
    assert len(env.calls) == 4, env.calls
    assert out.logits.shape == env.full.last_hidden_state.shape
    torch.testing.assert_close(out.logits, env.full.last_hidden_state, rtol = 1e-5, atol = 1e-6)


def test_logits_width_rerun_uses_the_callers_arguments():
    _run(_logits_width)


def _mapping(env):
    # Finding 7: the no-limiter fallback hands the forward's own mapping back: never with cut hidden states.
    lm, _ = env.attach(_MappingLM)
    out = lm(input_ids = env.ids)
    env.assert_states(out["hidden_states"])


def test_no_limiter_fallback_returns_every_state():
    _run(_mapping)


def test_passthrough_audit():
    attrs = frozenset({"model"})
    assert cp._decoder_output_passthrough(_LM.forward, attrs)
    assert cp._decoder_output_passthrough(_DroppingLM.forward, attrs)
    assert not cp._decoder_output_passthrough(_ConsumingLM.forward, attrs)
    assert not cp._decoder_output_passthrough(_MappingLM.forward, attrs)

    class _TupleLM:
        def forward(
            self,
            input_ids = None,
            **kwargs,
        ):
            outputs = self.model(input_ids = input_ids, **kwargs)
            self.probe = outputs.to_tuple()[1][0]
            return outputs[0]

    class _ValuesLM:
        def forward(
            self,
            input_ids = None,
            **kwargs,
        ):
            outputs = self.model(input_ids = input_ids, **kwargs)
            states = list(outputs.values)
            return states[0]

    # Output methods hand the hidden-state tuple on indirectly: never audited as a passthrough.
    assert not cp._decoder_output_passthrough(_TupleLM.forward, attrs)
    assert not cp._decoder_output_passthrough(_ValuesLM.forward, attrs)
    from transformers.models.llama.modeling_llama import LlamaForCausalLM
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

    for cls in (LlamaForCausalLM, Qwen2ForCausalLM):
        assert cp._decoder_output_passthrough(cls.forward, attrs), cls


def test_hidden_states_keyword_into_another_call_is_not_a_passthrough():
    # hidden_states = outputs.hidden_states counts as handing the tuple on only into the returned ModelOutput.
    attrs = frozenset({"model"})

    class _KeywordConsumerLM:
        def forward(
            self,
            input_ids = None,
            **kwargs,
        ):
            outputs = self.model(input_ids = input_ids, **kwargs)
            self.consume(hidden_states = outputs.hidden_states)
            return CausalLMOutputWithPast(
                logits = outputs.last_hidden_state, hidden_states = outputs.hidden_states
            )

    class _ReturnedConsumerLM:
        def forward(
            self,
            input_ids = None,
            **kwargs,
        ):
            outputs = self.model(input_ids = input_ids, **kwargs)
            return self.consume(hidden_states = outputs.hidden_states)

    class _AssignedOutputLM:
        def forward(
            self,
            input_ids = None,
            **kwargs,
        ):
            outputs = self.model(input_ids = input_ids, **kwargs)
            result = CausalLMOutputWithPast(
                logits = outputs.last_hidden_state, hidden_states = outputs.hidden_states
            )
            self.probe = result[1]
            return result

    for cls in (_KeywordConsumerLM, _ReturnedConsumerLM, _AssignedOutputLM):
        assert not cp._decoder_output_passthrough(cls.forward, attrs), cls
    assert cp._decoder_output_passthrough(_LM.forward, attrs)


class _CPDecoder(torch.nn.Module):
    def __init__(self, manager):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(4, 4)
        self.layers = torch.nn.ModuleList()
        self._unsloth_cp_rl_manager = manager


def _find_decoder(monkeypatch, forward):
    manager = object()
    monkeypatch.setattr(cp, "get_cp_manager", lambda: manager)
    model = torch.nn.Module()
    model.model = _CPDecoder(manager)
    found = cp.last_hidden_state_decoder(model, forward)
    assert found is None or found is model.model
    return found


def test_unknown_forward_wrapper_is_not_a_passthrough(monkeypatch):
    import functools

    def reading(forward):
        @functools.wraps(forward)
        def wrapper(self, *args, **kwargs):
            out = forward(self, *args, **kwargs)
            self.probe = out.hidden_states[1]
            return out

        return wrapper

    assert _find_decoder(monkeypatch, _LM.forward) is not None
    # The innermost forward is a clean passthrough; the wrapper reads an intermediate state.
    assert _find_decoder(monkeypatch, reading(_LM.forward)) is None

    # A marker copied outward by functools.wraps does not vouch for the wrapper.
    def marked(self, *args, **kwargs):
        return _LM.forward(self, *args, **kwargs)

    marked._unsloth_cp_hidden_states_passthrough = True
    assert _find_decoder(monkeypatch, marked) is not None
    assert _find_decoder(monkeypatch, reading(marked)) is None


def test_transformers_causal_lm_forwards_keep_the_truncation(monkeypatch):
    # The real decorators (can_return_tuple, deprecate_kwarg) are known wrappers: these CausalLMs still
    # gather only the last hidden state.
    from transformers.models.llama.modeling_llama import LlamaForCausalLM
    from transformers.models.mistral.modeling_mistral import MistralForCausalLM
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    for cls in (LlamaForCausalLM, Qwen2ForCausalLM, Qwen3ForCausalLM, MistralForCausalLM):
        assert _find_decoder(monkeypatch, cls.forward) is not None, cls
