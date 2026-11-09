# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""GRPO scores with real logits from the model's own head call when the head is hooked.

The hidden-states scorers read lm_head.weight directly and never call the head, so its forward / backward hooks
and torch's global module hooks would never see it."""

from __future__ import annotations

import ast
import functools
import inspect
import os
import textwrap

import pytest
import torch

os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
import unsloth.models.rl_replacements as rl

from torch.nn.modules import module as _torch_module

HIDDEN, VOCAB = 8, 16


class _TinyLM(torch.nn.Module):
    """Honours UNSLOTH_RETURN_HIDDEN_STATES like Unsloth's generated forwards: hidden states, else its own head."""

    def __init__(
        self,
        hidden = HIDDEN,
        vocab = VOCAB,
    ):
        super().__init__()
        torch.manual_seed(0)
        self.embed = torch.nn.Embedding(vocab, hidden)
        self.lm_head = torch.nn.Linear(hidden, vocab, bias = False)
        self.hidden_state_calls = 0

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids):
        hidden = self.embed(input_ids)
        if os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES", "0") == "1":
            self.hidden_state_calls += 1
            return type("Out", (), {"logits": hidden})()
        return type("Out", (), {"logits": self.lm_head(hidden)})()


def _owned_by(module_name):
    def hook(*args, **kwargs):
        return None

    hook.__module__ = module_name
    return hook


@pytest.fixture(autouse = True)
def _no_global_hooks():
    tables = (
        "_global_forward_hooks",
        "_global_forward_pre_hooks",
        "_global_backward_hooks",
        "_global_backward_pre_hooks",
    )
    assert not any(getattr(_torch_module, name, None) for name in tables)
    yield
    assert not any(getattr(_torch_module, name, None) for name in tables)


def test_plain_head_keeps_hidden_states():
    assert rl._unsloth_grpo_head_hooked(_TinyLM()) is False


@pytest.mark.parametrize(
    "register",
    [
        lambda head: head.register_forward_hook(lambda m, i, o: None),
        lambda head: head.register_forward_pre_hook(lambda m, i: None),
        lambda head: head.register_full_backward_hook(lambda m, gi, go: None),
        lambda head: head.register_full_backward_pre_hook(lambda m, go: None),
        lambda head: head.register_forward_hook(
            functools.partial(lambda tag, m, i, o: None, "user")
        ),
    ],
    ids = ["forward", "forward_pre", "backward", "backward_pre", "partial"],
)
def test_hooked_heads_fall_back(register):
    model = _TinyLM()
    handle = register(model.lm_head)
    assert rl._unsloth_grpo_head_hooked(model) is True
    handle.remove()
    assert rl._unsloth_grpo_head_hooked(model) is False


def test_patched_head_forward_falls_back():
    model = _TinyLM()
    model.lm_head.forward = lambda x: torch.nn.functional.linear(x, model.lm_head.weight) * 2
    assert rl._unsloth_grpo_head_hooked(model) is True


@pytest.mark.parametrize(
    "register",
    [
        lambda: _torch_module.register_module_forward_hook(lambda m, i, o: None),
        lambda: _torch_module.register_module_forward_pre_hook(lambda m, i: None),
        lambda: _torch_module.register_module_full_backward_hook(lambda m, gi, go: None),
        lambda: _torch_module.register_module_full_backward_pre_hook(lambda m, go: None),
    ],
    ids = ["forward", "forward_pre", "backward", "backward_pre"],
)
def test_global_module_hooks_fall_back(register):
    model = _TinyLM()
    handle = register()
    try:
        assert rl._unsloth_grpo_head_hooked(model) is True
    finally:
        handle.remove()
    assert rl._unsloth_grpo_head_hooked(model) is False


@pytest.mark.parametrize(
    "owner",
    [
        "unsloth_zoo.training_utils",
        "unsloth.models.vision",
        "deepspeed.runtime.zero.parameter_offload",
        "torch.distributed.fsdp._fully_shard._fsdp_state",
    ],
)
def test_internal_hooks_keep_hidden_states(owner):
    """Hooks the scorers already account for (ZeRO-3 gather, FSDP2 unshard, Unsloth's own) keep the fused route."""
    model = _TinyLM()
    model.lm_head.register_forward_pre_hook(_owned_by(owner))
    model.lm_head.register_forward_hook(_owned_by(owner))
    assert rl._unsloth_grpo_head_hooked(model) is False
    model.lm_head.register_forward_hook(lambda m, i, o: None)
    assert rl._unsloth_grpo_head_hooked(model) is True


def test_accelerate_device_hook_keeps_hidden_states():
    hooks = pytest.importorskip("accelerate.hooks")
    model = _TinyLM()
    hooks.add_hook_to_module(model.lm_head, hooks.ModelHook())
    assert "forward" in vars(model.lm_head)
    assert rl._unsloth_grpo_head_hooked(model) is False
    model.lm_head.register_forward_hook(lambda m, i, o: None)
    assert rl._unsloth_grpo_head_hooked(model) is True


def test_square_head_keeps_hidden_states():
    """vocab_size == hidden_size: the width dispatch could not tell real logits from hidden states."""
    model = _TinyLM(hidden = VOCAB, vocab = VOCAB)
    model.lm_head.register_forward_hook(lambda m, i, o: None)
    assert rl._unsloth_grpo_head_hooked(model) is False


def test_models_without_an_output_head_are_left_alone():
    assert rl._unsloth_grpo_head_hooked(torch.nn.Linear(2, 2)) is False


# zoo's grpo_accumulated_loss as unsloth embeds it: one hidden-states request, one width dispatch.
_ZOO_SHAPED = """
def grpo_accumulated_loss(trainer, input_ids):
    _prior = os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES")
    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"
    try:
        unwrapped_model = trainer.model
        lm_head = unwrapped_model.get_output_embeddings().weight
        new_hidden_states_chunk = unwrapped_model(input_ids).logits
        if new_hidden_states_chunk.shape[-1] == lm_head.shape[1]:
            logits = new_hidden_states_chunk @ lm_head.t()
        else:
            logits = new_hidden_states_chunk
        return torch.log_softmax(logits.float(), dim = -1)
    finally:
        if _prior is None:
            os.environ.pop("UNSLOTH_RETURN_HIDDEN_STATES", None)
        else:
            os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = _prior
"""


def _patched_gradient_function(source):
    source = rl._patch_grpo_accumulated_loss_head_hooks(
        rl._patch_grpo_accumulated_loss_hidden_states_dispatch(source)
    )
    namespace = {
        "os": os,
        "torch": torch,
        "_unsloth_grpo_returns_hidden_states": rl._unsloth_grpo_returns_hidden_states,
        "_unsloth_grpo_hidden_states_signal": rl._unsloth_grpo_hidden_states_signal,
        "_unsloth_grpo_head_hooked": rl._unsloth_grpo_head_hooked,
    }
    exec(compile(source, "<patched grpo_accumulated_loss>", "exec"), namespace)
    return namespace["grpo_accumulated_loss"]


def test_gradient_path_runs_a_hooked_head_and_its_backward_hook():
    function = _patched_gradient_function(_ZOO_SHAPED)
    model = _TinyLM()
    trainer = type("Trainer", (), {"model": model})()
    input_ids = torch.randint(0, VOCAB, (2, 5), generator = torch.Generator().manual_seed(1))

    unhooked = function(trainer, input_ids)
    assert model.hidden_state_calls == 1

    seen = {"forward": 0, "backward": 0}
    model.lm_head.register_forward_hook(
        lambda m, i, o: seen.__setitem__("forward", seen["forward"] + 1)
    )
    model.lm_head.register_full_backward_hook(
        lambda m, gi, go: seen.__setitem__("backward", seen["backward"] + 1)
    )
    hooked = function(trainer, input_ids)
    hooked.sum().backward()
    assert model.hidden_state_calls == 1  # no new hidden-states forward
    assert seen == {"forward": 1, "backward": 1}
    torch.testing.assert_close(hooked, unhooked)
    assert os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES") is None


def test_installed_zoo_gradient_path_requests_hidden_states_only_through_the_check():
    source = rl._patch_grpo_accumulated_loss_head_hooks(
        rl._patch_grpo_accumulated_loss_hidden_states_dispatch(rl.grpo_accumulated_loss)
    )
    assert rl._GRPO_HIDDEN_STATES_REQUEST.search(source) is None
    assert source.count("_unsloth_grpo_head_hooked(trainer.model)") >= 1
    compile(source, "<patched grpo_accumulated_loss>", "exec")


def test_source_patch_fails_loudly_without_a_hidden_states_request():
    with pytest.raises(RuntimeError, match = "requests hidden states"):
        rl._patch_grpo_accumulated_loss_head_hooks(
            "def grpo_accumulated_loss(trainer, x):\n    return x\n"
        )


def test_generated_trainer_carries_the_check_and_the_patched_gradient_path():
    items = rl.RL_PRE_ITEMS["grpo_trainer"]
    assert sum("def _unsloth_grpo_head_hooked(" in item for item in items) == 1
    embedded = [item for item in items if "def grpo_accumulated_loss(" in item]
    assert len(embedded) == 1
    assert rl._GRPO_HIDDEN_STATES_REQUEST.search(embedded[0]) is None


def _nograd_function_ast():
    source = rl.grpo_trainer__get_per_token_logps_and_entropies(
        "_get_per_token_logps_and_entropies", "def _get_per_token_logps_and_entropies(): pass"
    )
    tree = ast.parse(textwrap.dedent(source))
    (function,) = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_get_per_token_logps_and_entropies"
    ]
    return function


def test_nograd_path_requests_hidden_states_only_for_an_unhooked_head():
    function = _nograd_function_ast()
    checks = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and [ast.unparse(t) for t in node.targets] == ["_unsloth_head_hooked"]
    ]
    assert [ast.unparse(node.value) for node in checks] == [
        "_unsloth_grpo_head_hooked(unwrapped_model)"
    ]
    requests = [
        ast.unparse(node.value)
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and [ast.unparse(t) for t in node.targets] == ["os.environ['UNSLOTH_RETURN_HIDDEN_STATES']"]
    ]
    assert requests, "the no-grad path no longer sets UNSLOTH_RETURN_HIDDEN_STATES"
    for value in requests:
        assert value in ("'0'", "'0' if _unsloth_head_hooked else '1'"), value
    assert requests.count("'0' if _unsloth_head_hooked else '1'") >= 1


def test_nograd_prefix_grouper_is_off_for_a_hooked_head():
    """PrefixGrouper's extract_logps always projects with lm_head.weight."""
    function = _nograd_function_ast()
    gates = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and [ast.unparse(t) for t in node.targets] == ["_pg_engage"]
        and "UNSLOTH_GRPO_PREFIX_GROUPER_ON" in ast.unparse(node.value)
    ]
    assert len(gates) == 1
    assert "not _unsloth_head_hooked" in ast.unparse(gates[0].value)


def test_helper_is_self_contained_for_the_generated_trainer():
    """Embedded via inspect.getsource: only torch may come from outside."""
    source = textwrap.dedent(inspect.getsource(rl._unsloth_grpo_head_hooked))
    tree = ast.parse(source)
    loaded = {
        n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }
    stored = {
        n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and not isinstance(n.ctx, ast.Load)
    }
    stored |= {a.arg for a in ast.walk(tree) if isinstance(a, ast.arg)}
    stored |= {
        alias.asname or alias.name
        for n in ast.walk(tree)
        if isinstance(n, ast.ImportFrom)
        for alias in n.names
    }
    import builtins

    free = {name for name in loaded - stored if not hasattr(builtins, name)}
    assert free == {"torch"}, free
