# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A dispatched module must carry a hook, including after we rebuild it.

`dispatch_model` hooks every map entry; `post_patch` then installs a NEW
Embedding and Linear over the same weights, so `_hf_hook` dies with the old
module and a split model raises `index is on cuda:0, different from other
tensors on cuda:1`. `tie_word_embeddings = False` fails identically, so nothing
here is conditioned on the tie.

These RUN the real function against stubs: a rule fed a hand-written dict passes
on a function that repairs nothing.
"""

import sys
import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("accelerate")

from accelerate.hooks import AlignDevicesHook, add_hook_to_module  # noqa: E402


def _repair():
    from unsloth.models.vision import _repair_dispatch_hooks
    return _repair_dispatch_hooks


# `init_hook` moves real tensors, so a CUDA name needs that card and the CPU runner has none.
FAR = "meta"
NEAR = "cpu"


class _Model(torch.nn.Module):
    def __init__(
        self,
        device_map,
        hooked = (),
    ):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(4, 2)
        self.lm_head = torch.nn.Linear(2, 4)
        self.layer = torch.nn.Linear(2, 2)
        self.hf_device_map = dict(device_map)
        for name in hooked:
            add_hook_to_module(self.get_submodule(name), AlignDevicesHook())


def _hooked(model):
    return {n for n, m in model.named_modules() if hasattr(m, "_hf_hook")}


def test_a_tied_module_the_map_placed_gets_its_hook_back():
    model = _Model({"embed_tokens": FAR, "lm_head": FAR, "layer": NEAR})
    assert not hasattr(model.embed_tokens, "_hf_hook"), "fixture is not the broken state"

    repaired = _repair()(model)

    assert repaired == 2, (
        "the two modules the map put on the far card were not repaired, so the "
        "first embedding lookup still crosses devices unaided"
    )
    assert {"embed_tokens", "lm_head"} <= _hooked(model)


def test_a_bare_integer_names_the_card_the_map_meant(monkeypatch):
    """`hf_device_map` gives CUDA entries as bare ints; torch needs the type."""
    import accelerate.hooks as ah

    seen = {}

    def capture(module, hook, **kwargs):
        seen[id(module)] = hook
        return module

    monkeypatch.setattr(ah, "add_hook_to_module", capture)
    model = _Model({"embed_tokens": 1, "layer": 0})
    assert _repair()(model) == 2

    hook = seen[id(model.embed_tokens)]
    assert str(hook.execution_device) == "cuda:1", (
        f"the ids would be sent to {hook.execution_device!r}, not the card the "
        "map put the weight on"
    )
    assert hook.io_same_device is True, (
        "without io_same_device the output is left on the far card and the "
        "mismatch simply moves one operation downstream"
    )


def test_a_failed_attach_is_reported_not_swallowed():
    model = _Model({"embed_tokens": "cuda:99", "layer": NEAR})
    with pytest.warns(RuntimeWarning, match = "could not re-attach"):
        repaired = _repair()(model)
    assert repaired == 0, "the unattachable module was still counted as repaired"
    assert "embed_tokens" not in _hooked(model)


def test_a_module_that_already_has_a_hook_is_left_alone():
    """Double-hooking a module moves its inputs twice per forward."""
    model = _Model({"embed_tokens": FAR, "layer": NEAR}, hooked = ["embed_tokens"])
    original = model.embed_tokens._hf_hook

    assert _repair()(model) == 0, "nothing else here is repairable"
    assert model.embed_tokens._hf_hook is original, (
        "an already-dispatched module was hooked again, so its inputs move "
        "twice on every forward"
    )


def test_a_single_device_map_is_left_completely_alone():
    # Entries that WOULD attach, so dropping the early return changes the count.
    model = _Model({"embed_tokens": FAR, "lm_head": FAR, "layer": FAR})
    assert _repair()(model) == 0
    assert _hooked(model) == set(), "hooks were attached on a single-device load"


def test_no_map_at_all_is_not_an_error():
    """`device_map = {"": 0}` leaves `hf_device_map` None, and that path trains."""
    model = _Model({})
    model.hf_device_map = None
    assert _repair()(model) == 0


def test_a_cpu_or_disk_entry_is_never_hooked_here():
    """Offload is a different mechanism with hooks of its own."""
    model = _Model({"embed_tokens": "cpu", "lm_head": "disk", "layer": FAR})
    assert _repair()(model) == 1, "only `layer` is repairable here"
    assert "embed_tokens" not in _hooked(model) and "lm_head" not in _hooked(model), (
        "an offloaded module was given a dispatch hook, which fights the "
        "offload path's own pre-hook"
    )


def test_a_torch_device_cpu_entry_is_never_hooked_either():
    model = _Model({"embed_tokens": torch.device("cpu"), "lm_head": FAR, "layer": NEAR})
    assert _repair()(model) == 1, "only `lm_head` is repairable here"
    assert "embed_tokens" not in _hooked(model), (
        "an offloaded module spelled as a torch.device was read as an "
        "accelerator and given a dispatch hook"
    )


def test_the_llama_loader_stands_aside_under_vllm():
    """vLLM owns the weights; the HF tree this would hook is not what runs."""
    import ast
    import inspect
    import textwrap

    from unsloth.models.llama import FastLlamaModel

    # dedent, not lstrip: this one is a method, so every line is indented and lstrip would leave the body hanging off a
    # stripped `def`.
    tree = ast.parse(textwrap.dedent(inspect.getsource(FastLlamaModel.from_pretrained)))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "_repair_dispatch_hooks"
    ]
    assert calls, "the llama loader no longer repairs dispatch hooks at all"

    guarded = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.UnaryOp)
        and isinstance(node.test.op, ast.Not)
        and getattr(node.test.operand, "id", None) == "fast_inference"
        and any(
            isinstance(c, ast.Call) and getattr(c.func, "id", None) == "_repair_dispatch_hooks"
            for c in ast.walk(node)
        )
    ]
    assert guarded, (
        "the repair is no longer behind `if not fast_inference`, so a vLLM load "
        "gets accelerate hooks on a module tree vLLM does not execute"
    )


def test_a_name_the_model_does_not_have_is_skipped_not_invented():
    model = _Model({"embed_tokens": FAR, "does.not.exist": FAR, "layer": NEAR})
    assert _repair()(model) == 1, "the one real far-device name was not repaired"
    assert "embed_tokens" in _hooked(model)


def test_the_repair_stands_aside_for_an_offloaded_embedding():
    import ast
    import inspect

    from unsloth.models import vision

    src = inspect.getsource(vision._attach_bnb_multidevice_hooks)
    tree = ast.parse(src.lstrip())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "_repair_dispatch_hooks"
    ]
    assert calls, "the loader no longer repairs tied hooks at all"

    guarded = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.UnaryOp)
        and isinstance(node.test.op, ast.Not)
        and getattr(node.test.operand, "id", None) == "offload_embedding"
        and any(
            isinstance(c, ast.Call) and getattr(c.func, "id", None) == "_repair_dispatch_hooks"
            for c in ast.walk(node)
        )
    ]
    assert guarded, (
        "the repair is no longer behind `if not offload_embedding`, so it "
        "attaches a hook naming a card while the offload path sends the ids "
        "to the CPU weight"
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason = "needs two real devices; `cpu` plus one card is enough, a CPU-only runner is not",
)
def test_the_whole_sequence_against_real_accelerate():
    """dispatch, rebuild as `post_patch` does, repair, then train."""
    from accelerate import dispatch_model

    transformers = pytest.importorskip("transformers")

    config = transformers.AutoConfig.for_model(
        "llama",
        vocab_size = 128,
        hidden_size = 32,
        intermediate_size = 64,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 4,
        tie_word_embeddings = True,
    )
    torch.manual_seed(0)
    model = transformers.AutoModelForCausalLM.from_config(config).to(torch.float32).eval()

    device_map = {
        "model.embed_tokens": 0,
        "lm_head": 0,
        "model.norm": "cpu",
        "model.rotary_emb": "cpu",
    }
    for i in range(config.num_hidden_layers):
        device_map[f"model.layers.{i}"] = "cpu"
    dispatch_model(model, device_map = device_map, main_device = "cpu")
    assert hasattr(model.get_input_embeddings(), "_hf_hook"), (
        "accelerate did not hook the mapped embedding, so this fixture is not "
        "reproducing the state the repair exists for"
    )

    # The shape of unsloth_zoo.patching_utils, which post_patch runs.
    old_in = model.get_input_embeddings().weight
    model.set_input_embeddings(torch.nn.Embedding.from_pretrained(old_in))
    lm_head = torch.nn.Linear(1, 1, bias = None)
    del lm_head.weight
    lm_head.weight = old_in
    lm_head.in_features, lm_head.out_features = old_in.shape[1], old_in.shape[0]
    model.set_output_embeddings(lm_head)
    model.lm_head = lm_head
    model.tie_weights()

    assert not hasattr(
        model.get_input_embeddings(), "_hf_hook"
    ), "the rebuild kept the hook, so there is nothing here to repair"

    ids = torch.randint(0, 128, (2, 6))
    with pytest.raises(RuntimeError, match = "same device"):
        with torch.no_grad():
            model(ids)

    pointer_before = model.get_input_embeddings().weight.data_ptr()
    assert _repair()(model) == 2, "the two rebuilt modules were not repaired"

    assert model.get_input_embeddings().weight.data_ptr() == model.lm_head.weight.data_ptr(), (
        "the repair untied the pair, so a full finetune silently stops sharing "
        "one gradient between the embedding and the lm_head"
    )
    assert (
        model.get_input_embeddings().weight.data_ptr() == pointer_before
    ), "the repair reallocated a weight that was already on its mapped device"

    with torch.no_grad():
        model(ids)

    for parameter in model.parameters():
        parameter.requires_grad_(True)
    model.train()
    model(ids, labels = ids.clone()).loss.backward()
    input_grad = model.get_input_embeddings().weight.grad
    output_grad = model.lm_head.weight.grad
    assert input_grad is not None and output_grad is not None
    assert (
        input_grad.data_ptr() == output_grad.data_ptr()
    ), "the tied pair accumulated two separate gradients after the repair"


def test_a_torch_device_with_an_index_is_still_read_as_cpu():
    model = _Model({"embed_tokens": torch.device("cpu", 0), "lm_head": FAR, "layer": NEAR})
    assert _repair()(model) == 1, "only `lm_head` is repairable here"
    assert "embed_tokens" not in _hooked(model)


def test_the_repaired_hook_carries_the_models_skip_keys(monkeypatch):
    import accelerate.hooks as ah

    seen = {}
    monkeypatch.setattr(
        ah,
        "add_hook_to_module",
        lambda module, hook, **kw: (seen.__setitem__(id(module), hook), module)[1],
    )
    model = _Model({"embed_tokens": FAR, "layer": NEAR})
    model._skip_keys_device_placement = ["past_key_values"]
    _repair()(model)
    assert seen[id(model.embed_tokens)].skip_keys == [
        "past_key_values"
    ], "the repaired hook drops the skip keys, so it moves tensors dispatch_model excluded"


def test_io_same_device_follows_the_root_hook(monkeypatch):
    """dispatch sets it on the root only; on a submodule it double-copies."""
    import accelerate.hooks as ah

    seen = {}
    monkeypatch.setattr(
        ah,
        "add_hook_to_module",
        lambda module, hook, **kw: (seen.__setitem__(id(module), hook), module)[1],
    )

    with_root = _Model({"embed_tokens": FAR, "layer": NEAR})
    add_hook_to_module(with_root, AlignDevicesHook(io_same_device = True))
    _repair()(with_root)
    assert seen[id(with_root.embed_tokens)].io_same_device is False, (
        "the root already returns the output, so a submodule that does it too "
        "sends every activation back and forth once more per layer"
    )

    seen.clear()
    without_root = _Model({"embed_tokens": FAR, "layer": NEAR})
    assert not hasattr(without_root, "_hf_hook"), "fixture is not the rootless case"
    _repair()(without_root)
    assert seen[id(without_root.embed_tokens)].io_same_device is True, (
        "with no root hook nothing returns the far card's output, so the "
        "mismatch just moves one operation downstream"
    )


def _repairs_last_in(func):
    """Is `_repair_dispatch_hooks` called after the last module-replacing call?"""
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    repair_lines = [
        n.lineno
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "_repair_dispatch_hooks"
    ]
    replacer_lines = [
        n.lineno
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and getattr(getattr(n, "func", None), "attr", None)
        in ("resize_token_embeddings", "set_input_embeddings", "set_output_embeddings")
        or (isinstance(n, ast.Call) and getattr(n.func, "id", None) == "patch_model_and_tokenizer")
    ]
    return repair_lines, replacer_lines


def test_the_vision_loader_repairs_after_its_own_patching_pass():
    from unsloth.models.vision import FastBaseModel

    repair_lines, replacer_lines = _repairs_last_in(FastBaseModel.from_pretrained)
    assert repair_lines, "the vision loader never repairs dispatch hooks"
    assert replacer_lines, "no module-replacing call found; this guard has gone vacuous"
    assert max(repair_lines) > max(replacer_lines), (
        "the last repair runs before the last module replacement, so the hook it "
        "attaches is thrown away by the rebuild that follows"
    )


def test_the_loader_repairs_after_resizing_the_vocabulary():
    import ast
    import inspect

    from unsloth.models import loader

    tree = ast.parse(inspect.getsource(loader))
    resize = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and getattr(getattr(n, "func", None), "attr", None) == "resize_token_embeddings"
    ]
    assert resize, "no resize_token_embeddings call; this guard has gone vacuous"

    repairs = [
        n.lineno
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "_repair_dispatch_hooks"
    ]
    for call in resize:
        assert any(call.lineno < line < call.lineno + 30 for line in repairs), (
            f"the resize at line {call.lineno} is not followed by a repair, so the "
            "new embedding sits on its mapped card with nothing sending it the ids"
        )


class _CoarseModel(torch.nn.Module):
    """An endpoint covered by an ANCESTOR entry, which dispatch hooks anyway."""

    def __init__(self, device_map):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(4, 2)
        self.model.layer = torch.nn.Linear(2, 2)
        self.lm_head = torch.nn.Linear(2, 4)
        self.hf_device_map = dict(device_map)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head


def test_an_embedding_covered_only_by_an_ancestor_entry_is_repaired():
    """`{"model": far, "lm_head": near}` never names `model.embed_tokens`."""
    model = _CoarseModel({"model": FAR, "lm_head": NEAR})

    _repair()(model)

    assert "model.embed_tokens" in _hooked(model), (
        "the rebuilt embedding is covered by the 'model' entry and was left "
        "unhooked, so a coarse map still crashes at the first lookup"
    )


def test_a_root_entry_covers_both_rebuilt_modules():
    model = _CoarseModel({"": FAR, "model.layer": NEAR})

    _repair()(model)

    assert {"model.embed_tokens", "lm_head"} <= _hooked(model)


def test_the_map_wins_over_a_covering_ancestor():
    """Resolution is a fallback; overriding a real entry would relocate it."""
    model = _CoarseModel({"": NEAR, "model.embed_tokens": FAR})

    _repair()(model)

    hook = model.model.embed_tokens._hf_hook
    assert str(hook.execution_device) == FAR, (
        f"the embedding took {hook.execution_device} from the covering root "
        f"entry instead of the {FAR} the map names for it"
    )


def test_a_model_that_cannot_answer_for_its_embeddings_is_not_guessed_at():
    class _Awkward(_CoarseModel):
        def get_input_embeddings(self):
            raise NotImplementedError("this architecture does not say")

    model = _Awkward({"model": FAR, "lm_head": NEAR})

    repaired = _repair()(model)

    assert repaired >= 1, "one raising accessor aborted the whole repair"
    assert "model" in _hooked(model)


def _lift():
    import unsloth.models.vision as V
    return V._lift_endpoint_hooks_onto_adapters


class _WrappedModel(torch.nn.Module):
    """A model whose endpoints are LoRA-style wrappers over hooked modules."""

    def __init__(
        self,
        wrap_in = True,
        wrap_out = True,
    ):
        super().__init__()
        self.embed = _FakeLora(torch.nn.Embedding(4, 2)) if wrap_in else torch.nn.Embedding(4, 2)
        self.head = _FakeLora(torch.nn.Linear(2, 4)) if wrap_out else torch.nn.Linear(2, 4)

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.head


class _FakeLora(torch.nn.Module):
    """Stands in for `peft.tuners.lora.Linear`: `base_layer` carries the hook."""

    def __init__(self, base_layer):
        super().__init__()
        self.base_layer = base_layer


def test_a_hook_on_base_layer_is_lifted_onto_the_adapter_wrapper():
    model = _WrappedModel()
    for m in (model.embed.base_layer, model.head.base_layer):
        add_hook_to_module(m, AlignDevicesHook(execution_device = torch.device(FAR)))

    lifted = _lift()(model)

    assert lifted == 2, f"lifted {lifted}, so an adapter branch still reads the caller's tensor"
    assert hasattr(model.embed, "_hf_hook") and hasattr(model.head, "_hf_hook")


def test_the_lifted_hook_takes_the_base_layers_execution_device():
    """A guessed device is worse than none: it relocates a placed module."""
    model = _WrappedModel(wrap_out = False)
    add_hook_to_module(
        model.embed.base_layer,
        AlignDevicesHook(execution_device = torch.device(FAR), skip_keys = ["past_key_values"]),
    )

    _lift()(model)

    hook = model.embed._hf_hook
    assert hook.execution_device == torch.device(
        FAR
    ), f"the lifted hook points at {hook.execution_device}, not the base layer's {FAR}"
    assert hook.skip_keys == [
        "past_key_values"
    ], "the lifted hook drops the skip keys, so it moves tensors dispatch_model excluded"


def test_an_unwrapped_endpoint_is_left_alone():
    model = _WrappedModel(wrap_in = False, wrap_out = False)
    add_hook_to_module(model.embed, AlignDevicesHook(execution_device = torch.device(FAR)))

    assert _lift()(model) == 0, "something was lifted onto a module PEFT never wrapped"


def test_a_wrapper_whose_base_was_never_hooked_is_left_alone():
    assert _lift()(_WrappedModel()) == 0


def test_a_wrapper_that_already_has_a_hook_is_not_hooked_twice():
    model = _WrappedModel(wrap_out = False)
    add_hook_to_module(model.embed.base_layer, AlignDevicesHook(execution_device = torch.device(FAR)))
    add_hook_to_module(model.embed, AlignDevicesHook(execution_device = torch.device(FAR)))

    assert _lift()(model) == 0, "a second hook was stacked on the wrapper"


def test_a_lift_on_a_model_that_cannot_answer_for_its_embeddings_is_skipped():
    class _Awkward(_WrappedModel):
        def get_input_embeddings(self):
            raise NotImplementedError("this architecture does not say")

    model = _Awkward()
    add_hook_to_module(model.head.base_layer, AlignDevicesHook(execution_device = torch.device(FAR)))

    assert _lift()(model) == 1, "one raising accessor aborted the whole lift"
