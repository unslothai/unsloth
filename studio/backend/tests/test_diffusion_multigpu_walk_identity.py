# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pins the single-walk router checks (fingerprint, hook refusal, state mirror) to the verbatim previous implementation."""

from __future__ import annotations

import copy
import types

import pytest
import torch

from core.inference import diffusion_multigpu as mg


# Verbatim reference: the implementations before the single-walk rewrite.
def _ref_hooks_present(module):
    for name, sub in module.named_modules():
        kind = mg._registry_kind(sub)
        if kind == "other":
            return f"{name or '<root>'}._diffusers_hook ({', '.join(sorted(sub._diffusers_hook.hooks))})"
        if getattr(sub, "_hf_hook", None) is not None:
            return f"{name or '<root>'}._hf_hook"
        for attr in ("_forward_hooks", "_forward_pre_hooks"):
            for hook in (getattr(sub, attr, None) or {}).values():
                if mg._is_offload_object(hook):
                    continue
                if name and not mg._trusted_hook(hook):
                    return f"{name}.{attr}"
                if not name and not mg._hook_origin(hook).startswith(mg._TRUSTED_HOOK_MODULES):
                    return f"<root>.{attr} (a hook outside Unsloth: its per-call state cannot be mirrored)"
        override = sub.__dict__.get("forward")
        if override is None or isinstance(override, (types.MethodType, mg.CfgParallelRouter)):
            continue
        if not name and mg._resident_graph(sub) is not None:
            continue
        if kind == "offload" or mg._is_offload_object(override):
            continue
        if callable(override) and not mg._trusted_hook(override):
            return f"{name or '<root>'}.forward (instance override)"
    return None


def _ref_weights_fingerprint(module):
    parts = []
    for name, param in (*module.named_parameters(), *module.named_buffers()):
        try:
            version = param._version
        except Exception:  # noqa: BLE001
            version = -1
        parts.append((name, id(param), version))
    for name, sub in module.named_modules():
        for attr in mg._ADAPTER_ATTRS:
            value = sub.__dict__.get(attr)
            if isinstance(value, dict):
                value = {
                    k: (id(v), v._version) if hasattr(v, "_version") else v
                    for k, v in value.items()
                }
            if value is not None and not callable(value):
                parts.append((name, attr, repr(value)))
    return hash(tuple(parts))


def _ref_hook_state(pairs):
    out = []
    for i, (src, _dst) in enumerate(pairs):
        for key, value in src.__dict__.items():
            if key.startswith(mg._STATE_PREFIX) and isinstance(value, mg._STATE_TYPES):
                out.append((i, key, value))
    return tuple(out)


def _ref_mirror_hook_state(pairs):
    state = _ref_hook_state(pairs)
    wanted = {}
    for i, key, value in state:
        wanted.setdefault(i, {})[key] = value
    for i, (_src, dst) in enumerate(pairs):
        mine = wanted.get(i, {})
        for key in [
            k
            for k, v in dst.__dict__.items()
            if k.startswith(mg._STATE_PREFIX)
            and isinstance(v, mg._STATE_TYPES)
            and k not in mine
            and k not in mg._STATE_KEEP
        ]:
            del dst.__dict__[key]
        for key, value in mine.items():
            if (
                dst.__dict__.get(key, mg._DROP) is not value
                and dst.__dict__.get(key, mg._DROP) != value
            ):
                dst.__dict__[key] = value
    return state


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = torch.nn.Linear(4, 4)
        self.ff = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.GELU(), torch.nn.Linear(8, 4))
        self.norm = torch.nn.LayerNorm(4)
        self.register_buffer("freqs", torch.arange(4.0))
        self.register_buffer("scratch", torch.zeros(2), persistent = False)
        self.register_buffer("unset", None)
        self.register_parameter("gate", None)
        self.missing = None

    def forward(self, x):
        return self.norm(x + self.ff(self.attn(x)))


class _Dit(torch.nn.Module):
    def __init__(self, blocks = 3):
        super().__init__()
        self.embed = torch.nn.Linear(4, 4)
        self.blocks = torch.nn.ModuleList([_Block() for _ in range(blocks)])
        self.blocks[1].shared = self.blocks[0].attn  # one module reachable under two names
        self.out = torch.nn.Linear(4, 4)
        self.out.weight = self.embed.weight  # tied parameter
        self.register_buffer("scale", torch.ones(1))
        self.slot = None
        self.add_module("empty", None)

    def forward(self, hidden_states):
        x = self.embed(hidden_states)
        for block in self.blocks:
            x = block(x)
        return self.out(x)


def _assert_fingerprint_same(dit):
    assert mg.weights_fingerprint(dit) == _ref_weights_fingerprint(dit)


def _assert_hooks_same(dit):
    assert mg._hooks_present(dit) == _ref_hooks_present(dit)


def test_named_modules_matches_torch_including_duplicates_and_overrides():
    dit = _Dit()
    assert mg._named_modules(dit) == list(dit.named_modules())

    class Custom(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = torch.nn.Linear(2, 2)

        def named_modules(
            self,
            memo = None,
            prefix = "",
            remove_duplicate = True,
        ):
            yield prefix + ".custom", self

    dit.blocks[2].custom = Custom()
    assert mg._named_modules(dit) == list(dit.named_modules())
    dit.blocks[0].__dict__["named_modules"] = lambda memo = None, prefix = "", rd = True: iter(
        [(prefix, dit)]
    )
    assert mg._named_modules(dit) == list(dit.named_modules())
    root = Custom()
    assert mg._named_modules(root) == list(root.named_modules())


def test_fingerprint_identical_through_every_mutation():
    dit = _Dit()
    seen = []

    def step():
        _assert_fingerprint_same(dit)
        seen.append(mg.weights_fingerprint(dit))

    step()
    step()
    dit.blocks[0].attn.weight = torch.nn.Parameter(torch.randn(4, 4))  # parameter replaced
    step()
    with torch.no_grad():
        dit.blocks[1].ff[0].bias.add_(1.0)
    step()
    with torch.no_grad():
        dit.scale.mul_(2.0)
    step()
    dit.blocks[2].attn.scaling = {"default": 1.0}  # PEFT-like adapter state
    dit.blocks[2].attn._active_adapter = ["default"]
    dit.blocks[2].attn.merged_adapters = []
    dit.blocks[2].attn._disable_adapters = False
    step()
    dit.blocks[2].attn.scaling["default"] = 0.5  # adapter scale changed
    step()
    dit.blocks[2].attn._active_adapter = ["other"]
    step()
    pinned = torch.tensor(1.0)
    dit.blocks[0].ff[2].scaling = {"default": pinned}  # a pinned 0-d scale tensor
    step()
    pinned.mul_(0.25)  # in-place write to the pinned scale
    step()
    dit.blocks[0].disable_adapters = lambda: None  # a callable is not state
    step()
    dit.blocks[1].lora_A = torch.nn.ModuleDict(
        {"default": torch.nn.Linear(4, 2)}
    )  # adapter injected
    step()
    dit.blocks[1].register_buffer("packed", torch.zeros(3, dtype = torch.uint8))
    step()
    del dit.blocks[1].lora_A
    step()
    with torch.inference_mode():
        frozen = torch.ones(2)
    dit.blocks[0].register_buffer("frozen", frozen)  # an inference tensor (no version counter)
    step()
    # every change moved the fingerprint except the repeat and the callable, exactly as before
    assert [seen.index(s) for s in seen] == [0, 0, 2, 3, 4, 5, 6, 7, 8, 9, 9, 11, 12, 13, 14]


def test_fingerprint_identical_when_the_root_replaces_torch_traversal():
    class Odd(_Dit):
        def named_parameters(
            self,
            prefix = "",
            recurse = True,
            remove_duplicate = True,
        ):
            yield from reversed(list(super().named_parameters(prefix, recurse, remove_duplicate)))

    dit = Odd()
    assert not mg._stock_traversal(dit)
    _assert_fingerprint_same(dit)
    plain = _Dit()
    assert mg._stock_traversal(plain)
    plain.__dict__["named_buffers"] = lambda: iter([])
    assert not mg._stock_traversal(plain)
    _assert_fingerprint_same(plain)


def test_fingerprint_identical_on_a_peft_injected_model():
    peft = pytest.importorskip("peft")
    dit = _Dit()
    config = peft.LoraConfig(r = 2, lora_alpha = 4, target_modules = r".*\.(attn|ff\.0|ff\.2)")
    dit = peft.inject_adapter_in_model(config, dit)
    _assert_fingerprint_same(dit)
    _assert_hooks_same(dit)
    before = mg.weights_fingerprint(dit)
    for sub in dit.modules():
        if isinstance(getattr(sub, "scaling", None), dict):
            sub.scaling["default"] = 0.125
    _assert_fingerprint_same(dit)
    assert mg.weights_fingerprint(dit) != before


def _foreign_hook(
    module,
    args,
    output = None,
):
    return None


def _trusted(
    module,
    args,
    output = None,
):
    return None


_trusted.__module__ = "core.inference.diffusion_trim"


def _offload(module, args):
    return None


_offload.__module__ = "core.inference.diffusion_offload_prefetch"


def test_hooks_present_identical_through_every_hook_change():
    dit = _Dit()
    _assert_hooks_same(dit)
    assert mg._hooks_present(dit) is None

    handles = []
    handles.append(dit.blocks[1].ff[0].register_forward_pre_hook(_offload))  # offload: skipped
    _assert_hooks_same(dit)
    handles.append(dit.register_forward_hook(_trusted))  # Studio root hook: accepted
    _assert_hooks_same(dit)
    assert mg._hooks_present(dit) is None
    handles.append(
        dit.blocks[2].norm.register_forward_hook(_foreign_hook)
    )  # foreign hook below the root
    _assert_hooks_same(dit)
    assert mg._hooks_present(dit) == "blocks.2.norm._forward_hooks"
    handles.append(
        dit.blocks[0].attn.register_forward_pre_hook(_foreign_hook)
    )  # an earlier one wins
    _assert_hooks_same(dit)
    assert mg._hooks_present(dit) == "blocks.0.attn._forward_pre_hooks"
    for handle in handles[2:]:
        handle.remove()
    handles.append(dit.register_forward_pre_hook(_foreign_hook))  # foreign root hook
    _assert_hooks_same(dit)
    assert "a hook outside Unsloth" in mg._hooks_present(dit)
    handles[-1].remove()
    _assert_hooks_same(dit)
    assert mg._hooks_present(dit) is None


def test_hooks_present_identical_for_accelerate_registry_and_overrides():
    class Registry:
        def __init__(self, *names):
            self.hooks = {n: object() for n in names}

    dit = _Dit()
    dit.blocks[1].attn._hf_hook = object()
    _assert_hooks_same(dit)
    assert mg._hooks_present(dit) == "blocks.1.attn._hf_hook"
    dit.blocks[1].attn._hf_hook = None
    _assert_hooks_same(dit)
    dit.blocks[2].register_buffer(
        "_hf_hook", torch.zeros(1)
    )  # reached through torch's buffer fallback
    _assert_hooks_same(dit)
    assert mg._hooks_present(dit) == "blocks.2._hf_hook"
    del dit.blocks[2]._hf_hook
    _assert_hooks_same(dit)
    dit.blocks[0].ff._diffusers_hook = Registry("first_block_cache")  # step cache registry
    _assert_hooks_same(dit)
    assert "first_block_cache" in mg._hooks_present(dit)
    dit.blocks[0].ff._diffusers_hook = Registry()
    _assert_hooks_same(dit)
    dit.blocks[0].ff.forward = lambda x: x
    _assert_hooks_same(dit)
    assert mg._hooks_present(dit) == "blocks.0.ff.forward (instance override)"
    del dit.blocks[0].ff.forward
    dit.blocks[0].ff.forward = types.MethodType(torch.nn.Sequential.forward, dit.blocks[0].ff)
    _assert_hooks_same(dit)
    assert mg._hooks_present(dit) is None


def test_hooks_present_identical_for_unusual_attribute_lookup():
    class ClassLevel(torch.nn.Linear):
        _hf_hook = "class default"

    class Lookup(torch.nn.Linear):
        def __getattr__(self, name):
            if name == "_hf_hook":
                return "computed"
            return super().__getattr__(name)

    dit = _Dit()
    dit.blocks[2].odd = ClassLevel(4, 4)
    _assert_hooks_same(dit)
    assert mg._hooks_present(dit) == "blocks.2.odd._hf_hook"
    del dit.blocks[2].odd
    dit.blocks[1].odd = Lookup(4, 4)
    _assert_hooks_same(dit)
    assert mg._hooks_present(dit) == "blocks.1.odd._hf_hook"
    del dit.blocks[1].odd
    del dit.blocks[0].norm.__dict__["_forward_hooks"]  # torch's fallback answers None
    _assert_hooks_same(dit)
    assert mg._hooks_present(dit) is None


def _pairs(src, dst):
    return list(zip(src.modules(), dst.modules()))


def _dst_state(dst):
    return [
        {
            k: v
            for k, v in sub.__dict__.items()
            if isinstance(k, str) and k.startswith(mg._STATE_PREFIX)
        }
        for sub in dst.modules()
    ]


def test_hook_state_and_mirror_identical_through_every_state_change():
    src = _Dit()
    ref_dst, new_dst = copy.deepcopy(src), copy.deepcopy(src)
    for dst in (ref_dst, new_dst):
        dst._unsloth_replica_compiled = 0  # kept on the replica
        dst.blocks[2].attn._unsloth_stale = True  # stale on the replica: deleted
    ref_pairs, new_pairs = _pairs(src, ref_dst), _pairs(src, new_dst)

    def step():
        assert mg._hook_state(new_pairs) == _ref_hook_state(ref_pairs)
        assert mg._mirror_hook_state(new_pairs) == _ref_mirror_hook_state(ref_pairs)
        assert _dst_state(new_dst) == _dst_state(ref_dst)
        assert mg._hook_state(new_pairs) == _ref_hook_state(ref_pairs)

    step()
    src._unsloth_null_mask = True
    src.blocks[1].ff._unsloth_trim = 3
    step()
    src.blocks[1].ff._unsloth_trim = 4
    step()
    src.blocks[1].ff._unsloth_trim = True  # equal value, other type
    step()
    src.blocks[0]._unsloth_cache = object()  # not a scalar: never mirrored
    src.blocks[0]._unsloth_name = "x"
    step()
    del src.blocks[1].ff._unsloth_trim  # removed on the source: deleted on the copy
    step()
    src.blocks[2].norm.__dict__["a\0_unsloth_b"] = 1  # a separator inside a key
    step()
    src.blocks[2].norm.__dict__[7] = "non-string key"  # both versions fail the same way
    with pytest.raises(AttributeError):
        _ref_hook_state(ref_pairs)
    with pytest.raises(AttributeError):
        mg._hook_state(new_pairs)
    del src.blocks[2].norm.__dict__[7]
    new_dst.blocks[0].attn._unsloth_extra = 1.5  # replica-side state the source never had
    ref_dst.blocks[0].attn._unsloth_extra = 1.5
    step()
    src._unsloth_null_mask = None
    step()
    assert any(s for s in _dst_state(new_dst))


def test_router_checks_on_a_diffusers_denoiser():
    import importlib

    try:
        module = importlib.import_module("diffusers.models.transformers.transformer_flux")
    except (ImportError, RuntimeError) as exc:
        pytest.skip(f"diffusers is not importable here: {type(exc).__name__}")
    dit = module.FluxTransformer2DModel(
        patch_size = 1,
        in_channels = 8,
        num_layers = 1,
        num_single_layers = 1,
        attention_head_dim = 8,
        num_attention_heads = 2,
        joint_attention_dim = 16,
        pooled_projection_dim = 16,
        axes_dims_rope = (2, 2, 4),
    )
    _assert_fingerprint_same(dit)
    _assert_hooks_same(dit)
    replica = copy.deepcopy(dit)
    dit.transformer_blocks[0]._unsloth_flag = True
    pairs = _pairs(dit, replica)
    assert mg._hook_state(pairs) == _ref_hook_state(pairs)
    assert mg._mirror_hook_state(pairs) == _ref_hook_state(pairs)
    assert replica.transformer_blocks[0]._unsloth_flag is True
