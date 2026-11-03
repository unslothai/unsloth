# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Splits repeated denoiser blocks over CUDA cards for memory, not speed; lazy output hand-off only
for listed classes without residual inputs, else eager; UNSLOTH_DIFFUSION_LAYER_SPLIT=0 disables."""

from __future__ import annotations

import dataclasses
import inspect
import os
import threading
import weakref
from typing import Any, Mapping, Optional, Sequence

LAYER_SPLIT_ENV = "UNSLOTH_DIFFUSION_LAYER_SPLIT"

# Denoisers whose forward passes block outputs straight to the next block (checked against diffusers' loops).
LAZY_CLASSES: frozenset[str] = frozenset(
    {
        "Flux2Transformer2DModel",
        "Flux3Transformer2DModel",
        "FluxTransformer2DModel",
        "HunyuanImageTransformer2DModel",
        "QwenImageTransformer2DModel",
        "QwenImage21Transformer2DModel",
        "ZImageTransformer2DModel",
        "WanTransformer3DModel",
        "LTX2VideoTransformer3DModel",
        # HiDream's llama states / cats follow hidden_states.device (bitwise vs one card, test_diffusion_layer_split_cuda.py).
        "HiDreamImageTransformer2DModel",
        "HunyuanVideo15Transformer3DModel",
        "Ideogram4Transformer2DModel",
        "Krea2Transformer2DModel",
        "Lumina2Transformer2DModel",
        "MiniMaxH3Transformer3DModel",
    }
)
# Block lists run in another order than registered (forward order, so each card is entered once).
FORWARD_ORDER: dict[str, tuple[str, ...]] = {
    "Flux3Transformer2DModel": (
        "txt_mode_blocks",
        "mode_blocks.image",
        "mode_blocks.image_cond",
        "single_transformer_blocks",
    ),
}
# A non-None keyword whose name contains one of these feeds a root-level add between blocks: eager for that call.
_RESIDUAL_KWARG_PARTS: tuple[str, ...] = (
    "controlnet",
    "control_hidden",
    "residual",
    "hint",
    "vace",
)

_STEP_CACHE_HOOKS: frozenset[str] = frozenset({"fbc_leader_block_hook", "fbc_block_hook"})

_SPLIT_ATTR = "_unsloth_layer_split"
_SUBCLASSES: dict[type, type] = {}
_SUBCLASS_LOCK = threading.Lock()


class LayerSplitError(RuntimeError):
    """The denoiser cannot be split (no repeated blocks, hooks that own its placement, or the budgets are too small).
    The caller falls back to its one-card plan."""


def layer_split_mode() -> str:
    """``auto`` (default: split only a denoiser that fits no single selected card), ``off``, or ``force``.
    ``UNSLOTH_DIFFUSION_MULTIGPU=0`` turns it off too."""
    from .diffusion_multigpu import multigpu_mode

    if multigpu_mode() == "off":
        return "off"
    raw = (os.environ.get(LAYER_SPLIT_ENV) or "").strip().lower()
    if raw in ("0", "off", "false", "no"):
        return "off"
    if raw == "force":
        return "force"
    return "auto"


def _torch():
    import torch
    return torch


def _leaves(t: Any) -> list:
    from .diffusion_block_graph import _leaves as leaves
    return leaves(t)


def module_bytes(module: Any) -> int:
    """Bytes the module's weights take on a card: tensor subclasses (torchao) count their inner tensors."""
    seen: set = set()
    total = 0
    for t in (*module.parameters(), *module.buffers()):
        if id(t) in seen:
            continue
        seen.add(id(t))
        for leaf in _leaves(t):
            try:
                total += leaf.numel() * leaf.element_size()
            except Exception:  # noqa: BLE001 - opaque leaf: count the wrapper
                total += t.numel() * t.element_size()
                break
    return total


def _block_list_name(qual: str) -> str:
    """The block list a repeated block belongs to: the module path before its last list index. A list entry that
    wraps the block (HiDream's ``double_stream_blocks.3.block``) belongs to ``double_stream_blocks``, not to a group
    of its own, so consecutive entries route lazily and record as one segment."""
    parts = qual.split(".")
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].isdigit():
            return ".".join(parts[:i])
    return qual.rpartition(".")[0]


def find_blocks(transformer: Any) -> list[tuple[str, str, Any]]:
    """(qualified name, list name, block) for every repeated block, in forward order (registration order unless
    :data:`FORWARD_ORDER` names the lists). Nested repeated blocks are part of their outer block."""
    names: tuple[str, ...] = ()
    try:
        from .diffusion_regional_compile import ensure_repeated_blocks
        names = tuple(ensure_repeated_blocks(transformer) or ())
    except Exception:  # noqa: BLE001 - fall back to what the class declares
        pass
    if not names:
        names = tuple(getattr(transformer, "_repeated_blocks", None) or ())
    if not names:
        names = tuple(getattr(transformer, "_no_split_modules", None) or ())
    wanted = set(names)
    out: list[tuple[str, str, Any]] = []
    inside: Optional[str] = None
    for qual, module in transformer.named_modules():
        if module is transformer:
            continue
        if inside is not None and qual.startswith(inside + "."):
            continue
        inside = None
        base = getattr(type(module), "_unsloth_layer_split_base", type(module))
        if base.__name__ in wanted:
            out.append((qual, _block_list_name(qual), module))
            inside = qual
    order = FORWARD_ORDER.get(type(transformer).__name__)
    if order:
        rank = {g: i for i, g in enumerate(order)}
        out.sort(
            key = lambda item: rank.get(item[1], len(rank))
        )  # stable: keeps each list's own order
    return out


def _module_device(module: Any) -> Any:
    for t in module.parameters():
        return t.device
    for t in module.buffers():
        return t.device
    return None


def _flex_processor(transformer: Any) -> Optional[str]:
    try:
        processors = getattr(transformer, "attn_processors", None) or {}
    except Exception:  # noqa: BLE001 - no processor registry: nothing flex to find
        return None
    for processor in processors.values():
        name = type(processor).__name__
        if "Flex" in name:
            return name
    return None


def _diffusers_hook_names(module: Any) -> set:
    registry = getattr(module, "_diffusers_hook", None)
    return set(getattr(registry, "hooks", None) or {}) if registry is not None else set()


def step_cache_hooked(transformer: Any) -> bool:
    """Whether Studio's First-Block-Cache (``diffusion_cache``) is the ONLY diffusers hook on ``transformer``. It keeps
    its residuals on one card, so a split removes it first (:func:`_drop_step_cache`) instead of refusing."""
    try:
        names = set().union(*(_diffusers_hook_names(m) for m in transformer.modules()))
    except Exception:  # noqa: BLE001 - not a torch module
        return False
    return bool(names) and names <= _STEP_CACHE_HOOKS


def layer_split_refusal(transformer: Any, *, step_cache_ok: bool = False) -> Optional[str]:
    """Why this denoiser cannot be split, else None. ``step_cache_ok``: Studio's own First-Block-Cache hooks do not
    refuse (placement removes them)."""
    torch = _torch()
    if not isinstance(transformer, torch.nn.Module):
        return "no denoiser module"
    if getattr(transformer, "_compiled_call_impl", None) is not None:
        return "denoiser is compiled whole"
    cache_only = step_cache_ok and step_cache_hooked(transformer)
    for name, sub in transformer.named_modules():
        if getattr(sub, "_hf_hook", None) is not None:
            return f"{name or '<root>'} carries an accelerate hook"
        hooks = _diffusers_hook_names(sub)
        if hooks and not cache_only:
            return f"{name or '<root>'} carries diffusers hooks ({', '.join(sorted(hooks))})"
    for t in (*transformer.parameters(), *transformer.buffers()):
        if t.device.type == "meta":
            return "denoiser has meta weights"
        break
    flex = _flex_processor(transformer)
    if flex is not None:
        # its BlockMask's mask_mod closes over tensors built on the primary, which no boundary copy reaches
        return f"{flex} drives flex attention through a BlockMask built on the primary"
    if len(find_blocks(transformer)) < 2:
        return "denoiser has fewer than two repeated blocks"
    return None


@dataclasses.dataclass
class LayerRange:
    device: str
    blocks: list[str]
    mib: int

    def describe(self) -> dict:
        return {
            "device": self.device,
            "count": len(self.blocks),
            "first": self.blocks[0] if self.blocks else None,
            "last": self.blocks[-1] if self.blocks else None,
            "mib": self.mib,
        }


@dataclasses.dataclass
class LayerSplitPlan:
    primary: str
    rest_mib: int
    ranges: list[LayerRange]
    assignment: dict[str, str]  # block qualified name -> device

    def per_card_mib(self) -> dict[str, int]:
        out: dict[str, int] = {self.primary: self.rest_mib}
        for r in self.ranges:
            out[r.device] = out.get(r.device, 0) + r.mib
        return out


def _cuda(ordinal_or_device: Any) -> Any:
    torch = _torch()
    if isinstance(ordinal_or_device, int):
        return torch.device("cuda", ordinal_or_device)
    dev = torch.device(ordinal_or_device)
    if dev.type == "cuda" and dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    return dev


def plan_layer_split(
    transformer: Any,
    *,
    primary: Any,
    cards: Sequence[Any],
    budgets_mib: Mapping[Any, int],
    strategy: str = "fill",
) -> LayerSplitPlan:
    """Contiguous block ranges over ``cards`` (primary first; added if missing), in forward order.

    ``budgets_mib``: denoiser weights each card may take (keys: ordinal or device string); the primary's budget also
    pays for the non-block modules. ``fill`` packs each card up to its budget before the next (fewest cards, one
    crossing per used card); ``balance`` spreads the blocks in proportion to the budgets. Raises
    :class:`LayerSplitError` when the blocks do not fit."""
    primary_dev = _cuda(primary)
    order = [primary_dev]
    for c in cards:
        dev = _cuda(c)
        if dev not in order:
            order.append(dev)
    if len(order) < 2:
        raise LayerSplitError("a layer split needs at least two cards")

    def _budget(dev: Any) -> int:
        for key in (dev.index, str(dev), dev):
            if key in budgets_mib:
                return int(budgets_mib[key])
        raise LayerSplitError(f"no budget for {dev}")

    mib = 1 << 20
    blocks = find_blocks(transformer)
    if len(blocks) < 2:
        raise LayerSplitError("denoiser has fewer than two repeated blocks")
    sizes = [module_bytes(b) for _, _, b in blocks]
    rest = max(0, module_bytes(transformer) - sum(sizes))
    budgets = [_budget(d) * mib for d in order]
    budgets[0] -= rest
    if budgets[0] < 0:
        raise LayerSplitError(
            f"the primary's budget ({_budget(order[0])} MiB) does not cover the non-block modules "
            f"({rest // mib} MiB)"
        )
    total = sum(sizes)
    if total > sum(max(0, b) for b in budgets):
        raise LayerSplitError(
            f"blocks need {total // mib} MiB, the cards' budgets give {sum(max(0, b) for b in budgets) // mib} MiB"
        )
    if strategy == "balance":
        weight = sum(max(0, b) for b in budgets)
        targets = [min(b, total * max(0, b) / weight) for b in budgets]
    elif strategy == "fill":
        targets = list(budgets)
    else:
        raise LayerSplitError(f"unknown strategy {strategy!r}")

    assignment: dict[str, str] = {}
    ranges: list[LayerRange] = []
    card = 0
    used = 0
    current: list[str] = []
    current_bytes = 0

    def _close() -> None:
        if current:
            ranges.append(LayerRange(str(order[card]), list(current), current_bytes // mib))

    for (qual, _, _), size in zip(blocks, sizes):
        while card < len(order) - 1 and (
            used + size > budgets[card] or (strategy == "balance" and used >= targets[card])
        ):
            _close()
            current, current_bytes, used = [], 0, 0
            card += 1
        if used + size > budgets[card]:
            raise LayerSplitError(
                f"block {qual} ({size // mib} MiB) does not fit any remaining card"
            )
        assignment[qual] = str(order[card])
        current.append(qual)
        current_bytes += size
        used += size
    _close()
    return LayerSplitPlan(str(primary_dev), rest // mib, ranges, assignment)


def assigned_plan(
    transformer: Any, *, primary: Any, assignment: Mapping[str, Any]
) -> LayerSplitPlan:
    """A plan from an explicit block -> card map (qualified names from :func:`find_blocks`; card = ordinal or device).
    Every block must be assigned. Raises :class:`LayerSplitError` on an unknown or missing block."""
    primary_dev = _cuda(primary)
    mib = 1 << 20
    blocks = find_blocks(transformer)
    if len(blocks) < 2:
        raise LayerSplitError("denoiser has fewer than two repeated blocks")
    known = {qual for qual, _, _ in blocks}
    unknown = sorted(set(assignment) - known)
    if unknown:
        raise LayerSplitError(f"assignment names unknown blocks ({', '.join(unknown[:3])})")
    sizes = [module_bytes(b) for _, _, b in blocks]
    rest = max(0, module_bytes(transformer) - sum(sizes))
    out: dict[str, str] = {}
    ranges: list[LayerRange] = []
    for (qual, _, _), size in zip(blocks, sizes):
        if qual not in assignment:
            raise LayerSplitError(f"block {qual} has no card")
        dev = str(_cuda(assignment[qual]))
        out[qual] = dev
        if ranges and ranges[-1].device == dev:
            ranges[-1].blocks.append(qual)
            ranges[-1].mib += size // mib
        else:
            ranges.append(LayerRange(dev, [qual], size // mib))
    if len({r.device for r in ranges} | {str(primary_dev)}) < 2:
        raise LayerSplitError("assignment keeps every block on the primary")
    return LayerSplitPlan(str(primary_dev), rest // mib, ranges, out)


def cross_copy(t: Any, device: Any) -> Any:
    """Every input copy onto a block's card goes through here (the memo and the per-call hand-off): the one place to
    redirect, e.g. into a per-card graph's static buffer."""
    return t.to(device)


def cross_outputs(out: Any, device: Any) -> Any:
    """Every block output sent to another card (the next block's, or home) goes through here."""
    from .diffusion_multigpu import move_to_device
    return move_to_device(out, device, objects = False)


class _Memo:
    """Copies of one forward's tensors per card, keyed by the source's id and checked against a weak reference, its
    data pointer and its version, so a freed or rewritten source never serves a stale copy. Inference tensors (every
    render) keep no version, so the memo is only trusted across blocks in lazy mode, where the denoiser does no tensor
    op of its own between blocks; an eager call clears it after every block (``_split_call``)."""

    def __init__(self) -> None:
        self.entries: dict[int, tuple[Any, int, Any, dict]] = {}
        self.copies = 0

    def clear(self) -> None:
        self.entries.clear()

    @staticmethod
    def _version(t: Any) -> Any:
        try:
            return t._version
        except Exception:  # noqa: BLE001 - inference tensors keep no version counter
            return None

    def get(self, t: Any, device: Any) -> Any:
        torch = _torch()
        if type(t) is not torch.Tensor and not isinstance(t, torch.nn.Parameter):
            self.copies += 1
            return cross_copy(t, device)  # tensor subclasses: no stable data pointer to key on
        key = id(t)
        hit = self.entries.get(key)
        if hit is not None:
            ref, ptr, version, copies = hit
            if ref() is t and ptr == t.data_ptr() and version == self._version(t):
                moved = copies.get(device)
                if moved is None:
                    self.copies += 1
                    moved = copies[device] = cross_copy(t, device)
                return moved
        self.copies += 1
        moved = cross_copy(t, device)
        entries = self.entries

        def _drop(_ref: Any, key: int = key) -> None:
            current = entries.get(key)
            if current is not None and current[0] is _ref:
                entries.pop(key, None)

        try:
            ref = weakref.ref(t, _drop)
        except TypeError:
            return moved
        entries[key] = (ref, t.data_ptr(), self._version(t), {device: moved})
        return moved


def _probe(module: Any) -> Optional[tuple]:
    """(weak module, kind, name) of one tensor ``module`` itself or a descendant holds, read per call for its current
    device; None when it holds none. ``skip``: ids of submodules (the blocks) whose tensors are not the module's."""
    return _probe_outside(module, frozenset())


def _probe_outside(module: Any, skip: frozenset) -> Optional[tuple]:
    stack = [module]
    while stack:
        current = stack.pop()
        if id(current) in skip:
            continue
        for kind in ("_parameters", "_buffers"):
            for name, tensor in getattr(current, kind, {}).items():
                if tensor is not None:
                    return (weakref.ref(current), kind, name)
        stack.extend(reversed(list(current.children())))
    return None


def _probe_device(probe: Optional[tuple], default: Any) -> Any:
    if probe is None:
        return default
    owner = probe[0]()
    tensor = getattr(owner, probe[1], {}).get(probe[2]) if owner is not None else None
    return default if tensor is None else tensor.device


@dataclasses.dataclass
class _BlockState:
    device: Any  # the card placement put the block on (used only when the block holds no tensor)
    next_block: (
        Any  # weak reference to the next block of the same list, or None at the end of the list
    )
    split: "_Split"
    probe: Optional[tuple] = None

    def current(self) -> Any:
        return _probe_device(self.probe, self.device)


class _Split:
    def __init__(self, transformer: Any, home: Any, lazy_class: bool) -> None:
        self.transformer = weakref.ref(transformer)
        self.home = home
        self.lazy_class = lazy_class
        self.lazy_now = False
        self.memo = _Memo()
        self.handles: list = []
        # weak: blocks hold their state, which holds this split; a strong list would make every split a cycle
        self.blocks: list[tuple[Any, type]] = []
        self.owns_home = False
        self.home_probe: Optional[tuple] = None
        self.calls = 0
        self.crossings = 0
        self.lazy_calls = 0
        self.eager_calls = 0

    def current_home(self) -> Any:
        """Where the denoiser's own (non-block) weights are now: the placement's primary until something moves them."""
        return _probe_device(self.home_probe, self.home)

    def move_in(
        self,
        obj: Any,
        device: Any,
        home: Any = None,
    ) -> Any:
        from .diffusion_multigpu import move_to_device

        home = self.home if home is None else home
        memo = self.memo

        def _tensor(t: Any) -> Any:
            self.crossings += 1
            if t.device == home:
                return memo.get(t, device)
            memo.copies += 1
            return cross_copy(t, device)

        return move_to_device(obj, device, tensor_fn = _tensor, objects = False)


def _split_call(self: Any, *args: Any, **kwargs: Any) -> Any:
    state: Optional[_BlockState] = self.__dict__.get(_SPLIT_ATTR)
    base_call = type(self)._unsloth_layer_split_base.__call__
    if state is None:
        return base_call(self, *args, **kwargs)
    from .diffusion_multigpu import _device_context

    split = state.split
    device = state.current()
    home = split.current_home()
    args, kwargs = split.move_in((args, kwargs), device, home)
    with _device_context(device):
        out = base_call(self, *args, **kwargs)
    target = home if device != home else None
    if split.lazy_now:
        nxt = state.next_block() if state.next_block is not None else None
        nxt_state = nxt.__dict__.get(_SPLIT_ATTR) if nxt is not None else None
        if nxt_state is not None:
            nxt_device = nxt_state.current()
            target = nxt_device if nxt_device != device else None
    else:
        # eager: an in-place write between blocks is invisible (inference tensors carry no version), so no copy outlives its block
        split.memo.clear()
    if target is not None:
        out = cross_outputs(out, target)
    return out


def _split_class(cls: type) -> type:
    base = getattr(cls, "_unsloth_layer_split_base", None)
    if base is not None:
        return cls
    with _SUBCLASS_LOCK:
        sub = _SUBCLASSES.get(cls)
        if sub is None:
            # Same name and module: class-name checks (``_repeated_blocks``, block graph / cache lookups) still match.
            sub = type(
                cls.__name__,
                (cls,),
                {
                    "__call__": _split_call,
                    "__module__": cls.__module__,
                    "__qualname__": cls.__qualname__,
                    "_unsloth_layer_split_base": cls,
                },
            )
            _SUBCLASSES[cls] = sub
    return sub


_POSITIONAL_NAMES: "weakref.WeakKeyDictionary[type, tuple]" = weakref.WeakKeyDictionary()


def _positional_names(module: Any) -> tuple:
    """Names of ``module.forward``'s positional parameters in order, cached per class."""
    cls = type(module)
    names = _POSITIONAL_NAMES.get(cls)
    if names is None:
        try:
            params = inspect.signature(cls.forward).parameters.values()
            names = tuple(
                p.name
                for p in params
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.name != "self"
            )
        except (TypeError, ValueError):
            names = ()
        _POSITIONAL_NAMES[cls] = names
    return names


def _is_residual_name(name: Any) -> bool:
    lowered = str(name).lower()
    return any(part in lowered for part in _RESIDUAL_KWARG_PARTS)


def _residual_call(
    kwargs: Mapping[str, Any],
    args: Sequence[Any] = (),
    module: Any = None,
) -> bool:
    """The call feeds a residual the denoiser adds between blocks (ControlNet / VACE hints), by keyword or by position."""
    for key, value in kwargs.items():
        if value is not None and _is_residual_name(key):
            return True
    if args and module is not None:
        names = _positional_names(module)
        for name, value in zip(names, args):
            if value is not None and _is_residual_name(name):
                return True
        if len(args) > len(names) and any(_is_residual_name(n) for n in names):
            return True  # *args past the named slots: cannot tell, so stay exact
    return False


def _move_rest(transformer: Any, block_ids: set, device: Any, logger: Any) -> None:
    """Every parameter and buffer of ``transformer`` outside the blocks to ``device``."""
    from .diffusion_fast_load import fast_upload

    owners = []
    stack = [transformer]
    while stack:
        module = stack.pop()
        if id(module) in block_ids:
            continue
        owners.append(module)
        stack.extend(module.children())
    with fast_upload(owners, device, logger = logger):
        for module in owners:
            module._apply(lambda t: t.to(device), recurse = False)


def _move_weights(
    transformer: Any, plan: LayerSplitPlan, blocks: list, home: Any, logger: Any
) -> None:
    """Each block once to its card (blocks first, so the primary only ever holds its own share), then the rest."""
    torch = _torch()
    from .diffusion_fast_load import fast_upload

    by_device: dict[str, list] = {}
    for qual, _, block in blocks:
        by_device.setdefault(plan.assignment[qual], []).append(block)
    for dev_name, mods in by_device.items():
        dev = torch.device(dev_name)
        with fast_upload(mods, dev, logger = logger):
            for block in mods:
                block.to(dev)
    _move_rest(transformer, {id(b) for _, _, b in blocks}, home, logger)


def _install_boundary(transformer: Any, plan: LayerSplitPlan, blocks: list, split: _Split) -> None:
    torch = _torch()
    for i, (qual, group, block) in enumerate(blocks):
        dev = torch.device(plan.assignment[qual])
        nxt = blocks[i + 1] if i + 1 < len(blocks) else None
        nxt_ref = weakref.ref(nxt[2]) if nxt is not None and nxt[1] == group else None
        original = type(block)
        block.__dict__[_SPLIT_ATTR] = _BlockState(dev, nxt_ref, split, _probe(block))
        block.__class__ = _split_class(original)
        split.blocks.append(
            (weakref.ref(block), getattr(original, "_unsloth_layer_split_base", original))
        )
    split.home_probe = _probe_outside(transformer, frozenset(id(b) for _, _, b in blocks))

    def _pre(module: Any, args: Any, kwargs: Any) -> None:
        split.memo.clear()
        split.calls += 1
        split.lazy_now = split.lazy_class and not _residual_call(kwargs, args, module)
        if split.lazy_now:
            split.lazy_calls += 1
        else:
            split.eager_calls += 1

    def _post(module: Any, args: Any, kwargs: Any, output: Any) -> None:
        split.memo.clear()
        split.lazy_now = False

    split.handles.append(transformer.register_forward_pre_hook(_pre, with_kwargs = True))
    split.handles.append(
        transformer.register_forward_hook(_post, with_kwargs = True, always_call = True)
    )
    transformer.__dict__[_SPLIT_ATTR + "_root"] = split
    split.owns_home = "_unsloth_split_home" not in transformer.__dict__
    transformer._unsloth_split_home = split.home


def layer_split_of(module: Any) -> Optional[_Split]:
    if module is None:
        return None
    return getattr(module, "__dict__", {}).get(_SPLIT_ATTR + "_root")


def is_layer_split(module: Any) -> bool:
    return layer_split_of(module) is not None


def _drop_step_cache(pipe: Any, attr: str, transformer: Any, logger: Any) -> bool:
    """Take Studio's First-Block-Cache off ``transformer`` before a split (it keeps residuals on one card and its
    hooks own the blocks' forward). The load re-reads ``_unsloth_step_cache`` afterwards, so status reports it off.
    Raises :class:`LayerSplitError` when the hooks cannot be verified gone. True when a cache was removed."""
    if not step_cache_hooked(transformer):
        return False
    from . import diffusion_cache

    view = pipe if attr == "transformer" else type("_View", (), {"transformer": transformer})()
    diffusion_cache.maybe_toggle_step_cache(view, steps = 0, logger = logger)
    if step_cache_hooked(transformer) or getattr(transformer, "_unsloth_step_cache", None):
        raise LayerSplitError("the step cache on the denoiser could not be removed")
    if logger is not None:
        logger.info(
            "diffusion.layer_split: step cache removed from %s (it keeps residuals on one card)",
            attr,
        )
    return True


def place_layer_split(
    pipe: Any,
    *,
    primary: Any,
    cards: Sequence[Any] = (),
    budgets_mib: Optional[Mapping[Any, int]] = None,
    strategy: str = "fill",
    attr: str = "transformer",
    assignment: Optional[Mapping[str, Any]] = None,
    logger: Any = None,
) -> dict[str, Any]:
    """Split ``pipe.<attr>`` across ``cards`` (see :func:`plan_layer_split`), or by an explicit block -> card
    ``assignment`` (see :func:`assigned_plan`), and place it. Other components are left where they are. Returns the placement summary also shown by ``placement_status``. Raises :class:`LayerSplitError`
    (nothing moved) when the split is refused or does not fit; on a failure while moving, the boundary is removed and
    the error re-raised, and the caller re-places the pipeline."""
    torch = _torch()
    from . import diffusion_multigpu as mg

    if layer_split_mode() == "off":
        raise LayerSplitError(f"{LAYER_SPLIT_ENV}=0 or {mg.MULTIGPU_ENV}=0")
    transformer = getattr(pipe, attr, None)
    why = layer_split_refusal(transformer, step_cache_ok = True)
    if why is not None:
        raise LayerSplitError(why)
    if is_layer_split(transformer):
        undo_layer_split(pipe, attr = attr)
    if assignment is not None:
        assigned_plan(
            transformer, primary = primary, assignment = assignment
        )  # validate before touching the cache
    else:
        plan_layer_split(
            transformer, primary = primary, cards = cards, budgets_mib = budgets_mib, strategy = strategy
        )
    _drop_step_cache(pipe, attr, transformer, logger)
    if assignment is not None:
        plan = assigned_plan(transformer, primary = primary, assignment = assignment)
        strategy = "assigned"
    else:
        plan = plan_layer_split(
            transformer, primary = primary, cards = cards, budgets_mib = budgets_mib, strategy = strategy
        )
    home = torch.device(plan.primary)
    blocks = find_blocks(transformer)
    split = _Split(transformer, home, type(transformer).__name__ in LAZY_CLASSES)
    try:
        _move_weights(transformer, plan, blocks, home, logger)
        _install_boundary(transformer, plan, blocks, split)
        mg._SPLIT_EVER_PLACED = True
        mg.install_execution_device_patch()
    except Exception:
        _remove_boundary(transformer, split)
        raise

    info = {
        "primary": plan.primary,
        "strategy": strategy,
        "denoiser": attr,
        "rest_mib": plan.rest_mib,
        "ranges": [r.describe() for r in plan.ranges],
        "per_card_mib": plan.per_card_mib(),
        "lazy": split.lazy_class,
    }
    split.info = info  # type: ignore[attr-defined]
    try:
        record = mg._PLACEMENTS.get(pipe)
        if record is None:
            record = {
                "placement": {
                    str(n): str(_module_device(m))
                    for n, m in (getattr(pipe, "components", {}) or {}).items()
                    if isinstance(m, torch.nn.Module) and m is not transformer
                },
                "primary": plan.primary,
            }
            mg._PLACEMENTS[pipe] = record
        record.setdefault("placement", {})[attr] = "+".join(r.device for r in plan.ranges)
        record["layer_split"] = info
    except TypeError:  # not weak-referenceable
        pass
    if logger is not None:
        logger.info(
            "diffusion.layer_split: %s split %s",
            type(transformer).__name__,
            ", ".join(f"{r.device}: {len(r.blocks)} blocks {r.mib} MiB" for r in plan.ranges),
        )
    return info


def _remove_boundary(transformer: Any, split: _Split) -> None:
    for handle in split.handles:
        try:
            handle.remove()
        except Exception:  # noqa: BLE001
            pass
    split.handles.clear()
    for ref, base in split.blocks:
        block = ref()
        if block is None:
            continue
        block.__dict__.pop(_SPLIT_ATTR, None)
        if getattr(type(block), "_unsloth_layer_split_base", None) is base:
            block.__class__ = base
    split.blocks.clear()
    split.memo.clear()
    if transformer is not None:
        transformer.__dict__.pop(_SPLIT_ATTR + "_root", None)
        if split.owns_home:
            transformer.__dict__.pop("_unsloth_split_home", None)


def undo_layer_split(
    pipe: Any,
    *,
    attr: str = "transformer",
    device: Any = None,
) -> bool:
    """Remove the boundary (weights stay where they are unless ``device`` is given, then the denoiser moves there).
    True when a split was removed."""
    from . import diffusion_multigpu as mg

    transformer = getattr(pipe, attr, None)
    split = layer_split_of(transformer)
    if split is None:
        return False
    _remove_boundary(transformer, split)
    try:
        record = mg._PLACEMENTS.get(pipe)
    except TypeError:
        record = None
    if record is not None:
        record.pop("layer_split", None)
        if not record.get("moved"):  # no component split under it: the record was ours
            mg._PLACEMENTS.pop(pipe, None)
        else:
            record.get("placement", {}).pop(attr, None)
    if device is not None:
        transformer.to(device)
    return True


def layer_split_stats(module: Any) -> Optional[dict]:
    split = layer_split_of(module)
    if split is None:
        return None
    out = dict(getattr(split, "info", {}) or {})
    out.update(
        {
            "calls": split.calls,
            "lazy_calls": split.lazy_calls,
            "eager_calls": split.eager_calls,
            "input_handoffs": split.crossings,
            "input_copies": split.memo.copies,
        }
    )
    return out
