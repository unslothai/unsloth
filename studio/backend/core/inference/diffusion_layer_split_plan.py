# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Layer split tried after companion split, only when the one-card plan offloads the denoiser; one
card, Keep multiple models loaded and kill switches keep the current plan unchanged; engine is
stubbed until it lands."""

from __future__ import annotations

import contextlib
import dataclasses
import sys
import weakref
from typing import Any, Callable, Mapping, Optional, Sequence

LAYER_SPLIT_ENV = "UNSLOTH_DIFFUSION_LAYER_SPLIT"
_TE_ROLE = "text_encoder"
_DIT_ROLE = "dit"
_DENOISER_OFFLOAD_POLICIES = ("model", "sequential", "streaming")
# A block-only secondary holds one block's working set (B200 1024px: 0.6 GiB klein-4B, 1.2 GiB FLUX.2-dev):
# priced as twice the largest block, at least 1 GiB, never above the plan's denoise headroom.
_WORKING_SET_BLOCKS = 2
_WORKING_SET_MIN_MIB = 1024
_TE_ONLY_HEADROOM_MIB = (
    1024  # diffusion_multigpu.TEXT_ENCODER_HEADROOM_MIB: an encoder-only phase on a card
)


def secondary_reserve_mib() -> int:
    """ComfyUI's per-card floor (``minimum_inference_memory``: 0.8 GiB + EXTRA_RESERVED_VRAM, 400 MiB on Linux, 600 MiB
    on Windows plus 100 MiB on a 16 GB+ card, taken here unconditionally) for a card that only runs blocks. The primary
    keeps Studio's base overhead when that is larger."""
    return 800 + (700 if sys.platform == "win32" else 400)


def block_working_set_mib(
    denoisers: Mapping[str, "DenoiserLayout"], denoise_headroom_mib: int
) -> int:
    largest = max((max(d.block_mib) for d in denoisers.values() if d.block_mib), default = 0)
    estimate = max(_WORKING_SET_MIN_MIB, _WORKING_SET_BLOCKS * int(largest))
    return min(estimate, int(denoise_headroom_mib)) if denoise_headroom_mib > 0 else estimate


def layer_split_mode() -> str:
    """``off`` / ``auto`` (default) / ``force``; off whenever the whole multi-GPU lever is off."""
    from .diffusion_layer_split import layer_split_mode as engine_mode
    return engine_mode()


@dataclasses.dataclass(frozen = True)
class BlockRange:
    denoiser: str
    start: int
    stop: int
    card: int
    mib: int


@dataclasses.dataclass(frozen = True)
class LayerSplit:
    primary: int
    cards: tuple[int, ...]
    ranges: tuple[BlockRange, ...]
    text_encoders_card: Optional[int]
    card_required_mib: tuple[tuple[int, int], ...]
    card_limit_mib: tuple[tuple[int, int], ...]
    card_weights_mib: tuple[tuple[int, int], ...] = ()
    reason: str = ""
    text_encoders_streamed: bool = False
    card_reserve_mib: tuple[tuple[int, int], ...] = ()

    def working_mib(self, ordinal: int) -> tuple[int, int]:
        """(activation working set, fixed reserve) the plan held on card ``ordinal`` beside its weights, at the default
        request. A request larger than that scales the first only."""
        required = dict(self.card_required_mib).get(ordinal, 0)
        weights = dict(self.card_weights_mib).get(ordinal, 0)
        reserve = dict(self.card_reserve_mib).get(ordinal, 0)
        return max(0, int(required) - int(weights) - int(reserve)), int(reserve)

    @property
    def boundaries(self) -> int:
        """Device crossings per pass of every denoiser, the return of the output to the primary included."""
        out = 0
        for name in dict.fromkeys(r.denoiser for r in self.ranges):
            path = [
                self.primary,
                *(r.card for r in self.ranges if r.denoiser == name),
                self.primary,
            ]
            out += sum(1 for a, b in zip(path, path[1:]) if a != b)
        return out

    def card_of(self, denoiser: str, index: int) -> Optional[int]:
        for r in self.ranges:
            if r.denoiser == denoiser and r.start <= index < r.stop:
                return r.card
        return None

    def as_public_dict(self) -> dict[str, Any]:
        return {
            "primary": self.primary,
            "cards": list(self.cards),
            "ranges": [dataclasses.asdict(r) for r in self.ranges],
            "text_encoders_card": self.text_encoders_card,
            "text_encoders_streamed": self.text_encoders_streamed,
            "card_required_mib": {str(o): m for o, m in self.card_required_mib},
            "card_limit_mib": {str(o): m for o, m in self.card_limit_mib},
            "card_weights_mib": {str(o): m for o, m in self.card_weights_mib},
            "card_reserve_mib": {str(o): m for o, m in self.card_reserve_mib},
            "boundaries": self.boundaries,
            "reason": self.reason,
        }


@dataclasses.dataclass(frozen = True)
class DenoiserLayout:
    """Storage MiB of one denoiser: ``stem`` (everything outside the sequential blocks) and each block in order."""

    stem_mib: int
    block_mib: tuple[int, ...]


def _partition(blocks: Sequence[int], capacities: Sequence[int]) -> Optional[list[int]]:
    """Cut points (len(capacities) + 1, first 0, last len(blocks)) of a contiguous partition with part i <= capacity i,
    proportional to the capacities when that fits, else greedy first-fit. None when nothing fits."""
    n = len(blocks)
    total = sum(blocks)
    cap_sum = sum(max(0, c) for c in capacities)
    if n == 0:
        return [0] * (len(capacities) + 1)
    if cap_sum < total:
        return None

    def check(cuts: list[int]) -> bool:
        return all(
            sum(blocks[cuts[i] : cuts[i + 1]]) <= capacities[i] for i in range(len(capacities))
        )

    cuts = [0]
    acc = 0
    prefix = [0]
    for b in blocks:
        prefix.append(prefix[-1] + b)
    for i, cap in enumerate(capacities[:-1]):
        acc += max(0, cap)
        target = total * acc / cap_sum
        best = min(range(cuts[-1], n + 1), key = lambda j: (abs(prefix[j] - target), j))
        cuts.append(best)
    cuts.append(n)
    if check(cuts):
        return cuts
    cuts = [0]
    for cap in capacities:
        j = cuts[-1]
        used = 0
        while j < n and used + blocks[j] <= cap:
            used += blocks[j]
            j += 1
        cuts.append(j)
    if cuts[-1] != n:
        return None
    return cuts


_SEARCH_BUDGET = 200_000


def _search_cuts(
    layouts: Sequence[Sequence[int]], capacities: Sequence[int]
) -> Optional[list[list[int]]]:
    """Cut points for several denoisers sharing the cards, by backtracking: each denoiser contiguous over the cards in
    order, at least one keeping a block off the primary (a denoiser wholly on the primary is placed there unsplit),
    each card's parts within its capacity. The fallback when cutting each denoiser in turn by proportion leaves a later one no fit.
    None when nothing fits within the search budget."""
    k = len(capacities)
    prefixes = []
    for blocks in layouts:
        prefix = [0]
        for b in blocks:
            prefix.append(prefix[-1] + int(b))
        prefixes.append(prefix)
    visited = [0]

    def options(d: int, remaining: list[int]):
        prefix = prefixes[d]
        n = len(prefix) - 1

        def rec(i: int, start: int, cuts: list[int]):
            if i == k - 1:
                if prefix[n] - prefix[start] <= remaining[i]:
                    yield [*cuts, n]
                return
            for j in range(start, n + 1):
                if prefix[j] - prefix[start] > remaining[i]:
                    break
                yield from rec(i + 1, j, [*cuts, j])

        yield from rec(0, 0, [0])

    def dfs(d: int, remaining: list[int], split: bool) -> Optional[list[list[int]]]:
        if d == len(layouts):
            return [] if split else None
        prefix = prefixes[d]
        n = len(prefix) - 1
        for cuts in options(d, remaining):
            visited[0] += 1
            if visited[0] > _SEARCH_BUDGET:
                return None
            left = [remaining[i] - (prefix[cuts[i + 1]] - prefix[cuts[i]]) for i in range(k)]
            rest = dfs(d + 1, left, split or cuts[1] < n)
            if rest is not None:
                return [cuts, *rest]
        return None

    return dfs(0, [int(c) for c in capacities], False)


def plan_layer_split(
    *,
    denoisers: Mapping[str, DenoiserLayout],
    companions: Mapping[str, tuple[int, str]],
    primary: int,
    card_limits: Mapping[int, int],
    headroom_mib: int,
    overhead_mib: int,
    phase_headroom_mib: Optional[tuple[int, int, int]] = None,
    secondary_overhead_mib: Optional[int] = None,
    working_set_mib: Optional[int] = None,
) -> tuple[Optional[LayerSplit], str]:
    """Pure decision: ``(LayerSplit, reason)`` or ``(None, decline reason)``.

    ``card_limits`` maps every usable card (the primary included) to its resident limit. Fewest cards first: the
    primary, then the secondaries by limit (largest first). ``phase_headroom_mib`` = ``(encode, denoise, decode)``.

    The primary holds the stems, the VAE, the latents and the activation peak: it is charged the plan's full headroom
    and ``max(overhead_mib, secondary reserve)``. A secondary runs blocks only: one block's working set
    (``working_set_mib``, default :func:`block_working_set_mib`) plus ``secondary_overhead_mib`` (default
    :func:`secondary_reserve_mib`), and the encode headroom when the text encoders live there."""
    if primary not in card_limits:
        return None, "primary card limit unknown"
    if not denoisers or any(not d.block_mib for d in denoisers.values()):
        return None, "denoiser blocks unknown"
    secondaries = sorted(
        (o for o in card_limits if o != primary), key = lambda o: (-card_limits[o], o)
    )
    if not secondaries:
        return None, "no secondary card"
    stems = sum(d.stem_mib for d in denoisers.values())
    te_mib = sum(m for m, role in companions.values() if role == _TE_ROLE)
    other_mib = sum(m for m, role in companions.values() if role not in (_TE_ROLE, _DIT_ROLE))
    if phase_headroom_mib is not None:
        encode, denoise, decode = (int(x or 0) for x in phase_headroom_mib)
    else:
        denoise = decode = int(headroom_mib)
        encode = _TE_ONLY_HEADROOM_MIB
    sec_reserve = (
        secondary_reserve_mib() if secondary_overhead_mib is None else int(secondary_overhead_mib)
    )
    pri_reserve = max(int(overhead_mib), sec_reserve)
    working = (
        block_working_set_mib(denoisers, denoise)
        if working_set_mib is None
        else int(working_set_mib)
    )
    reserve = {primary: pri_reserve, **{o: sec_reserve for o in secondaries}}
    blocks_total = sum(sum(d.block_mib) for d in denoisers.values())
    need_total = blocks_total + stems + te_mib + other_mib
    best_shortfall: Optional[int] = None
    for k in range(2, len(secondaries) + 2):
        cards = [primary, *secondaries[: k - 1]]
        # TE home: last card first, primary last; streaming the TE from host (one paged encode per prompt) is the last resort
        te_homes = (
            [(o, False) for o in reversed(cards)] + [(None, True)] if te_mib else [(None, False)]
        )
        for te_home, te_streamed in te_homes:
            fixed = {o: 0 for o in cards}
            head = {o: working for o in cards}
            fixed[primary] += stems + other_mib
            head[primary] = max(denoise, decode)
            if te_home is not None:
                fixed[te_home] += te_mib
                head[te_home] = max(head[te_home], encode)
            elif te_streamed:
                head[primary] = max(head[primary], encode)
            caps = [card_limits[o] - fixed[o] - head[o] - reserve[o] for o in cards]
            if any(c < 0 for c in caps):
                continue
            ranges: list[BlockRange] = []
            remaining = list(caps)
            ok = True
            names = list(denoisers)
            for idx, name in enumerate(names):
                layout = denoisers[name]
                left = sum(sum(denoisers[n].block_mib) for n in names[idx:])
                share = sum(layout.block_mib) / left if left else 1.0
                caps_d = [int(r * share) if idx < len(names) - 1 else r for r in remaining]
                cuts = _partition(layout.block_mib, caps_d) or _partition(
                    layout.block_mib, remaining
                )
                n = len(layout.block_mib)
                if cuts is not None and n > 1 and cuts[1] == n:
                    # the cut rounded every block onto the primary, which the engine refuses: move its last block to the furthest card with room
                    room = [
                        c
                        for c in range(len(cards) - 1, 0, -1)
                        if layout.block_mib[-1] <= remaining[c]
                    ]
                    cuts = (
                        [0, *([n - 1] * room[0]), *([n] * (len(cards) - room[0]))] if room else None
                    )
                if cuts is None:
                    ok = False
                    break
                for i, card in enumerate(cards):
                    lo, hi = cuts[i], cuts[i + 1]
                    if hi > lo:
                        mib = sum(layout.block_mib[lo:hi])
                        ranges.append(BlockRange(name, lo, hi, card, mib))
                        remaining[i] -= mib
            if not ok and len(names) > 1:
                # cut in turn, an earlier denoiser can take the room a later one needed: search the cuts jointly
                joint = _search_cuts([denoisers[n].block_mib for n in names], caps)
                if joint is not None:
                    ok, ranges, remaining = True, [], list(caps)
                    for name, cuts in zip(names, joint):
                        blocks = denoisers[name].block_mib
                        for i, card in enumerate(cards):
                            lo, hi = cuts[i], cuts[i + 1]
                            if hi > lo:
                                mib = sum(blocks[lo:hi])
                                ranges.append(BlockRange(name, lo, hi, card, mib))
                                remaining[i] -= mib
            if not ok:
                continue
            used = {r.card for r in ranges} | {primary}
            if te_home is not None:
                used.add(te_home)
            if len(used) < 2:
                continue  # everything landed on the primary: no split needed (and the caller only asks when it is)
            required = {
                o: card_limits[o] - caps[i] + (caps[i] - remaining[i])
                for i, o in enumerate(cards)
                if o in used
            }
            split = LayerSplit(
                primary = primary,
                cards = tuple(o for o in cards if o in used),
                ranges = tuple(ranges),
                text_encoders_card = te_home,
                text_encoders_streamed = te_streamed,
                card_required_mib = tuple(sorted(required.items())),
                card_limit_mib = tuple(sorted((o, int(card_limits[o])) for o in used)),
                card_weights_mib = tuple(
                    sorted(
                        (o, fixed[o] + caps[i] - remaining[i])
                        for i, o in enumerate(cards)
                        if o in used
                    )
                ),
                card_reserve_mib = tuple(sorted((o, int(reserve[o])) for o in used)),
            )
            per_card = ", ".join(
                f"cuda:{o} {required[o]}/{card_limits[o]} MiB" for o in split.cards
            )
            reason = f"multi-GPU layer split: {', '.join(names)} cut over {len(split.cards)} cards ({per_card})"
            return dataclasses.replace(split, reason = reason), reason
        available = (
            sum(card_limits[o] - reserve[o] for o in cards)
            - max(denoise, decode, encode if te_mib else 0)
            - (k - 1) * working
        )
        # the text encoders can always stream from host memory, so they do not count against the cards
        shortfall = need_total - te_mib - available
        best_shortfall = shortfall if best_shortfall is None else min(best_shortfall, shortfall)
    shortfall_text = (
        f"short by about {best_shortfall} MiB"
        if best_shortfall is not None and best_shortfall > 0
        else "no cut of the blocks fits every card"
    )
    return None, (
        f"layer split does not fit the selected cards ({need_total - te_mib} MiB of denoiser and VAE weights with "
        f"the text encoders streamed, {shortfall_text} after the per-card reserves and "
        f"{max(denoise, decode, encode if te_mib else 0)} MiB "
        f"of activations on cuda:{primary})"
    )


_BLOCK_LIST_ATTRS: tuple[str, ...] = (
    "transformer_blocks",
    "single_transformer_blocks",
    "blocks",
    "layers",
)


def split_blocks(module: Any) -> Optional[list[tuple[str, Any]]]:
    """The denoiser's repeated blocks in forward order as ``(qualified name, module)`` (the engine's own list, so a
    plan's indices name the blocks it places); None when the engine cannot cut this denoiser."""
    from .diffusion_layer_split import find_blocks, layer_split_refusal
    try:
        if layer_split_refusal(module, step_cache_ok = True) is not None:
            return None
        return [(qual, block) for qual, _, block in find_blocks(module)] or None
    except Exception:  # noqa: BLE001 - no torch, or a module the engine cannot walk
        return None


def engine_available() -> bool:
    """Whether the layer split engine can run here (CUDA torch)."""
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def _denoisers(split: LayerSplit) -> list[str]:
    return list(dict.fromkeys(r.denoiser for r in split.ranges))


def place_layer_split(
    pipe: Any,
    *,
    primary: str,
    split: LayerSplit,
    logger: Any = None,
) -> dict:
    """Place every block range on its card, the stems and non-TE companions on ``primary``, the text encoders on
    ``split.text_encoders_card``, and install the device-crossing hooks. Raises on failure."""
    import torch

    from . import diffusion_multigpu as mg
    from .diffusion_fast_load import fast_upload
    from .diffusion_layer_split import LayerSplitError
    from .diffusion_layer_split import place_layer_split as engine_place

    primary_dev = torch.device(primary)
    if primary_dev.type == "cuda" and primary_dev.index is None:
        primary_dev = torch.device("cuda", torch.cuda.current_device())
    names = _denoisers(split)
    te_card = split.text_encoders_card
    te_dev = torch.device("cuda", int(te_card)) if te_card is not None else primary_dev
    components = {
        str(n): m
        for n, m in (getattr(pipe, "components", {}) or {}).items()
        if isinstance(m, torch.nn.Module) and str(n) not in names
    }
    streamed = (
        [n for n in components if n.startswith("text_encoder")]
        if split.text_encoders_streamed
        else []
    )
    moved = [
        n
        for n in components
        if n.startswith("text_encoder") and te_dev != primary_dev and n not in streamed
    ]
    placement: dict[str, str] = {}
    for name, module in components.items():
        if name in streamed:
            continue
        device = te_dev if name in moved else primary_dev
        with fast_upload([module], device, logger = logger):
            module.to(device)
        placement[name] = str(device)
    if streamed:
        from .diffusion_memory import stream_text_encoders_onto
        for name in stream_text_encoders_onto(pipe, primary_dev, logger):
            placement[name] = f"cpu (streamed to {primary_dev})"
    for name in moved:
        mg.install_boundary(components[name], primary_dev)
    if moved or streamed:
        for name, module in components.items():
            if name not in moved and name not in streamed:
                module._unsloth_split_home = primary_dev
        with contextlib.suppress(TypeError):
            mg._PLACEMENTS[pipe] = {
                "placement": placement,
                "primary": str(primary_dev),
                "secondary": str(te_dev) if moved else None,
                "moved": moved,
                "streamed": streamed,
            }
    out: dict[str, Any] = {}
    for name in names:
        module = getattr(pipe, name, None)
        blocks = split_blocks(module)
        if not blocks:
            raise LayerSplitError(f"{name}: no blocks to split")
        assignment: dict[str, int] = {}
        for r in split.ranges:
            if r.denoiser == name:
                for qual, _ in blocks[r.start : r.stop]:
                    assignment[qual] = int(r.card)
        if assignment and all(card == primary_dev.index for card in assignment.values()):
            with fast_upload([module], primary_dev, logger = logger):
                module.to(primary_dev)
            out[name] = {"primary": str(primary_dev), "denoiser": name, "whole_on_primary": True}
            continue
        out[name] = engine_place(
            pipe, primary = primary_dev, attr = name, assignment = assignment, logger = logger
        )
    return out


def undo_layer_split(pipe: Any) -> None:
    """Remove the crossing hooks a (partial) ``place_layer_split`` installed. Never raises."""
    from . import diffusion_multigpu as mg
    from .diffusion_layer_split import undo_layer_split as engine_undo

    try:
        record = dict(
            mg._PLACEMENTS.get(pipe) or {}
        )  # read first: the engine's undo drops a record it owns
    except TypeError:
        record = {}
    for name in mg.DENOISER_ATTRS:
        with contextlib.suppress(Exception):
            engine_undo(pipe, attr = name)
    with contextlib.suppress(Exception):
        if record.get("streamed"):
            from .diffusion_memory import _remove_group_offload_hooks
            for name in record["streamed"]:
                module = getattr(pipe, name, None)
                if module is not None:
                    _remove_group_offload_hooks(module)
    with contextlib.suppress(Exception):
        mg.undo_split(pipe)


_APPLIED: "weakref.WeakKeyDictionary[Any, LayerSplit]" = weakref.WeakKeyDictionary()
_DECLINES: "weakref.WeakKeyDictionary[Any, str]" = weakref.WeakKeyDictionary()


def _record_decline(pipe: Any, reason: str, logger: Any) -> None:
    with contextlib.suppress(TypeError):
        _DECLINES[pipe] = reason
    if logger is not None:
        logger.info("diffusion.multigpu: layer split not used (%s)", reason)


def _is_rocm() -> bool:
    from .diffusion_multigpu import _rocm_torch
    return _rocm_torch()


UNIFIED_DECLINE = "unified memory"


def _rocm_card_unified(ordinal: int) -> Optional[bool]:
    """True for a ROCm APU (Strix Halo / Point, Phoenix: the GPU's memory is the host's), False for a discrete card,
    None when unreadable. Studio's own classifier (the driver's integrated flag, then the gfx target)."""
    try:
        import torch

        from utils.hardware.hardware import _rocm_props_unified_status
        return _rocm_props_unified_status(torch.cuda.get_device_properties(int(ordinal)))
    except Exception:  # noqa: BLE001
        return None


def _unified_reason(cards: Sequence[int]) -> str:
    names = ", ".join(f"cuda:{int(o)}" for o in cards)
    what = "is an integrated GPU (APU)" if len(cards) == 1 else "are integrated GPUs (APUs)"
    return (
        f"{UNIFIED_DECLINE}: {names} {what} sharing system memory, so cutting the denoiser across cards frees no "
        "memory"
    )


def note_unified_primary(
    pipe: Any,
    selection: Any,
    logger: Any = None,
) -> None:
    """``refine_plan_for_multigpu`` stopped on a unified-memory primary: on ROCm say why in the layer split status (a
    multi-APU selection is otherwise silent). CUDA keeps today's silent no-op."""
    if selection is None or layer_split_mode() == "off" or not _is_rocm():
        return
    _record_decline(pipe, _unified_reason((int(selection.primary),)), logger)


def _rocm_discrete_secondaries(
    pipe: Any, primary: int, secondaries: Sequence[int], logger: Any
) -> Optional[tuple[int, ...]]:
    """ROCm: the secondaries a split may use (discrete cards); None after recording a decline. A unified primary
    declines (its blocks and the host share one pool); a unified secondary is dropped (its blocks would sit in host
    memory). An unreadable card is kept: the per-card budgets below still price it."""
    if _rocm_card_unified(primary) is True:
        _record_decline(pipe, _unified_reason((int(primary),)), logger)
        return None
    unified = tuple(int(o) for o in secondaries if _rocm_card_unified(int(o)) is True)
    kept = tuple(int(o) for o in secondaries if int(o) not in unified)
    if not kept:
        _record_decline(pipe, _unified_reason(unified), logger)
        return None
    if unified and logger is not None:
        logger.info(
            "diffusion.multigpu: layer split skips %s (%s)",
            ", ".join(f"cuda:{o}" for o in unified),
            UNIFIED_DECLINE,
        )
    return kept


def plan_offloads_denoiser(plan: Any) -> bool:
    """The one-card plan pages the denoiser's weights per call (what a layer split exists to remove)."""
    policy = getattr(plan, "offload_policy", "none")
    if policy in _DENOISER_OFFLOAD_POLICIES:
        return True
    return policy == "group" and bool(getattr(plan, "stream_transformer", True))


def _denoiser_layouts(pipe: Any) -> Optional[dict[str, DenoiserLayout]]:
    from .diffusion_memory import _storage_nbytes
    from .diffusion_multigpu import DENOISER_ATTRS

    mib = 1 << 20
    out: dict[str, DenoiserLayout] = {}
    for name in DENOISER_ATTRS:
        module = getattr(pipe, name, None)
        if module is None:
            continue
        blocks = split_blocks(module)
        if not blocks:
            return None
        seen: set[int] = set()

        def size(m: Any) -> int:
            nbytes = 0
            for t in (*m.parameters(), *m.buffers()):
                if id(t) in seen:
                    continue
                seen.add(id(t))
                nbytes += sum(_storage_nbytes(t))
            return nbytes

        block_bytes = [size(b) for _, b in blocks]
        stem_bytes = size(module)  # what is left after the blocks claimed theirs
        out[name] = DenoiserLayout(
            stem_mib = -(-stem_bytes // mib), block_mib = tuple(-(-b // mib) for b in block_bytes)
        )
    return out or None


def _layout_refusal(pipe: Any) -> str:
    """The engine's reason the first unsplittable denoiser of ``pipe`` cannot be cut (for the decline status)."""
    from .diffusion_layer_split import layer_split_refusal
    from .diffusion_multigpu import DENOISER_ATTRS

    for name in DENOISER_ATTRS:
        module = getattr(pipe, name, None)
        if module is None:
            continue
        try:
            why = layer_split_refusal(module, step_cache_ok = True)
        except Exception as exc:  # noqa: BLE001
            why = f"{type(exc).__name__}: {exc}"
        if why is not None:
            return f"{name}: {why}"
    return "no sequential block list the engine can cut"


# Backends gated on the primary's arch only (flash3, flash4, Sage, FA2): split only onto identical cards.
_PRIMARY_ARCH_BACKENDS = frozenset({"_flash_3_hub", "flash_4_hub", "sage", "sage_hub", "flash"})


def _backend_cards(
    pipe: Any,
    primary: int,
    secondaries: Sequence[int],
    logger: Any = None,
) -> tuple[tuple[int, ...], Optional[str]]:
    """``(secondaries the engaged attention backend runs on, that backend)``; a card of unknown capability is kept."""
    from . import diffusion_multigpu as mg

    backend = None
    with contextlib.suppress(Exception):
        from .diffusion_attention import ATTENTION_BACKEND_ATTR
        for name in mg.DENOISER_ATTRS:
            module = getattr(pipe, name, None)
            tag = getattr(module, ATTENTION_BACKEND_ATTR, None) if module is not None else None
            if tag in _PRIMARY_ARCH_BACKENDS:
                backend = tag
                break
    cards = tuple(int(o) for o in secondaries)
    if _is_rocm():
        return _rocm_backend_cards(pipe, primary, cards, logger)
    if backend is None:
        return cards, None
    need = mg._capability(primary)
    if need is None:
        return cards, backend
    return tuple(o for o in cards if (mg._capability(o) or need) == need), backend


# ROCm kernels built per gfx target: ROCm flash_attn (CK / Triton, #13792) and AITER.
_ROCM_ARCH_BACKENDS = frozenset({"flash", "aiter"})


def _rocm_gfx(ordinal: int) -> str:
    try:
        import torch

        from .rocm_bf16 import _device_gfx_arch
        return _device_gfx_arch(torch, int(ordinal))
    except Exception:  # noqa: BLE001
        return ""


def _denoiser_dtype(module: Any) -> Any:
    try:
        return next(module.parameters()).dtype
    except Exception:  # noqa: BLE001
        return None


def _rocm_flash_runs_on(ordinal: int, dtype: Any) -> bool:
    try:
        from types import SimpleNamespace

        from .diffusion_attention import _rocm_flash_attn_runs
        return bool(
            _rocm_flash_attn_runs(
                SimpleNamespace(device = "cuda", torch_device = f"cuda:{int(ordinal)}", dtype = dtype)
            )
        )
    except Exception:  # noqa: BLE001
        return False


MATH_ONLY_DECLINE = "math-only attention"
_PROBE_DTYPES = ("torch.float16", "torch.bfloat16", "torch.float32")


def _rocm_attention_dtype(pipe: Any) -> Any:
    """The compute dtype the SDPA probe should run at: the denoiser's, else the pipeline's; None (the probe's fp16
    default) when only a quantized storage dtype is readable."""
    from . import diffusion_multigpu as mg

    for name in mg.DENOISER_ATTRS:
        module = getattr(pipe, name, None)
        if module is not None:
            dtype = _denoiser_dtype(module)
            if str(dtype) in _PROBE_DTYPES:
                return dtype
    dtype = getattr(pipe, "dtype", None)
    return dtype if str(dtype) in _PROBE_DTYPES else None


def _rocm_sdpa_math_only(ordinal: int, dtype: Any) -> bool:
    """diffusion_attention's per-device SDPA probe: True only when math ran and every fused kernel refused on this
    card at this dtype (an unanswered probe is False)."""
    try:
        from types import SimpleNamespace

        from .diffusion_attention import sdpa_math_only
        return bool(
            sdpa_math_only(
                SimpleNamespace(device = "cuda", torch_device = f"cuda:{int(ordinal)}", dtype = dtype)
            )
        )
    except Exception:  # noqa: BLE001
        return False


def _rocm_drop_math_only(
    pipe: Any,
    primary: int,
    cards: tuple[int, ...],
    logger: Any = None,
) -> tuple[int, ...]:
    """Drop a secondary whose native SDPA is math-only when the primary's is not: the plan's activation figures were
    priced for the primary's fused kernel, and math SDPA materialises the full score matrix on the card running the
    block. A math-only primary priced that peak already, so cards matching it are kept."""
    if not cards:
        return cards
    dtype = _rocm_attention_dtype(pipe)
    if _rocm_sdpa_math_only(primary, dtype):
        return cards
    dropped = tuple(o for o in cards if _rocm_sdpa_math_only(o, dtype))
    if dropped and logger is not None:
        logger.info(
            "diffusion.multigpu: layer split skips %s (%s at %s)",
            ", ".join(f"cuda:{o}" for o in dropped),
            MATH_ONLY_DECLINE,
            dtype,
        )
    return tuple(o for o in cards if o not in dropped)


def _math_only_reason(primary: int, cards: Sequence[int]) -> str:
    names = ", ".join(f"cuda:{int(o)}" for o in cards)
    has = "has" if len(cards) == 1 else "have"
    return (
        f"{MATH_ONLY_DECLINE}: {names} {has} no fused SDPA kernel at the model dtype (cuda:{int(primary)} does), so its "
        "attention would materialise the full score matrix"
    )


def _rocm_backend_cards(
    pipe: Any,
    primary: int,
    cards: tuple[int, ...],
    logger: Any = None,
) -> tuple[tuple[int, ...], Optional[str]]:
    """ROCm: native SDPA runs on any gfx target (the load dtype is already matched by ``make_selection``), so every
    card whose SDPA is not math-only where the primary's is fused is kept. A per-target kernel (flash_attn, AITER) the
    primary verified keeps only cards of the primary's gfx target, and flash only where its own probe runs and matches
    on that card (as the primary's did at load)."""
    from . import diffusion_multigpu as mg

    backend = module = None
    with contextlib.suppress(Exception):
        from .diffusion_attention import ATTENTION_BACKEND_ATTR
        for name in mg.DENOISER_ATTRS:
            candidate = getattr(pipe, name, None)
            tag = (
                getattr(candidate, ATTENTION_BACKEND_ATTR, None) if candidate is not None else None
            )
            if tag in _ROCM_ARCH_BACKENDS:
                backend, module = tag, candidate
                break
    if backend is None:
        return _rocm_drop_math_only(pipe, primary, cards, logger), None
    need = _rocm_gfx(primary)
    if not need:
        return (), backend
    kept = tuple(o for o in cards if _rocm_gfx(o) == need)
    if backend == "flash":
        dtype = _denoiser_dtype(module)
        kept = tuple(o for o in kept if _rocm_flash_runs_on(o, dtype))
    return kept, backend


def refine_plan_for_layer_split(
    pipe: Any,
    plan: Any,
    *,
    selection: Any,
    sizes: Mapping[str, tuple[int, str]],
    primary_limit_mib: int,
    secondary_limit_mib: Callable[[int], Optional[int]],
    headroom_mib: int,
    overhead_mib: int,
    logger: Any = None,
) -> Any:
    """Called by ``refine_plan_for_multigpu`` once the companion assignment found nothing (or under ``force``). The
    plan unchanged unless a layer split fits; then a resident plan carrying ``layer_split`` and the one-card plan as
    ``fallback_plan``. The caller has already checked the selection, mode, explicit offload and CUDA gates."""
    mode = layer_split_mode()
    if mode == "off" or not engine_available():
        return plan
    if not plan_offloads_denoiser(plan) and mode != "force":
        return plan
    candidates: Sequence[int] = selection.secondaries
    if _is_rocm():
        candidates = _rocm_discrete_secondaries(pipe, int(selection.primary), candidates, logger)
        if candidates is None:
            return plan
    layouts = _denoiser_layouts(pipe)
    if layouts is None:
        _record_decline(pipe, f"denoiser cannot be split ({_layout_refusal(pipe)})", logger)
        return plan
    limits: dict[int, int] = {int(selection.primary): int(primary_limit_mib)}
    secondaries, backend = _backend_cards(pipe, int(selection.primary), candidates, logger)
    if not secondaries:
        _record_decline(
            pipe,
            _math_only_reason(int(selection.primary), candidates)
            if backend is None and _is_rocm()
            else f"attention backend {backend} was verified for cuda:{int(selection.primary)}'s architecture only",
            logger,
        )
        return plan
    for o in secondaries:
        limit = secondary_limit_mib(int(o))
        if limit is not None:
            limits[int(o)] = int(limit)
    companions = {n: v for n, v in sizes.items() if v[1] != _DIT_ROLE}
    split, reason = plan_layer_split(
        denoisers = layouts,
        companions = companions,
        primary = int(selection.primary),
        card_limits = limits,
        headroom_mib = int(headroom_mib),
        overhead_mib = int(overhead_mib),
    )
    if split is None:
        _record_decline(pipe, reason, logger)
        return plan
    reason = f"{reason}; single card would use '{plan.offload_policy}'"
    if logger is not None:
        logger.info("diffusion.multigpu: %s", reason)
    with contextlib.suppress(TypeError):
        _DECLINES.pop(pipe, None)
    return dataclasses.replace(
        plan,
        offload_policy = "none",
        vae_tiling = plan.vae_tiling,
        vae_slicing = plan.vae_slicing,
        stream_text_encoders = False,
        stream_transformer = True,
        resident_transformer_mib = None,
        resident_text_encoder_mib = None,
        reasons = tuple(plan.reasons) + (reason,),
        layer_split = dataclasses.replace(split, reason = reason),
        fallback_plan = plan,
    )


def apply_layer_split_plan(
    pipe: Any,
    plan: Any,
    *,
    placement: str,
    logger: Any = None,
) -> bool:
    """Place a layer split plan. True when placed; on any failure the split is undone, the pipe returns to the host and
    False is returned (the caller then applies ``fallback_plan``)."""
    split = getattr(plan, "layer_split", None)
    if split is None:
        return False
    try:
        place_layer_split(pipe, primary = placement, split = split, logger = logger)
    except Exception as exc:  # noqa: BLE001 - fall back to one card
        if logger is not None:
            logger.warning(
                "diffusion.multigpu: layer split placement failed (%s); falling back to one card",
                exc,
            )
        with contextlib.suppress(Exception):
            undo_layer_split(pipe)
        forget_applied_layer_split(pipe)
        with contextlib.suppress(Exception):
            pipe.to("cpu")
        _drop_stranded_fused_records(pipe)
        with contextlib.suppress(Exception):
            import torch
            torch.cuda.empty_cache()
        _record_decline(pipe, f"placement failed ({exc})", None)
        return False
    with contextlib.suppress(TypeError):
        _APPLIED[pipe] = split
    for name in _denoisers(split):
        module = getattr(pipe, name, None)
        if module is not None:
            module.__dict__[_RECORD_ATTR] = split
    return True


_RECORD_ATTR = "_unsloth_layer_split_record"


def _drop_stranded_fused_records(pipe: Any) -> None:
    """Drop fused records a failed placement left on a card. Never raises."""
    from .diffusion_multigpu import DENOISER_ATTRS
    for name in DENOISER_ATTRS:
        module = getattr(pipe, name, None)
        if module is None:
            continue
        with contextlib.suppress(Exception):
            from .diffusion_int8_fused import drop_stranded_fused_records
            drop_stranded_fused_records(module)


def applied_layer_split(pipe: Any) -> Optional[LayerSplit]:
    """The split placed on ``pipe``, or on a denoiser it shares with the pipeline that placed it."""
    if pipe is None:
        return None
    try:
        split = _APPLIED.get(pipe)
    except TypeError:
        split = None
    if split is not None:
        return split
    from .diffusion_layer_split import is_layer_split as engine_split
    from .diffusion_multigpu import DENOISER_ATTRS

    for name in DENOISER_ATTRS:
        module = getattr(pipe, name, None)
        record = getattr(module, "__dict__", {}).get(_RECORD_ATTR) if module is not None else None
        if record is not None and engine_split(module):
            return record
    return None


def forget_applied_layer_split(pipe: Any) -> None:
    """Teardown: the placed split's record goes (the pipeline's and its denoisers'), a decline reason stays."""
    from .diffusion_multigpu import DENOISER_ATTRS

    with contextlib.suppress(TypeError):
        _APPLIED.pop(pipe, None)
    for name in DENOISER_ATTRS:
        module = getattr(pipe, name, None)
        if module is not None:
            getattr(module, "__dict__", {}).pop(_RECORD_ATTR, None)


def is_layer_split(pipe: Any) -> bool:
    return applied_layer_split(pipe) is not None


def forget_layer_split(pipe: Any) -> None:
    """Unload / re-place: drop the record (the engine's ``undo_layer_split`` removes the hooks)."""
    forget_applied_layer_split(pipe)
    with contextlib.suppress(TypeError):
        _DECLINES.pop(pipe, None)


def layer_split_status(pipe: Any) -> Optional[dict[str, Any]]:
    """``status()["multi_gpu"]["layer_split"]``: the applied split, ``{"declined": reason}`` after an attempt that did
    not engage, or None when none was considered."""
    split = applied_layer_split(pipe)
    if split is not None:
        return split.as_public_dict()
    try:
        declined = _DECLINES.get(pipe) if pipe is not None else None
    except TypeError:
        declined = None
    return {"declined": declined} if declined is not None else None


def layer_split_card_floors(pipe: Any) -> Optional[tuple]:
    """``vram_split_cards`` for the video guard: ``(ordinal, resident MiB, decodes, home, denoises, block headroom
    MiB, reserve MiB)`` per card. The VAE and the stems stay on the primary, and every card runs blocks. The last two
    fields are what the plan held on a card that holds only blocks: one block's working set at the family's default
    request (it scales with the request) and the fixed per-card reserve (it does not); None on the primary, which the
    whole-request estimate prices."""
    split = applied_layer_split(pipe)
    if split is None:
        return None
    weights = dict(split.card_weights_mib)
    return tuple(
        (
            o,
            int(weights.get(o, 0)),
            o == split.primary,
            o == split.primary,
            True,
            None if o == split.primary else split.working_mib(o)[0],
            None if o == split.primary else split.working_mib(o)[1],
        )
        for o in split.cards
    )


def secondary_block_cards(pipe: Any) -> tuple[int, ...]:
    """Cards other than the primary that run blocks (each needs its own generate-time activation check)."""
    split = applied_layer_split(pipe)
    if split is None:
        return ()
    return tuple(sorted({r.card for r in split.ranges if r.card != split.primary}))
