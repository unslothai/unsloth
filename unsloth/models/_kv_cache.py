# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""A transformers StaticCache whose full-attention layers grow by power-of-two buckets.

Within a bucket tensors keep shape and address (graph/compile friendly); growth is eager between buckets. The growth
check reads only the host counter `cumulative_length_int`, never the device."""

import inspect
import os
import sys

import torch

try:
    from transformers.cache_utils import StaticLayer as _StaticLayer
except Exception:  # transformers without layered static caches
    _StaticLayer = None

__all__ = [
    "GrowableLayer",
    "enabled",
    "make_growable",
    "supports",
    "is_growable",
    "needs_growth",
    "fills_bucket",
    "reserve",
    "frozen",
    "written",
    "epoch",
    "capacity",
    "bucket",
]


def enabled():
    return _StaticLayer is not None and os.environ.get("UNSLOTH_GROWABLE_KV", "0") == "1"


def _min_slots():
    try:
        return max(1, int(os.environ.get("UNSLOTH_GROWABLE_KV_MIN", "256")))
    except ValueError:
        return 256


def bucket(needed, limit):
    """Smallest power of two >= needed (and >= the minimum), clamped to `limit`."""
    size = max(_min_slots(), 1 << max(int(needed) - 1, 0).bit_length())
    return min(size, int(limit)) if limit else size


class _Group:
    """Shared by the growable layers of one cache: they grow together, and graphs key on the epoch."""

    __slots__ = ("members", "epoch", "__weakref__")

    def __init__(self):
        self.members = []
        self.epoch = 0


if _StaticLayer is not None:

    class GrowableLayer(_StaticLayer):
        """A transformers StaticLayer that allocates `capacity` slots (a bucket) instead of `max_cache_len`."""

        def __init__(
            self,
            max_cache_len,
            group = None,
        ):
            super().__init__(max_cache_len = max_cache_len)
            self.capacity = None
            self.cumulative_length_int = 0
            self.group = group if group is not None else _Group()

        def _plan(self, q_len):
            """Capacity for q_len more tokens; grows the group if short. Never during a capture: the host counter is
            already past the step and the copy would be recorded, not run."""
            needed = self.cumulative_length_int + int(q_len)
            if (self.capacity is None or needed > self.capacity) and not _frozen():
                _grow_group(self.group, bucket(needed, max(int(self.max_cache_len), needed)))
            return self.capacity

        def lazy_initialization(self, key_states, *rest):
            if self.capacity is None:
                self.capacity = bucket(key_states.shape[-2], self.max_cache_len)
            limit = self.max_cache_len
            self.max_cache_len = self.capacity
            try:
                super().lazy_initialization(key_states, *rest)
            finally:
                self.max_cache_len = limit

        def update(self, key_states, value_states, *args, **kwargs):
            if torch.compiler.is_compiling():
                return super().update(key_states, value_states, *args, **kwargs)
            q_len = key_states.shape[-2]
            self._plan(q_len)
            out = super().update(key_states, value_states, *args, **kwargs)
            self.cumulative_length_int += q_len
            return out

        def get_mask_sizes(self, *args, **kwargs):
            query = args[0] if args else kwargs.get("query_length", kwargs.get("cache_position"))
            q_len = query.shape[0] if isinstance(query, torch.Tensor) else int(query)
            if not torch.compiler.is_compiling():
                return self._plan(q_len), 0
            # Compiled mask code must not mutate the cache: generate reserves beforehand, so only read capacity.
            if self.capacity is None:
                return bucket(q_len, self.max_cache_len), 0
            return self.capacity, 0

        def get_max_cache_shape(self):
            return self.max_cache_len

        def reset(self):
            super().reset()
            self.cumulative_length_int = 0

        def _grow(self, size):
            if size <= (self.capacity or 0):
                return
            old = self.capacity
            self.capacity = size
            if not getattr(self, "is_initialized", False):
                return
            for name in ("keys", "values"):
                tensor = getattr(self, name)
                shape = list(tensor.shape)
                shape[-2] = size
                grown = tensor.new_zeros(shape)
                grown[:, :, :old].copy_(tensor)
                torch._dynamo.mark_static_address(grown)
                setattr(self, name, grown)

else:  # pragma: no cover
    GrowableLayer = None


_FREEZE = [0]


def _frozen():
    if _FREEZE[0]:
        return True
    try:
        return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
    except Exception:
        return False


class frozen:
    """No growth while active (decode graph captures): steps use the capacity reserved before them."""

    def __enter__(self):
        _FREEZE[0] += 1
        return self

    def __exit__(self, *exc):
        _FREEZE[0] -= 1
        return False


def _grow_group(group, size):
    for layer in group.members:
        layer._grow(size)
    group.epoch += 1


_SUPPORTS = {}


def _sizes_by_max_cache_shape(module):
    # Old-style models size the mask / ALiBi by get_max_cache_shape(), which would disagree with the bucket.
    if module is None:
        return False
    if hasattr(module, "_prepare_4d_causal_attention_mask_with_cache_position"):
        return True
    source = sys.modules.get(type(module).__module__)
    try:
        return "get_max_cache_shape(" in inspect.getsource(source) if source is not None else False
    except (OSError, TypeError):
        return True


def supports(model):
    """Whether this model's mask code reads the cache's allocated length (transformers' masking_utils), so its
    full-attention layers may grow. Models that size masks or ALiBi by get_max_cache_shape() keep StaticCache."""
    key = type(model)
    if key not in _SUPPORTS:
        base = getattr(model, getattr(model, "base_model_prefix", ""), None) or model
        decoder = base.get_decoder() if hasattr(base, "get_decoder") else None
        _SUPPORTS[key] = not any(
            _sizes_by_max_cache_shape(m)
            for m in {id(x): x for x in (model, base, decoder) if x is not None}.values()
        )
    return _SUPPORTS[key]


def make_growable(cache):
    """Swaps the uninitialised full-attention StaticLayers of a transformers StaticCache for growable ones, in place.
    Returns the cache. Initialised layers (a cache that already holds tokens) are left as they are."""
    if GrowableLayer is None or cache is None:
        return cache
    layers = getattr(cache, "layers", None)
    if not layers:
        return cache
    group = None
    for i, layer in enumerate(layers):
        if type(layer) is not _StaticLayer or getattr(layer, "is_initialized", False):
            continue
        if group is None:
            group = _Group()
        grown = GrowableLayer(max_cache_len = layer.max_cache_len, group = group)
        group.members.append(grown)
        layers[i] = grown
    return cache


def _members(cache):
    for layer in getattr(cache, "layers", None) or ():
        if GrowableLayer is not None and type(layer) is GrowableLayer:
            yield layer


def is_growable(cache):
    return any(True for _ in _members(cache))


def fills_bucket(cache, q_len):
    """Whether a step writing q_len tokens fills the last slots of this cache's bucket: a graph captured there
    can never replay (the next step grows the cache and moves its tensors)."""
    for layer in _members(cache):
        if layer.capacity is not None and layer.cumulative_length_int + q_len == layer.capacity:
            return True
    return False


def needs_growth(cache, q_len):
    """Whether a step writing q_len tokens grows this cache (host counters only, no device read)."""
    for layer in _members(cache):
        if layer.capacity is None or layer.cumulative_length_int + q_len > layer.capacity:
            return True
    return False


def reserve(
    cache,
    total,
    written = None,
):
    """Grows a growable cache, eagerly, to hold `total` tokens. `written` re-syncs the host counters, which a
    compiled update or a replay may leave stale."""
    members = list(_members(cache))
    if not members:
        return
    if written is not None and written >= 0:
        for layer in members:
            layer.cumulative_length_int = int(written)
    layer = members[0]
    if layer.capacity is None or int(total) > layer.capacity:
        _grow_group(layer.group, bucket(total, max(int(layer.max_cache_len), int(total))))


def written(cache):
    """Tokens written to a growable cache (host counter), 0 when it has no growable layer."""
    return max((layer.cumulative_length_int for layer in _members(cache)), default = 0)


def epoch(cache):
    for layer in _members(cache):
        return layer.group.epoch
    return None


def capacity(cache):
    """Allocated slots of the growable layers (None when the cache has none)."""
    caps = [layer.capacity for layer in _members(cache) if layer.capacity is not None]
    return max(caps) if caps else None


def short_layers(cache, max_cache_len):
    """Whether a sliding / chunked layer was cut at the creating call's limit (min(window, limit)), so a longer
    call up to `max_cache_len` would roll it early."""
    limit = max((int(layer.max_cache_len) for layer in _members(cache)), default = None)
    if limit is None:
        return False
    for layer in getattr(cache, "layers", None) or ():
        if GrowableLayer is not None and type(layer) is GrowableLayer:
            continue
        length = getattr(layer, "max_cache_len", None)
        if type(length) is int and limit <= length < int(max_cache_len):
            return True
    return False


def raise_limit(cache, max_cache_len):
    """A kept growable cache reused by a longer call: its layers may grow up to the new limit."""
    for layer in _members(cache):
        if int(max_cache_len) > int(layer.max_cache_len):
            layer.max_cache_len = int(max_cache_len)
