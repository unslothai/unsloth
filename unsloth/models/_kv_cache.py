# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""A transformers StaticCache whose full-attention layers grow by power-of-two buckets.

A StaticCache allocates prompt + max_new_tokens slots for every full-attention layer at prefill and attention reads
all of them every step, however early the reply stops. Here each full layer allocates the smallest bucket that holds
what is written (at least UNSLOTH_GROWABLE_KV_MIN slots, never more than the limit generate sized for) and doubles
when a step would pass it. Within a bucket every tensor keeps its shape and address, so a step is as compile- and
CUDA-graph-friendly as a StaticCache step; growth happens eagerly between buckets and moves the tensors (one graph
capture / compile per bucket). Sliding and chunked layers stay transformers' own: they are capped at their window.

Writes are transformers' own StaticLayer.update (an index_copy_ at device positions). The number of written tokens is
also kept on the host (`cumulative_length_int`, advanced in Python, never read back from the device), which is what
the growth check reads."""

import os

import torch

try:
    from transformers.cache_utils import StaticLayer as _StaticLayer
except Exception:  # transformers without layered static caches
    _StaticLayer = None

__all__ = [
    "GrowableLayer",
    "enabled",
    "make_growable",
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
            """Capacity this layer needs to take q_len more tokens; grows the group if it is short. Never inside a
            graph capture: the host counter there is already past the step (the warm-up ran it), and a capture
            would record the copy into the new buffers without running it. generate reserves beforehand."""
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
                # Fixed-shape path only: no host branch and no Python counter inside a compiled graph.
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
                # Grow before the forward that writes, so the mask and every layer's keys share one length.
                return self._plan(q_len), 0
            # A compiled mask builder (Unsloth compiles transformers' masks) must not mutate the cache: generate
            # grows it eagerly beforehand (reserve()); before the first write the prefill bucket is what update()
            # will allocate. Only the capacity is read, so a compiled mask recompiles once per bucket.
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
    """Grows a growable cache, eagerly, so it holds `total` tokens (host ints only). generate calls it before a
    step's mask is built, so a mask built inside compiled code reads the grown capacity. `written` (the tokens
    the cache holds before this step, from generate's own lengths) re-syncs the host counters: a step that ran
    update() under torch.compile or as a graph replay without the replay path advancing them leaves them behind,
    and they must never drive a growth decision stale."""
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


def raise_limit(cache, max_cache_len):
    """A kept growable cache reused by a longer call: its layers may grow up to the new limit."""
    for layer in _members(cache):
        if int(max_cache_len) > int(layer.max_cache_len):
            layer.max_cache_len = int(max_cache_len)
