# SPDX-License-Identifier: AGPL-3.0-only
"""MLX inference backend for Apple Silicon; drop-in InferenceBackend replacement using mlx-lm/mlx-vlm."""

import copy
import difflib
import functools
import hashlib
import importlib
import inspect
import os
import re
import shutil
import sys
import threading
import time
from collections import Counter, OrderedDict
from contextlib import ExitStack, closing, contextmanager, nullcontext
from typing import Optional, Generator
from core.inference.worker import RowRefused
from core.inference import context_refusal
from core.inference.native_tool_tokens import (
    NativeToolTokenDecoder,
    closes_an_open_envelope,
    decoder_preserves_token,
    reasoning_control_tokens,
    stop_token_text,
)
from core.inference.runtime_context import (
    MAX_REQUESTABLE_CONTEXT,
    UNSET_GENERATION_BUDGET,
    generation_budget_for_window,
    runtime_context_length,
)
from core.inference.chat_template_helpers import (
    # Aliased to this module's historic names; bodies moved to the shared helper (#10092).
    alternating_turns,
    count_structured_images as _count_vlm_images,
    count_structured_videos as _count_vlm_videos,
    detect_reasoning_channel_markers,
    make_reasoning_normalizer,
    messages_have_tool_history as _vlm_messages_have_tool_history,
    messages_with_attached_image,
    neutralize_for_render,
    normalize_reasoning_snapshots,
    prompt_opens_reasoning_channel,
    strip_open_reasoning_prefill,
    ThoughtUnresumableError,
    trailing_assistant_resume_kind,
    trailing_assistant_text,
    vlm_prompt_issue as _vlm_prompt_issue,
)
from core.inference.mcp_images import (
    image_marker_parts,
    pixels_in_marker_order,
    top_up_image_markers,
    trim_image_turns,
)
from utils.models.model_config import (
    is_audio_input_type,
    load_mlx_adapter_processor,
    load_mlx_adapter_tokenizer,
)
from utils.utils import is_metal_queue_dead
from loggers import get_logger

logger = get_logger(__name__)


# mlx-vlm prefix reuse: a forward is not shape-invariant, so reuse must prefill on mlx-vlm's grid

VLM_PREFILL_STEP = 2048
VLM_PROMPT_CACHE_ENTRIES = 6


def _prefill_layer_step(model, step):
    if not step:
        return None
    try:
        from unsloth_zoo.mlx.offload import wide_prefill_step
        wide = wide_prefill_step(model)
    except Exception:
        return None
    own = vlm_prefill_step()
    return own if wide == own and step % own == 0 else None


def vlm_prefill_step():
    """mlx-vlm's own step, so an unreused request prefills as mlx-vlm does. It divides every
    boundary, so a step no grid can be built on falls back here rather than reaching a request."""
    try:
        from mlx_vlm.generate.common import DEFAULT_PREFILL_STEP_SIZE
        step = int(DEFAULT_PREFILL_STEP_SIZE)
    except Exception:
        return VLM_PREFILL_STEP
    return step if step > 0 else VLM_PREFILL_STEP


def shape_stable_prefix(
    token_count,
    origin = 0,
    step = VLM_PREFILL_STEP,
):
    """Rows whole chunks produced from ``origin``; mlx-vlm holds the last token back."""
    rows = token_count - 1
    if rows < origin:
        return 0
    return origin + (rows - origin) // step * step


def _is_cache(value):
    """Every mlx-vlm cache class carries ``state``; read off the type so no property runs."""
    return hasattr(value, "__dict__") and hasattr(type(value), "state")


def _is_opaque_cache(value):
    """Attributes in __slots__ are invisible to vars(), so the copy must stop, not return the original."""
    return not hasattr(value, "__dict__") and hasattr(type(value), "state")


def _rebuilt_tuple(original, items):
    """Rebuild as the tuple's own type: TurboQuant dispatches on its state class."""
    rebuild = getattr(type(original), "_make", None)
    return rebuild(items) if rebuild is not None else tuple(items)


def _copy_value(value, mx):
    if isinstance(value, mx.array):
        # Safe to share: MLX writes in place only into an unreferenced buffer; slices are copied out.
        return mx.contiguous(value)
    if isinstance(value, list):
        return [_copy_value(item, mx) for item in value]
    if isinstance(value, tuple):
        return _rebuilt_tuple(value, (_copy_value(item, mx) for item in value))
    if isinstance(value, dict):
        return {key: _copy_value(item, mx) for key, item in value.items()}
    if _is_cache(value):
        duplicate = copy.copy(value)
        for name, item in list(vars(duplicate).items()):
            setattr(duplicate, name, _copy_value(item, mx))
        return duplicate
    if _is_opaque_cache(value):
        raise TypeError(f"{type(value).__name__} cannot be copied")
    return value


def _arrays(value, mx):
    if isinstance(value, mx.array):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _arrays(item, mx)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _arrays(item, mx)
    elif _is_cache(value):
        for item in vars(value).values():
            yield from _arrays(item, mx)


def _release_value(value, mx):
    if isinstance(value, mx.array):
        return None
    if isinstance(value, list):
        value[:] = [_release_value(item, mx) for item in value]
    elif isinstance(value, tuple):
        return _rebuilt_tuple(value, (_release_value(item, mx) for item in value))
    elif isinstance(value, dict):
        for key, item in list(value.items()):
            value[key] = _release_value(item, mx)
    elif _is_cache(value):
        for name, item in list(vars(value).items()):
            setattr(value, name, _release_value(item, mx))
    return value


def release_cache_entries(entries):
    import mlx.core as mx
    entries[:] = [_release_value(entry, mx) for entry in entries]


def copy_cache_entries(entries):
    import mlx.core as mx

    for entry in entries:
        if not (_is_cache(entry) or isinstance(entry, (list, tuple))):
            raise TypeError(f"{type(entry).__name__} cannot be copied")
    copies = [_copy_value(entry, mx) for entry in entries]
    mx.eval(list(_arrays(copies, mx)))
    return copies


def cache_entries_nbytes(entries):
    import mlx.core as mx
    return sum(array.nbytes for array in _arrays(entries, mx))


def cache_entries_offset(entries):
    """Tokens the entries hold, else None: only ``offset`` counts prompt rows."""
    for entry in entries:
        offset = getattr(entry, "offset", None)
        if isinstance(offset, int):
            return offset
        nested = vars(entry).values() if _is_cache(entry) else entry
        if isinstance(nested, (list, tuple)) or _is_cache(entry):
            found = cache_entries_offset(
                [item for item in nested if _is_cache(item) or isinstance(item, (list, tuple))]
            )
            if found is not None:
                return found
    return None


def media_prefix_end(token_ids, media_token_ids):
    """Rows through the last media placeholder, 0 if none: what mlx-vlm resumes past."""
    media = {int(token) for token in media_token_ids}
    if not media:
        return 0
    for index in range(len(token_ids) - 1, -1, -1):
        if int(token_ids[index]) in media:
            return index + 1
    return 0


def _chunk_rows(args, kwargs):
    for chunk in (kwargs.get("inputs_embeds"), kwargs.get("inputs"), args[0] if args else None):
        shape = getattr(chunk, "shape", None)
        if shape and len(shape) > 1:
            return shape[1]
    return 0


def _place_per_layer_inputs(kwargs, rows, start):
    """Slice ``per_layer_inputs`` to this forward's own rows: mlx-vlm computes it for the whole
    uncached prompt and the model slices it at an offset a resume shifts. A single-token forward
    gets none and computes its own, as the reference model does."""
    inputs = kwargs.get("per_layer_inputs")
    shape = getattr(inputs, "shape", None)
    if not shape:
        return
    if rows == 1:
        kwargs.pop("per_layer_inputs")
    elif rows > 1 and shape[1] != rows and 0 <= start and start + rows <= shape[1]:
        kwargs["per_layer_inputs"] = inputs[:, start : start + rows]


class _ForwardRecord:
    __slots__ = (
        "forwards",
        "capture_at",
        "capture_rows",
        "snapshot",
        "tail_at",
        "tail",
        "resume_offset",
        "on_resume",
    )

    def __init__(self):
        self.forwards = 0
        self.capture_at = 0
        self.capture_rows = 0
        self.snapshot = None
        self.tail_at = 0
        self.tail = None
        self.resume_offset = None
        self.on_resume = None


_RECORDING_CLASSES = {}


def _prompt_wide_position_ids(args, kwargs):
    """Whether ``position_ids`` spans more than the chunk being fed: mlx-vlm primes mRoPE
    prompt-wide and not every family slices the kwarg (glm4v feeds it through as is), so a
    prompt-wide one is withheld for the model's own primed state."""
    shape = getattr(kwargs.get("position_ids"), "shape", None)
    if not shape:
        return False
    chunk = kwargs.get("inputs", args[0] if args else None)
    chunk_shape = getattr(chunk, "shape", None) or getattr(
        kwargs.get("inputs_embeds"), "shape", None
    )
    return bool(chunk_shape) and len(chunk_shape) > 1 and shape[-1] != chunk_shape[1]


def _forward_capturing(base, model, args, kwargs, cache, rows):
    from unsloth_zoo.mlx.offload import capture_prefill_rows

    def copy(entry):
        try:
            return copy_cache_entries([entry])[0]
        except Exception as exc:
            logger.info("MLX VLM prompt cache: cache layout not copied (%s)", exc)

    with capture_prefill_rows(rows, copy) as captured:
        output = base.__call__(model, *args, **kwargs)
    snapshot = [captured.get(id(entry)) for entry in cache]
    return output, snapshot if all(entry is not None for entry in snapshot) else None


def _recording_class(base):
    """``base`` with a forward that counts itself and copies the cache on cue."""
    cls = _RECORDING_CLASSES.get(base)
    if cls is not None:
        return cls

    def __call__(self, *args, **kwargs):
        # A plain attribute: nn.Module stores dicts and lists in its module tree.
        record = self.__dict__.get("_studio_forward_record")
        cache = kwargs.get("cache")
        if record is not None:
            offset = cache_entries_offset(cache) if cache is not None else None
            if record.resume_offset is None:
                record.resume_offset = offset or 0
                if record.on_resume is not None:
                    record.on_resume(record.resume_offset)
            # Only after a resume: a Qwen VL model fed no positions reuses the previous request's.
            if record.resume_offset and _prompt_wide_position_ids(args, kwargs):
                kwargs.pop("position_ids")
            _place_per_layer_inputs(
                kwargs, _chunk_rows(args, kwargs), (offset or 0) - record.resume_offset
            )
        # As the next forward begins, not as the boundary one returns: generate_step converts the cache between the
        # two (KV quantization).
        boundary = record is not None and record.forwards == record.capture_at
        if (
            record is not None
            and record.forwards
            and (boundary and not record.capture_rows or record.forwards == record.tail_at)
            and cache is not None
        ):
            try:
                copied = copy_cache_entries(cache)
            except Exception as exc:
                logger.info("MLX VLM prompt cache: cache layout not copied (%s)", exc)
            else:
                if boundary:
                    record.snapshot = copied
                else:
                    record.tail = copied
        if boundary and record.capture_rows and cache is not None:
            output, record.snapshot = _forward_capturing(
                base, self, args, kwargs, cache, record.capture_rows
            )
        else:
            output = base.__call__(self, *args, **kwargs)
        if record is not None:
            record.forwards += 1
        return output

    cls = type(f"Recording{base.__name__}", (base,), {"__call__": __call__, "_studio_base": base})
    _RECORDING_CLASSES[base] = cls
    return cls


def _offset_aware_policy(host):
    """``chunked_prefill_policy`` of ``host`` fed only the token types past the cache: with the
    vision block already there, the suffix is text and may chunk."""
    base_policy = type(host).chunked_prefill_policy

    def policy(
        *,
        prompt_cache = None,
        prefill_kwargs = None,
        **rest,
    ):
        offset = cache_entries_offset(prompt_cache) if prompt_cache is not None else 0
        if offset and prefill_kwargs:
            prefill_kwargs = dict(prefill_kwargs)
            for key in ("mm_token_type_ids", "token_type_ids"):
                shape = getattr(prefill_kwargs.get(key), "shape", None)
                if shape and len(shape) > 1 and shape[1] > offset:
                    prefill_kwargs[key] = prefill_kwargs[key][:, offset:]
        return base_policy(host, prompt_cache = prompt_cache, prefill_kwargs = prefill_kwargs, **rest)

    return policy


class RecordingForward:
    def __init__(self, language_model):
        self._language_model = language_model
        self.record = _ForwardRecord()
        self._base = None

    def __enter__(self):
        model = self._language_model
        base = getattr(type(model), "_studio_base", type(model))
        self._base = base
        model.__class__ = _recording_class(base)
        model.__dict__["_studio_forward_record"] = self.record
        return self

    def __exit__(self, *_exc):
        model = self._language_model
        model.__dict__.pop("_studio_forward_record", None)
        model.__class__ = self._base
        return False

    def unrecorded(self, *args, **kwargs):
        """A forward the record does not count: rows prepared outside the generation."""
        return self._base.__call__(self._language_model, *args, **kwargs)


def _cache_classes(name):
    modules = (sys.modules.get(m) for m in ("mlx_vlm.models.cache", "mlx_lm.models.cache"))
    return tuple(getattr(m, name) for m in modules if hasattr(m, name))


def _rows_in_order(entry, classes):
    """Rows are positions 0..offset-1: plain KV, or a ring that has not dropped a row yet."""
    kv, rings = classes
    if entry.keys is None:
        return False
    return type(entry) in kv or (
        type(entry) in rings and entry.offset == entry._idx == entry.keys.shape[2]
    )


def _shared_kv_pairs(entries, base, classes, pairs):
    """All or none: False when ``base`` cannot serve every in-order entry."""
    if isinstance(entries, (list, tuple)):
        if not isinstance(base, (list, tuple)) or len(entries) != len(base):
            return False
        return all(_shared_kv_pairs(e, b, classes, pairs) for e, b in zip(entries, base))
    if type(entries) is not type(base):
        return False
    if type(entries) in classes[0] + classes[1]:
        if not _rows_in_order(entries, classes):
            return True
        if not _rows_in_order(base, classes) or entries.offset > base.offset:
            return False
        if entries.keys.shape[2] > base.keys.shape[2]:
            return False
        pairs.append((entries, base))
        return True
    nested = getattr(entries, "caches", None)
    if isinstance(nested, (list, tuple)):
        return _shared_kv_pairs(nested, base.caches, classes, pairs)
    return True


def share_kv_rows(entries, base):
    """Point in-order entries at ``base``'s leading rows; returns bytes now read through ``base``."""
    pairs = []
    classes = (_cache_classes("KVCache"), _cache_classes("RotatingKVCache"))
    if not _shared_kv_pairs(entries, base, classes, pairs):
        return 0
    shared = 0
    for entry, source in pairs:
        rows = entry.keys.shape[2]
        entry.keys = source.keys[..., :rows, :]
        entry.values = source.values[..., :rows, :]
        shared += entry.keys.nbytes + entry.values.nbytes
    return shared


def compact_sliding_windows(entries):
    """Drop ring rows past the window after a multi-row update; no later update reads them."""
    rings = _cache_classes("RotatingKVCache")
    pending = list(entries)
    compacted = []
    while pending:
        entry = pending.pop()
        nested = entry if isinstance(entry, (list, tuple)) else getattr(entry, "caches", None)
        if isinstance(nested, (list, tuple)):
            pending.extend(nested)
        elif type(entry) in rings and entry.keys is not None:
            excess = entry.keys.shape[2] - entry.max_size
            if excess > 0 and entry._idx == entry.keys.shape[2]:
                entry.keys = entry._trim(excess, entry.keys)
                entry.values = entry._trim(excess, entry.values)
                entry._idx = entry.max_size
                compacted.extend((entry.keys, entry.values))
    if compacted:
        import mlx.core as mx
        mx.eval(compacted)


class VLMPromptSnapshotStore:
    """LRU boundary snapshots under a byte budget; a shorter prefix reads KV rows from its base.
    A replay (one per key) serves only a prompt of that exact length and reads through no base."""

    def __init__(
        self,
        max_bytes,
        max_entries = VLM_PROMPT_CACHE_ENTRIES,
    ):
        self._max_bytes = max_bytes
        self._max_entries = max_entries
        self._entries = OrderedDict()
        self._bases = {}
        self._replays = {}
        self.nbytes = 0

    def __len__(self):
        return len(self._entries)

    def lookup(self, key, token_ids, limit):
        """Longest stored prefix ending by ``limit``, or the replay holding all but the last token."""
        best = None
        for item, (entries, _nbytes) in self._entries.items():
            stored_key, prefix = item
            n = len(prefix)
            fits = n == len(token_ids) - 1 if self._is_replay(item) else n <= limit
            if stored_key != key or n == 0 or not fits:
                continue
            if best is not None and n <= len(best[0]):
                continue
            if tuple(token_ids[:n]) == prefix:
                best = (prefix, entries)
        if best is None:
            return None, 0
        self._touch((key, best[0]))
        return best[1], len(best[0])

    def store(
        self,
        key,
        prefix_ids,
        entries,
        replay = False,
    ):
        compact_sliding_windows(entries)
        nbytes = cache_entries_nbytes(entries)
        if nbytes > self._max_bytes:
            logger.debug(
                "MLX VLM prompt cache: skipping %.2f GB snapshot over the %.2f GB budget",
                nbytes / 1e9,
                self._max_bytes / 1e9,
            )
            return False
        item = (key, tuple(prefix_ids))
        replay = replay or self._is_replay(item)
        readers = [other for other, base in self._bases.items() if base == item]
        self._pop(item)
        self._entries[item] = (entries, nbytes)
        self.nbytes += nbytes
        replaced = self._replays.get(key)
        if replay:
            self._replays[key] = item
        longer = [other for other in reversed(self._entries) if _extends(other, item)]
        base = self._bases.get(longer[0], longer[0]) if longer and not replay else item
        rebased = {
            other
            for other in list(self._entries)
            if other != base
            and other not in (replaced, self._replays.get(key))
            and (other == item or _extends(item, other))
            and self._rebase(other, base)
        }
        for other in readers:
            if other not in rebased:
                self.discard(other)
        if replay and replaced not in (None, item):
            self._hand_down(replaced)
            self.discard(replaced)
        self._touch(item)
        while self._entries and (
            self.nbytes > self._max_bytes or len(self._entries) > self._max_entries
        ):
            self.discard(next(iter(self._entries)))
        return item in self._entries

    def _rebase(self, item, base):
        entries, nbytes = self._entries[item]
        shared = share_kv_rows(entries, self._entries[base][0])
        if shared:
            owned = cache_entries_nbytes(entries) - shared
            self._entries[item] = (entries, owned)
            self.nbytes += owned - nbytes
            self._bases[item] = base
        return bool(shared)

    def _hand_down(self, item):
        """Re-home ``item``'s readers first, else an edit's new replay drops the old branch."""
        readers = [other for other, base in self._bases.items() if base == item]
        for other in sorted(readers, key = lambda other: len(other[1]), reverse = True):
            del self._bases[other]
            longer = [
                longest
                for longest in reversed(self._entries)
                if longest != item and self._bases.get(longest) != item and _extends(longest, other)
            ]
            longer.sort(key = lambda longest: len(longest[1]), reverse = True)
            if longer and self._rebase(other, self._bases.get(longer[0], longer[0])):
                self._touch(other)
                continue
            entries, nbytes = self._entries[other]
            try:
                entries = copy_cache_entries(entries)
            except Exception as exc:
                logger.info("MLX VLM prompt cache: snapshot not handed down (%s)", exc)
                self.discard(other)
                continue
            owned = cache_entries_nbytes(entries)
            self._entries[other] = (entries, owned)
            self.nbytes += owned - nbytes
            self._entries.move_to_end(other)

    def _touch(self, item):
        self._entries.move_to_end(item)
        base = self._bases.get(item)
        if base is not None:
            # Base is evicted last.
            self._entries.move_to_end(base)

    def _is_replay(self, item):
        return self._replays.get(item[0]) == item

    def _pop(self, item):
        dropped = self._entries.pop(item, None)
        self._bases.pop(item, None)
        if self._is_replay(item):
            del self._replays[item[0]]
        if dropped is not None:
            self.nbytes -= dropped[1]

    def discard(self, item):
        self._pop(item)
        for other in [other for other, base in self._bases.items() if base == item]:
            self.discard(other)

    def retain(self, item):
        keep = (item, self._bases.get(item))
        for other in [other for other in self._entries if other not in keep]:
            self.discard(other)

    def retain_key(self, key):
        for other in [other for other in self._entries if other[0] != key]:
            self.discard(other)

    def resume_entries(self):
        """The newest turn's longest boundary snapshot, then its replay."""
        newest = next(reversed(self._entries), None)
        if newest is None:
            return []
        key = newest[0]
        replay = self._replays.get(key)
        boundaries = [i for i in self._entries if i[0] == key and not self._is_replay(i)]
        # _touch keeps the longest last, but after an edit that can be an older base the replay's
        # turn no longer extends: prefer the longest one it does.
        fits = [i for i in boundaries if replay is not None and replay[1][: len(i[1])] == i[1]]
        boundary = max(fits, key = lambda i: len(i[1])) if fits else next(reversed(boundaries), None)
        return [
            (item[0], list(item[1]), self._entries[item][0], self._is_replay(item))
            for item in (boundary, replay)
            if item is not None
        ]

    def clear(self):
        self._entries.clear()
        self._bases.clear()
        self._replays.clear()
        self.nbytes = 0


def _extends(item, other):
    (key, prefix), (other_key, other_prefix) = item, other
    return (
        key == other_key
        and len(prefix) > len(other_prefix)
        and prefix[: len(other_prefix)] == other_prefix
    )


class VLMPromptCacheSession:
    """The ``prompt_cache_state`` one generation hands to mlx-vlm: ``find_prefix_length`` picks the
    snapshot to resume from, swaps ``cache`` to it and arms the capture, and ``finish`` stores
    it. ``releases_unserved`` frees the rest, the headroom a media pass needs. A ``media_block``
    prefills the rows through the last vision token for models that cannot chunk them, becoming
    the grid's origin, and ``policy_hosts`` are the models whose ``chunked_prefill_policy`` would
    then refuse to chunk the suffix."""

    def __init__(
        self,
        store,
        key,
        language_model,
        make_cache,
        media_token_ids = (),
        releases_unserved = False,
        media_block = None,
        policy_hosts = (),
        step = VLM_PREFILL_STEP,
        window = None,
    ):
        self.step = step
        self.window = window or step
        self._store = store
        self._key = key
        self._media_token_ids = tuple(media_token_ids)
        self._releases_unserved = releases_unserved
        self.media_block = media_block
        self._origin = 0
        # Produced for this request: prefill work rather than reuse, for the stats.
        self.produced_tokens = 0
        self.produced_seconds = 0.0
        self._policy_hosts = tuple(policy_hosts)
        self._forward = RecordingForward(language_model)
        self.cache = make_cache()
        self.reused_tokens = 0
        self._token_ids = None
        self._media_end = 0

    def __enter__(self):
        self._forward.__enter__()
        self._patched_hosts = [
            host for host in self._policy_hosts if hasattr(type(host), "chunked_prefill_policy")
        ]
        for host in self._patched_hosts:
            host.chunked_prefill_policy = _offset_aware_policy(host)
        return self

    def __exit__(self, *exc):
        try:
            for host in self._patched_hosts:
                # policy_hosts is (model, language_model), which name one object twice when the wrapper exposes none,
                # so unpatch idempotently rather than del twice.
                host.__dict__.pop("chunked_prefill_policy", None)
        finally:
            # The class goes back regardless: one left wrapped serves every later request.
            self._forward.__exit__(*exc)
        return False

    def find_prefix_length(self, token_ids):
        token_ids = list(token_ids)
        self._token_ids = token_ids
        origin = self.media_block.rows(token_ids) if self.media_block is not None else 0
        self._origin = origin
        boundary = shape_stable_prefix(len(token_ids), origin, self.step)
        self._media_end = media_prefix_end(token_ids, self._media_token_ids)
        record = self._forward.record
        if boundary < self._media_end:
            # Inside the media span: the prompt has to grow past it first.
            self._keep(None)
            return 0
        entries, prefix_len = self._store.lookup(self._key, token_ids, boundary)
        if entries is None and origin:
            # Produced here instead, in the one square forward the vision tokens need.
            self._keep(None)
            started = time.perf_counter()
            try:
                entries, prefix_len = self.media_block.prefill(self._forward, origin), origin
                self.produced_tokens = origin
            except Exception as exc:
                # Prefilled by mlx-vlm as before, off this grid: no capture.
                logger.info("MLX VLM prompt cache: media block not prefilled (%s)", exc)
                self._keep(None)
                return 0
            finally:
                # Spent either way, and outside mlx-vlm's prompt timer.
                self.produced_seconds = time.perf_counter() - started
        if entries is not None:
            # The copy is taken at the first forward, once mlx-vlm has resumed.
            self.cache = entries
            record.on_resume = self._detach_served
        self._keep((self._key, tuple(token_ids[:prefix_len])) if prefix_len else None)
        rows = boundary - prefix_len
        record.capture_at, record.capture_rows = (
            divmod(rows, self.window) if rows >= 0 else (rows, 0)
        )
        if prefix_len <= boundary < len(token_ids) - 1:
            record.tail_at = record.capture_at + 1
        self.reused_tokens = prefix_len
        return prefix_len

    def _keep(self, item):
        # At lookup, before the media pass a declined request starts.
        if self._releases_unserved:
            self._store.retain(item)

    def _detach_served(self, offset):
        prefix_ids = self._token_ids[: self.reused_tokens]
        if offset != self.reused_tokens:
            # Declined: freed now, before the fresh-cache request's media pass evaluates.
            self._store.discard((self._key, tuple(prefix_ids)))
            try:
                release_cache_entries(self.cache)
            except Exception as exc:
                logger.info("MLX VLM prompt cache: declined snapshot not released (%s)", exc)
            return
        # About to be advanced by this generation: the store keeps a copy.
        try:
            copy = copy_cache_entries(self.cache)
        except Exception as exc:
            # Costs the next turn its reuse, never this answer.
            logger.info("MLX VLM prompt cache: served snapshot not kept (%s)", exc)
            self._store.discard((self._key, tuple(prefix_ids)))
            return
        self._store.store(self._key, prefix_ids, copy)

    def update(self, _token_ids, _cache):
        """mlx-vlm's after-generation hook; ``finish`` stores, as it also runs on a consumer that
        stops reading early."""

    def finish(self):
        if self._token_ids is None:
            return False
        record = self._forward.record
        stored = self._store_boundary(record.snapshot)
        replay = len(self._token_ids) - 1
        if (
            record.tail is not None
            and record.resume_offset == self.reused_tokens
            and cache_entries_offset(record.tail) == replay
        ):
            stored = (
                self._store.store(self._key, self._token_ids[:replay], record.tail, replay = True)
                or stored
            )
        return stored

    def _store_boundary(self, snapshot):
        if snapshot is None:
            return False
        # Read off the snapshot: a declined offer captures an earlier boundary.
        held = cache_entries_offset(snapshot)
        if (
            not held
            or held < self._origin
            or (held - self._origin) % self.step
            or held < self._media_end
            or held > shape_stable_prefix(len(self._token_ids), self._origin, self.step)
        ):
            logger.debug("MLX VLM prompt cache: snapshot holds %r rows, not stored", held)
            return False
        return self._store.store(self._key, self._token_ids[:held], snapshot)


class VLMBatchRowCache:
    """The ``prompt_cache_state`` one batched row hands to unsloth-zoo, on the single path's grid
    and store: ``open`` resumes the longest snapshot ending by the row's boundary and names the
    chunk ends to bank, the boundary and the replay; ``checkpoint`` stores each. With no
    ``store`` the row prefills cold on the same cache layout; ``releases_unserved`` frees what
    does not serve it, as the single path does."""

    def __init__(
        self,
        store,
        key,
        make_cache,
        media_token_ids = (),
        releases_unserved = False,
        step = VLM_PREFILL_STEP,
    ):
        self._store = store
        self._key = key
        self._make_cache = make_cache
        self._media_token_ids = tuple(media_token_ids)
        self._releases_unserved = releases_unserved
        self._step = step
        self._token_ids = []
        self._replay = None

    def open(self, token_ids):
        self._token_ids = token_ids = list(token_ids)
        boundary = shape_stable_prefix(len(token_ids), 0, self._step)
        media_end = media_prefix_end(token_ids, self._media_token_ids)
        if self._store is None:
            return self._make_cache(), ()
        if boundary < media_end:
            self._keep(None)
            return self._make_cache(), ()
        cache = None
        entries, prefix = self._store.lookup(self._key, token_ids, boundary)
        if entries is not None and prefix >= media_end:
            try:
                # The row advances its cache in place; the store keeps its own.
                cache = copy_cache_entries(entries)
            except Exception as exc:
                logger.info("MLX VLM prompt cache: snapshot not copied for a batched row (%s)", exc)
        if cache is None:
            cache, prefix = self._make_cache(), 0
        self._keep((self._key, tuple(token_ids[:prefix])) if prefix else None)
        replay = len(token_ids) - 1
        self._replay = replay if replay != boundary else None
        return cache, [length for length in (boundary, replay) if length > prefix]

    def _keep(self, item):
        if self._releases_unserved:
            self._store.retain(item)

    def checkpoint(self, token_count, cache):
        # Runs inside the batch's prefill: a failure costs a later turn its reuse, never this row.
        try:
            self._store.store(
                self._key,
                self._token_ids[:token_count],
                copy_cache_entries(cache),
                replay = token_count == self._replay,
            )
        except Exception as exc:
            logger.info("MLX VLM prompt cache: batched row snapshot not stored (%s)", exc)


def _mlx_adapter_modules(model):
    """Return bypassable adapter entries and unsupported wrapper paths."""
    adapters = []
    unsupported = []
    for path, module in model.named_modules():
        if not path or not (hasattr(module, "lora_a") and hasattr(module, "lora_b")):
            continue
        base = getattr(module, "linear", None)
        if base is None:
            base = getattr(module, "embedding", None)
        if base is None:
            unsupported.append(path)
        else:
            adapters.append((path, module, base))
    return adapters, unsupported


_MLX_FUSION_UNAVAILABLE = set()


def _mlx_fusion_unavailable(name, error):
    """Log once per cause: this is a per-request path, so repeating would flood."""
    key = (name, type(error).__name__, str(error))
    if key not in _MLX_FUSION_UNAVAILABLE:
        _MLX_FUSION_UNAVAILABLE.add(key)
        logger.warning(
            "Optional MLX fusion %s is unavailable, continuing with native inference: %s: %s",
            name,
            type(error).__name__,
            error,
        )


def _mlx_inference_patch(name):
    """An optional Zoo fusion helper, or None. Never raises: these are a throughput
    optimization, so no Zoo state may fail a load or a request that worked without it."""
    try:
        patches = importlib.import_module("unsloth_zoo.mlx.inference")
    except ModuleNotFoundError as error:
        if error.name != "unsloth_zoo.mlx.inference":
            _mlx_fusion_unavailable(name, error)  # a transitive import inside Zoo failed
        return None
    except Exception as error:
        # A partial or skewed install raises ImportError rather than ModuleNotFoundError.
        _mlx_fusion_unavailable(name, error)
        return None
    return getattr(patches, name, None)


@contextmanager
def _mlx_optional_fusion(name, model, *args):
    """Hold an optional Zoo fusion scope, or yield the model unfused. Guard ENTRY only:
    an open scope must unwind through the ExitStack as an unguarded `with` would, or an
    interrupted generation stops restoring the module tree. A dead Metal queue is not guarded."""
    patch = _mlx_inference_patch(name)
    with ExitStack() as scope:
        active = model
        if patch is not None:
            try:
                entered = scope.enter_context(patch(model, *args))
            except Exception as error:
                if is_metal_queue_dead(error):
                    raise
                _mlx_fusion_unavailable(name, error)
            else:
                if entered is not None:
                    active = entered
        yield active


def _mlx_scope_refused(name, error):
    if is_metal_queue_dead(error):
        raise error
    _mlx_fusion_unavailable(name, error)


# What Studio entered by hand before Zoo's registry; a Zoo without it keeps exactly these.
_MLX_LEGACY_SCOPES = (
    "fused_moe_gate_up",
    "fused_decode_conv_silu",
    "fused_residual_norm",
    "fused_moe_router",
    "fused_moe_routed_experts",
    "fused_residual_norm_handoff",
    "fused_packed_gated_delta",
)


@contextmanager
def _mlx_inference_scopes(model, call, int8_prefill):
    """Hold the Zoo scopes Zoo picks for `call` (its call traits), so a new Zoo scope needs no Studio change.
    A Zoo without the registry keeps the scopes Studio listed before it (none for the text batch)."""
    compose = _mlx_inference_patch("inference_scopes")
    with ExitStack() as scope:
        if compose is None and "batch" not in call:
            for name in _MLX_LEGACY_SCOPES:
                scope.enter_context(_mlx_optional_fusion(name, model))
            if "speculative" in call:
                scope.enter_context(_mlx_optional_fusion("exact_speculative_verify", model))
        elif compose is not None:
            try:
                scope.enter_context(
                    compose(
                        model,
                        call,
                        int8_prefill = int8_prefill,
                        on_error = _mlx_scope_refused,
                    )
                )
            except Exception as error:
                if is_metal_queue_dead(error):
                    raise
                _mlx_fusion_unavailable("inference_scopes", error)
        yield model


def _mlx_warm_inference(model, int8_prefill):
    """Zoo's warm-up generation, so the first request skips the scopes' kernel builds and first-use
    checks. A Zoo without it, or a failed warm-up, leaves them to the first request."""
    warm = _mlx_inference_patch("warm_inference")
    if warm is None:
        return
    try:
        warm(model, int8_prefill = int8_prefill)
    except Exception as error:
        if is_metal_queue_dead(error):
            raise
        logger.warning(
            "MLX warm-up failed, so the first request runs it: %s: %s",
            type(error).__name__,
            error,
        )


def _int8_prefill_zoo():
    """unsloth_zoo's MLX inference module when it can run int8 prefill, else None."""
    try:
        zoo = importlib.import_module("unsloth_zoo.mlx.inference")
        if "int8_prefill" in inspect.signature(zoo.nax_quantized_linear).parameters:
            return zoo
    except Exception:
        pass
    return None


def _int8_prefill_status(requested, model):
    """Whether a load runs int8 prefill: asked for, and the loaded model can use it here."""
    status = {"requested": bool(requested), "active": False, "reason": "", "scope": None}
    if not requested:
        return status
    zoo = _int8_prefill_zoo()
    available = getattr(zoo, "int8_prefill_available", None)
    if available is None:
        status["reason"] = "unsupported_zoo"
        return status
    try:
        verdict = available(model)
    except Exception as exc:
        if is_metal_queue_dead(exc):
            raise
        logger.warning("MLX int8 prefill check failed: %s", exc)
        status["reason"] = "probe_failed"
        return status
    status["reason"] = verdict.reason
    if verdict.available:
        status["active"] = True
        status["scope"] = zoo.nax_quantized_linear
    else:
        logger.info("MLX int8 prefill not applied: %s", verdict.reason)
    return status


def _ane_prefill_zoo():
    """unsloth_zoo's MLX inference module when it can run Neural Engine prefill, else None."""
    try:
        zoo = importlib.import_module("unsloth_zoo.mlx.inference")
        if "ane_prefill" in inspect.signature(zoo.ane_prefill_mlp).parameters:
            return zoo
    except Exception:
        pass
    return None


def _ane_prefill_status(requested, model):
    """Whether a load runs Neural Engine prefill: asked for, and the loaded model can use it here. Read again after
    the warm-up, whose first prepare settles which layers attach."""
    status = {"requested": bool(requested), "active": False, "reason": "", "scope": None}
    if not requested:
        return status
    zoo = _ane_prefill_zoo()
    if zoo is None:
        status["reason"] = "unsupported_zoo"
        return status
    try:
        verdict = zoo.ane_prefill_available(model)
    except Exception as exc:
        if is_metal_queue_dead(exc):
            raise
        logger.warning("MLX Neural Engine prefill check failed: %s", exc)
        status["reason"] = "probe_failed"
        return status
    status["reason"] = verdict.reason
    if verdict.available:
        status["active"] = True
        status["scope"] = zoo.ane_prefill_mlp
    else:
        logger.info("MLX Neural Engine prefill not applied: %s", verdict.reason)
    return status


_MLX_OFFLOAD_PARTS = (("ple", "PLE tables"), ("experts", "quantized routed experts"))


def normalize_mlx_offload(value):
    """An offload knob as a load reads it: ``"auto"`` (also null), the fraction read from disk, or
    None for off."""
    if value in (None, "auto"):
        return "auto"
    return None if value == "off" else float(value)


def mlx_resolve_offload(
    model_dir,
    knobs,
    *,
    pinned,
    load_in_4bit: bool,
    retains_history: bool,
    kv_bits = None,
    vision = None,
    drafter = None,
) -> tuple:
    """The ``(ple, expert)`` fractions a load of *model_dir* reads from disk (None for off), from
    knobs as ``normalize_mlx_offload`` gives them: an ``"auto"`` knob decided by unsloth_zoo's 50/50
    rule on this load's pricing, and off where that cannot be planned; any other knob as given;
    then the planned prefill step for experts on disk, else None. ``vision=True`` prices the
    mlx-vlm route a speculative text load takes; ``drafter``, a ``(path, builtin)`` to make room for."""
    fixed = tuple(None if knob == "auto" else knob for knob in knobs) + (None,)
    if "auto" not in knobs or not model_dir:
        return fixed
    from core.inference.mlx_memory import mlx_offload_plan, mlx_offload_planner

    if mlx_offload_planner() is None:
        return fixed
    budget = mlx_memory_budget(retains_history = retains_history)
    ceiling = pinned or mlx_checkpoint_window(model_dir)
    if budget is None or not ceiling:
        return fixed
    if kv_bits is not None and mlx_kv_quant_is_refused(model_dir, kv_bits):
        kv_bits = None
    with mlx_rng_preserved():
        try:
            plan = mlx_offload_plan(
                model_dir,
                budget_bytes = budget,
                max_ctx = int(ceiling),
                context = pinned or None,
                ple = knobs[0],
                experts = knobs[1],
                kv_bits = kv_bits,
                load_in_4bit = load_in_4bit,
                vision = vision,
                drafter = drafter,
            )
        except Exception as exc:
            logger.info("MLX auto offload not planned for %s: %s", model_dir, exc)
            return fixed
    return fixed if plan is None else plan


# Most of each offloadable part an auto drafter may add to disk (break-even was ~5 points on sampled decode, #15342).
MLX_AUTO_DRAFTER_ROOM = 0.04


def mlx_resolve_drafter_offload(model_dir, knobs, drafter, *, auto: bool, **priced) -> tuple:
    """``(plan, baseline)``: ``mlx_resolve_offload``'s plan with room made for *drafter*, and for an *auto*
    drafter the target-only plan its costless check measures against. An explicit drafter keeps its room
    unless it is priced not to fit there; an auto drafter gets room only when no part puts more than
    ``MLX_AUTO_DRAFTER_ROOM`` more of itself on disk and it then fits costlessly; otherwise, the target-only plan."""
    if not drafter:
        return mlx_resolve_offload(model_dir, knobs, **priced), None
    from types import SimpleNamespace

    def fits(plan, **kwargs):
        return mlx_drafter_fit(
            model_dir,
            mlx_checkpoint_window(model_dir),
            priced["pinned"],
            SimpleNamespace(path = drafter[0], builtin = drafter[1]),
            load_in_4bit = priced["load_in_4bit"],
            **kwargs,
            **mlx_offload_route(plan),
        )[0]

    room = mlx_resolve_offload(model_dir, knobs, drafter = drafter, **priced)
    if not auto:
        # Room a chosen drafter does not fit even under MLX's own cap would only slow the load.
        if "auto" in knobs and fits(room, reserve = False) is False:
            return mlx_resolve_offload(model_dir, knobs, **priced), None
        return room, None
    alone = mlx_resolve_offload(model_dir, knobs, **priced)
    if room == alone:
        return room, alone
    if all(
        (more or 0) - (less or 0) <= MLX_AUTO_DRAFTER_ROOM
        for more, less in zip(room[:2], alone[:2])
    ) and fits(room, costless = True, baseline = mlx_offload_route(alone)):
        return room, alone
    return alone, alone


def mlx_offload_route(plan) -> dict:
    """The ``offload`` / ``prefill_chunk`` pricing kwargs of a ``(ple, experts, prefill_step)`` plan, both
    always present so they override a priced route."""
    return {"offload": tuple(plan[:2]) if any(plan[:2]) else None, "prefill_chunk": plan[2]}


def mlx_checkpoint_window(model_dir) -> Optional[int]:
    from types import SimpleNamespace

    from core.inference.mlx_memory import _snapshot_config

    try:
        config = _snapshot_config(model_dir)
    except Exception:
        return None
    if config is None:
        return None
    native = mlx_native_context_length(SimpleNamespace(config = config))
    return None if native is None else min(int(native), MAX_REQUESTABLE_CONTEXT)


def _takes_prefill_step(target) -> bool:
    fields = getattr(target, "__dataclass_fields__", None)
    if fields is not None:
        return "prefill_step_size" in fields
    try:
        params = inspect.signature(target).parameters
    except (TypeError, ValueError):
        return False
    return "prefill_step_size" in params or any(p.kind is p.VAR_KEYWORD for p in params.values())


def _mlx_offload_request(knobs, plan):
    """The offload a load asks for: the ``(ple, experts)`` knobs as sent, the fraction each wants,
    the ``FastMLXModel.from_pretrained`` kwargs, and why none is passed when unsloth_zoo cannot
    offload. unsloth_zoo hands an ``"auto"`` knob to *plan* once it has the checkpoint (the one it
    resolved and downloaded), and never does for a load it cannot offload; with no *plan* it is off."""
    from core.inference.mlx_memory import mlx_offload_planner

    requested = dict(zip(("ple", "experts"), knobs))
    wanted = {part: normalize_mlx_offload(value) for part, value in requested.items()}
    if plan is None:
        wanted = {part: None if value == "auto" else value for part, value in wanted.items()}
    offload = {
        "requested": requested,
        "wanted": wanted,
        "reason": None,
        "kwargs": {},
        "prefill_step": None,
    }
    if not any(wanted.values()):
        return offload
    if mlx_offload_planner() is None:
        offload["reason"] = "the installed unsloth_zoo cannot read weights from disk"
    elif "auto" in wanted.values():

        def decide(model_dir):
            *fractions, step = plan(model_dir, tuple(wanted.values()))
            # A repeat call sees fractions, not "auto", and plans no step: keep the first one.
            if not offload["decided"]:
                offload["prefill_step"] = step
            wanted.update(zip(wanted, fractions))
            offload["decided"] = True
            return dict(wanted)

        offload["decided"] = False
        offload["kwargs"]["offload"] = decide
    else:
        offload["kwargs"]["offload"] = dict(wanted)
    return offload


def _load_offloading(load, source, load_kwargs, offload):
    """``load(source, **load_kwargs)``, again without offload when unsloth_zoo refuses it for this load
    (an adapter, a bitsandbytes checkpoint, runtime quantization); ``offload`` keeps the reason."""
    try:
        return load(source, **load_kwargs)
    except ValueError as exc:
        # unsloth_zoo's offload refusals all open so.
        if not offload["kwargs"] or not str(exc).startswith("Unsloth: offload "):
            raise
        logger.info("MLX offload not applied to %s: %s", source, exc)
        offload["reason"] = str(exc)
    load_kwargs.pop("offload", None)
    offload["kwargs"] = {}
    return load(source, **load_kwargs)


def _mlx_offload_status(offload, model):
    """Per knob the fraction read from disk, the request and why it is not applied; and the bytes
    the loaded model reads from disk."""
    reports = getattr(model, "_unsloth_offload", None) or {}
    status = {"disk_bytes": sum(report["on_disk_bytes"] for report in reports.values()) or None}
    for part, what in _MLX_OFFLOAD_PARTS:
        applied = offload["wanted"][part] if part in reports else None
        reason = None
        if offload["wanted"][part] not in (None, "auto") and applied is None:
            reason = offload["reason"] or (
                "unsloth_zoo cannot offload this load (an adapter, a bitsandbytes checkpoint or "
                "runtime quantization)"
                if offload.get("decided") is False
                else f"the model has no {what} unsloth_zoo can read from disk"
            )
        status[part] = {
            "applied": applied,
            "requested": offload["requested"][part],
            "reason": reason,
        }
    return status


# Keyed by checkpoint fingerprint: the panel asks again on every model switch.
_int8_prefill_checkpoint_cache: dict = {}
_int8_prefill_checkpoint_lock = threading.Lock()


def mlx_int8_prefill_checkpoint_status(model_dir) -> tuple[bool, str]:
    """(available, reason) of int8 prefill for a downloaded MLX checkpoint, without loading it."""
    check = getattr(_int8_prefill_zoo(), "int8_prefill_checkpoint_available", None)
    if check is None:
        return False, "unsupported_zoo"
    try:
        from core.inference.mlx_memory import _checkpoint_fingerprint
        key = (os.path.realpath(model_dir), _checkpoint_fingerprint(model_dir))
    except Exception:
        key = None
    with _int8_prefill_checkpoint_lock:
        if key in _int8_prefill_checkpoint_cache:
            return _int8_prefill_checkpoint_cache[key]
    try:
        verdict = check(model_dir)
        answer = (bool(verdict.available), verdict.reason)
    except Exception as exc:
        logger.debug("MLX int8 prefill checkpoint check failed for %s: %s", model_dir, exc)
        return False, "probe_failed"
    if key is not None and answer[1] != "probe_failed":
        with _int8_prefill_checkpoint_lock:
            _int8_prefill_checkpoint_cache[key] = answer
            while len(_int8_prefill_checkpoint_cache) > 32:
                _int8_prefill_checkpoint_cache.pop(next(iter(_int8_prefill_checkpoint_cache)))
    return answer


_ane_prefill_checkpoint_cache: dict = {}
_ane_prefill_checkpoint_lock = threading.Lock()


def mlx_ane_prefill_checkpoint_status(model_dir) -> tuple[bool, str]:
    """(available, reason) of Neural Engine prefill for a downloaded MLX checkpoint, without loading it."""
    check = getattr(_ane_prefill_zoo(), "ane_prefill_checkpoint_available", None)
    if check is None:
        return False, "unsupported_zoo"
    try:
        from core.inference.mlx_memory import _checkpoint_fingerprint
        key = (os.path.realpath(model_dir), _checkpoint_fingerprint(model_dir))
    except Exception:
        key = None
    with _ane_prefill_checkpoint_lock:
        if key in _ane_prefill_checkpoint_cache:
            return _ane_prefill_checkpoint_cache[key]
    try:
        verdict = check(model_dir)
        answer = (bool(verdict.available), verdict.reason)
    except Exception as exc:
        logger.debug("MLX Neural Engine prefill checkpoint check failed for %s: %s", model_dir, exc)
        return False, "probe_failed"
    if key is not None and answer[1] != "probe_failed":
        with _ane_prefill_checkpoint_lock:
            _ane_prefill_checkpoint_cache[key] = answer
            while len(_ane_prefill_checkpoint_cache) > 32:
                _ane_prefill_checkpoint_cache.pop(next(iter(_ane_prefill_checkpoint_cache)))
    return answer


def _vlm_generation_context():
    import mlx.core as mx
    from mlx_vlm.generate import generation_stream
    return mx.stream(generation_stream)


def _iter_vlm_responses(responses):
    iterator = iter(responses)
    try:
        while True:
            # Token views outside VLM's inner step must use its generation stream too.
            with _vlm_generation_context():
                try:
                    response = next(iterator)
                except StopIteration:
                    return
            yield response
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            with _vlm_generation_context():
                close()


@contextmanager
def _temporary_mlx_adapter_state(model, use_adapter):
    """Select base or adapter modules for one request, then restore the tree."""
    if use_adapter is None:
        yield
        return
    if isinstance(use_adapter, str):
        raise NotImplementedError(
            "Unsloth MLX: named adapter selection is not supported; use True for "
            "the loaded adapter or False for the base model."
        )
    if use_adapter is not True and use_adapter is not False:
        raise TypeError("Unsloth MLX: use_adapter must be None, True, False, or a string.")

    adapters, unsupported = _mlx_adapter_modules(model)
    if use_adapter is True:
        if not adapters and not unsupported:
            logger.warning("MLX adapter requested, but the active model has no adapter layers")
        yield
        return
    if unsupported:
        raise RuntimeError(
            "Unsloth MLX: cannot disable adapter layers without their base modules: "
            + ", ".join(unsupported[:5])
        )
    if not adapters:
        yield
        return

    from mlx.utils import tree_unflatten

    base_modules = tree_unflatten([(path, base) for path, _, base in adapters])
    adapter_modules = tree_unflatten([(path, wrapper) for path, wrapper, _ in adapters])
    try:
        model.update_modules(base_modules)
        yield
    finally:
        model.update_modules(adapter_modules)


def _mlx_vlm_model_config(model):
    """Return the loaded MLX model config and its type, preferring whichever of config / _config
    actually carries a model_type."""

    def _model_type(cfg):
        return cfg.get("model_type") if isinstance(cfg, dict) else getattr(cfg, "model_type", None)

    configs = [
        cfg
        for cfg in (getattr(model, "config", None), getattr(model, "_config", None))
        if cfg is not None
    ]
    for cfg in configs:
        model_type = _model_type(cfg)
        if model_type is not None:
            return cfg, model_type
    return (configs[0] if configs else None), None


# Matched on the context term (mlx-lm spells it four ways); not max_length/n_sequences
_MLX_CONTEXT_KEY = re.compile(
    r"(?:position(?:s|_embeddings)|(?:^|_)ctx|context_len(?:gth)?"
    r"|seq(?:uence)?_len(?:gth)?|model_max_length)$"
)
# A vision tower's token count reads like a length but bounds one image, not the window.
_MLX_CONTEXT_KEY_EXCLUDE = re.compile(r"(?:^|_)image_")
# Below one cache block is not a window; above, configs carry sentinel "unlimited" lengths.
_MLX_MIN_PLAUSIBLE_CONTEXT = 256
_MLX_MAX_PLAUSIBLE_CONTEXT = 1 << 24


def _mlx_config_candidates(model):
    """Config-ish objects that may carry text-tower dims, most specific first."""
    language_model = getattr(model, "language_model", None)
    config, _ = _mlx_vlm_model_config(model)
    text_config = None
    if config is not None:
        text_config = (
            config.get("text_config")
            if isinstance(config, dict)
            else getattr(config, "text_config", None)
        )
    return [
        cfg
        for cfg in (
            getattr(language_model, "args", None),
            text_config,
            getattr(model, "args", None),
            config,
        )
        if cfg is not None
    ]


def _positive_int(value):
    if isinstance(value, bool):
        return None
    try:
        value_int = int(value)
    except (TypeError, ValueError, OverflowError):
        # json.loads accepts Infinity: unusable, but must not abort the load.
        return None
    return value_int if value_int > 0 else None


def mlx_native_context_length(model):
    """Takes the widest length across every config, since a stub text config can understate the window."""
    lengths = []
    for cfg in _mlx_config_candidates(model):
        try:
            items = cfg.items() if isinstance(cfg, dict) else vars(cfg).items()
        except TypeError:
            continue
        lengths += [
            length
            for name, value in items
            if _MLX_CONTEXT_KEY.search(str(name))
            and not _MLX_CONTEXT_KEY_EXCLUDE.search(str(name))
            and (length := _positive_int(value)) is not None
            and _MLX_MIN_PLAUSIBLE_CONTEXT <= length <= _MLX_MAX_PLAUSIBLE_CONTEXT
        ]
    return max(lengths) if lengths else None


def _ascii_registry_key(value):
    """An mlx-vlm registry key lowered within ASCII, else None. Not `casefold`: it folds non-ASCII
    onto ASCII, so "ſmolvlm" would reach `smolvlm`."""
    if not isinstance(value, str) or not value.isascii():
        return None
    return value.lower()


def _resumes_thought(messages, continue_final_message) -> bool:
    return bool(continue_final_message) and (
        trailing_assistant_resume_kind(messages) == "reasoning_content"
    )


def _think_prefix(prompt, special_tokens, messages, continue_final_message, **kwargs) -> str:
    """``detect_think_prefill``; refuses a resumed thought whose ``</think>`` decoding would strip."""
    from core.inference.chat_template_helpers import detect_think_prefill

    resumes = _resumes_thought(messages, continue_final_message)
    prefix = detect_think_prefill(prompt, special_tokens, resumes_thought = resumes, **kwargs)
    if resumes and not prefix:
        raise ThoughtUnresumableError()
    return prefix


def _render_registered_vlm_prompt(
    processor,
    model,
    messages,
    num_images,
    num_audios = 0,
    continue_final_message = False,
    tools = None,
):
    """Resumes a partial from the swept messages, since a raw partial could close the turn."""
    from mlx_vlm import prompt_utils

    config, model_type = _mlx_vlm_model_config(model)
    if config is None:
        return None
    model_config = getattr(prompt_utils, "MODEL_CONFIG", {})
    if model_type not in model_config:
        # ASCII fold: casefold would route 'ſmolvlm' to `smolvlm`
        folded = _ascii_registry_key(model_type)
        matches = (
            [
                key
                for key in model_config
                if isinstance(key, str) and _ascii_registry_key(key) == folded
            ]
            if folded is not None
            else []
        )
        # An ambiguous fold is not evidence for either key: stay fail-closed.
        if len(matches) != 1:
            return None
        canonical = matches[0]
        # Copy: mutating the loaded config would leak into capability and export logic
        config = dict(config) if isinstance(config, dict) else dict(config.__dict__)
        config["model_type"] = canonical

    # mlx-vlm's formatters carry no reasoning opener to reopen a thought on.
    if _resumes_thought(messages, continue_final_message):
        return None
    # Recovery path: sweeps the caller's original list rather than reusing a copy (#7066).
    swept, tools, _ = neutralize_for_render(processor, messages, tools)
    partial = trailing_assistant_text(swept) if continue_final_message else None
    rendered = prompt_utils.apply_chat_template(
        processor,
        config,
        swept[:-1] if partial else swept,
        add_generation_prompt = True,
        num_images = num_images,
        num_audios = num_audios,
        **({"tools": tools} if tools else {}),
    )
    if isinstance(rendered, str) and rendered.strip():
        # A prefilled open "<think>" would resume the answer inside the reasoning block.
        return f"{strip_open_reasoning_prefill(rendered)}{partial}" if partial else rendered
    raise RuntimeError("mlx-vlm's registered renderer returned an empty prompt.")


# Rate the chat route decodes uploads to; mlx-vlm does not resample arrays.
_AUDIO_INPUT_SAMPLE_RATE = 16000
_AUDIO_PROBE_MESSAGES = [{"role": "user", "content": "audio"}]
# Same turn with and without an image part, so a diff isolates the image marker.
_IMAGE_PROBE_MESSAGES = [
    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "hi"}]}
]
# A pair differing only in image count, so a diff isolates the marker; the question is distinctive.
_MULTI_IMAGE_PROBE_TEXT = "what is in these"
_MULTI_IMAGE_PROBE_MESSAGES = [
    [
        {
            "role": "user",
            "content": [{"type": "image"}] * count
            + [{"type": "text", "text": _MULTI_IMAGE_PROBE_TEXT}],
        }
    ]
    for count in (1, 2)
]
_TEXT_PROBE_MESSAGES = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
_VIDEO_PROBE_MESSAGES = [
    {"role": "user", "content": [{"type": "video"}, {"type": "text", "text": "hi"}]}
]


def _mlx_vlm_decodes_video() -> bool:
    """Both halves of the decoder, ``load_video`` (0.5.0+) and cv2, so a clip missing either is
    refused by name rather than by a ModuleNotFoundError mid-stream."""
    try:
        from mlx_vlm import utils as vlm_utils
    except ImportError:
        return False
    if not callable(getattr(vlm_utils, "load_video", None)):
        return False
    try:
        import cv2  # noqa: F401
    except ImportError:
        return False
    return True


def _video_placeholder(processor):
    """The model's own video placeholder, when it names one."""
    for source in (processor, getattr(processor, "tokenizer", None)):
        token = getattr(source, "video_token", None)
        if isinstance(token, str) and token:
            return token
    return None


def _marks_video(target, processor) -> bool:
    """Whether ``target`` renders a video part as a marker: a render that merely differs from the
    text one can still be prose, as in the image probe."""
    from core.inference.chat_template_helpers import apply_chat_template_for_generation

    marked = apply_chat_template_for_generation(target, _VIDEO_PROBE_MESSAGES)
    if not marked or marked == apply_chat_template_for_generation(target, _TEXT_PROBE_MESSAGES):
        return False
    if _vlm_prompt_issue(marked, _VIDEO_PROBE_MESSAGES):
        return False
    placeholder = _video_placeholder(processor)
    return placeholder is None or placeholder in marked


def _video_render_target(processor):
    """What renders a turn with video. mlx-vlm's Gemma 4 processor rebuilds each message from media
    counts and drops a video part it holds no clip for; its tokenizer carries the same template and
    renders the part as written. A processor that keeps the marker keeps rendering, since some
    deliberately differ from their tokenizer (MiniCPM-V's non-thinking prefix)."""
    from core.inference.chat_template_helpers import chat_render_target

    target = chat_render_target(processor)
    tokenizer = getattr(processor, "tokenizer", None)
    template = getattr(processor, "chat_template", None)
    if (
        target is not processor
        or not template
        or getattr(tokenizer, "chat_template", None) != template
    ):
        return target
    try:
        drops = not _marks_video(processor, processor)
    except Exception:
        drops = True
    return tokenizer if drops else processor


def _mlx_lm_reads_images(model) -> bool:
    """An image model unsloth-zoo loaded through mlx-lm, with an unsloth-zoo that prefills its images."""
    if getattr(model, "_unsloth_supports_images", None) is not True:
        return False
    try:
        from unsloth_zoo.mlx.generate import prefill_image_prompt  # noqa: F401
    except ImportError:
        return False
    return True


def _mlx_reads_video(processor) -> bool:
    """Processor carries a video component AND the template places a video marker."""
    if processor is None or getattr(processor, "video_processor", None) is None:
        return False
    if not _mlx_vlm_decodes_video():
        return False
    try:
        return _marks_video(_video_render_target(processor), processor)
    except BaseException:
        # A load must never fail on a probe, matching _image_marker_survives.
        return False


def _write_video_clip(video_b64: str) -> str:
    import base64
    import tempfile

    from utils.media_playlist import refuse_playlist

    data = base64.b64decode(video_b64)
    # cv2.VideoCapture and mlx-vlm both open this through FFmpeg, which follows playlist entries.
    refuse_playlist(data)
    with tempfile.NamedTemporaryFile(prefix = "unsloth-video-", delete = False) as handle:
        handle.write(data)
        return handle.name


# mlx-vlm holds two native-resolution RGB copies of the sampled frames before the processor resizes.
_VIDEO_DECODE_BUDGET_BYTES = 2 << 30


# load_video's own defaults, unchanged 0.5.0 to 0.7.0, for a release that settles a knob nowhere.
_VIDEO_SAMPLING_FALLBACK = {"fps": 2.0, "min_frames": 4}


def _sampled(name, value):
    return _VIDEO_SAMPLING_FALLBACK[name] if value is None else value


def _declared_video_sampling(processor) -> dict:
    """Read the way mlx-vlm's ``processor_video_sampling`` reads it, so both branches agree."""
    component = getattr(processor, "video_processor", None)
    if component is None:
        return {}
    for owner in (component, processor):
        hook = getattr(owner, "video_sampling_defaults", None)
        if callable(hook):
            declared = hook()
            return dict(declared) if isinstance(declared, dict) else vars(declared)
    return {name: getattr(component, name, None) for name in ("fps", "min_frames", "max_frames")}


def _video_sampling_target(processor) -> tuple[float, int, Optional[int]]:
    """mlx-vlm's (fps, min_frames, nframes); older releases pass only ``fps`` to ``load_video``."""
    import inspect

    from mlx_vlm import utils as vlm_utils

    resolve = getattr(vlm_utils, "resolve_video_sampling", None)
    if resolve is not None:
        sampling = resolve(processor, {})
        # Unset is unreachable today; the fallback keeps the rate divisible for a future release.
        return (
            _sampled("fps", sampling.fps),
            _sampled("min_frames", sampling.min_frames),
            sampling.nframes,
        )
    # Below 0.7.0 prepare_inputs forwards only fps, so nothing else applies the model's rate
    defaults = inspect.signature(vlm_utils.load_video).parameters

    def _default(name):
        parameter = defaults.get(name)
        if parameter is None or parameter.default is inspect.Parameter.empty:
            return None
        return parameter.default

    declared_fps = _declared_video_sampling(processor).get("fps")
    return (
        _sampled("fps", _default("fps") if declared_fps is None else declared_fps),
        _sampled("min_frames", _default("min_frames")),
        None,
    )


class _ClipsTooLarge(RuntimeError):
    pass


def _video_frame_rate(
    clip_path: str,
    processor,
    clips: int = 1,
) -> Optional[float]:
    """The sampling ``fps`` keeping the decoded stack within this clip's share of the budget; None
    when the clip has no readable frame geometry or count, which mlx-vlm then refuses itself."""
    import cv2

    capture = cv2.VideoCapture(clip_path)
    try:
        if not capture.isOpened():
            return None
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        native_fps = capture.get(cv2.CAP_PROP_FPS)
    finally:
        capture.release()
    if min(width, height, total_frames) <= 0:
        return None
    if not native_fps or native_fps <= 0:
        # The one measurement mlx-vlm substitutes rather than refusing; match its substitution.
        native_fps = 1.0
    wanted_fps, min_frames, nframes = _video_sampling_target(processor)
    # mlx-vlm holds every decoded clip until the processor runs, so the clips share one budget.
    budget = _VIDEO_DECODE_BUDGET_BYTES // clips
    affordable = budget // (2 * 3 * width * height)
    needed = min_frames if nframes is None else nframes
    if affordable < needed:
        raise _ClipsTooLarge(
            f"The video's {width}x{height} frames are too large to decode: {needed} of them "
            f"exceed the {budget >> 20} MiB frame budget."
        )
    return min(wanted_fps, affordable * native_fps / total_frames)


def _video_sampling_fps(
    clip_paths,
    processor,
    rows = 1,
) -> Optional[float]:
    """One rate for every clip, since mlx-vlm before 0.7.3 applies a single ``fps`` to all.

    ``rows`` copies of the clips share the decode budget when a batch decodes them together.
    """
    rates = [_video_frame_rate(path, processor, len(clip_paths) * rows) for path in clip_paths]
    return min((rate for rate in rates if rate is not None), default = None)


def _discard_video_clip(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


_SPEECH_REPLY_SYSTEM_PROMPT = "Respond with interleaved text and audio."


def _mlx_lm_serves(model_type) -> bool:
    from importlib.util import find_spec

    from mlx_lm.utils import MODEL_REMAPPING
    return find_spec(f"mlx_lm.models.{MODEL_REMAPPING.get(model_type, model_type)}") is not None


def _zoo_declares_multimodal(model_name, hf_token = None) -> bool:
    """unsloth_zoo's verdict from mlx-vlm's own configs, for mlx-vlm-only families the transformers
    registry does not know (an audio-only speech model, a tower under another name)."""
    try:
        from utils.models.model_config import _read_raw_config_json

        config = _read_raw_config_json(model_name, hf_token)
        model_type = (config or {}).get("model_type")
        # A type mlx-lm serves keeps its text path, and skips zoo's scan of every mlx-vlm family.
        if not isinstance(model_type, str) or not model_type or _mlx_lm_serves(model_type):
            return False
        from unsloth_zoo.mlx.loader import _is_vlm

        return _is_vlm(config)
    except Exception as exc:  # noqa: BLE001 -- detection must never fail a load
        logger.debug("unsloth_zoo multimodal detection skipped for %s (%s)", model_name, exc)
        return False


def _classify_mlx_audio_type(
    model,
    processor,
    is_vision,
    config_audio_type = None,
):
    """The probe may only retract audio_vlm once it has run, else the pre-load audio_type stands."""

    def _probe_says_no():
        # Authoritative for audio_vlm only; anything else passes through.
        return None if config_audio_type == "audio_vlm" else config_audio_type

    if not is_vision or processor is None:
        return config_audio_type
    # A model load must never fail on a probe; BaseException since any escape aborts the load
    try:
        from unsloth_zoo.mlx.utils import (
            audio_extractor_sampling_rate,
            audio_input_capability,
        )

        if audio_extractor_sampling_rate(processor) != _AUDIO_INPUT_SAMPLE_RATE:
            logger.info(
                "MLX audio input unavailable: feature extractor is not %d Hz, and "
                "the chat route decodes uploads to that rate.",
                _AUDIO_INPUT_SAMPLE_RATE,
            )
            return _probe_says_no()
        args = (processor, model, _AUDIO_PROBE_MESSAGES, 0)
        marked = _render_registered_vlm_prompt(*args, num_audios = 1)
        if not marked or marked == _render_registered_vlm_prompt(*args, num_audios = 0):
            logger.info(
                "MLX audio input unavailable: mlx-vlm's renderer for this family "
                "places no audio marker."
            )
            return _probe_says_no()
        capability = audio_input_capability(model, processor, texts = marked)
        if capability.capable:
            return "audio_vlm"
        logger.info("MLX audio input unavailable for this model: %s", capability.reason)
        return _probe_says_no()
    except BaseException as exc:
        logger.info(
            "MLX audio capability check did not run (%s); keeping the pre-load "
            "classification %r. Audio input needs an unsloth_zoo release providing "
            "audio_input_capability.",
            type(exc).__name__,
            config_audio_type,
        )
    return config_audio_type


def _mlx_config_field(model, name):
    """Reads a config field from a dict or an object, since a getattr-only read misses dict configs."""
    for cfg in (getattr(model, "config", None), getattr(model, "_config", None)):
        value = cfg.get(name) if isinstance(cfg, dict) else getattr(cfg, name, None)
        if value is not None:
            return value
    return None


def _mlx_stop_token_ids(tokenizer, model = None):
    """Prefers mlx_vlm's stopping criteria: tokenizer and config stop ids disagree on some repos."""
    for source in (
        getattr(getattr(tokenizer, "stopping_criteria", None), "eos_token_ids", None),
        _mlx_config_field(model, "eos_token_id"),
        getattr(tokenizer, "eos_token_ids", None),
        getattr(tokenizer, "eos_token_id", None),
    ):
        if source is None:
            continue
        if isinstance(source, (list, tuple, set, frozenset)):
            # Empty/unset falls through (not an answer); a bare id is kept since 0 is a valid token
            if not source:
                continue
            return tuple(source)
        # int() so a numpy scalar cannot raise here, inside a caller's finally.
        return (int(source),)
    return ()


def _mlx_stop_sequences(stop):
    """The sequences a request asked to stop on, as a list. An empty one is dropped rather than
    matched: it is found at position 0 of every reply and would end each turn before its first
    token."""
    return [x for x in ([stop] if isinstance(stop, str) else stop or []) if x]


def _rebuilt_detokenizer(detokenizer):
    """Rebuilds the naive detokenizer from its tokenizer, since copying it raises on some versions."""
    tokenizer = getattr(detokenizer, "_tokenizer", None)
    if tokenizer is None:
        return None
    try:
        return type(detokenizer)(tokenizer)
    except Exception as exc:  # noqa: BLE001 -- third-party detokenizers vary; caller falls back
        logger.debug("MLX streaming detokenizer could not be rebuilt (%s)", exc)
        return None


def _mlx_stream_detokenizer(source):
    """Takes a fresh read of the detokenizer where possible; copying raises on the naive kind."""
    detokenizer = getattr(source, "detokenizer", None)
    if detokenizer is None:
        return None
    independent = getattr(source, "detokenizer", None)
    if independent is None or independent is detokenizer:
        try:
            independent = copy.copy(detokenizer)
        except Exception as exc:  # noqa: BLE001 -- third-party detokenizers vary; caller falls back
            # mlx-vlm < 0.6.0 cannot copy the naive detokenizer; rebuild it as __copy__ would
            independent = _rebuilt_detokenizer(detokenizer)
            if independent is None:
                logger.debug("MLX streaming detokenizer unavailable (%s)", exc)
                return None
    try:
        independent.reset()
    except Exception as exc:  # noqa: BLE001 -- as above
        logger.debug("MLX streaming detokenizer unavailable (%s)", exc)
        return None
    return independent


class _MlxStreamedText:
    """A reply rebuilt from the ids the runtime sampled, for turns carrying native controls.

    A streaming detokenizer holds a piece until the ones after it settle, so a control reaches
    ``response.text`` inside a later segment, with nothing left to say which characters came from
    it: deciding suppression there deletes prose that merely spells a control, and re-decoding the
    id on its own step writes a second copy ahead of the text still held.

    ``generation_tokens``, not a finish reason, says which ids the reply owes -- the supported
    mlx-vlm floor has no such field, and a turn broken out of on a stop token or the token limit
    reports a position no yield covered. Hence also ``finish``, run by the caller once the runtime
    stops yielding.

    Stop ids are held rather than written, since both runtimes stop before their detokenizer sees
    one, and released only to close a tool envelope. The run is held whole: the ids read as ending
    a turn come from the model config, which lists more than the one the runtime stopped on.

    Neither a denoised block, which names one id for many, nor a missing detokenizer leaves ids to
    rebuild from; the runtime's own text stands and the allowlist cannot speak for it."""

    def __init__(self, detokenizer, decoder, tokenizer, stop_ids):
        self._detokenizer = detokenizer
        self._decoder = decoder
        self._tokenizer = tokenizer
        self._stop_ids = stop_ids
        self._held_stops = []
        self._filled = 0

    def feed(self, response) -> str:
        if self._detokenizer is None:
            # Every id but the stop id is already in the text, so only the last can still be a closer owed.
            stop_id = getattr(response, "token", None)
            self._held_stops = (
                [int(stop_id)] if stop_id is not None and int(stop_id) in self._stop_ids else []
            )
            return getattr(response, "text", None) or ""
        filled = getattr(response, "generation_tokens", None)
        if filled is not None:
            filled = int(filled)
            if filled <= self._filled:
                return ""
            several = filled > self._filled + 1
            self._filled = filled
            if several:
                return self._drain() + (getattr(response, "text", None) or "")
        token_id = getattr(response, "token", None)
        if token_id is None:
            return ""
        return self.feed_token(int(token_id))

    def feed_token(self, token_id: int) -> str:
        """One sampled id, for a runtime that reports ids one at a time."""
        if token_id in self._stop_ids:
            self._held_stops.append(token_id)
            return ""
        if self._decoder.suppresses(token_id):
            return ""
        if self._held_stops:
            # Text followed them, so they ended nothing.
            held, self._held_stops = self._held_stops, []
            for stop_id in held:
                if not self._decoder.suppresses(stop_id):
                    self._detokenizer.add_token(stop_id)
        self._detokenizer.add_token(token_id)
        return self._detokenizer.last_segment

    def finish(self, sampled: str) -> str:
        """The text still held once the runtime stops yielding."""
        if self._detokenizer is None:
            return self._closing_stop(sampled)
        segment = self._drain()
        return segment + self._closing_stop(sampled + segment)

    def _drain(self) -> str:
        self._detokenizer.finalize()
        return self._detokenizer.last_segment

    def _closing_stop(self, settled: str) -> str:
        held, self._held_stops = self._held_stops, []
        closing = ""
        for stop_id in held:
            if not self._decoder.keeps(stop_id):
                continue
            text = stop_token_text(self._tokenizer, stop_id) or ""
            if settled.endswith(text):
                continue
            # Nested markup ends on a closer each, so the run can owe more than one.
            if closes_an_open_envelope(settled + text, text):
                closing += text
        return closing


def _require_row_detokenizer(streamed):
    """A batch row feeds ids one at a time, so unlike the single path it has no runtime text to fall back on."""
    if streamed is not None and streamed._detokenizer is None:
        raise RowRefused("no independent detokenizer to decode this row's ids")


def _native_reply_text(decoder, token_ids, stop_ids, prefix) -> str:
    """A native-protocol reply re-decoded from its ids. A runtime may stop on an allowlisted
    control (<|end_message|>): the trailing stop id stays only when it closes an open envelope,
    judged with the prefill, since a restored <think> lives in the prompt."""
    ids = list(token_ids)
    if ids and ids[-1] in stop_ids:
        whole = decoder.decode(ids)
        closer = whole[len(decoder.decode(ids[:-1])) :]
        if not closes_an_open_envelope(prefix + whole, closer):
            ids = ids[:-1]
    return decoder.decode(ids)


def _mlx_stop_cut(text: str, stops) -> tuple[int, bool]:
    """Holds back text that could grow into a stop; a rewriting caller must withhold until turn ends"""
    resolved = text.rstrip("\ufffd")
    cut = len(resolved)
    for sequence in stops:
        found = resolved.find(sequence)
        if found != -1:
            cut = min(cut, found)
    if cut < len(resolved):
        return cut, True
    held = 0
    for sequence in stops:
        for size in range(min(len(sequence) - 1, len(resolved)), held, -1):
            if resolved.endswith(sequence[:size]):
                held = size
                break
    return len(resolved) - held, False


def _mlx_finish_reason(response, stop_ids, generated_n, max_tokens):
    """At the limit, a stop token can look like exhaustion, so the last token's identity decides."""
    reason = getattr(response, "finish_reason", None)
    if reason in ("stop", "length"):
        return reason
    if generated_n < max_tokens:
        return "stop"
    token = getattr(response, "token", None)
    return "stop" if token is not None and token in tuple(stop_ids) else "length"


def _build_generation_stats(
    prompt_n,
    prompt_tps,
    gen_n,
    gen_tps,
    cached_n = 0,
    finish_reason = None,
    drafts = None,
):
    """Map mlx stream stats onto the usage/timings shape llama-server emits, plus the reason
    generation ended. ``drafts`` is a speculative reply's (drafted, accepted) token counts."""
    prompt_n = int(prompt_n or 0)
    gen_n = int(gen_n or 0)
    cached_n = int(cached_n or 0)
    prompt_tps = float(prompt_tps or 0.0)
    gen_tps = float(gen_tps or 0.0)
    prompt_ms = (prompt_n / prompt_tps * 1000.0) if prompt_tps > 0 else 0.0
    predicted_ms = (gen_n / gen_tps * 1000.0) if gen_tps > 0 else 0.0
    total_prompt_n = prompt_n + cached_n
    return {
        "usage": {
            "prompt_tokens": total_prompt_n,
            "completion_tokens": gen_n,
            "total_tokens": total_prompt_n + gen_n,
            # Report cached tokens in the OpenAI field as llama-server does, not only timings.cache_n
            "prompt_tokens_details": {"cached_tokens": cached_n},
        },
        "timings": {
            "prompt_n": prompt_n,
            "prompt_ms": prompt_ms,
            "prompt_per_token_ms": (prompt_ms / prompt_n) if prompt_n > 0 else 0.0,
            "prompt_per_second": prompt_tps,
            "predicted_n": gen_n,
            "predicted_ms": predicted_ms,
            "predicted_per_token_ms": (predicted_ms / gen_n) if gen_n > 0 else 0.0,
            "predicted_per_second": gen_tps,
            "cache_n": cached_n,
            **({} if drafts is None else {"draft_n": drafts[0], "draft_n_accepted": drafts[1]}),
        },
        # Latched at generation exit so a later cancel cannot rewrite the finish reason
        "finish_reason": finish_reason,
    }


PROMPT_CACHE_ENTRIES = 6

# Every bit width mx.quantize supports; unrelated to llama.cpp's cache_type_kv names.
MLX_KV_BITS_CHOICES = (8, 6, 5, 4, 3, 2)
MLX_TURBOQUANT_BITS_CHOICES = (4, 3.5, 3, 2)
# Quantization group size; a head dim that is not a multiple makes mx.quantize raise.
MLX_KV_GROUP_SIZE = 64
MLX_PREFILL_CHUNK_FALLBACK = 2048


def _generation_step(*, vision: bool, drafted: bool):
    """The function that would run this generation's prefill."""
    if vision:
        from mlx_vlm.generate.ar import generate_step as vlm_generate_step
        return vlm_generate_step
    if drafted:
        from mlx_lm.generate import speculative_generate_step
        return speculative_generate_step
    from mlx_lm.generate import generate_step

    return generate_step


def _generation_default(setting: str, fallback: int, *, vision: bool, drafted: bool) -> int:
    """A prefill setting as the runtime that would run generation defaults to it."""
    import inspect
    try:
        found = (
            inspect.signature(_generation_step(vision = vision, drafted = drafted))
            .parameters[setting]
            .default
        )
        if isinstance(found, bool) or not isinstance(found, int) or found <= 0:
            raise ValueError(f"the runtime states no usable {setting}")
        return found
    except Exception as exc:
        logger.debug("Cannot read the runtime's %s: %s", setting, exc)
        return fallback


def mlx_vlm_snapshot_store_available() -> bool:
    """Whether a vision load can build the snapshot store, which decides two things about it."""
    if _prompt_cache_max_bytes() <= 0:
        return False
    try:
        from mlx_vlm.generate import GenerationResult
    except Exception as exc:
        logger.debug("Cannot tell whether mlx-vlm carries the snapshot grid: %s", exc)
        return False
    return hasattr(GenerationResult, "cached_tokens")


def mlx_vlm_pins_the_prefill_step(config) -> bool:
    """Whether EVERY request to this vision load prefills on the pinned grid."""
    if not mlx_vlm_snapshot_store_available():
        return False
    return bool(MLXInferenceBackend._vlm_media_token_ids(config))


def mlx_prefill_chunk(
    *,
    vision: bool = False,
    drafted: bool = False,
    config = None,
) -> int:
    """Most tokens one prefill step takes; the last step of a prompt may be shorter."""
    if vision and mlx_vlm_pins_the_prefill_step(config):
        return vlm_prefill_step()
    return _generation_default(
        "prefill_step_size", MLX_PREFILL_CHUNK_FALLBACK, vision = vision, drafted = drafted
    )


# Surfaced with the resolved setting so an API client sees the reuse cost too.
MLX_KV_QUANT_NO_REUSE = (
    "The installed mlx-lm cannot measure a quantized cache entry, so prompt-cache "
    "reuse across turns is disabled while this is on."
)
# The cap _configure_memory_limits installs, so a fitted context names a load MLX would allocate.
MLX_MEMORY_LIMIT_FRACTION = 0.85


MLX_TURBOQUANT_TEXT_LOAD = (
    "TurboQuant is not available for this model: it could not be loaded through the runtime "
    "TurboQuant requires. The model is served without it."
)


def _turboquant_refusal(*, is_vision, is_distributed, is_lora):
    """Why a text model cannot take the mlx-vlm route TurboQuant lives on, else ""."""
    if is_vision:
        return ""
    if is_distributed:
        return "TurboQuant is not available under distributed inference. Run on one device, or turn it off."
    if is_lora:
        return "TurboQuant is not available for a LoRA adapter on a text model. Merge it first, or turn TurboQuant off."
    return ""


MLX_KV_QUANT_AUTO = "auto"
# TurboQuant's widths repeat mx.quantize's, so the scheme rides in the value: "4" alone is ambiguous.
MLX_TURBOQUANT_PREFIX = "tq-"


def _kv_quant_choice(bits):
    return str(int(bits)) if float(bits).is_integer() else str(float(bits))


MLX_KV_QUANT_CHOICES = (
    (MLX_KV_QUANT_AUTO,)
    + tuple(_kv_quant_choice(bits) for bits in MLX_KV_BITS_CHOICES)
    + tuple(MLX_TURBOQUANT_PREFIX + _kv_quant_choice(bits) for bits in MLX_TURBOQUANT_BITS_CHOICES)
)


def parse_mlx_kv_quant(value):
    """The (bit width, TurboQuant) pair for a choice. Total: an unknown one reads as auto."""

    if value is None:
        return None, False
    text = str(value).strip().lower()
    if text in ("", MLX_KV_QUANT_AUTO):
        return None, False
    if text not in MLX_KV_QUANT_CHOICES:
        logger.warning(
            "MLX kv quantization %r unsupported (choose %s); ignoring",
            value,
            " or ".join(MLX_KV_QUANT_CHOICES),
        )
        return None, False
    turboquant = text.startswith(MLX_TURBOQUANT_PREFIX)
    width = float(text.removeprefix(MLX_TURBOQUANT_PREFIX))
    return (int(width) if width.is_integer() else width), turboquant


def encode_mlx_kv_quant(bits, turboquant = False):
    if bits is None:
        return MLX_KV_QUANT_AUTO
    return (MLX_TURBOQUANT_PREFIX if turboquant else "") + _kv_quant_choice(bits)


def _kv_entry_nbytes(entry):
    """Reads the nbytes property directly, since upstream's cache admission sums that same property."""
    try:
        return int(entry.nbytes)
    except Exception:
        return None


def _normalize_mlx_kv_bits(value):
    """Supported bit width, or None when unset or out of domain."""
    if value is None:
        return None
    try:
        bits = int(value)
    except (TypeError, ValueError):
        logger.warning("MLX kv_bits=%r is not an integer; ignoring", value)
        return None
    if bits not in MLX_KV_BITS_CHOICES:
        logger.warning(
            "MLX kv_bits=%s unsupported (choose %s); ignoring",
            bits,
            " or ".join(str(b) for b in MLX_KV_BITS_CHOICES),
        )
        return None
    return bits


def _mlx_rng_key_words():
    """A key that is not two words is logged before declining, as the rewind cannot work on it."""
    try:
        import mlx.core as mx
        words = mx.random.state[0].tolist()
    except Exception:
        return None
    if len(words) != 2:
        logger.warning(
            "MLX exposes a %d-word random key; Unsloth can only rewind the "
            "two-word form, so sizing a load will not restore the PRNG and "
            "sampling after a load may differ from an unsized run.",
            len(words),
        )
        return None
    return _as_uint32_pair(int(words[0]), int(words[1]))


def _as_uint32_pair(high, low):
    """Range-checked, not masked: masking an out-of-range word would silently restore a wrong key."""
    converted = []
    for word in (high, low):
        if not -(2**31) <= word < 2**32:
            logger.warning(
                "MLX exposed a random key word of %d, which is not a 32-bit "
                "word; sizing a load will not restore the PRNG and sampling "
                "after a load may differ from an unsized run.",
                word,
            )
            return None
        converted.append(word & 0xFFFFFFFF)
    return (converted[0], converted[1])


def _restore_mlx_rng_key(words):
    """Unguarded on purpose: the range check in _as_uint32_pair removes the only way it can raise."""
    if words is None:
        return
    import mlx.core as mx

    pair = _as_uint32_pair(int(words[0]), int(words[1]))
    if pair is None:
        return
    mx.random.seed((pair[0] << 32) | pair[1])


@contextmanager
def mlx_rng_preserved():
    """Leave the PRNG where it was found around weightless builds."""
    rng_key = _mlx_rng_key_words()
    try:
        yield
    finally:
        _restore_mlx_rng_key(rng_key)


def _kv_entry_windowed(entry):
    return any(getattr(entry, name, None) is not None for name in ("max_size", "window_size"))


def _allocate_empty_quantized_exactly(cls):
    """First block at mx.quantize's ``dim * bits // 32``; upstream's ``dim // (32 // bits)`` is a word too wide at 3/5/6 bits."""
    original = cls.update_and_fetch
    if getattr(original, "_unsloth_exact_empty_alloc", False):
        return

    @functools.wraps(original)
    def update_and_fetch(self, keys, values):
        if self.keys is None:
            import mlx.core as mx

            batch, heads, steps, _ = keys.shape
            shape = (batch, heads, (self.step + steps - 1) // self.step * self.step)

            def block(dim):
                return (
                    mx.zeros((*shape, dim * self.bits // 32), dtype = mx.uint32),
                    mx.zeros((*shape, dim // self.group_size), dtype = keys.dtype),
                    mx.zeros((*shape, dim // self.group_size), dtype = keys.dtype),
                )

            self.keys, self.values = block(keys.shape[-1]), block(values.shape[-1])
        return original(self, keys, values)

    update_and_fetch._unsloth_exact_empty_alloc = True
    cls.update_and_fetch = update_and_fetch


def _kv_quant_targets(entries):
    """``(indices, held)``: windowed entries keep their ring; past two entries the last
    full-attention one stays float, as mlx-vlm's should_quantize_kv_layer does."""
    targets = [
        index
        for index, entry in enumerate(entries)
        if getattr(entry, "to_quantized", None) is not None and not _kv_entry_windowed(entry)
    ]
    if len(entries) > 2 and targets:
        targets.pop()
        return targets, 1
    return targets, 0


def _quantize_kv_entries(entries, bits):
    for index in _kv_quant_targets(entries)[0]:
        entry = entries[index]
        entries[index] = entry.to_quantized(group_size = MLX_KV_GROUP_SIZE, bits = bits)
        if getattr(entry, "keys", True) is None:
            _allocate_empty_quantized_exactly(type(entries[index]))
    return entries


def _kv_quant_probe(
    language_model,
    entries,
    bits,
    evaluate = True,
):
    """``(converted, skipped, failure, retainable)``: runs a token through the converted cache,
    since static proxies (declared head_dim, window spelling) proved wrong both ways. Unevaluated,
    the same graph is only built, which is what a weightless tower affords."""
    import mlx.core as mx

    settle = mx.eval if evaluate else (lambda _arrays: None)

    targets, held = _kv_quant_targets(entries)
    skipped = len(entries) - len(targets) - held
    if not targets:
        # Verdict already known, so skip the cost of a full model call.
        return 0, skipped, None, True

    # The forward passes below draw random numbers, so keep sampled output stable.
    rng_key = _mlx_rng_key_words()
    try:
        try:
            language_model(mx.array([[0]]), cache = entries)
            settle([getattr(entry, "state", None) for entry in entries])
        except Exception as exc:
            if is_metal_queue_dead(exc):
                raise
            return 0, 0, f"its cache could not be exercised ({type(exc).__name__})", True
        try:
            _quantize_kv_entries(entries, bits)
            settle([entries[index].state for index in targets])
        except Exception as exc:
            if is_metal_queue_dead(exc):
                raise
            return 0, 0, f"MLX cannot quantize it ({type(exc).__name__})", True
        try:
            language_model(mx.array([[0]]), cache = entries)
            settle([getattr(entry, "state", None) for entry in entries])
        except Exception as exc:
            if is_metal_queue_dead(exc):
                raise
            return 0, 0, f"it cannot attend over a quantized cache ({type(exc).__name__})", True
        # Same helper insertion uses, so the caveat matches what insertion sees.
        retainable = all(_kv_entry_nbytes(entries[index]) is not None for index in targets)
        return len(targets), skipped, None, retainable
    finally:
        _restore_mlx_rng_key(rng_key)


# Bare mx.synchronize() waits on the default stream, which generation does not use
_GENERATION_STREAM_MODULES = (
    "mlx_lm.generate",
    "mlx_vlm.generate",
    "mlx_vlm.generate.dispatch",
    "mlx_vlm.generate.ar",
    "mlx_vlm.speculative.common",
)


def _drain_generation_streams(mx):
    """Best effort: a stream raises when synchronized off its creating thread, so the drain is skipped."""

    synchronize = getattr(mx, "synchronize", None)
    if not callable(synchronize):
        return
    drained = []
    for name in _GENERATION_STREAM_MODULES:
        try:
            stream = getattr(sys.modules.get(name), "generation_stream", None)
            # 0.6.x aliases one object across every mlx_vlm.generate name.
            if stream is None or any(stream is seen for seen in drained):
                continue
            drained.append(stream)
            synchronize(stream)
        except Exception as error:
            logger.debug("MLX stream drain skipped for %s: %s", name, error)
    try:
        synchronize()
    except Exception as error:
        logger.debug("MLX default stream drain skipped: %s", error)


def _kv_quant_eligibility(
    model,
    is_vlm,
    bits = MLX_KV_BITS_CHOICES[0],
):
    """``(verdict, reason, retainable)`` for this model, verdict in full/partial/none/refused."""
    try:
        entries = _make_kv_entries(model, is_vlm)
    except Exception as exc:
        logger.warning("MLX KV quantization eligibility probe failed: %s", exc)
        return "none", "this model's KV cache layout could not be inspected", True

    if not entries:
        return "none", "this model builds no KV cache to quantize", True

    total = len(entries)
    windowed = sum(_kv_entry_windowed(entry) for entry in entries)
    language_model = getattr(model, "language_model", model) if is_vlm else model
    converted, skipped, failure, retainable = _kv_quant_probe(language_model, entries, bits)
    # Released only here, once the probe's own locals are gone, or the pages stay in the allocator.
    import mlx.core as mx

    entries.clear()
    del entries
    _drain_generation_streams(mx)
    mx.clear_cache()
    if failure is not None:
        return "refused", f"this model's KV cache cannot be quantized: {failure}", True
    if not converted:
        if skipped < total:
            reason = (
                "it has a single full-attention layer, and the last full-attention "
                "layer stays unquantized"
            )
        else:
            reason = "this model's KV cache layout cannot be quantized"
        return "none", reason, True
    if not skipped:
        return "full", "", retainable
    if windowed == skipped:
        reason = (
            "its sliding-window layers keep a fixed-size cache, so only the "
            "full-attention layers are quantized"
        )
    else:
        reason = "only some of this model's layers use a quantizable KV cache"
    return "partial", reason, retainable


def _make_kv_entries(
    model,
    is_vlm,
    max_kv_size = None,
):
    language_model = getattr(model, "language_model", model) if is_vlm else model
    if is_vlm:
        from mlx_vlm.models import cache as vlm_cache
        return vlm_cache.make_prompt_cache(language_model, max_kv_size = max_kv_size)
    from mlx_lm.models import cache as lm_cache

    return lm_cache.make_prompt_cache(language_model, max_kv_size = max_kv_size)


# Override replaces, never creates: renderers pick their target by whether a template exists
_TEMPLATE_NOT_CAPTURED = object()
MLX_TEMPLATE_NO_TARGET = (
    "this model builds its prompt without a chat template, and supplying one "
    "would drop the markers that place its images and audio"
)
MLX_TEMPLATE_CALLABLE = (
    "the installed mlx-lm supplies this model's template as code rather than text, "
    "and prefers it over any override"
)
MLX_TEMPLATE_NAMED_SET = (
    "this model ships a set of named templates rather than one, and replacing the "
    "set with a single template would lose its tool-calling variant"
)
MLX_TEMPLATE_RENDER_FAILED = "it could not render a conversation: {error}"
MLX_TEMPLATE_NOT_SETTABLE = "this model's template cannot be replaced: {error}"
MLX_TEMPLATE_DROPS_IMAGE = (
    "it does not mark where images go, so this model could not accept image input with it"
)
MLX_TEMPLATE_DROPS_AUDIO = (
    "it does not mark where audio goes, so this model could not accept audio input with it"
)


def _template_install_targets(tokenizer, processor):
    """Objects whose chat_template would be read at render time. The processor and the tokenizer can
    be the same object when a processor exposes no nested tokenizer, so the result is
    de-duplicated by identity."""
    seen, targets = [], []
    for candidate in (processor, tokenizer):
        if candidate is None:
            continue
        if any(candidate is other for other in seen):
            continue
        seen.append(candidate)
        targets.append(candidate)
    return targets


def _template_render_targets(tokenizer, processor):
    """Objects a render selector would actually read the template from. Deferred to
    chat_render_target rather than restated (#7066): it also requires the processor to be able to
    render, and mlx-vlm's get_chat_template applies the same rule for audio."""
    from core.inference.chat_template_helpers import chat_render_target

    target = chat_render_target(processor, tokenizer)
    return [target] if target is not None else []


def _usable_template(value):
    """Whether a chat_template value is one this override can replace."""
    return isinstance(value, str) and value.strip() != ""


def _native_template_source(tokenizer, processor):
    """The render target's template, so the editor's model default is the one generation really uses."""
    for target in _template_render_targets(tokenizer, processor):
        if _usable_template(getattr(target, "chat_template", None)):
            return target
    return tokenizer


def _template_override_status(override, tokenizer, processor):
    """Resolve a requested override against this model, without applying it. Returns the targets to
    install onto and a status dict; an empty target list with a reason means the override cannot
    be honored."""
    status = {
        "requested": override,
        "applied": None,
        "reason": None,
        # Kept so a later check (the audio marker) can put the model back.
        "restore": [],
    }
    if not override or not override.strip():
        return [], status
    try:
        candidates = _template_install_targets(tokenizer, processor)
        existing = [(c, getattr(c, "chat_template", None)) for c in candidates]
        rendering = _template_render_targets(tokenizer, processor)
    except Exception as exc:
        status["reason"] = MLX_TEMPLATE_NOT_SETTABLE.format(error = exc)
        return [], status
    # Check only rendering objects; an unread object's unreplaceable template cannot block the override.
    blocked = [(c, getattr(c, "chat_template", None)) for c in rendering]
    if any(getattr(c, "_chat_template", None) is not None for c in rendering):
        # apply_chat_template prefers a callable template over the attribute, so an assignment would be inert.
        status["reason"] = MLX_TEMPLATE_CALLABLE
        return [], status
    if any(isinstance(value, (dict, list)) for _, value in blocked):
        status["reason"] = MLX_TEMPLATE_NAMED_SET
        return [], status
    if not any(_usable_template(value) for _, value in blocked):
        # Text models need one; with a processor, creating one bypasses mlx-vlm's marker fallback
        if processor is not None or not blocked:
            status["reason"] = MLX_TEMPLATE_NO_TARGET
            return [], status
        return [c for c, _ in blocked], status
    # Install on every object holding a replaceable string, so both selectors keep choosing what they chose before.
    return [c for c, value in existing if _usable_template(value)], status


def _audio_marker_survives(processor, model):
    """Whether the installed template still marks where audio goes."""
    if processor is None:
        return True
    args = (processor, model, _AUDIO_PROBE_MESSAGES, 0)
    try:
        marked = _render_registered_vlm_prompt(*args, num_audios = 1)
        return bool(marked) and marked != _render_registered_vlm_prompt(*args, num_audios = 0)
    except BaseException:
        # A load must never fail on a probe, matching _classify_mlx_audio_type.
        return False


def _image_placeholder(tokenizer, processor):
    """The model's own image placeholder, when it names one."""
    for source in (processor, tokenizer):
        token = getattr(source, "image_token", None)
        if isinstance(token, str) and token:
            return token
    return None


_IMAGE_TOKEN_ID_KEYS = ("image_token_index", "image_token_id", "boi_token_index")


def _declared_image_tokens(
    tokenizer,
    processor,
    model = None,
):
    """Config token ids too: granite-vision names its token only by id, on a bare tokenizer."""
    named = []

    def add(token):
        if isinstance(token, str) and token and token not in named:
            named.append(token)

    for source in (processor, tokenizer):
        for attr in ("image_token", "boi_token"):
            add(getattr(source, attr, None))
    config, _model_type = _mlx_vlm_model_config(model) if model is not None else (None, None)
    for key in _IMAGE_TOKEN_ID_KEYS:
        token_id = config.get(key) if isinstance(config, dict) else getattr(config, key, None)
        for source in (getattr(processor, "tokenizer", None), tokenizer, processor):
            try:
                add(source.convert_ids_to_tokens(token_id))
            except BaseException:
                continue
            break
    return named


def _scaling_image_marker(
    tokenizer,
    processor,
    model = None,
):
    """Per-image marker text, or None; a load must never fail on this probe, so diff renders not names"""
    from core.inference.chat_template_helpers import apply_chat_template_for_generation

    targets = _template_render_targets(tokenizer, processor)
    if not targets:
        return None
    try:
        one, two = (
            apply_chat_template_for_generation(targets[0], probe)
            for probe in _MULTI_IMAGE_PROBE_MESSAGES
        )
        if not isinstance(one, str) or not isinstance(two, str):
            return None
        if _MULTI_IMAGE_PROBE_TEXT not in one or _MULTI_IMAGE_PROBE_TEXT not in two:
            return None
        opcodes = difflib.SequenceMatcher(None, one, two, autojunk = False).get_opcodes()
        inserted = [two[j1:j2] for tag, _i1, _i2, j1, j2 in opcodes if tag == "insert"]
        if len(inserted) != 1 or any(tag in ("replace", "delete") for tag, *_rest in opcodes):
            return None
        marker = inserted[0]
        if one.count(marker) != 1 or two.count(marker) != 2:
            return None
        named = _declared_image_tokens(tokenizer, processor, model)
        if any(token in marker for token in named):
            return marker
    except BaseException:
        return None
    return None


def _model_type_takes_several_images(model):
    """Studio renders the model's own template, so mlx-vlm's deny-list has to be read here."""
    try:
        from mlx_vlm.prompt_utils import SINGLE_IMAGE_ONLY_MODELS
    except BaseException:
        return False
    _config, model_type = _mlx_vlm_model_config(model)
    return bool(model_type) and model_type not in SINGLE_IMAGE_ONLY_MODELS


def _image_marker_survives(
    tokenizer,
    processor,
    placeholder = None,
):
    """Fails when an image renders the same as none, emits the content object, or drops the placeholder."""
    from core.inference.chat_template_helpers import apply_chat_template_for_generation

    targets = _template_render_targets(tokenizer, processor)
    if not targets:
        return True
    target = targets[0]
    try:
        marked = apply_chat_template_for_generation(target, _IMAGE_PROBE_MESSAGES)
        if not marked or marked == apply_chat_template_for_generation(target, _TEXT_PROBE_MESSAGES):
            return False
        if _vlm_prompt_issue(marked, _IMAGE_PROBE_MESSAGES):
            return False
        return placeholder is None or placeholder in marked
    except BaseException:
        # A load must never fail on a probe, matching _audio_marker_survives.
        return False


def _revoke_override_dropping(status, survives, reason):
    """Undo an installed override that stopped marking where media goes. Capability was classified
    against the native template, so an override that no longer renders the marker would leave the
    model advertising an input it can no longer place."""
    if not status["applied"] or survives():
        return status
    _restore_templates(status["restore"])
    status["applied"] = None
    status["restore"] = []
    status["reason"] = reason
    return status


def _revoke_override_that_drops_audio(status, processor, model):
    return _revoke_override_dropping(
        status,
        lambda: _audio_marker_survives(processor, model),
        MLX_TEMPLATE_DROPS_AUDIO,
    )


def _revoke_override_that_drops_image(
    status,
    tokenizer,
    processor,
    placeholder = None,
):
    return _revoke_override_dropping(
        status,
        lambda: _image_marker_survives(tokenizer, processor, placeholder),
        MLX_TEMPLATE_DROPS_IMAGE,
    )


def _restore_templates(installed):
    """Put back the templates an install replaced, newest first."""
    for target, value in reversed(installed):
        target.chat_template = value


def _install_template_override(override, tokenizer, processor, probe):
    """Install a chat template override, or report why it was not honored. ``probe`` renders a short
    conversation with whatever is installed; a template that cannot render would otherwise raise
    on every generation instead of once here, which is how a hand-edited template usually fails."""
    targets, status = _template_override_status(override, tokenizer, processor)
    if not targets:
        return status
    # Restore only what was assigned: a refused target would also refuse the restore
    installed = []
    template = MLX_TEMPLATE_RENDER_FAILED
    try:
        for target in targets:
            original = target.chat_template
            template = MLX_TEMPLATE_NOT_SETTABLE
            target.chat_template = override
            template = MLX_TEMPLATE_RENDER_FAILED
            installed.append((target, original))
        probe()
    except Exception as exc:
        _restore_templates(installed)
        status["reason"] = template.format(error = exc)
        return status
    status["applied"] = override
    status["restore"] = installed
    return status


def _kv_entry_is_bounded(entry, window):
    """Only a declared cap within the window counts; an undeclared cache grows without bound."""
    cap = getattr(entry, "max_size", None) or getattr(entry, "chunk_size", None)
    if not cap or cap > window:
        return False
    # Combined stdout+stderr, or None when unknown (exec, timeout, nonzero exit); None is never evidence a feature is absent, so callers stay conservative.
    return cap > getattr(entry, "keep", 0)


def mlx_memory_budget(*, retains_history: bool = True) -> Optional[int]:
    """Bytes a load's own footprint may occupy here, or None where Metal cannot say."""
    import mlx.core as mx

    if not mx.metal.is_available():
        return None
    recommended = mx.device_info().get("max_recommended_working_set_size")
    if not recommended or recommended <= 0:
        return None
    budget = int(recommended * MLX_MEMORY_LIMIT_FRACTION)
    if retains_history:
        budget -= _prompt_cache_max_bytes(recommended / 1e9)
    return budget if budget > 0 else None


def _snapshot_dir(model, model_name: str) -> Optional[str]:
    """Where the files this load actually opened are, or None if none are local."""
    resolved = getattr(model, "_src_path", None)
    if resolved and os.path.isdir(str(resolved)):
        return str(resolved)
    if os.path.isdir(model_name):
        return model_name
    from utils.utils import hf_cache_snapshot_dir

    snapshot = hf_cache_snapshot_dir(model_name)
    return str(snapshot) if snapshot is not None else None


def mlx_fit_to_memory(
    model_dir,
    ceiling,
    *,
    load_in_4bit: bool,
    retains_history: bool,
    kv_bits = None,
    **route,
) -> Optional[int]:
    """The context a load of *model_dir* is held to, priced at the KV width its cache takes.
    ``vision=True`` in *route* prices the mlx-vlm route a speculative text load takes."""
    budget = mlx_memory_budget(retains_history = retains_history)
    if budget is None or not ceiling or not model_dir:
        return None
    with mlx_rng_preserved():
        try:
            from core.inference.mlx_memory import mlx_fit_context

            fitted = mlx_fit_context(
                model_dir,
                budget_bytes = budget,
                max_ctx = int(ceiling),
                kv_bits = kv_bits,
                load_in_4bit = load_in_4bit,
                **route,
            )
            logger.debug("MLX fit for %s: %s tokens under %.1f GB", model_dir, fitted, budget / 1e9)
            return fitted
        except Exception as exc:
            logger.debug("MLX context fit unavailable for %s: %s", model_dir, exc)
            return None


def _fitted_context(
    model,
    model_name: str,
    ceiling,
    *,
    load_in_4bit: bool,
    retains_history: bool,
    kv_bits = None,
    is_vlm = False,
    eligibility = None,
    **route,
):
    """The fit for a resident model and the eligibility verdict it was priced under."""
    try:
        model_dir = _snapshot_dir(model, model_name)
    except Exception as exc:
        logger.debug("MLX snapshot unavailable for %s: %s", model_name, exc)
        return None, None
    verdict = eligibility
    if verdict is None and kv_bits is not None:
        verdict = _kv_quant_eligibility(model, is_vlm, kv_bits)
    fitted = mlx_fit_to_memory(
        model_dir,
        ceiling,
        load_in_4bit = load_in_4bit,
        retains_history = retains_history,
        kv_bits = kv_bits if verdict is None or verdict[0] in ("full", "partial") else None,
        **route,
    )
    return fitted, verdict


def _fit_rank(outcome, context) -> tuple:
    # An unpriceable baseline outranks every fit, so nothing is costless against it.
    return {"no_fit": 0, "fitted": 1, "fits": 2}.get(outcome, 3), context or 0


def mlx_drafter_fit(
    model_dir,
    ceiling,
    pinned,
    source,
    *,
    load_in_4bit: bool,
    costless = False,
    offload = None,
    prefill_chunk = None,
    baseline = None,
    reserve = True,
):
    """``(attaches, fitted context or None)`` for *source*'s drafter on an mlx-vlm load of *model_dir*:
    only a priced fit reserves it, a pinned context must fit whole, and a ``costless`` one must fit
    at least the context the load fits without it, on the *baseline* route (``mlx_offload_route``)
    when the offload made room for it. ``attaches`` is None when the fit could not be priced.
    ``reserve=False`` prices against MLX's own cap, spending the prompt cache's share."""
    budget = mlx_memory_budget(retains_history = reserve and mlx_vlm_snapshot_store_available())
    ceiling = pinned or ceiling
    if budget is None or not ceiling or not model_dir:
        return None, None
    with mlx_rng_preserved():
        try:
            from core.inference.mlx_memory import mlx_fit_outcome

            fit = functools.partial(
                mlx_fit_outcome,
                model_dir,
                budget_bytes = budget,
                max_ctx = int(ceiling),
                load_in_4bit = load_in_4bit,
                vision = True,
                offload = offload,
                prefill_chunk = prefill_chunk,
            )
            outcome, fitted = fit(drafter = (source.path, source.builtin))
            if outcome == "unsizable":
                return None, fitted
            if costless and _fit_rank(outcome, fitted) < _fit_rank(*fit(**(baseline or {}))):
                return False, fitted
        except Exception as exc:
            logger.info("MLX drafter %s not priced: %s", source.path, exc)
            return None, None
    return outcome == "fits" or (outcome == "fitted" and not pinned), fitted


def mlx_prompt_cache_room(
    model_dir,
    n_ctx,
    source,
    *,
    load_in_4bit: bool,
    offload = None,
    prefill_chunk = None,
):
    """Bytes MLX's own cap leaves the prompt cache beside *source*'s drafter at *n_ctx*; None where unsized."""
    budget = mlx_memory_budget(retains_history = False)
    if budget is None or not n_ctx or not model_dir:
        return None
    with mlx_rng_preserved():
        try:
            from core.inference.mlx_memory import mlx_headroom
            return mlx_headroom(
                model_dir,
                budget_bytes = budget,
                n_ctx = int(n_ctx),
                load_in_4bit = load_in_4bit,
                vision = True,
                drafter = (source.path, source.builtin),
                offload = offload,
                prefill_chunk = prefill_chunk,
            )
        except Exception as exc:
            logger.info("MLX prompt cache room beside %s not priced: %s", source.path, exc)
            return None


def _kv_window_enforced(model, is_vlm, window):
    """Whether a cache built at *window* is bounded in every layer, None where none could be built
    to judge: both runtimes defer to a model's own make_cache and ignore max_kv_size there."""
    try:
        entries = _make_kv_entries(model, is_vlm, max_kv_size = window)
        # Inside the guard: load_model calls this unguarded, so a raise would fail the load.
        flattened = list(_flatten_kv_entries(entries))
        return bool(flattened) and all(_kv_entry_is_bounded(entry, window) for entry in flattened)
    except Exception as exc:
        logger.debug("MLX context limit probe failed: %s", exc)
        return None


def _asking_the_architecture(model_dir, question):
    """Put *question* to the tower the sizing would price, built weightless from the checkpoint."""
    with mlx_rng_preserved():
        try:
            import mlx.core as mx

            from core.inference.mlx_memory import (
                _PROBE_SHORT,
                _probe_models,
                _runtime_dtype,
                _snapshot_config,
                weightless_forward,
            )

            config = _snapshot_config(model_dir)
            if config is None:
                return None
            with weightless_forward():
                for build, make_prompt_cache, _ in _probe_models(config, _runtime_dtype()):
                    model = build()
                    try:
                        model(
                            mx.zeros((1, _PROBE_SHORT), dtype = mx.int32),
                            cache = make_prompt_cache(model),
                        )
                    except Exception:
                        continue
                    return question(model, make_prompt_cache)
        except Exception as exc:
            logger.debug("MLX architecture probe unavailable for %s: %s", model_dir, exc)
        return None


# Slider steps resend the width, and each answer is a weightless build.
_kv_refusal_cache: dict = {}
_kv_refusal_lock = threading.Lock()


def mlx_kv_quant_is_refused(model_dir, bits) -> bool:
    """Whether a load would keep this checkpoint's cache at full width despite *bits*, by the
    eligibility probe's own rule (nothing to convert, or a converted cache the model cannot
    attend over)."""

    def refused(model, make_prompt_cache):
        converted, _skipped, failure, _retainable = _kv_quant_probe(
            model, make_prompt_cache(model), bits, evaluate = False
        )
        return failure is not None or not converted

    try:
        from core.inference.mlx_memory import _checkpoint_fingerprint
        key = (os.path.realpath(model_dir), bits, _checkpoint_fingerprint(model_dir))
    except Exception:
        return _asking_the_architecture(model_dir, refused) is True
    with _kv_refusal_lock:
        if key in _kv_refusal_cache:
            return _kv_refusal_cache[key]
    answer = _asking_the_architecture(model_dir, refused) is True
    with _kv_refusal_lock:
        _kv_refusal_cache[key] = answer
        while len(_kv_refusal_cache) > 32:
            _kv_refusal_cache.pop(next(iter(_kv_refusal_cache)))
    return answer


def _turboquant_reserved_positions(count, bits):
    """Entry positions mlx-vlm reserves, learned from a control cache rather than a fixed number."""

    from mlx_vlm.generate.common import maybe_quantize_kv_cache
    from mlx_vlm.models.cache import KVCache
    from mlx_vlm.turboquant import TurboQuantKVCache

    control = [KVCache() for _ in range(count)]
    maybe_quantize_kv_cache(control, 0, MLX_KV_GROUP_SIZE, bits, kv_quant_scheme = "turboquant")
    return {
        index for index, entry in enumerate(control) if not isinstance(entry, TurboQuantKVCache)
    }


def _turboquant_status(
    model,
    bits,
    refusal = "",
):
    import mlx.core as mx

    status = dict(requested_kv_bits = bits, kv_bits = None, eligibility = "none", reason = "", note = "")
    if refusal:
        status["eligibility"] = "refused"
        status["reason"] = refusal
        logger.info("MLX TurboQuant not applied: %s", refusal)
        return status

    try:
        from mlx_vlm.generate.common import maybe_quantize_kv_cache
    except ImportError:
        # TurboQuant predates generate.common (mlx-vlm 0.6.0).
        status["eligibility"] = "refused"
        status["reason"] = "TurboQuant requires a newer mlx-vlm. Update the MLX packages and retry."
        return status
    from mlx_vlm.models.cache import make_prompt_cache
    from mlx_vlm.turboquant import TurboQuantKVCache

    language_model = getattr(model, "language_model", model)
    try:
        entries = make_prompt_cache(language_model)
    except Exception as exc:
        if is_metal_queue_dead(exc):
            raise
        logger.warning("MLX TurboQuant eligibility probe failed: %s", exc)
        status["reason"] = "this model's KV cache layout could not be inspected"
        return status
    if not entries:
        status["reason"] = "this model builds no KV cache to quantize"
        return status

    rng_key = _mlx_rng_key_words()
    try:
        reserved = _turboquant_reserved_positions(len(entries), bits)
        maybe_quantize_kv_cache(entries, 0, MLX_KV_GROUP_SIZE, bits, kv_quant_scheme = "turboquant")
        converted = declined = 0
        for index, entry in enumerate(entries):
            leaves = list(_flatten_kv_entries([entry]))
            quantized = sum(isinstance(leaf, TurboQuantKVCache) for leaf in leaves)
            converted += quantized
            if index not in reserved:
                declined += len(leaves) - quantized
        if not converted:
            status["reason"] = (
                "This model's cache keeps its native format; no layers can use TurboQuant."
            )
            return status
        for _ in range(2):
            result = language_model(mx.array([[0]]), cache = entries)
            mx.eval(getattr(result, "logits", result), [entry.state for entry in entries])
        status["kv_bits"] = bits
        status["eligibility"] = "partial" if declined else "full"
        if declined:
            status["note"] = "Some layers keep their native cache format."
        return status
    except Exception as exc:
        if is_metal_queue_dead(exc):
            raise
        logger.warning("MLX TurboQuant eligibility probe failed: %s", exc)
        status["eligibility"] = "refused"
        status["reason"] = f"this model's KV cache cannot use TurboQuant ({type(exc).__name__})"
        return status
    finally:
        _restore_mlx_rng_key(rng_key)
        # After the probe's own locals are gone, or the pages stay in the allocator.
        entries.clear()
        del entries
        _drain_generation_streams(mx)
        mx.clear_cache()


def _kv_quant_status(
    requested_bits,
    model,
    is_vlm,
    eligibility = None,
):
    """Resolve a requested bit width against this model into a status dict."""
    status = {
        "requested_kv_bits": requested_bits,
        "kv_bits": None,
        "eligibility": None,
        "reason": "",
        "note": "",
    }
    if requested_bits is None:
        return status
    verdict, reason, retainable = eligibility or _kv_quant_eligibility(
        model, is_vlm, requested_bits
    )
    status["eligibility"] = verdict
    status["reason"] = reason
    if verdict in ("full", "partial"):
        status["kv_bits"] = requested_bits
        status["note"] = "" if retainable else MLX_KV_QUANT_NO_REUSE
    else:
        logger.info("MLX KV quantization not applied: %s", reason)
    return status


PROMPT_CACHE_MEMORY_FRACTION = 0.15
PROMPT_CACHE_FALLBACK_BYTES = 2 * 1024**3


def _mlx_prompt_cache_api():
    try:
        from mlx_lm.models.cache import (
            LRUPromptCache,
            can_trim_prompt_cache,
            make_prompt_cache,
            trim_prompt_cache,
        )
    except ImportError:
        return None
    return LRUPromptCache, make_prompt_cache, can_trim_prompt_cache, trim_prompt_cache


def _prompt_cache_max_bytes(recommended_gb = None):
    override = os.environ.get("UNSLOTH_MLX_PROMPT_CACHE_BYTES")
    if override:
        try:
            return max(int(override), 0)
        except ValueError:
            logger.warning("Ignoring non-integer UNSLOTH_MLX_PROMPT_CACHE_BYTES=%r", override)
    if recommended_gb:
        return int(recommended_gb * 1e9 * PROMPT_CACHE_MEMORY_FRACTION)
    return PROMPT_CACHE_FALLBACK_BYTES


def _flatten_kv_entries(cache):
    for entry in cache:
        nested = getattr(entry, "caches", None)
        if nested is None:
            yield entry
        else:
            yield from _flatten_kv_entries(nested)


def _kv_prefix_coverage(cache):
    """Recurrent entries carry no offset, so an attention sibling must attest the token count."""
    covered = None
    for entry in _flatten_kv_entries(cache):
        offset = getattr(entry, "offset", None)
        if offset is None:
            # Upstream trims once all entries agree, so only unrewindable counts.
            if getattr(entry, "is_trimmable", lambda: False)():
                return None
            continue
        if getattr(entry, "start_position", 0):
            return None
        window = getattr(entry, "max_size", None)
        if window is not None and offset > window:
            return None
        if covered is None:
            covered = offset
        elif covered != offset:
            return None
    return covered


class _MLXPromptCacheHistory:
    def __init__(
        self,
        max_entries,
        max_bytes,
        max_kv_size = None,
        prepare = None,
    ):
        api = _mlx_prompt_cache_api()
        if api is None:
            raise RuntimeError("mlx-lm is too old for LRUPromptCache")
        lru_cls, make, can_trim, trim = api
        self._max_kv_size = max_kv_size
        self._prepare = prepare
        self._make_prompt_cache = make
        self._can_trim = can_trim
        self._trim = trim
        self._max_bytes = max_bytes
        self._lru = lru_cls(max_size = max_entries, max_bytes = max_bytes)
        self._oversized = None
        self._newest = None

    def _take_oversized(self, key, tokens):
        held, self._oversized = self._oversized, None
        if held is None:
            return None
        held_key, held_tokens, cache, nbytes = held
        if held_key != key:
            return None
        common = next(
            (i for i, (a, b) in enumerate(zip(held_tokens, tokens)) if a != b),
            min(len(held_tokens), len(tokens)),
        )
        prefix = min(common, len(tokens) - 1)
        # A prefix within budget is the LRU's to serve, so chats under it behave as before.
        if prefix * nbytes <= self._max_bytes * len(held_tokens):
            return None
        excess = len(held_tokens) - prefix
        if excess and not (self._can_trim(cache) and self._trim(cache, excess) == excess):
            return None
        return cache, list(tokens[prefix:])

    def release_oversized(self):
        self._oversized = None

    def fetch(self, model, key, tokens):
        held = self._take_oversized(key, tokens)
        if held is not None:
            return held
        cache, rest = self._lru.fetch_nearest_cache(key, list(tokens))
        if cache is not None:
            if rest:
                return cache, list(rest)
            if self._can_trim(cache) and self._trim(cache, 1) == 1:
                return cache, list(tokens[-1:])
        if len(tokens) > 1:
            head = list(tokens[:-1])
            cache, rest = self._lru.fetch_nearest_cache(key, head)
            if cache is not None:
                covered = len(head) - len(rest)
                return cache, list(tokens[covered:])
        if self._max_kv_size is None:
            cache = self._make_prompt_cache(model)
        else:
            cache = self._make_prompt_cache(model, max_kv_size = self._max_kv_size)
        return cache if self._prepare is None else self._prepare(cache), list(tokens)

    def insert(self, key, tokens, cache):
        sizes = [_kv_entry_nbytes(entry) for entry in cache]
        if any(size is None for size in sizes):
            logger.debug("MLX prompt cache: skipping state whose size is unmeasurable")
            return
        nbytes = sum(sizes)
        covered = _kv_prefix_coverage(cache)
        if covered is None:
            logger.debug("MLX prompt cache: skipping cache with unverifiable prefix coverage")
            return
        tokens = list(tokens)
        if covered > len(tokens):
            logger.debug(
                "MLX prompt cache: cache covers %d tokens but only %d were tracked",
                covered,
                len(tokens),
            )
            return
        tokens = tokens[:covered]
        if not tokens:
            return
        self._newest = (key, tokens)
        if nbytes > self._max_bytes:
            # In the LRU it would evict itself and every other conversation.
            logger.debug(
                "MLX prompt cache: holding %.2f GB entry over the %.2f GB budget for its next turn",
                nbytes / 1e9,
                self._max_bytes / 1e9,
            )
            self._oversized = (key, tokens, cache, nbytes)
            return
        self._lru.insert_cache(key, tokens, cache)

    def resume_entries(self):
        if self._newest is None:
            return []
        key, tokens = self._newest
        held = self._oversized
        if held is not None and held[0] == key and held[1] == tokens:
            return [(key, tokens, held[2], False)]
        try:
            # The trie's own entry: fetch_nearest_cache would deep-copy it.
            return [(key, tokens, self._lru._trie.get(key, tokens).prompt_cache, False)]
        except (AttributeError, KeyError):
            return []


def _resume_save_max_bytes():
    # Mirrors core.inference.llama_cpp._SLOT_SAVE_MAX_BYTES.
    try:
        return int(os.environ.get("UNSLOTH_SLOT_SAVE_MAX_BYTES") or (10 << 30))
    except ValueError:
        return 10 << 30


_RESUME_CACHE_MODULES = ("mlx.", "mlx_lm.", "mlx_vlm.", "unsloth_zoo.mlx.")


def save_resume_cache(path, key, tokens, entries):
    """Arrays as safetensors tensors, the cache layout pickled in metadata: any cache class fits."""
    import base64
    import io
    import pickle

    import mlx.core as mx

    arrays = {}

    class _Pickler(pickle.Pickler):
        def persistent_id(self, obj):
            if isinstance(obj, mx.array):
                name = f"a{len(arrays)}"
                arrays[name] = obj
                return name
            return None

    layout = io.BytesIO()
    _Pickler(layout, protocol = pickle.HIGHEST_PROTOCOL).dump((key, list(tokens), entries))
    # Renamed into place, so a reader never sees a partial file and a removed dir fails the write.
    partial = f"{path}.partial.safetensors"
    mx.save_safetensors(
        partial, arrays, {"layout": base64.b64encode(layout.getvalue()).decode("ascii")}
    )
    os.replace(partial, path)


def load_resume_cache(path):
    """``(key, tokens, entries)`` from ``save_resume_cache``; refuses any non-MLX class."""
    import base64
    import io
    import pickle

    import mlx.core as mx

    arrays, metadata = mx.load(path, return_metadata = True)

    class _Unpickler(pickle.Unpickler):
        def persistent_load(self, name):
            return arrays[name]

        def find_class(self, module, name):
            found = None
            if module.startswith(_RESUME_CACHE_MODULES) and "." not in name:
                found = super().find_class(module, name)
            if not isinstance(found, type):
                raise pickle.UnpicklingError(f"{module}.{name} is not a cache class")
            return found

    key, tokens, entries = _Unpickler(io.BytesIO(base64.b64decode(metadata["layout"]))).load()
    # Read now: the caller deletes the file once the restore returns.
    mx.eval(list(arrays.values()))
    return key, tokens, entries


def _mlx_distributed_rank_size(group = None):
    if group is None:
        return 0, 1
    rank = int(group.rank())
    world_size = int(group.size())
    if world_size < 1:
        raise ValueError(f"Invalid MLX distributed world_size={world_size}.")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"Invalid MLX distributed rank={rank} for world_size={world_size}.")
    return rank, world_size


def _mlx_distributed_backend_from_env():
    if os.environ.get("MLX_JACCL_COORDINATOR") and os.environ.get("MLX_IBV_DEVICES"):
        return "jaccl"
    return None


def _init_mlx_distributed():
    """Initialize MLX distributed state, falling back to singleton metadata."""
    import mlx.core as mx

    group = None
    rank = 0
    world_size = 1
    distributed = getattr(mx, "distributed", None)
    init = getattr(distributed, "init", None) if distributed is not None else None
    if callable(init):
        backend = _mlx_distributed_backend_from_env()
        if backend is None:
            group = init()
        else:
            try:
                group = init(backend = backend)
            except TypeError:
                group = init()
        if group is not None:
            rank, world_size = _mlx_distributed_rank_size(group)
    return group, rank, world_size


def _normalize_mlx_seed(seed):
    """Reduces seeds modulo 2**64: mx.random.key raises on negatives and anything at or above 2**64."""
    return int(seed) % (2**64)


def _make_seeded_mlx_sampler(
    seed,
    *,
    temp,
    top_p,
    min_p,
    top_k,
    min_tokens_to_keep = 1,
):
    """Uses a request-scoped key, since mx.random.seed state would leak into later unseeded requests."""
    import mlx.core as mx
    from mlx_lm.sample_utils import apply_top_p, apply_min_p, apply_top_k

    if temp == 0:
        # argmax draws no randomness; seeding it would be meaningless, not wrong.
        return lambda logprobs: mx.argmax(logprobs, axis = -1)

    stages = []
    if 0 < top_p < 1.0:
        stages.append(lambda x: apply_top_p(x, top_p))
    if min_p != 0.0:
        stages.append(lambda x: apply_min_p(x, min_p, min_tokens_to_keep))
    if top_k > 0:
        stages.append(lambda x: apply_top_k(x, top_k))

    state = {"key": mx.random.key(_normalize_mlx_seed(seed))}

    def _sampler(logprobs):
        for stage in stages:
            logprobs = stage(logprobs)
        state["key"], subkey = mx.random.split(state["key"])
        return mx.random.categorical(logprobs * (1 / temp), key = subkey)

    return _sampler


def _penalized_tokens(tokens, state, window):
    """The tokens a penalty scores: the whole completion, or with ``window`` the last ``window``
    tokens of prompt and completion, as llama-server's ``repeat_last_n`` scores them."""
    if window is not None:
        return tokens[-window:]
    if state["prompt_len"] is None:
        # First call is prompt-only; latch its length.
        state["prompt_len"] = int(tokens.shape[0])
        return tokens[:0]
    return tokens[state["prompt_len"] :]


def _make_mlx_presence_penalty_processor(penalty: float, window = None):
    """Presence penalty as an mlx_lm/mlx_vlm logits processor, matching the safetensors path.
    generate_step calls processors as ``fn(tokens, logits)`` with ``tokens`` the full running
    sequence; the first call is prompt-only, so latch that length and penalize only after it."""
    state = {"prompt_len": None}

    def _processor(tokens, logits):
        generated = _penalized_tokens(tokens, state, window)
        if generated.size == 0:
            return logits
        import mlx.core as mx

        vocab = logits.shape[-1]
        # Bound ids to [0, vocab): MLX has no bounds checks (UB on OOB); strays go to scratch slot vocab
        valid = (generated >= 0) & (generated < vocab)
        safe = mx.where(valid, generated, vocab).astype(mx.int32)
        # (vocab + 1)-wide mask: duplicates idempotent, scratch column dropped
        mask = mx.zeros((vocab + 1,), dtype = logits.dtype)
        mask[safe] = penalty
        logits = logits - mask[:vocab]
        return logits

    _processor.history_only = True
    return _processor


def _row_logits_processors(processors):
    """A reply's processors, each starting at the prompt's last token."""
    if not processors:
        return []

    def _shown(processor):
        state = {"offset": None}

        def _narrowed(tokens, logits):
            if state["offset"] is None:
                state["offset"] = max(int(tokens.shape[0]) - 1, 0)
            return processor(tokens[state["offset"] :], logits)

        return _narrowed

    return [_shown(processor) for processor in processors]


def _make_mlx_frequency_penalty_processor(penalty: float, window = None):
    """Frequency penalty as an mlx_lm/mlx_vlm logits processor. Identical to the presence processor
    except the scatter *accumulates*, so a token repeated N times in the completion is charged N
    x penalty. It counts occurrences and scales once, in float32: accumulating the penalty itself
    in a float16 logits dtype rounds on every repeat, which drifts by tens of logits over a long
    run (1000 repeats at 0.3 lands on -274.25, not -300)."""
    state = {"prompt_len": None}

    def _processor(tokens, logits):
        generated = _penalized_tokens(tokens, state, window)
        if generated.size == 0:
            return logits
        import mlx.core as mx

        vocab = logits.shape[-1]
        valid = (generated >= 0) & (generated < vocab)
        safe = mx.where(valid, generated, vocab).astype(mx.int32)
        counts = mx.zeros((vocab + 1,), dtype = mx.float32).at[safe].add(1.0)
        return logits - (penalty * counts[:vocab]).astype(logits.dtype)

    _processor.history_only = True
    return _processor


def _make_mlx_logit_bias_processor(logit_bias: dict):
    """Routes out-of-range ids to a scratch slot, since MLX does no bounds checking on logit indices."""
    state = {"safe": None, "values": None, "vocab": None}

    def _processor(tokens, logits):
        import mlx.core as mx

        vocab = logits.shape[-1]
        if state["vocab"] != vocab:
            pairs = [(int(t), float(v)) for t, v in logit_bias.items()]
            state["safe"] = mx.array(
                [t if 0 <= t < vocab else vocab for t, _ in pairs], dtype = mx.int32
            )
            state["values"] = mx.array([v for _, v in pairs], dtype = mx.float32)
            state["vocab"] = vocab
        mask = mx.zeros((vocab + 1,), dtype = mx.float32).at[state["safe"]].add(state["values"])
        return logits + mask[:vocab].astype(logits.dtype)

    _processor.history_only = True
    return _processor


_DIFFUSION_RESPONSE_FORMAT = (
    "response_format is not supported on this model: it generates by "
    "diffusion rather than one token at a time, so no grammar can "
    "constrain the next token. Load an autoregressive model to use "
    "guided decoding."
)


def _vlm_generation_is_diffusion(model) -> bool:
    try:
        from mlx_vlm.generate.diffusion import is_diffusion_model
    except Exception:
        return False
    try:
        return bool(is_diffusion_model(model))
    except Exception:
        return False


def _build_grammar_constraint(
    response_format,
    tokenizer,
    prompt,
    *,
    reasoning_markers = None,
    tools = None,
    reasoning_is_extracted = False,
    reply_keeps_special_tokens = False,
):
    if response_format is None or response_format == {"type": "text"}:
        return None
    from core.inference.grammar_constraint import build_constraint

    constraint = build_constraint(
        response_format,
        tokenizer,
        prompt,
        reasoning_markers = reasoning_markers,
        tools = tools,
        reasoning_is_extracted = reasoning_is_extracted,
        reply_keeps_special_tokens = reply_keeps_special_tokens,
    )
    if constraint is not None:
        logger.info("Guided decoding active: response_format=%s", response_format.get("type"))
    return constraint


def _raw_prompt_constraint(grammar, tokenizer):
    if grammar is None:
        return None
    from core.inference.grammar_constraint import GrammarConstraint

    # Raw replies decode with skip_special_tokens, so a special id spelling a literal would vanish.
    return GrammarConstraint(grammar, tokenizer, strips_special_tokens = True)


class _PrefillStopped(Exception):
    pass


def _stream_until_cancelled(stream_generate, cancel_event, *args, **kwargs):
    def check(*_):
        if cancel_event is not None and cancel_event.is_set():
            raise _PrefillStopped

    try:
        yield from stream_generate(*args, prompt_progress_callback = check, **kwargs)
    except _PrefillStopped:
        return


@contextmanager
def _slots_until_cancelled(model, cancel_event):
    """mlx-vlm's ``generate_audio`` takes no stop hook; LFM3 samples through ``stream_slots``, so end that loop."""
    slots = getattr(model, "stream_slots", None)
    if cancel_event is None or slots is None:
        yield
        return

    def until_cancelled(*args, **kwargs):
        for slot in slots(*args, **kwargs):
            if cancel_event.is_set():
                return
            yield slot

    model.stream_slots = until_cancelled
    try:
        yield
    finally:
        del model.stream_slots


class LogprobsUnsupported(ValueError):
    """``n_probs`` asked of a model that does not sample one token per step."""

    public = True
    openai_param = "logprobs"


def _make_mlx_raw_logprob_recorder():
    """A first logits processor queueing each step's log-softmax of the model's own logits, and
    the queue. llama-server reports ``n_probs`` from the logits before any sampler, so a token
    forced by ``logit_bias`` keeps its real logprob; mlx_lm and mlx_vlm compute theirs after the
    processors. Neither runtime calls processors during prefill, so entry i is generated token i."""
    from collections import deque

    import mlx.core as mx

    recorded = deque()

    def _processor(tokens, logits):
        raw = logits.astype(mx.float32)
        recorded.append(raw - mx.logsumexp(raw, axis = -1, keepdims = True))
        return logits

    return _processor, recorded


def _token_bytes_text(tokenizer, token_id):
    kw = dict(skip_special_tokens = False, clean_up_tokenization_spaces = False)
    # SentencePiece decoders strip a sequence's leading space, so "▁the" alone decodes to
    # "the"; after a copy of itself it keeps the " the" llama-server reports.
    alone = tokenizer.decode([token_id], **kw)
    twice = tokenizer.decode([token_id, token_id], **kw)
    text = twice[len(alone) :] if twice.startswith(alone) else alone
    return text, list(text.encode("utf-8"))


def _mlx_token_logprob(raw, token_id, n_probs, tokenizer) -> dict:
    """One llama-server ``logprobs.content`` entry: the sampled token's logprob and the
    ``n_probs`` most likely tokens, all read from the unprocessed logits."""
    import mlx.core as mx

    raw = raw.reshape(-1)
    n = min(int(n_probs), raw.shape[0])
    top = mx.argpartition(-raw, kth = n - 1)[:n]
    values = raw[top]
    order = mx.argsort(-values)
    # llama-server writes the lowest float for -inf, which JSON cannot carry.
    floor = -3.4028234663852886e38
    entry_ids, entry_values = top[order].tolist(), values[order].tolist()
    text, data = _token_bytes_text(tokenizer, int(token_id))
    top_logprobs = []
    for tid, value in zip(entry_ids, entry_values):
        tid_text, tid_bytes = _token_bytes_text(tokenizer, int(tid))
        top_logprobs.append(
            {"id": int(tid), "token": tid_text, "bytes": tid_bytes, "logprob": max(value, floor)}
        )
    return {
        "id": int(token_id),
        "token": text,
        "bytes": data,
        "logprob": max(raw[int(token_id)].item(), floor),
        "top_logprobs": top_logprobs,
    }


def _with_prompt_history(processor, prompt, window):
    """*processor* also scoring the prompt tokens the runtime withholds: mlx_lm and mlx_vlm hand
    processors only what is left of the prompt after prefill and prompt-cache reuse."""
    import mlx.core as mx

    head = None

    def _processor(tokens, logits):
        nonlocal head
        if head is None:
            seen = int(tokens.shape[0])
            head = mx.array(list(prompt)[: max(len(prompt) - seen, 0)][-window:], dtype = mx.int32)
        return processor(mx.concatenate([head, tokens.astype(mx.int32)]), logits)

    # Still a function of the token history alone, so a speculative draft keeps running.
    _processor.history_only = getattr(processor, "history_only", False)
    return _processor


def _mlx_sampling_processors(
    *,
    repetition_penalty = None,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    logit_bias = None,
    grammar_constraint = None,
    penalty_window = None,
    penalty_prompt = None,
):
    """Logits processors for the sampling knobs, or ``None`` when all are inert. Bias runs before
    the penalties, matching llama-server's sampler order. mlx_lm supplies only the repetition
    penalty here: its presence and frequency processors window the last 20 tokens *including the
    prompt*, while the penalties below score the whole completion and exclude it, so using them
    would make the same request sample differently depending on the backend. ``penalty_window``
    is llama-server's ``repeat_last_n``: one window for all three penalties, ``0`` disabling them.
    ``penalty_prompt`` is the full prompt, so the window reaches the part the runtime withholds."""
    processors = []
    if grammar_constraint is not None:
        from core.inference.grammar_constraint import make_grammar_logits_processor
        processors.append(make_grammar_logits_processor(grammar_constraint))
    if logit_bias:
        processors.append(_make_mlx_logit_bias_processor(logit_bias))
    if penalty_window == 0:
        return processors or None
    penalties = []
    if repetition_penalty is not None and float(repetition_penalty) not in (0.0, 1.0):
        from mlx_lm.sample_utils import make_logits_processors

        window = {} if penalty_window is None else {"repetition_context_size": penalty_window}
        repetition = make_logits_processors(repetition_penalty = float(repetition_penalty), **window)
        for processor in repetition:
            processor.history_only = True
        penalties.extend(repetition)
    if presence_penalty:
        penalties.append(
            _make_mlx_presence_penalty_processor(float(presence_penalty), penalty_window)
        )
    if frequency_penalty:
        penalties.append(
            _make_mlx_frequency_penalty_processor(float(frequency_penalty), penalty_window)
        )
    if penalty_window is not None and penalty_prompt:
        penalties = [
            _with_prompt_history(penalty, penalty_prompt, penalty_window) for penalty in penalties
        ]
    processors.extend(penalties)
    return processors or None


# Families whose template emits the markers (mlx-vlm 0.6.0-0.6.8 list): tokenization must not add them; laguna left out on purpose, unlike the current helper
_VLM_INLINE_SPECIAL_TOKEN_FAMILIES = ("gemma3", "gemma3n", "gemma4", "gemma4_unified")
# First mlx-vlm whose gemma-4 mask overlay skips a resumed suffix's non-square mask (#1445).
_VLM_MEDIA_BLOCK_MIN_VERSION = "0.6.4"


def _vlm_add_special_tokens(model_type, processor):
    try:
        from mlx_vlm.utils import should_add_special_tokens
        return should_add_special_tokens(model_type, processor)
    except Exception:
        # The rule those releases inline, which Studio's runtime gate still accepts.
        return (
            getattr(processor, "chat_template", None) is None
            if model_type in _VLM_INLINE_SPECIAL_TOKEN_FAMILIES
            else True
        )


def _vlm_prepared_inputs(model, processor, prompt, images):
    """The processor's inputs for *prompt* and *images*, as mlx-vlm's ``stream_generate`` prepares them."""
    from mlx_vlm.utils import prepare_inputs

    config = getattr(model, "config", None)
    read = config.get if isinstance(config, dict) else lambda attr: getattr(config, attr, None)
    return prepare_inputs(
        processor,
        images = images,
        prompts = prompt,
        image_token_index = read("image_token_index"),
        add_special_tokens = _vlm_add_special_tokens(read("model_type"), processor),
    )


def _vlm_generate_kwargs(inputs):
    """Prepared inputs as ``stream_generate`` kwargs, which then skips its own preparation."""
    inputs = dict(inputs)
    return {
        "input_ids": inputs.pop("input_ids"),
        "pixel_values": inputs.pop("pixel_values", None),
        "mask": inputs.pop("attention_mask", None),
        **inputs,
    }


class _VLMMediaBlock:
    """A prompt's rows through its last vision token, prefilled by Studio: models whose vision
    tokens attend to each other prefill in one forward that lands off the grid, and this is that
    forward cut where the grid can continue."""

    def __init__(self, model, processor, prompt, images, make_cache):
        self._model = model
        self._make_cache = make_cache
        self._inputs = _vlm_prepared_inputs(model, processor, prompt, images)

    def generate_kwargs(self):
        return _vlm_generate_kwargs(self._inputs)

    def rows(self, token_ids):
        ids = self._inputs["input_ids"]
        if ids.shape[0] != 1 or ids[0].tolist() != list(token_ids):
            return 0
        types = self._inputs.get("mm_token_type_ids", self._inputs.get("token_type_ids"))
        if types is None:
            return 0
        visual = [index for index, kind in enumerate(types[0].tolist()) if kind in (1, 2)]
        return visual[-1] + 1 if visual else 0

    def prefill(self, forward, rows):
        import mlx.core as mx

        inputs = self._inputs
        extra = {
            key: value
            for key, value in inputs.items()
            if key not in ("input_ids", "pixel_values", "attention_mask")
        }
        features = self._model.get_input_embeddings(
            input_ids = inputs["input_ids"], pixel_values = inputs.get("pixel_values"), **extra
        )
        kwargs = {
            key: extra[key][:, :rows]
            for key in ("mm_token_type_ids", "token_type_ids")
            if key in extra
        }
        if features.per_layer_inputs is not None:
            kwargs["per_layer_inputs"] = features.per_layer_inputs[:, :rows]
        entries = self._make_cache()
        forward.unrecorded(
            inputs = None, inputs_embeds = features.inputs_embeds[:, :rows], cache = entries, **kwargs
        )
        mx.eval([entry.state for entry in entries])
        return entries


class _RowStream:
    def __init__(
        self,
        *,
        prefix = "",
        sequences = (),
        normalizer = None,
        decode = None,
    ):
        self._prefix = prefix
        self._sequences = sequences
        self._normalizer = normalizer
        self._decode = decode
        self.token_ids = []
        self.sampled = ""
        self.stopped = False
        self._released = 0
        self._cut = 0
        self._output = prefix

    def feed(
        self,
        token = None,
        text = "",
    ):
        if token is not None:
            self.token_ids.append(token)
        if self._decode is not None:
            self.sampled = self._decode(self.token_ids)
        else:
            self.sampled += text or ""
        if self._sequences:
            self._cut, self.stopped = _mlx_stop_cut(self.sampled, self._sequences)
        else:
            self._cut = len(self.sampled)
        if self._normalizer is not None:
            return self._normalize(self.sampled[self._released : self._cut])
        if not self._sequences:
            # Re-decoded snapshots hold back an unfinished character, as in _generate_text.
            held = self.sampled.rstrip("\ufffd") if self._decode is not None else self.sampled
            return [self._prefix + held]
        if self._decode is not None:
            return []
        if self._cut > self._released:
            self._released = self._cut
            return [self._prefix + self.sampled[: self._cut]]
        return []

    def flush(self, text):
        """Text a decoder held until the reply ended; ``settle`` delivers it for normalized or re-decoded rows."""
        if not self._sequences or (self._normalizer is None and self._decode is None):
            # feed releases up to a stop it finds here, which settle no longer would once stopped.
            return self.feed(text = text)
        self.sampled += text
        self._cut, self.stopped = _mlx_stop_cut(self.sampled, self._sequences)
        return []

    def settle(self):
        if not self._sequences:
            if self._decode is not None and self.sampled.endswith("\ufffd"):
                return [self._prefix + self.sampled]
            return []
        if not self.stopped:
            self._cut = len(self.sampled)
        if self._normalizer is not None:
            return self._normalize(self.sampled[self._released : self._cut])
        if self._decode is not None:
            settled = self._prefix + self.sampled[: self._cut]
            return [settled] if settled != self._prefix else []
        if not self.stopped and self._released < len(self.sampled):
            return [self._prefix + self.sampled]
        return []

    def drain(self, *, cancelled):
        if self._normalizer is None:
            return []
        drained = cancelled and not self.stopped
        tail = self._normalizer.drain() if drained else self._normalizer.finish()
        return self._append(tail)

    def _normalize(self, raw):
        self._released = self._cut
        return self._append(self._normalizer.feed(raw))

    def _append(self, delta):
        if not delta:
            return []
        self._output += delta
        return [self._output]


def _messages_with_system_prompt(messages, system_prompt):
    full = [{"role": "system", "content": system_prompt}] if system_prompt else []
    full.extend(messages)
    return full


def _engine_fields(engine, name):
    return getattr(getattr(engine, name, None), "__dataclass_fields__", ())


def _asks_for_row_processors(request):
    return (
        _mlx_sampling_processors(
            repetition_penalty = request.get("repetition_penalty", 1.0),
            presence_penalty = request.get("presence_penalty", 0.0),
            frequency_penalty = request.get("frequency_penalty", 0.0),
            logit_bias = request.get("logit_bias"),
        )
        is not None
    )


def _row_processor_gap(engine, requests):
    """Why these replies' penalties cannot travel into a batch, or None."""
    if not any(_asks_for_row_processors(request) for request in requests):
        return None
    unavailable = getattr(engine, "row_logits_processors_unavailable_reason", None)
    return unavailable() if callable(unavailable) else None


def _constrained(request):
    return request.get("response_format") not in (None, {"type": "text"}) or (
        request.get("grammar") is not None
    )


def _request_batch_gap(request):
    """What a reply asks for that only the one-at-a-time decode serves, or None."""
    if request.get("images") or request.get("images_base64"):
        return "a reply replays tool images"
    if request.get("n_probs"):
        return "a reply reports token probabilities"
    return None


def _carries_video(request):
    return bool(request.get("videos") or request.get("videos_base64"))


def _batch_clips(stack, plans, processor):
    """Each row's clips as files mlx-vlm reads, removed when ``stack`` closes.

    Rows with the same clips share their files, so unsloth-zoo batches them as one group, and that
    group decodes its clips together, so its rows share one decode budget. Rows whose clips are too
    large to share it get files of their own, so each decodes alone with the whole budget.
    """
    files, shared = [], {}
    stack.callback(lambda: [_discard_video_clip(path) for path in files])

    def write(clip):
        files.append(_write_video_clip(clip))
        return files[-1]

    rows = Counter(tuple(plan.videos or ()) for plan in plans)
    clips = []
    for plan in plans:
        videos = tuple(plan.videos or ())
        paths = [
            shared[clip] if clip in shared else shared.setdefault(clip, write(clip))
            for clip in videos
        ]
        if not paths:
            clips.append({})
            continue
        try:
            rate = _video_sampling_fps(paths, processor, rows[videos])
        except _ClipsTooLarge:
            paths = [write(clip) for clip in videos]
            rate = _video_sampling_fps(paths, processor)
        clips.append({"videos": paths, "fps": rate})
    return clips


def _batched_vision_engine_gap(engine):
    def fields(name):
        return _engine_fields(engine, name)

    if not hasattr(engine, "stream_batch"):
        return "the installed unsloth-zoo has no batched vision generation"
    if "seed" not in fields("SamplingParams"):
        return "the installed unsloth-zoo cannot seed a batched reply"
    if "prompt_token_count" not in fields("GenerationResult"):
        return "the installed unsloth-zoo does not report what a batched prompt cost"
    if "logits_processors" not in fields("GenerationRequest"):
        return "the installed unsloth-zoo cannot carry a penalty into a batched reply"
    return None


class _TextRowPlan:
    __slots__ = (
        "prompt",
        "think_prefix",
        "stream",
        "sampler",
        "processors",
        "max_tokens",
        "streamed",
    )

    def __init__(
        self,
        *,
        prompt,
        think_prefix,
        stream,
        sampler,
        processors,
        max_tokens,
        streamed = None,
    ):
        self.prompt = prompt
        self.think_prefix = think_prefix
        self.stream = stream
        self.streamed = streamed
        self.sampler = sampler
        self.processors = processors
        self.max_tokens = max_tokens


class _TextBatchRow:
    __slots__ = (
        "handle",
        "plan",
        "uid",
        "cache_key",
        "all_tokens",
        "prompt_len",
        "cached",
        "generated",
        "reason",
        "processing_at",
        "ready_at",
        "failure",
    )

    def __init__(self, *, handle, plan, cache_key, all_tokens, prompt_len, cached):
        self.handle = handle
        self.plan = plan
        self.uid = None
        self.cache_key = cache_key
        self.all_tokens = all_tokens
        self.prompt_len = prompt_len
        self.cached = cached
        self.generated = 0
        self.reason = None
        self.processing_at = None
        self.ready_at = None
        self.failure = None


def _failing_alone(row, processors):
    """A row's processors, whose failure is kept on the row rather than raised through the batch."""

    def guarded(processor):
        def _processor(tokens, logits):
            if row.failure is None:
                try:
                    return processor(tokens, logits)
                except Exception as exc:
                    row.failure = exc
            return logits

        # A drafted batch drafts only for rows whose processors all read just the history.
        _processor.history_only = getattr(processor, "history_only", False)
        return _processor

    return [guarded(processor) for processor in processors]


def _streamed_tail(row):
    """What a row decoded from ids still holds when it ends, released as the single path's finish."""
    stream = row.plan.stream
    if row.plan.streamed is None or stream.stopped:
        return []
    tail = stream.flush(row.plan.streamed.finish(stream.sampled))
    if stream.stopped:
        row.reason = "stop"
    return tail


def _row_kwargs(request):
    """The sampling and render options one batched reply is planned with."""
    return dict(
        temperature = request.get("temperature", 0.7),
        top_p = request.get("top_p", 0.9),
        top_k = request.get("top_k", 40),
        min_p = request.get("min_p", 0.0),
        max_new_tokens = request.get("max_new_tokens", 256),
        repetition_penalty = request.get("repetition_penalty", 1.0),
        tools = request.get("tools"),
        enable_thinking = request.get("enable_thinking"),
        reasoning_effort = request.get("reasoning_effort"),
        preserve_thinking = request.get("preserve_thinking"),
        continue_final_message = request.get("continue_final_message", False),
        presence_penalty = request.get("presence_penalty", 0.0),
        seed = request.get("seed"),
        frequency_penalty = request.get("frequency_penalty", 0.0),
        logit_bias = request.get("logit_bias"),
        stop = request.get("stop"),
        prompt_ids = request.get("prompt_ids"),
        penalty_window = request.get("penalty_window"),
    )


class _TextBatchSession:
    """mlx-lm's batch with the reply set left open."""

    def __init__(
        self,
        backend,
        *,
        width,
        adapter_state = None,
        owns_model = False,
    ):
        from mlx_lm.generate import BatchGenerator

        self.backend = backend
        self._adapter_state = adapter_state
        self._rows = {}
        self._by_uid = {}
        self._settled = {}
        self._held = ExitStack()
        try:
            if owns_model:
                self._held.enter_context(backend._generation_lock)
                self._held.enter_context(
                    _temporary_mlx_adapter_state(backend._model, adapter_state)
                )
            self._held.enter_context(backend._prefill_decision_scopes())
            self._held.enter_context(backend._generation_scopes("batch"))
            self.generator = BatchGenerator(
                backend._model,
                stop_tokens = [
                    [token] for token in _mlx_stop_token_ids(backend._tokenizer, backend._model)
                ],
                completion_batch_size = width,
                prefill_batch_size = width,
                **backend._prefill_step_kwargs(BatchGenerator),
            )
        except BaseException:
            self._held.close()
            raise

    @property
    def rows_in_flight(self) -> int:
        return len(self._rows)

    @property
    def handles(self):
        return list(self._rows)

    @property
    def ending(self):
        return []

    def take_stats(self, handle):
        """The usage a retired reply settled with, once."""
        return self._settled.pop(handle, None)

    def admit(self, request, handle):
        """Take one reply in, answering with the text preceding its first token."""
        backend = self.backend
        image = request.get("image") if backend._reads_vision_input() else None
        messages = request.get("messages") or []
        row_kwargs = _row_kwargs(request)
        plan = backend._plan_text_row(
            _messages_with_system_prompt(messages, request.get("system_prompt", ""))
            if image is None
            else backend._conversation_to_render(messages, request.get("system_prompt", ""), image),
            tool_protocol_active = request.get("tool_protocol_active"),
            response_format = request.get("response_format"),
            reasoning_is_extracted = request.get("reasoning_is_extracted", False),
            grammar = request.get("grammar"),
            **row_kwargs,
        )
        began = time.perf_counter()
        if image is None:
            prefix, remainder, cache, cache_key, all_tokens = backend._batch_prompt_row(
                plan.prompt,
                self._adapter_state,
            )
        else:
            from unsloth_zoo.mlx.generate import prefill_image_prompt

            # mlx-lm batches token ids only: the image prefills here, and stays out of the token-keyed cache.
            cache = backend._make_row_cache()
            all_tokens = prefill_image_prompt(
                backend._model, backend._tokenizer, plan.prompt, [image], cache
            )
            prefix, remainder, cache_key = all_tokens[:-1], all_tokens[-1:], None
            if row_kwargs["max_new_tokens"] is None:
                plan.max_tokens = backend._unset_generation_budget(plan.prompt, all_tokens)
        row = _TextBatchRow(
            handle = handle,
            plan = plan,
            cache_key = cache_key,
            all_tokens = all_tokens,
            prompt_len = len(remainder) if image is None else len(all_tokens),
            cached = len(prefix) if image is None else 0,
        )
        if image is not None:
            row.processing_at = began
        uids = self.generator.insert(
            [remainder],
            max_tokens = [plan.max_tokens],
            caches = [cache],
            all_tokens = [prefix],
            samplers = [plan.sampler],
            logits_processors = [_failing_alone(row, _row_logits_processors(plan.processors))],
        )
        if len(uids) != 1:
            self.generator.remove(list(uids))
            raise RuntimeError(
                f"mlx-lm admitted {len(uids)} rows for one reply; a batch cannot "
                "tell whose tokens are whose."
            )
        row.uid = uids[0]
        self._rows[handle] = row
        self._by_uid[row.uid] = row
        logger.info(
            "Admitted reply %s to the batch: prompt=%d, cached=%d, max_tokens=%d, model=%s",
            handle,
            row.prompt_len,
            row.cached,
            plan.max_tokens,
            type(backend._model).__name__,
        )
        return plan.think_prefix

    def step(self, waiting = None):
        """Report what the batch produced, retiring every reply that ended."""
        began = time.perf_counter()
        prompt_events, events = self.generator.next()
        now = time.perf_counter() if (events or prompt_events) else None
        for event in prompt_events:
            row = self._by_uid.get(event.uid)
            if row is not None and row.processing_at is None:
                row.processing_at = began
        # A processor runs a step ahead, so the single path raises before yielding this token;
        # reported first, so a sibling ending in this step cannot answer the request as done.
        for row in [row for row in self._rows.values() if row.failure is not None]:
            self.generator.remove([row.uid])
            self._rows.pop(row.handle, None)
            self._by_uid.pop(row.uid, None)
            yield row.handle, row.failure
        for event in events:
            row = self._by_uid.get(event.uid)
            if row is None:
                continue
            if row.processing_at is None:
                row.processing_at = began
            if row.ready_at is None:
                row.ready_at = now
            yield from self._feed(row, event)
        self._release_held_cache_beside_live_rows()

    def _feed(self, row, event):
        stream = row.plan.stream
        # A stop id is held, not written: the end-of-reply flush decides whether it closes markup.
        streamed = row.plan.streamed
        text = "" if streamed is None else streamed.feed_token(int(event.token))
        row.generated += 1
        yield from (
            (row.handle, snapshot) for snapshot in stream.feed(token = event.token, text = text)
        )
        if not (stream.stopped or event.finish_reason is not None):
            return
        row.reason = "stop" if stream.stopped else event.finish_reason
        banked = self.generator.remove(
            [row.uid],
            return_prompt_caches = event.finish_reason is None,
        )
        self.backend._bank_row_cache(
            row.cache_key,
            event,
            banked.get(row.uid),
            row.all_tokens,
            stream,
        )
        yield from self._retire(row, cancelled = False)

    def withdraw(self, handles):
        rows = [self._rows[handle] for handle in handles if handle in self._rows]
        if not rows:
            return
        taken = self.generator.remove([row.uid for row in rows], return_prompt_caches = True)
        for row in rows:
            row.reason = "stop"
            self.backend._bank_row_cache(
                row.cache_key,
                None,
                taken.get(row.uid),
                row.all_tokens,
                row.plan.stream,
            )
            yield from self._retire(row, cancelled = True)
        self._release_held_cache_beside_live_rows()

    def _release_held_cache_beside_live_rows(self):
        # Decided once every row that ended together is retired: holding it beside a row still
        # decoding would raise peak memory.
        history = self.backend._prompt_cache_history
        if self._rows and history is not None:
            history.release_oversized()

    def _retire(self, row, *, cancelled):
        self._rows.pop(row.handle, None)
        self._by_uid.pop(row.uid, None)
        stream = row.plan.stream
        yield from ((row.handle, snap) for snap in _streamed_tail(row))
        yield from ((row.handle, snap) for snap in stream.settle())
        yield from ((row.handle, snap) for snap in stream.drain(cancelled = cancelled))
        ready_at = row.ready_at or time.perf_counter()
        decoded = time.perf_counter() - ready_at
        prefill_s = ready_at - (row.processing_at or ready_at)
        self._settled[row.handle] = _build_generation_stats(
            row.prompt_len,
            (row.prompt_len / prefill_s) if prefill_s > 0 else 0.0,
            row.generated,
            row.generated / max(decoded, 1e-9),
            row.cached,
            finish_reason = row.reason,
        )
        yield row.handle, None

    def close(self):
        try:
            try:
                if self._by_uid:
                    self.generator.remove(list(self._by_uid))
            finally:
                self.generator.close()
        finally:
            self._rows.clear()
            self._by_uid.clear()
            self._settled.clear()
            self._held.close()


class _VLMRowPlan:
    __slots__ = (
        "prompt",
        "images",
        "videos",
        "think_prefix",
        "stream",
        "kwargs",
        "sampling",
        "max_tokens",
        "processors",
        "streamed",
        "speculates",
    )

    def __init__(
        self,
        *,
        prompt,
        images,
        think_prefix,
        stream,
        kwargs,
        sampling,
        max_tokens,
        processors = None,
        streamed = None,
        speculates = True,
        videos = None,
    ):
        self.prompt = prompt
        self.images = images
        self.videos = videos
        self.think_prefix = think_prefix
        self.stream = stream
        self.kwargs = kwargs
        self.sampling = sampling
        self.max_tokens = max_tokens
        self.processors = processors
        self.streamed = streamed
        self.speculates = speculates


class _VisionBatchRow:
    __slots__ = (
        "handle",
        "plan",
        "row",
        "generated",
        "reason",
        "prompt_tokens",
        "cached_tokens",
        "admitted_at",
        "ready_at",
        "cancelled",
        "drafts",
        "failure",
    )

    def __init__(
        self,
        *,
        handle,
        plan,
        row = None,
    ):
        self.handle = handle
        self.plan = plan
        self.row = row
        self.generated = 0
        self.reason = None
        self.prompt_tokens = 0
        self.cached_tokens = 0
        self.admitted_at = time.perf_counter()
        self.ready_at = None
        self.cancelled = False
        self.drafts = None
        self.failure = None


def _vision_batch_reports_ids(engine, speculative):
    """Whether vision rows can be decoded from their sampled ids, as the one-at-a-time decode does."""
    return "token_ids" in _engine_fields(engine, "GenerationEvent") and (
        speculative is None or "drafts" in _engine_fields(engine, "GenerationRequest")
    )


def _row_prompt_cache_gap():
    try:
        from unsloth_zoo.mlx.generate import row_prompt_cache_unavailable_reason
    except ImportError:
        return "the installed unsloth-zoo cannot resume a batched row"
    return row_prompt_cache_unavailable_reason()


def _row_quantized_cache_gap():
    try:
        from unsloth_zoo.mlx.generate import row_quantized_prompt_cache_unavailable_reason
    except ImportError:
        return "the installed unsloth-zoo cannot batch a quantized KV cache"
    return row_quantized_prompt_cache_unavailable_reason()


class _VisionBatchSession:
    def __init__(
        self,
        backend,
        *,
        width,
        adapter_state = None,
        owns_model = False,
    ):
        from unsloth_zoo.mlx import generate as engine
        from unsloth_zoo.mlx.generate import BatchStream, GenerationDefaults

        self.backend = backend
        self._adapter_state = adapter_state
        self._resumes_rows = _row_prompt_cache_gap() is None
        self._rows = {}
        self._by_row = {}
        self._settled = {}
        self._held = ExitStack()
        try:
            if owns_model:
                self._held.enter_context(backend._generation_lock)
                self._held.enter_context(
                    _temporary_mlx_adapter_state(backend._model, adapter_state)
                )
            self._held.enter_context(backend._prefill_decision_scopes())
            self._speculative = getattr(backend, "_speculative_draft", None)
            self._decodes_ids = _vision_batch_reports_ids(engine, self._speculative)
            self.stream = BatchStream(
                backend._model,
                backend._processor,
                defaults = GenerationDefaults(
                    prefill_batch_size = 1,
                    completion_batch_size = width,
                    # Rows quantize their own caches where the single path's decode does.
                    **backend._kv_runtime_quant_kwargs(),
                    **backend._prefill_step_kwargs(GenerationDefaults, grid = True),
                ),
                **({} if self._speculative is None else {"speculative": self._speculative}),
            )
        except BaseException:
            self._held.close()
            raise

    @property
    def rows_in_flight(self) -> int:
        return len(self._rows)

    @property
    def handles(self):
        return list(self._rows)

    @property
    def ending(self):
        return [handle for handle, row in self._rows.items() if row.cancelled]

    def take_stats(self, handle):
        return self._settled.pop(handle, None)

    def admit(self, request, handle):
        from unsloth_zoo.mlx.generate import (
            BatchRowRefused,
            GenerationRequest,
            SamplingParams,
        )

        backend = self.backend
        plan = backend._plan_vlm_request(request, decodes_ids = self._decodes_ids)
        row = _VisionBatchRow(handle = handle, plan = plan)
        state = (
            backend._vlm_batch_row_cache(self._adapter_state, plan.images)
            if self._resumes_rows
            else None
        )
        if state is None and backend._kv_quant_bits() is not None:
            # mlx-vlm's own batch cache would be float.
            raise RowRefused("this row cannot carry the load's quantized KV cache")
        if state is None and plan.images:
            backend._release_vlm_snapshots()
        resume = {} if state is None else {"prompt_cache_state": state}
        admitted_at = time.perf_counter()
        try:
            row_number = self.stream.add(
                GenerationRequest(
                    **backend._vlm_batch_prompt(plan.prompt),
                    image = plan.images[0] if plan.images else None,
                    max_tokens = plan.max_tokens,
                    sampling = SamplingParams(**plan.sampling),
                    logits_processors = plan.processors and _failing_alone(row, plan.processors),
                    **resume,
                    **({} if self._speculative is None or plan.speculates else {"drafts": False}),
                )
            )
        except BatchRowRefused as refusal:
            raise RowRefused(str(refusal)) from refusal
        row.row = row_number
        # A speculative batch prefills the row inside add(), so its clock starts before it.
        row.admitted_at = admitted_at if self._speculative is not None else time.perf_counter()
        self._rows[handle] = row
        self._by_row[row_number] = row
        logger.info(
            "Admitted vision reply %s to the batch: images=%d, max_tokens=%d, model=%s",
            handle,
            len(plan.images or ()),
            plan.max_tokens,
            type(backend._model).__name__,
        )
        return plan.think_prefix

    def step(self, waiting = None):
        now, stopped = None, {}
        # Only a speculative step spans many tokens, reported as they are decoded; only its unsloth-zoo has the call.
        # The batch cannot change mid-step, so a row that stopped itself ends the step and leaves after it.
        failing = {}
        events = (
            self.stream.iter_step(
                lambda: bool(stopped or failing) or (waiting is not None and waiting())
            )
            if self._speculative is not None
            else self.stream.step()
        )
        # As in _TextBatchSession.step: reported before the step's tokens, which it never yields.
        yield from self._fail([row for row in self._rows.values() if row.failure is not None])
        for event in events:
            row = self._by_row.get(event.index)
            if row is None:
                continue
            if row.failure is not None:
                # Only a drafted step decodes as it is read, so a row can fail mid-step and leave after it.
                failing[row.handle] = row
                continue
            if now is None:
                now = time.perf_counter()
            if row.ready_at is None:
                row.ready_at = now
            streamed = row.plan.streamed
            text = (
                event.delta
                if streamed is None
                else "".join(streamed.feed_token(int(token)) for token in event.token_ids)
            )
            if text and not row.cancelled:
                yield from ((row.handle, snapshot) for snapshot in row.plan.stream.feed(text = text))
                if row.plan.stream.stopped:
                    row.cancelled = True
                    row.reason = "stop"
            if event.result is not None:
                result = event.result
                if not row.cancelled:
                    row.reason = result.finish_reason
                self._record(row, result, len(result.token_ids) + (result.finish_reason == "stop"))
                yield from self._retire(row, cancelled = row.cancelled)
            elif row.cancelled:
                stopped[row.handle] = row
        yield from self._fail(list(failing.values()))
        for row in stopped.values():
            if row.handle in self._rows:
                yield from self._retake(row)

    def _fail(self, rows):
        for row in rows:
            self.stream.withdraw(row.row)
            self._rows.pop(row.handle, None)
            self._by_row.pop(row.row, None)
            yield row.handle, row.failure

    def withdraw(self, handles):
        for handle in list(handles):
            row = self._rows.get(handle)
            if row is None:
                continue
            row.cancelled = True
            row.reason = "stop"
            yield from self._retake(row)

    def _retake(self, row):
        managed = self.stream.withdraw(row.row)
        if managed is not None:
            self._record(row, managed, len(managed.token_ids))
            yield from self._retire(row, cancelled = True)

    def _record(self, row, result, generated):
        row.prompt_tokens, row.generated = result.prompt_token_count, generated
        row.cached_tokens = getattr(result, "cached_token_count", 0)
        if self._speculative is not None:
            row.drafts = (result.draft_tokens, result.accepted_draft_tokens)

    def _retire(self, row, *, cancelled):
        self._rows.pop(row.handle, None)
        self._by_row.pop(row.row, None)
        stream = row.plan.stream
        yield from ((row.handle, snap) for snap in _streamed_tail(row))
        yield from ((row.handle, snap) for snap in stream.settle())
        yield from ((row.handle, snap) for snap in stream.drain(cancelled = cancelled))
        ready_at = row.ready_at or time.perf_counter()
        decoded = time.perf_counter() - ready_at
        prefill_s = ready_at - row.admitted_at
        prefilled = row.prompt_tokens - row.cached_tokens
        self._settled[row.handle] = _build_generation_stats(
            prefilled,
            (prefilled / prefill_s) if prefill_s > 0 else 0.0,
            row.generated,
            row.generated / max(decoded, 1e-9),
            cached_n = row.cached_tokens,
            finish_reason = row.reason or "stop",
            drafts = row.drafts,
        )
        yield row.handle, None

    def close(self):
        try:
            self.stream.close()
        finally:
            self._rows.clear()
            self._by_row.clear()
            self._settled.clear()
            self._held.close()


class MLXInferenceBackend:
    def __init__(self):
        self.models = {}
        self.active_model_name = None
        self.loading_models = set()
        self.loaded_local_models = []
        self.device = "mlx"
        self._generation_lock = threading.Lock()
        # usage, timings and terminal reason of the latest generation, shipped on gen_done.
        self.last_generation_stats = None
        self.last_batch_generation_stats = []

        self._model = None
        self._model_fusion = ExitStack()
        self._tokenizer = None
        self._processor = None
        self._is_vlm = False
        self._reads_vision = None
        self._turboquant_refusal = ""
        self._speculative_draft = None
        self._prefill_step = None
        self._prefill_layer_step = None
        self._multi_image_marker = None
        self._config = {}
        self._distributed_group = None
        self._distributed_rank = 0
        self._distributed_world_size = 1

        # Recorded for unload to release pinned memory back to the OS.
        self._memory_limits_applied = {}

        # Load-time knobs read by all generation paths; bound early so a failed load still leaves dicts.
        self._kv_quant = _kv_quant_status(None, None, False)
        self._int8_prefill = _int8_prefill_status(False, None)
        self._ane_prefill = _ane_prefill_status(False, None)
        self._kv_cache_window = None
        self._kv_context_budget = None
        self._served_context = None
        self._resume_load_args = None
        self._resume_files = None
        self._template_override = _template_override_status(None, None, None)[1]

        self._prompt_cache_history = None
        self._prompt_cache_unavailable = False
        self._prompt_cache_room = None
        self._vlm_snapshot_store = None
        self._vlm_snapshot_store_unavailable = False
        self._vlm_is_diffusion_model = None

    def _prompt_cache(self):
        if self._prompt_cache_history is not None or self._prompt_cache_unavailable:
            return self._prompt_cache_history
        max_bytes = self._prompt_cache_bytes()
        if max_bytes <= 0:
            self._prompt_cache_unavailable = True
            logger.info("MLX prompt cache disabled by budget")
            return None
        try:
            self._prompt_cache_history = _MLXPromptCacheHistory(
                PROMPT_CACHE_ENTRIES,
                max_bytes,
                self._kv_cache_window,
                prepare = self._prepare_kv_entries,
            )
        except Exception as exc:
            self._prompt_cache_unavailable = True
            logger.info("MLX prompt cache unavailable (%s); prefilling every request", exc)
            return None
        logger.info(
            "MLX prompt cache: %d entries, %.2f GB budget",
            PROMPT_CACHE_ENTRIES,
            max_bytes / 1e9,
        )
        return self._prompt_cache_history

    def _prompt_cache_bytes(self):
        max_bytes = _prompt_cache_max_bytes(self._memory_limits_applied.get("recommended_gb"))
        room = getattr(self, "_prompt_cache_room", None)
        return max_bytes if room is None else min(max_bytes, room)

    def _clear_prompt_cache(self):
        self._prompt_cache_history = None
        self._prompt_cache_unavailable = False
        self._prompt_cache_room = None
        self._vlm_snapshot_store = None
        self._vlm_snapshot_store_unavailable = False
        self._vlm_is_diffusion_model = None

    def _resume_weight_files(self):
        from pathlib import Path

        name = self.active_model_name
        files = []
        for root in sorted(
            {d for d in (_snapshot_dir(self._model, name), _snapshot_dir(None, name)) if d}
        ):
            weights = [*Path(root).glob("*.safetensors"), *Path(root).glob("*.json")]
            for path in sorted(weights):
                st = path.stat()
                files.append((str(path), st.st_size, st.st_mtime_ns))
        return tuple(files)

    def _resume_fingerprint(self):
        """What a restore must match: the weights this load read and every KV-affecting setting."""
        import mlx.core as mx

        files = self._resume_files
        if files is None:
            files = self._resume_weight_files()
        versions = [mx.__version__]
        for module in ("mlx_lm", "mlx_vlm"):
            versions.append(getattr(sys.modules.get(module), "__version__", None))
        return (
            self.active_model_name,
            files,
            tuple(versions),
            self._is_vlm,
            self._kv_quant_bits(),
            bool(getattr(self, "_turboquant", False)),
            self._kv_cache_window,
            bool(self._int8_prefill.get("active")),
            bool(self._ane_prefill.get("active")),
            self._prefill_step,
            self._resume_load_args,
        )

    def _resume_stores(self):
        return (("text", self._prompt_cache_history), ("vlm", self._vlm_snapshot_store))

    def save_prompt_cache_for_resume(self, directory):
        """Saves each store's newest turn into ``directory``; the manifest, or None."""
        if not self.active_model_name or self._distributed_world_size > 1:
            return None
        # The KV was computed from the weights read at load; files replaced since would mislabel it.
        if self._resume_files is False or (
            self._resume_files is not None and self._resume_files != self._resume_weight_files()
        ):
            return None
        # Never blocks: the worker loop advancing a batch is the thread that would release it.
        if not self._generation_lock.acquire(blocking = False):
            return None
        try:
            kept = [
                (store_name, store.resume_entries() if store is not None else ())
                for store_name, store in self._resume_stores()
            ]
            # llama-slots' bounds: a cap, and 1 GiB of disk left after it.
            total = sum(cache_entries_nbytes(item[2]) for _, items in kept for item in items)
            if total > _resume_save_max_bytes() or shutil.disk_usage(directory).free < total + (
                1 << 30
            ):
                logger.debug("MLX prompt cache: %d bytes do not fit the save budget", total)
                return None
            slots = []
            for store_name, items in kept:
                for index, (key, tokens, entries, replay) in enumerate(items):
                    filename = f"{store_name}-{index}.safetensors"
                    save_resume_cache(os.path.join(directory, filename), key, tokens, entries)
                    slots.append({"filename": filename, "store": store_name, "replay": replay})
            if not slots:
                return None
            return {
                "engine": "mlx",
                "dir": directory,
                "fingerprint": self._resume_fingerprint(),
                "slots": slots,
            }
        finally:
            self._generation_lock.release()

    def restore_prompt_cache_for_resume(self, manifest):
        """Restores a manifest saved under this exact load; the entry count."""
        if not self.active_model_name or manifest.get("engine") != "mlx":
            return 0
        if manifest.get("fingerprint") != self._resume_fingerprint():
            logger.info("MLX prompt cache: the saved context belongs to another load; not restored")
            return 0
        restored = 0
        with self._generation_lock:
            for slot in manifest.get("slots") or ():
                if slot.get("store") == "text":
                    store = self._prompt_cache()
                elif slot.get("store") == "vlm":
                    store = self._vlm_prompt_cache_store()
                else:
                    store = None
                if store is None:
                    continue
                key, tokens, entries = load_resume_cache(
                    os.path.join(manifest["dir"], slot["filename"])
                )
                if slot["store"] == "text":
                    store.insert(key, tokens, entries)
                else:
                    store.store(key, tokens, entries, replay = bool(slot.get("replay")))
                restored += 1
        return restored

    def _release_vlm_snapshots(self):
        store = getattr(self, "_vlm_snapshot_store", None)
        if store is not None:
            store.clear()

    def _vlm_prompt_cache_store(self):
        if self._vlm_snapshot_store is not None or self._vlm_snapshot_store_unavailable:
            return self._vlm_snapshot_store
        max_bytes = self._prompt_cache_bytes()
        if max_bytes <= 0:
            self._vlm_snapshot_store_unavailable = True
            logger.info("MLX VLM prompt cache disabled by budget")
            return None
        # Needs mRoPE priming, reuse reporting and diffusion routing, all added with cached_tokens
        try:
            from mlx_vlm.generate import GenerationResult
            from mlx_vlm.generate.diffusion import is_diffusion_model
            if not hasattr(GenerationResult, "cached_tokens"):
                raise ImportError("GenerationResult has no cached_tokens")
        except ImportError as exc:
            self._vlm_snapshot_store_unavailable = True
            logger.info(
                "MLX VLM prompt cache needs a newer mlx-vlm (%s); prefilling every request", exc
            )
            return None
        self._vlm_is_diffusion_model = is_diffusion_model
        self._vlm_snapshot_store = VLMPromptSnapshotStore(max_bytes)
        logger.info("MLX VLM prompt cache: %.2f GB budget", max_bytes / 1e9)
        return self._vlm_snapshot_store

    @staticmethod
    def _vlm_media_token_ids(config):
        ids = set()
        for attr in ("image_token_id", "image_token_index", "video_token_id", "video_token_index"):
            value = config.get(attr) if isinstance(config, dict) else getattr(config, attr, None)
            if value is not None:
                ids.add(int(value))
        return ids

    @staticmethod
    def _vlm_image_digest(images):
        digest = hashlib.sha256()
        for image in images:
            # What the tower sees: RGB, so palette indices and alpha decide nothing.
            rgb = image.convert("RGB")
            digest.update(f"{rgb.size}:{image.getexif().get(0x0112, 1)}:".encode())
            digest.update(rgb.tobytes())
        return digest.hexdigest()[:16]

    @staticmethod
    def _vlm_bidirectional_vision(model):
        config = getattr(getattr(model, "language_model", model), "config", None)
        return getattr(config, "use_bidirectional_attention", None) == "vision"

    def _vlm_prompt_cache_session(
        self,
        adapter_state,
        images = None,
        prompt = None,
        has_video = False,
    ):
        store = self._vlm_prompt_cache_store()
        if store is None or self._vlm_is_diffusion_model(self._model):
            # Diffusion generation never consults the prefix hook.
            return None
        if has_video:
            # Keyed by token ids the clip does not vary: a later clip would resume the first's mRoPE grid
            return None
        media_ids = self._vlm_media_token_ids(getattr(self._model, "config", None))
        if images and not media_ids:
            # Neither side can tell where the image's rows end without them.
            return None
        try:
            language_model, key, make_cache = self._vlm_snapshot_scope(adapter_state, images)
            block = self._vlm_media_block(prompt, images, make_cache)
            return VLMPromptCacheSession(
                store,
                key,
                language_model,
                make_cache,
                media_token_ids = media_ids,
                releases_unserved = bool(images),
                media_block = block,
                policy_hosts = (self._model, language_model) if block is not None else (),
                step = vlm_prefill_step(),
                window = self._session_window(),
            )
        except Exception as exc:
            # A layout that cannot be built once cannot be built later.
            self._vlm_snapshot_store = None
            self._vlm_snapshot_store_unavailable = True
            logger.info("MLX VLM prompt cache unavailable (%s); prefilling every request", exc)
            return None

    def _vlm_snapshot_scope(self, adapter_state, images):
        """The language model, store key and cache factory every path resumes and banks under."""
        from mlx_vlm.models.cache import make_prompt_cache

        language_model = getattr(self._model, "language_model", self._model)
        window = self._kv_cache_window
        # Base-vs-LoRA compare must not serve one side's KV to the other.
        key = f"{self.active_model_name}|{adapter_state!r}"
        if images:
            key += f"|{self._vlm_image_digest(images)}"
        make_cache = lambda: self._prepare_kv_entries(
            make_prompt_cache(language_model, max_kv_size = window)
        )
        return language_model, key, make_cache

    def _vlm_batch_row_cache(
        self,
        adapter_state,
        images = None,
    ):
        """A batched row's ``prompt_cache_state``, or None while the store is off on an unquantized
        load. Otherwise every row gets one, so all rows' caches share the factory's layout."""
        store = self._vlm_prompt_cache_store()
        if (store is None and self._kv_quant_bits() is None) or _vlm_generation_is_diffusion(
            self._model
        ):
            return None
        media_ids = self._vlm_media_token_ids(getattr(self._model, "config", None))
        try:
            _language_model, key, make_cache = self._vlm_snapshot_scope(adapter_state, images)
        except Exception as exc:
            logger.info("MLX VLM prompt cache unavailable for a batched row (%s)", exc)
            return None
        if store is None:
            pass
        elif images and (not media_ids or self._vlm_bidirectional_vision(self._model)):
            # Unplaceable, or prefilled by the single path as a media block: nothing serves it.
            store.clear()
            store = None
        elif images:
            # Before the vision pass, which zoo runs before the row can look anything up.
            store.retain_key(key)
        return VLMBatchRowCache(
            store,
            key,
            make_cache,
            media_ids,
            releases_unserved = bool(images),
            step = vlm_prefill_step(),
        )

    @staticmethod
    def _vlm_media_block_available():
        try:
            import mlx_vlm
            from packaging.version import Version
            return Version(mlx_vlm.__version__) >= Version(_VLM_MEDIA_BLOCK_MIN_VERSION)
        except Exception:
            return False

    def _vlm_media_block(self, prompt, images, make_cache):
        if not images or prompt is None or not self._vlm_bidirectional_vision(self._model):
            return None
        if not self._vlm_media_block_available():
            return None
        try:
            return _VLMMediaBlock(self._model, self._processor, prompt, images, make_cache)
        except Exception as exc:
            # This request prefills the image as before; the store stays.
            logger.info("MLX VLM prompt cache: media block not prepared (%s)", exc)
            return None

    def _prepare_prompt_cache(self, prompt, adapter_state):
        history = self._prompt_cache()
        if history is None:
            return prompt, None, None, None, 0
        try:
            tokens = list(prompt) if isinstance(prompt, list) else self._encode_prompt(prompt)
            if not tokens:
                return prompt, None, None, None, 0
            key = f"{self.active_model_name}|{adapter_state!r}"
            cache, rest = history.fetch(self._model, key, tokens)
        except Exception as exc:
            logger.debug("MLX prompt cache lookup failed: %s", exc)
            return prompt, None, None, None, 0
        return rest, cache, key, tokens, len(tokens) - len(rest)

    def _unset_generation_budget(
        self,
        prompt,
        prompt_tokens = None,
        images = None,
        audio = None,
        videos = None,
    ):
        """Free context for an unset limit, or the default if the prompt cannot be counted."""
        try:
            prompt_n = (
                len(prompt_tokens)
                if prompt_tokens is not None
                else self._count_prompt_tokens(prompt, images, audio, videos)
            )
        except Exception as exc:
            logger.debug("MLX prompt count for an unset budget failed: %s", exc)
            return UNSET_GENERATION_BUDGET
        return generation_budget_for_window(self._served_context, prompt_n, None)

    def _generation_limit(
        self,
        prompt,
        max_new_tokens,
        prompt_tokens = None,
        images = None,
        audio = None,
        videos = None,
        cap = None,
    ):
        if prompt_tokens is None and isinstance(prompt, list):
            prompt_tokens = prompt
        budget = getattr(self, "_kv_context_budget", None)
        if not budget:
            # No budget: size from the placeholder count as before, never preparing the media.
            if max_new_tokens is None:
                max_new_tokens = self._unset_generation_budget(prompt, prompt_tokens)
                if cap is not None:
                    max_new_tokens = min(max_new_tokens, cap)
            return max_new_tokens
        # Max Tokens == window is the UI's "no cap"; admitting it as-is would refuse every prompt.
        if max_new_tokens is not None and int(max_new_tokens) == int(budget):
            max_new_tokens = None
        if prompt_tokens is None:
            try:
                prompt_tokens = range(self._count_prompt_tokens(prompt, images, audio, videos))
            except Exception as exc:
                logger.debug("MLX prompt count for the context budget failed: %s", exc)
        if max_new_tokens is None:
            max_new_tokens = self._unset_generation_budget(prompt, prompt_tokens)
            if cap is not None:
                max_new_tokens = min(max_new_tokens, cap)
        if prompt_tokens is not None:
            # Refuse only a prompt with no room; a longer ask is cut at the budget like llama-server, since callers size from estimates
            self._check_context_budget(prompt, 1, prompt_tokens)
            max_new_tokens = max(1, min(int(max_new_tokens), int(budget) - len(prompt_tokens)))
        return max_new_tokens

    def _prefill_decision_scopes(self):
        """Outermost scopes of a generation, whose Int8 and Neural Engine Prefill decisions every Zoo
        scope nested in them inherits. Each is pinned off when inactive, so Zoo's full-precision NAX
        routes still run and no scope reads either env switch."""
        with ExitStack() as stack:  # a scope that fails to enter unwinds the ones before it
            for status, zoo, name in (
                (getattr(self, "_int8_prefill", None), _int8_prefill_zoo, "nax_quantized_linear"),
                (getattr(self, "_ane_prefill", None), _ane_prefill_zoo, "ane_prefill_mlp"),
            ):
                scope = (status or {}).get("scope")
                if scope is not None:
                    stack.enter_context(scope(self._model, True))
                elif zoo() is not None:
                    stack.enter_context(_mlx_optional_fusion(name, self._model, False))
            return stack.pop_all()

    def _generation_scopes(self, *call):
        """Zoo's scopes for a generation of this load, under its Int8 Prefill decision. Nested in
        _prefill_decision_scopes, which stays outermost."""
        active = (getattr(self, "_int8_prefill", None) or {}).get("active", False)
        return _mlx_inference_scopes(self._model, call, active)

    def _kv_quant_bits(self):
        return (getattr(self, "_kv_quant", None) or {}).get("kv_bits")

    def _prepare_kv_entries(self, entries):
        bits = self._kv_quant_bits()
        # TurboQuant converts in mlx-vlm: mx.quantize cannot express its fractional widths.
        if bits is None or getattr(self, "_turboquant", False):
            return entries
        return _quantize_kv_entries(entries, bits)

    def _session_window(self):
        # TurboQuant converts the cache between forwards and a KV window rotates it per forward,
        # so neither can take in-window snapshots.
        step = getattr(self, "_prefill_step", None)
        if (
            step
            and getattr(self, "_prefill_layer_step", None)
            and getattr(self, "_kv_cache_window", None) is None
            and not self._kv_runtime_quant_kwargs()
        ):
            return step
        return vlm_prefill_step()

    def _prefill_step_kwargs(
        self,
        target = None,
        grid = False,
    ):
        step = getattr(self, "_prefill_step", None)
        if step is not None and grid:
            step = vlm_prefill_step()
        if step is None or not (target is None or _takes_prefill_step(target)):
            return {}
        return {"prefill_step_size": step}

    def _kv_runtime_quant_kwargs(self):
        """TurboQuant's width for mlx-vlm's converter; start 0, since its default leaves 5000 tokens unquantized."""
        bits = self._kv_quant_bits()
        if bits is None or not getattr(self, "_turboquant", False):
            return {}
        return {"kv_bits": bits, "kv_quant_scheme": "turboquant", "quantized_kv_start": 0}

    def _kv_quant_generate_kwargs(self):
        """A pre-quantized cache, not kv_bits: both runtimes would convert a rotating entry."""
        if self._kv_quant_bits() is None or getattr(self, "_turboquant", False):
            return {}
        return {
            "prompt_cache": self._prepare_kv_entries(_make_kv_entries(self._model, self._is_vlm))
        }

    def _check_context_budget(
        self,
        prompt,
        max_new_tokens = 0,
        prompt_tokens = None,
        images = None,
        audio = None,
        videos = None,
    ):
        """Refuse a request over the budget; an uncountable prompt is admitted."""
        budget = getattr(self, "_kv_context_budget", None)
        if not budget:
            return
        try:
            prompt_n = (
                len(prompt_tokens)
                if prompt_tokens is not None
                else self._count_prompt_tokens(prompt, images, audio, videos)
            )
        except Exception as exc:
            logger.debug("MLX prompt count for the context budget failed: %s", exc)
            return
        needed = prompt_n + max(0, int(max_new_tokens or 0))
        if needed > budget:
            raise context_refusal.ContextBudgetExceeded(needed, int(budget))

    def _kv_window_generate_kwargs(self):
        """The cache bound for a generation that builds its own cache, empty when unset. Both
        runtimes read it only when no prompt_cache is passed, so this covers the request that
        reaches generation without one rather than duplicating the bound."""
        window = getattr(self, "_kv_cache_window", None)
        return {} if window is None else {"max_kv_size": window}

    def _encode_prompt(self, prompt):
        """The tokens a generation sends for this prompt. A template that already emitted the BOS
        would otherwise get a second one, shifting every cached-prefix comparison and overstating
        a counted prompt by one."""
        bos = getattr(self._tokenizer, "bos_token", None)
        return list(
            self._tokenizer.encode(
                prompt, add_special_tokens = bos is None or not prompt.startswith(bos)
            )
        )

    def _reads_vision_input(self):
        """Whether this model reads images, which mlx-vlm serving it no longer implies."""
        reads = getattr(self, "_reads_vision", None)
        if reads is None:
            return bool(getattr(self, "_is_vlm", False))
        return bool(reads)

    def _count_media_prompt_tokens(
        self,
        prompt,
        images = None,
        audio = None,
        videos = None,
    ):
        """Prompt length as the processor expands media (as mlx-vlm's server does), else None."""
        if not (images or audio or videos):
            return None
        try:
            from mlx_vlm.utils import prepare_inputs

            config = getattr(self._model, "config", None)
            read = (
                config.get if isinstance(config, dict) else lambda attr: getattr(config, attr, None)
            )
            kwargs = {}
            if images:
                kwargs["images"] = images
            if audio:
                kwargs["audio"] = audio
            if videos:
                # Same fps as generation, not the processor's default rate.
                kwargs["videos"] = videos
                fps = _video_sampling_fps(videos, self._processor)
                if fps is not None:
                    kwargs["fps"] = fps
            inputs = prepare_inputs(
                self._processor,
                prompts = prompt,
                image_token_index = read("image_token_index"),
                add_special_tokens = _vlm_add_special_tokens(read("model_type"), self._processor),
                **kwargs,
            )
            ids = inputs["input_ids"]
            return int(getattr(ids, "size", 0)) or len(ids)
        except Exception as exc:
            logger.debug("MLX media prompt count failed: %s", exc)
            return None

    def _count_prompt_tokens(
        self,
        prompt,
        images = None,
        audio = None,
        videos = None,
    ):
        """Prompt length as generation tokenizes it; vision models follow mlx_vlm's marker rule."""
        if not self._reads_vision_input():
            return len(self._encode_prompt(prompt))
        expanded = self._count_media_prompt_tokens(prompt, images, audio, videos)
        if expanded is not None:
            return expanded
        model_type = getattr(getattr(self._model, "config", None), "model_type", None)
        add_special = _vlm_add_special_tokens(model_type, self._processor)
        return len(self._tokenizer.encode(prompt, add_special_tokens = add_special))

    def _configure_memory_limits(self):
        """Apply Metal memory caps before loading a model. memory_limit = 85% of recommended
        working-set; wired_limit = min(recommended, memory_limit). Recorded so unload can lower
        wired_limit back to release pinned RAM."""
        import mlx.core as mx

        if not mx.metal.is_available():
            return
        info = mx.device_info()
        rec_bytes = info.get("max_recommended_working_set_size")
        if not rec_bytes or rec_bytes <= 0:
            return
        rec_gb = rec_bytes / 1e9
        memory_limit_gb = rec_gb * MLX_MEMORY_LIMIT_FRACTION
        wired_limit_gb = min(rec_gb, memory_limit_gb)
        mx.set_memory_limit(int(memory_limit_gb * 1e9))
        mx.set_wired_limit(int(wired_limit_gb * 1e9))
        self._memory_limits_applied = {
            "memory_limit_gb": memory_limit_gb,
            "wired_limit_gb": wired_limit_gb,
            "recommended_gb": rec_gb,
        }
        logger.info(
            "MLX memory caps: memory_limit=%.2f GB, wired_limit=%.2f GB",
            memory_limit_gb,
            wired_limit_gb,
        )

    def _resolve_context_lengths(self, model, max_seq_length):
        """Clamps served context to MAX_REQUESTABLE_CONTEXT, as some models declare far larger windows; served length sizes the KV cache, native is unclamped"""
        native = mlx_native_context_length(model)
        served = runtime_context_length(model, max_seq_length) or native
        if served:
            served = min(int(served), MAX_REQUESTABLE_CONTEXT)
            logger.info("MLX context: served=%s native=%s", served, native)
        return served, native, native

    def _kv_cache_window_enforceable(self, served):
        """Whether a cache built for this model would really cap at *served*. A probe that could not
        run counts as not enforceable: claiming a bound nobody confirmed is what makes the
        setting look like it works."""
        if not served or served <= 0:
            return False
        enforced = _kv_window_enforced(self._model, self._is_vlm, int(served))
        if enforced is False:
            logger.warning(
                "MLX context limit of %d tokens is not enforced: this model builds at "
                "least one cache layer that declares no limit at that length.",
                int(served),
            )
        elif enforced is None:
            logger.warning(
                "MLX context limit of %d tokens is not enforced: no cache could be built "
                "to confirm it.",
                int(served),
            )
        return enforced

    def _resolve_kv_policy(
        self,
        is_vlm,
        kv_bits,
        max_seq_length,
        served,
        fitted = False,
        eligibility = None,
    ):
        """With kv_bits, only an enforceable pin or fitted window becomes a budget, since a rotating cache cannot be quantized"""
        pinned = _positive_int(max_seq_length) is not None or fitted
        # Tri-state: True bounded, False confirmed unbounded, None unjudgeable; only True installs a bound, so clients can tell the other two apart
        confirmed = self._kv_cache_window_enforceable(served)
        enforceable = confirmed is True
        if getattr(self, "_turboquant", False) and kv_bits is not None:
            quant = _turboquant_status(
                self._model, kv_bits, getattr(self, "_turboquant_refusal", "")
            )
        else:
            quant = _kv_quant_status(
                _normalize_mlx_kv_bits(kv_bits),
                self._model,
                is_vlm,
                eligibility,
            )
        if quant["kv_bits"] is None and enforceable:
            logger.info("MLX KV cache limited to %d tokens", int(served))
            return quant, int(served), True, None
        # No bound installed: False wherever the probe answered, None where it could not.
        budget = int(served) if quant["kv_bits"] is not None and pinned and enforceable else None
        if budget:
            logger.info("MLX context limited to %d tokens per request, not by the cache", budget)
        return quant, None, (None if confirmed is None else False), budget

    def load_model(
        self,
        config,
        max_seq_length = 2048,
        load_in_4bit = True,
        hf_token = None,
        trust_remote_code = False,
        gpu_ids = None,
        dtype = None,
        parallel_mode = None,
        distributed_group = None,
        kv_quant = None,
        chat_template_override = None,
        int8_prefill = False,
        ane_prefill = False,
        speculative_type = None,
        spec_draft_n_max = None,
        spec_draft_model = None,
        spec_drafters_allowed = None,
        offload = (None, None),
    ) -> bool:
        import mlx.core as mx

        from core.inference import mlx_speculative

        # Keep the token so the native-template fallback can fetch a gated model's repo template during generation.
        self._hf_token = hf_token
        self._resume_load_args = (bool(load_in_4bit), repr(dtype), repr(tuple(offload or ())))
        model_name = config.identifier if hasattr(config, "identifier") else str(config)
        is_vision = getattr(config, "is_vision", False)
        is_lora = getattr(config, "is_lora", False)
        zoo_multimodal = not is_vision and _zoo_declares_multimodal(
            getattr(config, "base_model", None) if is_lora else model_name, hf_token
        )
        kv_bits, turboquant = parse_mlx_kv_quant(kv_quant)
        self._turboquant = turboquant
        if self._turboquant:
            try:
                from mlx_vlm.turboquant import TurboQuantKVCache
            except ImportError as exc:
                raise RuntimeError(
                    "TurboQuant requires a newer mlx-vlm. Update the MLX packages and retry."
                ) from exc
        distributed_rank, distributed_size = _mlx_distributed_rank_size(distributed_group)
        is_distributed = distributed_group is not None and distributed_size > 1
        self._turboquant_refusal = (
            _turboquant_refusal(is_vision = is_vision, is_distributed = is_distributed, is_lora = is_lora)
            if self._turboquant
            else ""
        )
        spec_mode = mlx_speculative.mlx_spec_mode(speculative_type)
        spec_reason = None
        speculates = mlx_speculative.speculates_on_route(
            spec_mode, bool(is_vision), spec_draft_model
        )
        if speculates:
            spec_reason = mlx_speculative.speculation_refusal(
                kv_quant = kv_bits is not None or self._turboquant,
                distributed = is_distributed,
                lora = is_lora,
            )
            speculates = spec_reason is None
            if (
                spec_mode == "auto" and not spec_draft_model
            ):  # not a request, so no refusal to report
                spec_reason = None
        self._speculative_draft = None
        # mlx-lm cannot batch a quantized cache; mlx-vlm, which also ships text architectures, can.
        batches_quantized_text = (
            kv_bits is not None
            and not (is_distributed or is_lora)
            and _row_quantized_cache_gap() is None
        )
        # Speculation runs on mlx-vlm models and caches.
        use_vlm = (
            is_vision
            or zoo_multimodal
            or speculates
            or (self._turboquant and not self._turboquant_refusal)
            or batches_quantized_text
        )
        self._distributed_group = distributed_group
        self._distributed_rank = distributed_rank
        self._distributed_world_size = distributed_size

        # GGUF is served by llama-server; here means route detection flaked, so fail loudly
        if getattr(config, "is_gguf", False):
            raise RuntimeError(
                f"MLXInferenceBackend cannot load GGUF model '{model_name}': "
                f"GGUF models must be served by llama-server in the parent "
                f"process. The /api/inference/load route should have "
                f"detected this repo as GGUF before dispatching to the MLX "
                f"orchestrator -- this fallback indicates a transient HF "
                f"Hub failure during initial detection. Retry the request."
            )

        if hf_token:
            import os
            os.environ["HF_TOKEN"] = hf_token
        self._configure_memory_limits()

        logger.info(
            "Loading %s via %s (is_lora=%s, distributed=%s, rank=%s/%s, mode=%s)",
            model_name,
            "mlx-vlm" if use_vlm else "mlx-lm",
            is_lora,
            is_distributed,
            distributed_rank,
            distributed_size,
            parallel_mode,
        )
        if is_distributed and parallel_mode not in ("pipeline", "tensor"):
            raise ValueError(
                "Unsloth: distributed MLX inference requires parallel_mode='pipeline' "
                "or parallel_mode='tensor'."
            )
        if is_distributed and is_lora:
            raise ValueError(
                "Unsloth: distributed MLX inference for LoRA adapter repos "
                "is not supported yet. Merge/export the adapter into an MLX model "
                "before distributed inference."
            )

        try:
            from unsloth_zoo.mlx.loader import FastMLXModel
        except ImportError as e:
            raise ImportError(
                "Unsloth: MLX inference requires unsloth-zoo with the MLX modules "
                "(unsloth_zoo.mlx.loader). Reinstall via install.sh on Apple Silicon."
            ) from e

        load_kwargs = {
            "max_seq_length": max_seq_length,
            "dtype": dtype,
            "load_in_4bit": load_in_4bit,
            "token": hf_token,
            "trust_remote_code": trust_remote_code,
            "text_only": not use_vlm,
            "quantize_embeddings": not is_lora,
        }
        if is_distributed:
            if parallel_mode == "pipeline":
                load_kwargs["pipeline_group"] = distributed_group
            else:
                load_kwargs["tensor_group"] = distributed_group
        precision_folder = (
            None if getattr(config, "is_gguf", False) else getattr(config, "gguf_variant", None)
        )
        load_source = model_name
        if precision_folder:
            from hub.utils.mlx_variants import download_precision_folder
            load_source = download_precision_folder(model_name, precision_folder, hf_token)

        knobs = offload

        # Auto is planned only where the load is priced: not distributed, an adapter or an explicit dtype.
        def offload_request():
            if is_distributed or is_lora or dtype is not None:
                return _mlx_offload_request(knobs, None)
            priced = dict(
                pinned = _positive_int(max_seq_length),
                load_in_4bit = load_in_4bit,
                retains_history = not use_vlm or mlx_vlm_snapshot_store_available(),
                kv_bits = None if self._turboquant else _normalize_mlx_kv_bits(kv_bits),
                vision = True if speculates and not is_vision else None,
            )

            def plan(model_dir, wanted):
                drafter = speculates and mlx_speculative.offload_drafter(
                    spec_mode,
                    spec_draft_model,
                    model_dir = model_dir,
                    target_name = model_name,
                    allowed = spec_drafters_allowed,
                )
                chosen, alone = mlx_resolve_drafter_offload(
                    model_dir, wanted, drafter or None, auto = spec_mode == "auto", **priced
                )
                # A repeat call sees the fractions and plans no step: keep the first baseline.
                if "auto" in wanted:
                    offload_baseline.update(route = alone and mlx_offload_route(alone))
                return chosen

            return _mlx_offload_request(knobs, plan)

        offload_baseline = {"route": None}

        offload = offload_request()
        load_kwargs.update(offload["kwargs"])

        # Freed before the replacement weights are allocated, for headroom.
        self._model_fusion.close()
        self._clear_prompt_cache()
        model = None
        try:
            model, tokenizer_or_processor = _load_offloading(
                FastMLXModel.from_pretrained, load_source, load_kwargs, offload
            )
        except Exception as exc:
            if is_vision or not use_vlm or is_metal_queue_dead(exc):
                raise
            logger.warning(
                "Load of %s through mlx-vlm for %s failed (%s); serving through mlx-lm",
                model_name,
                "speculative decoding" if speculates else "its KV quantization",
                exc,
            )
        _eligibility = None
        if model is not None and batches_quantized_text and not (is_vision or self._turboquant):
            _eligibility = _kv_quant_eligibility(model, True, _normalize_mlx_kv_bits(kv_bits))
            verdict, reason, _retainable = _eligibility
            if verdict not in ("full", "partial"):
                # Unquantized, the load gains nothing from mlx-vlm's batch.
                logger.info(
                    "MLX KV quantization not applied to %s (%s); serving through mlx-lm",
                    model_name,
                    reason,
                )
                model = tokenizer_or_processor = _eligibility = None
        # unsloth-zoo builds an image checkpoint mlx-vlm lacks through mlx-lm: the text paths
        # serve it, reading its images only through an unsloth-zoo that prefills them.
        served_by_mlx_lm = use_vlm and getattr(model, "_is_vlm_model", None) is False
        if served_by_mlx_lm and not _mlx_lm_reads_images(model):
            is_vision = False
        if model is None or served_by_mlx_lm:
            if speculates:
                speculates = False
                spec_reason = (
                    None
                    if spec_mode == "auto" and not spec_draft_model
                    else mlx_speculative.RUNTIME_ERROR
                )
            elif self._turboquant:
                self._turboquant_refusal = MLX_TURBOQUANT_TEXT_LOAD
            use_vlm = False
        if model is None:
            import gc

            # Outside the except: its traceback keeps the failed load's weights alive.
            gc.collect()
            _drain_generation_streams(mx)
            mx.clear_cache()
            load_kwargs["text_only"] = True
            # Auto planned on the mlx-vlm route, and decide() consumed its knobs: plan this route afresh.
            offload = offload_request()
            load_kwargs.update(offload["kwargs"])
            model, tokenizer_or_processor = _load_offloading(
                FastMLXModel.from_pretrained, load_source, load_kwargs, offload
            )

        zoo_loaded_multimodal = zoo_multimodal and not served_by_mlx_lm
        if zoo_loaded_multimodal:
            is_vision = getattr(model, "_unsloth_supports_images", None) is True
        self._reads_vision = bool(is_vision)
        if use_vlm:
            processor = tokenizer_or_processor
            if is_lora:
                processor = load_mlx_adapter_processor(processor, model_name, hf_token)
            self._model = model
            self._processor = processor
            self._tokenizer = getattr(processor, "tokenizer", processor)
            self._is_vlm = True
        else:
            tokenizer = tokenizer_or_processor
            if is_lora:
                tokenizer = load_mlx_adapter_tokenizer(tokenizer, model_name, hf_token)
            self._model = model
            self._tokenizer = tokenizer
            self._processor = None
            self._is_vlm = False

        _config_audio_type = getattr(config, "audio_type", None)
        if served_by_mlx_lm and _config_audio_type == "audio_vlm":
            # Omni audio input rides mlx-vlm's processor, which an mlx-lm load does not have.
            _config_audio_type = None
        _audio_type = _classify_mlx_audio_type(
            model,
            self._processor,
            is_vision or zoo_loaded_multimodal,
            config_audio_type = _config_audio_type,
        )
        _served_ctx, _native_ctx, _max_ctx = self._resolve_context_lengths(
            self._model, max_seq_length
        )
        _offload = _mlx_offload_status(offload, self._model)
        _offloaded = (_offload["ple"]["applied"], _offload["experts"]["applied"])
        _offloaded = {"offload": _offloaded} if any(_offloaded) else {}
        self._prefill_step = offload["prefill_step"] if _offload["experts"]["applied"] else None
        self._prefill_layer_step = _prefill_layer_step(self._model, self._prefill_step)
        if self._prefill_step:
            _offloaded["prefill_chunk"] = self._prefill_step
        _priceable = not (is_distributed or is_lora or dtype is not None)
        spec_kind = _drafter_ctx = None
        _stretched = []
        _vlm_route = speculates and not is_vision
        if speculates and _vlm_generation_is_diffusion(self._model):
            # Neither the single path nor the resident batch may decode a diffusion model speculatively.
            speculates = False
            spec_reason = (
                None
                if spec_mode == "auto" and not spec_draft_model
                else mlx_speculative.RUNTIME_ERROR
            )
        if speculates:
            model_dir = _snapshot_dir(self._model, model_name)
            resolution = mlx_speculative.resolve_speculation(
                spec_mode,
                spec_draft_model,
                model_dir = model_dir,
                target_name = model_name,
                allowed = spec_drafters_allowed,
            )

            def _fits(source, reserve = True):
                if not _priceable:
                    return None, None
                return mlx_drafter_fit(
                    model_dir,
                    _served_ctx,
                    _positive_int(max_seq_length),
                    source,
                    load_in_4bit = load_in_4bit,
                    costless = spec_mode == "auto",
                    baseline = offload_baseline["route"],
                    reserve = reserve,
                    **_offloaded,
                )

            def _stretch(source):
                ok, context = _fits(source, reserve = False)
                if ok is not False:
                    _stretched.append(source)
                return ok, context

            (
                self._speculative_draft,
                spec_kind,
                spec_reason,
                _drafter_ctx,
            ) = mlx_speculative.build_draft(
                self._model,
                resolution,
                fits = _fits,
                stretch = _stretch,
                draft_n_max = _positive_int(spec_draft_n_max),
            )
            logger.info(
                "MLX speculative decoding: %s (mode %s, reason %s)",
                spec_kind or "off",
                spec_mode,
                spec_reason,
            )
        # TurboQuant's layout is not one the planner prices, and a refused one stays full width:
        # fit either at full width, which only ever under-promises.
        _requested_bits = None if self._turboquant else _normalize_mlx_kv_bits(kv_bits)
        _fitted_ctx, _eligibility = (
            (_drafter_ctx, _eligibility)
            if _drafter_ctx is not None
            else (None, _eligibility)
            if not _priceable or _positive_int(max_seq_length) is not None
            else _fitted_context(
                self._model,
                model_name,
                _served_ctx,
                load_in_4bit = load_in_4bit,
                retains_history = not use_vlm or mlx_vlm_snapshot_store_available(),
                kv_bits = _requested_bits,
                is_vlm = use_vlm,
                eligibility = _eligibility,
                **_offloaded,
                **({"vision": True} if _vlm_route else {}),
            )
        )
        if _fitted_ctx:
            logger.info("MLX context fitted to %d tokens from %d", _fitted_ctx, _served_ctx or 0)
            _served_ctx = _fitted_ctx
        self._served_context = _served_ctx
        # A stretched drafter spent the prompt cache's share: cap the cache to what is left beside it.
        if (
            _priceable
            and _stretched
            and getattr(self._speculative_draft, "drafter", None) is not None
        ):
            self._prompt_cache_room = mlx_prompt_cache_room(
                model_dir, _served_ctx, _stretched[-1], load_in_4bit = load_in_4bit, **_offloaded
            )
            if self._prompt_cache_room is not None:
                logger.info(
                    "MLX prompt cache holds at most %.2f GB beside the chosen drafter",
                    self._prompt_cache_room / 1e9,
                )
        # Classified now so a cache the model cannot attend over is refused here, not on a token.
        (
            self._kv_quant,
            self._kv_cache_window,
            _ctx_enforced,
            self._kv_context_budget,
        ) = self._resolve_kv_policy(
            use_vlm,
            kv_bits,
            max_seq_length,
            _served_ctx,
            fitted = _fitted_ctx is not None,
            eligibility = _eligibility,
        )
        if self._kv_quant["kv_bits"] is not None:
            logger.info(
                "MLX KV cache quantization: %s-bit (%s eligibility)",
                self._kv_quant["kv_bits"],
                self._kv_quant["eligibility"],
            )
        self._int8_prefill = _int8_prefill_status(int8_prefill, self._model)
        self._ane_prefill = _ane_prefill_status(ane_prefill, self._model)

        # Captured before install so the shipped template is kept, from the render target, not the nested tokenizer, which would overwrite the processor template
        native_source = _native_template_source(self._tokenizer, self._processor)
        native_template = getattr(native_source, "chat_template", None)
        native_marks_audio = bool(
            chat_template_override
            and is_audio_input_type(_audio_type)
            and _audio_marker_survives(self._processor, self._model)
        )
        image_placeholder = _image_placeholder(self._tokenizer, self._processor)
        self._template_override = _install_template_override(
            chat_template_override,
            self._tokenizer,
            self._processor,
            lambda: self._render_template_probe(use_vlm),
        )
        if native_marks_audio:
            _revoke_override_that_drops_audio(self._template_override, self._processor, self._model)
        # Unconditional for vision, not gated on the native template: an override rendering plain text silently drops the image
        if is_vision:
            _revoke_override_that_drops_image(
                self._template_override, self._tokenizer, self._processor, image_placeholder
            )
        # After the override settles, so the probe measures the template generation will use.
        self._multi_image_marker = (
            _scaling_image_marker(self._tokenizer, self._processor, self._model)
            if is_vision and _model_type_takes_several_images(self._model)
            else None
        )
        # Release after the media checks: the pairs reference tokenizer and processor and would outlive the unload
        self._template_override["restore"] = []
        if self._template_override["reason"]:
            logger.info(
                "MLX chat template override not applied: %s",
                self._template_override["reason"],
            )

        self.active_model_name = model_name
        try:
            self._resume_files = self._resume_weight_files()
        except OSError:
            self._resume_files = False
        self.models[model_name] = {
            # Per-model token for the native-template fallback (matches transformers).
            "hf_token": hf_token,
            "gguf_variant": precision_folder,
            "precision_folder": precision_folder,
            # Per-model trust_remote_code reused by the native-template reload (matches transformers).
            "trust_remote_code": trust_remote_code,
            "model": self._model,
            "tokenizer": self._tokenizer,
            "processor": self._processor,
            "is_vision": is_vision,
            "is_lora": getattr(config, "is_lora", False),
            # For a LoRA adapter the native chat template lives on the base model.
            "base_model": getattr(config, "base_model", None)
            if getattr(config, "is_lora", False)
            else None,
            # Mirrors utils.models.model_config semantics (is_audio == TTS).
            "is_audio": _audio_type is not None and _audio_type != "audio_vlm",
            "audio_type": _audio_type,
            "has_audio_input": is_audio_input_type(_audio_type),
            "speaks": _audio_type == "audio_vlm"
            and getattr(self._model, "joint_audio_text_generation", False) is True,
            "has_video_input": is_vision and _mlx_reads_video(self._processor),
            "context_length": _served_ctx,
            # Matches llama.cpp requested_n_ctx; served window alone cannot tell if a size was asked for.
            "requested_context_length": _positive_int(max_seq_length) or 0,
            "native_context_length": _native_ctx,
            "max_context_length": _max_ctx,
            # Whether the window bounds the cache: True enforced, False unbounded, None unjudged.
            "context_length_enforced": _ctx_enforced,
            "context_length_fitted": _fitted_ctx,
            "context_unbounded_when_batched": self._vision_batch_is_available(),
            "mlx_context_budget": self._kv_context_budget,
            "mlx_kv_bits": self._kv_quant["kv_bits"],
            "mlx_kv_bits_requested": self._kv_quant["requested_kv_bits"],
            "mlx_kv_quant": encode_mlx_kv_quant(self._kv_quant["kv_bits"], self._turboquant),
            "mlx_kv_quant_requested": encode_mlx_kv_quant(
                self._kv_quant["requested_kv_bits"], self._turboquant
            ),
            "mlx_kv_quant_eligibility": self._kv_quant["eligibility"],
            "mlx_kv_quant_reason": self._kv_quant["reason"],
            "mlx_kv_quant_note": self._kv_quant["note"],
            "mlx_int8_prefill": self._int8_prefill["active"],
            "mlx_int8_prefill_requested": self._int8_prefill["requested"],
            "mlx_int8_prefill_reason": self._int8_prefill["reason"],
            "mlx_ane_prefill": self._ane_prefill["active"],
            "mlx_ane_prefill_requested": self._ane_prefill["requested"],
            "mlx_ane_prefill_reason": self._ane_prefill["reason"],
            "mlx_offload": _offload,
            "chat_template_override_requested": self._template_override["requested"],
            "chat_template_override_reason": self._template_override["reason"],
            "speculative_type": spec_mode,
            "spec_draft_n_max": _positive_int(spec_draft_n_max),
            "spec_draft_model": spec_draft_model or None,
            "spec_drafter_kind": spec_kind,
            "spec_fallback_reason": spec_reason,
        }
        # Capture chat_template_info for the worker IPC reply and route capability classification.
        self._populate_chat_template_info(model_name, native_template)

        # The worker owns fixed base weights until unload; adapter requests stay scoped.
        if not is_lora:
            self._model_fusion.enter_context(_mlx_optional_fusion("fused_moe_gate_up", model))
        # Every rank of a distributed load must run each forward together.
        if not is_distributed:
            with mlx_rng_preserved(), self._prefill_decision_scopes():
                _mlx_warm_inference(model, self._int8_prefill["active"])
            if self._ane_prefill["active"]:
                self._ane_prefill = _ane_prefill_status(True, model)
                self.models[model_name].update(
                    mlx_ane_prefill = self._ane_prefill["active"],
                    mlx_ane_prefill_reason = self._ane_prefill["reason"],
                )
        logger.info("Model %s loaded successfully", model_name)
        return True

    def _render_template_probe(self, is_vision: bool) -> str:
        """Renders through the target generation uses, since the recovery renderer returns None off-list."""
        from core.inference.chat_template_helpers import (
            apply_chat_template_for_generation,
            chat_render_target,
        )

        messages = [{"role": "user", "content": "hi"}]
        target = (
            chat_render_target(self._processor)
            if is_vision and self._processor is not None
            else self._tokenizer
        )
        rendered = apply_chat_template_for_generation(target, messages)
        if not rendered or not rendered.strip():
            raise ValueError("the template produced an empty prompt")
        return rendered

    def _populate_chat_template_info(
        self,
        model_name: str,
        native_template = _TEMPLATE_NOT_CAPTURED,
    ) -> None:
        """Records the shipped template, not an override, since capability classification reads this one."""
        entry = self.models.get(model_name)
        if not entry:
            return
        tok = entry.get("tokenizer")
        if tok is None:
            proc = entry.get("processor")
            tok = getattr(proc, "tokenizer", None) if proc else None
        info = {
            "has_template": False,
            "template": None,
            "format_type": "generic",
            "special_tokens": {},
            "template_name": None,
            # The body _generate_vlm renders an image turn with, not the tokenizer body.
            "processor_template": None,
            "renders_image": False,
            "accepts_multiple_images": False,
        }
        from core.inference.chat_template_helpers import (
            chat_render_target as _chat_render_target,
        )

        _proc = entry.get("processor")
        # A processor template alone does not mean the render selects it (#7066).
        _proc_tpl = (
            getattr(_proc, "chat_template", None) if _chat_render_target(_proc) is _proc else None
        )
        if isinstance(_proc_tpl, (str, dict, list, tuple)) and _proc_tpl:
            info["processor_template"] = _proc_tpl
        # An image model served through mlx-lm marks images with its tokenizer's template.
        info["renders_image"] = self._reads_vision_input() and (
            _proc is not None or not getattr(self, "_is_vlm", False)
        )
        info["accepts_multiple_images"] = info["renders_image"] and bool(
            getattr(self, "_multi_image_marker", None)
        )
        try:
            tpl = (
                getattr(tok, "chat_template", None)
                if native_template is _TEMPLATE_NOT_CAPTURED
                else native_template
            )
            if tpl:
                info["has_template"] = True
                info["template"] = tpl
                lower = tpl.lower()
                if "start_header_id" in lower and "end_header_id" in lower:
                    info["format_type"] = "llama3"
                elif "[inst]" in lower and "[/inst]" in lower:
                    info["format_type"] = "mistral"
                elif "<|im_start|>" in lower and "<|im_end|>" in lower:
                    info["format_type"] = "chatml"
                else:
                    info["format_type"] = "custom"
                special = {}
                for attr in ("bos_token", "eos_token", "pad_token"):
                    val = getattr(tok, attr, None)
                    if val:
                        special[attr] = val
                info["special_tokens"] = special
        except Exception as exc:
            logger.warning("MLX chat_template_info capture failed: %s", exc)
        entry["chat_template_info"] = info

    def unload_model(self, model_name: str) -> bool:
        import mlx.core as mx
        import gc

        self._model_fusion.close()
        if self._ane_prefill["active"] and self._model is not None:
            # The Neural Engine worker can hold its last request's program past the model; a failed release must not
            # stop the unload.
            try:
                _ane_prefill_zoo().release_ane_prefill(self._model)
            except Exception as exc:
                logger.warning("MLX Neural Engine prefill release failed: %s", exc)
            self._ane_prefill = _ane_prefill_status(False, None)
        if model_name in self.models:
            del self.models[model_name]
        self._speculative_draft = None
        self._prefill_step = None
        self._prefill_layer_step = None
        self._model = None
        self._tokenizer = None
        self._processor = None
        self._distributed_group = None
        self._distributed_rank = 0
        self._distributed_world_size = 1
        if self.active_model_name == model_name:
            self.active_model_name = None
        self._clear_prompt_cache()
        gc.collect()
        _drain_generation_streams(mx)
        mx.clear_cache()

        if mx.metal.is_available() and self._memory_limits_applied and not self.models:
            try:
                mx.set_wired_limit(0)
                logger.info("MLX wired_limit released back to OS on unload")
            except Exception as e:
                logger.warning("Failed to release wired_limit: %s", e)
            self._memory_limits_applied = {}
        logger.info("Model %s unloaded", model_name)
        return True

    def _render_text_prompt(
        self,
        messages,
        *,
        tools = None,
        enable_thinking = None,
        reasoning_effort = None,
        preserve_thinking = None,
        continue_final_message = False,
    ):
        """Render the prompt a text generation sends, with its template metadata. Shared with
        counting, so a count cannot price a prompt the model never sees."""
        from core.inference.chat_template_helpers import (
            apply_chat_template_for_generation,
            render_with_native_template_fallback,
        )

        prompt = apply_chat_template_for_generation(
            self._tokenizer,
            messages,
            tools = tools,
            enable_thinking = enable_thinking,
            reasoning_effort = reasoning_effort,
            preserve_thinking = preserve_thinking,
            continue_final_message = continue_final_message,
        )
        if prompt is None:
            raise RuntimeError("apply_chat_template returned None — tokenizer may be incompatible")

        # Transformers parity: fall back to the native template if the template dropped tools
        model_info = self.models.get(self.active_model_name, {})
        return render_with_native_template_fallback(
            formatted_prompt = prompt,
            tokenizer = self._tokenizer,
            model_info = model_info,
            active_model_name = self.active_model_name,
            messages = messages,
            tools = tools,
            enable_thinking = enable_thinking,
            reasoning_effort = reasoning_effort,
            preserve_thinking = preserve_thinking,
            continue_final_message = continue_final_message,
            hf_token = model_info.get("hf_token"),
            return_metadata = True,
        )

    @staticmethod
    def _with_system_prompt(messages, system_prompt):
        """The conversation a request carrying this system prompt turns into. Shared with
        generation, which takes the system prompt beside the messages rather than inside them."""
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)
        return full_messages

    def _conversation_to_render(
        self,
        messages,
        system_prompt,
        image,
        video = 0,
    ):
        """The turns this request renders, whichever way its reply is decoded."""
        if (self._is_vlm or self._reads_vision_input()) and (image is not None or video):
            from core.inference.chat_template_helpers import (
                chat_render_target as _chat_render_target,
            )
            full_messages = messages_with_attached_image(
                messages,
                system_prompt = system_prompt,
                structured_content = (
                    self._processor is not None
                    and _chat_render_target(self._processor) is self._processor
                ),
                image = int(image is not None),
                video = video,
            )
        else:
            full_messages = self._with_system_prompt(messages, system_prompt)
        return full_messages

    def count_chat_tokens(
        self,
        messages,
        system_prompt = "",
        *,
        tools = None,
        enable_thinking = None,
        reasoning_effort = None,
        preserve_thinking = None,
    ) -> int:
        """Prompt tokens this model would receive for these messages. Renders and tokenizes exactly
        as generation does."""
        if self._model is None:
            raise RuntimeError("No model loaded")
        full_messages = self._with_system_prompt(messages, system_prompt)

        if self._is_vlm:
            # images=None: an image anywhere makes the structured-item check raise.
            prompt, _target, _markers = self._render_vlm_prompt(
                full_messages,
                None,
                tools = tools,
                enable_thinking = enable_thinking,
                reasoning_effort = reasoning_effort,
                preserve_thinking = preserve_thinking,
            )
            return self._count_prompt_tokens(prompt)

        render_result = self._render_text_prompt(
            full_messages,
            tools = tools,
            enable_thinking = enable_thinking,
            reasoning_effort = reasoning_effort,
            preserve_thinking = preserve_thinking,
        )
        return self._count_prompt_tokens(render_result.prompt)

    def generate_chat_response(
        self,
        messages,
        system_prompt = "",
        image = None,
        images = None,
        image_ordinal = None,
        temperature = 0.7,
        top_p = 0.9,
        top_k = 40,
        min_p = 0.0,
        max_new_tokens = 256,
        repetition_penalty = 1.0,
        cancel_event = None,
        # Reasoning / tool kwargs, rendered via apply_chat_template_for_generation (transformers parity).
        tools = None,
        enable_thinking = None,
        reasoning_effort = None,
        preserve_thinking = None,
        continue_final_message = False,
        presence_penalty = 0.0,
        seed = None,
        frequency_penalty = 0.0,
        logit_bias = None,
        stop = None,
        response_format = None,
        reasoning_is_extracted = False,
        _adapter_state = None,
        # Unrestricted mode uses an EMPTY tools list, so bool(tools) cannot detect the protocol
        tool_protocol_active = None,
        videos = None,
    ) -> Generator[str, None, None]:
        if self._model is None:
            raise RuntimeError("No model loaded")
        if videos:
            if not _mlx_vlm_decodes_video():
                raise RuntimeError("The installed mlx-vlm does not read video.")
            if not self._reads_vision_input():
                raise RuntimeError("The loaded model does not read video.")

        # Reset so a failed run cannot surface stale stats.
        self.last_generation_stats = None

        full_messages = self._with_system_prompt(messages, system_prompt)

        # History first, the attachment last: MLX binds pixels to markers in order.
        attached = list(images or []) + ([image] if image is not None else [])
        # Shared with the transformers vision path so both render the same turns (#10092).
        if self._reads_vision_input() and (attached or videos):
            # Processor templates want part lists, the tokenizer fallback wants strings.
            from core.inference.chat_template_helpers import (
                chat_render_target as _chat_render_target,
            )

            _renders_via_processor = (
                self._processor is not None
                and _chat_render_target(self._processor) is self._processor
            )
            full_messages = messages_with_attached_image(
                messages,
                system_prompt = system_prompt,
                structured_content = _renders_via_processor,
                image = len(attached),
                video = len(videos or ()),
            )
            # That helper skips convos with markers, but replayed MCP picture markers belong to other pictures, so the render would be one short
            _prior_markers = image_marker_parts(full_messages)
            full_messages = top_up_image_markers(
                full_messages, len(attached), ordinal = image_ordinal
            )
            if image is not None:
                # Read off the conversation: the attachment's turn can precede a
                # replayed picture, and pixels bind to markers in document order.
                attached = pixels_in_marker_order(
                    full_messages, _prior_markers, list(images or []), image
                )
            if len(attached) > 1 and not self._multi_image_marker:
                # The template marks one image: serve the attachment, else the newest picture
                keep = next(
                    (i for i, one in enumerate(attached) if one is image), len(attached) - 1
                )
                logger.info(
                    "Serving 1 of %d images: '%s' marks one image per request.",
                    len(attached),
                    self.active_model_name,
                )
                full_messages = list(full_messages)
                trim_image_turns(full_messages, attached, limit = 1, keep = (keep,))

        if self._is_vlm:
            stream = self._generate_vlm(
                full_messages,
                attached if self._reads_vision_input() else None,
                temperature,
                top_p,
                top_k,
                min_p,
                max_new_tokens,
                repetition_penalty,
                cancel_event,
                tools = tools,
                enable_thinking = enable_thinking,
                reasoning_effort = reasoning_effort,
                preserve_thinking = preserve_thinking,
                continue_final_message = continue_final_message,
                presence_penalty = presence_penalty,
                seed = seed,
                frequency_penalty = frequency_penalty,
                logit_bias = logit_bias,
                _adapter_state = _adapter_state,
                stop = stop,
                tool_protocol_active = tool_protocol_active,
                response_format = response_format,
                reasoning_is_extracted = reasoning_is_extracted,
                videos = videos,
            )
        else:
            stream = self._generate_text(
                full_messages,
                temperature,
                top_p,
                top_k,
                min_p,
                max_new_tokens,
                repetition_penalty,
                cancel_event,
                images = attached if self._reads_vision_input() else None,
                tools = tools,
                enable_thinking = enable_thinking,
                reasoning_effort = reasoning_effort,
                preserve_thinking = preserve_thinking,
                continue_final_message = continue_final_message,
                presence_penalty = presence_penalty,
                seed = seed,
                frequency_penalty = frequency_penalty,
                logit_bias = logit_bias,
                _adapter_state = _adapter_state,
                stop = stop,
                tool_protocol_active = tool_protocol_active,
                response_format = response_format,
                reasoning_is_extracted = reasoning_is_extracted,
            )
        yield from stream

    def _vocab_size(self):
        """The model's token-id range, as llama-server's ``n_vocab`` bounds a prompt's ids."""
        config = getattr(self._model, "config", None)
        for owner in (
            getattr(self._model, "args", None),
            getattr(config, "text_config", None),
            config,
        ):
            size = getattr(owner, "vocab_size", None)
            if isinstance(size, int) and size > 0:
                return size
        return len(getattr(self._tokenizer, "_tokenizer", self._tokenizer))

    def resolve_completion_prompt(self, prompt) -> dict:
        """Token ids and echo text for one llama-server completion prompt, tokenized as its
        ``tokenize_mixed`` does: specials parsed, only a leading string gets BOS."""
        if self._model is None:
            raise RuntimeError("No model loaded")
        if isinstance(prompt, dict):
            prompt = prompt.get("prompt_string")
        pieces = [prompt] if isinstance(prompt, str) else prompt
        if not isinstance(pieces, list):
            raise ValueError("Invalid prompt.")
        vocab = self._vocab_size()
        ids, text, run = [], [], []
        for index, piece in enumerate(pieces + [""]):
            if isinstance(piece, int) and not isinstance(piece, bool):
                if not 0 <= piece < vocab:
                    raise ValueError("Prompt contains invalid tokens.")
                run.append(piece)
                continue
            if not isinstance(piece, str):
                raise ValueError("Invalid prompt.")
            if run:
                text.append(
                    self._tokenizer.decode(
                        run, skip_special_tokens = False, clean_up_tokenization_spaces = False
                    )
                )
                ids.extend(run)
                run = []
            if index < len(pieces):
                ids.extend(self._tokenizer.encode(piece, add_special_tokens = index == 0))
                text.append(piece)
        if not ids:
            raise ValueError("The prompt is empty.")
        return {"ids": ids, "text": "".join(text), "n_vocab": vocab}

    def generate_completion(
        self,
        prompt_ids,
        temperature = 0.7,
        top_p = 0.9,
        top_k = 40,
        min_p = 0.0,
        max_new_tokens = 256,
        repetition_penalty = 1.0,
        cancel_event = None,
        presence_penalty = 0.0,
        seed = None,
        frequency_penalty = 0.0,
        logit_bias = None,
        stop = None,
        penalty_window = None,
        grammar = None,
        n_probs = 0,
    ) -> Generator[str, None, None]:
        """Cumulative text sampled after ``prompt_ids`` with no chat template.
        ``penalty_window`` is llama-server's ``repeat_last_n``; ``n_probs`` > 0 adds each
        token's ``logprobs.content`` entry to the stats as ``token_logprobs``."""
        if self._model is None:
            raise RuntimeError("No model loaded")
        self.last_generation_stats = None
        sampling = dict(
            presence_penalty = presence_penalty,
            seed = seed,
            frequency_penalty = frequency_penalty,
            logit_bias = logit_bias,
            stop = stop,
            prompt_ids = list(prompt_ids),
            penalty_window = penalty_window,
            grammar = grammar,
        )
        if n_probs:
            sampling["n_probs"] = int(n_probs)
        args = (temperature, top_p, top_k, min_p, max_new_tokens, repetition_penalty, cancel_event)
        if self._is_vlm:
            yield from self._generate_vlm(None, None, *args, **sampling)
        else:
            yield from self._generate_text(None, *args, **sampling)

    def _mark_stopped(self):
        """Record that a text sequence ended the turn, not the runtime."""
        if isinstance(self.last_generation_stats, dict):
            self.last_generation_stats["finish_reason"] = "stop"

    def _generate_text(
        self,
        messages,
        temperature,
        top_p,
        top_k,
        min_p,
        max_new_tokens,
        repetition_penalty,
        cancel_event,
        *,
        images = None,
        tools = None,
        enable_thinking = None,
        reasoning_effort = None,
        preserve_thinking = None,
        continue_final_message = False,
        presence_penalty = 0.0,
        seed = None,
        frequency_penalty = 0.0,
        logit_bias = None,
        _adapter_state = None,
        tool_protocol_active = None,
        stop = None,
        response_format = None,
        reasoning_is_extracted = False,
        prompt_ids = None,
        penalty_window = None,
        grammar = None,
        n_probs = 0,
    ):
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        if prompt_ids is not None:
            # A raw completion prompt: no template, so no prefill or reasoning channel.
            prompt, reasoning_channel_markers, think_prefix = list(prompt_ids), None, ""
            preserve_native_channels = False
            constraint = _raw_prompt_constraint(grammar, self._tokenizer)
        else:
            render_result = self._render_text_prompt(
                messages,
                tools = tools,
                enable_thinking = enable_thinking,
                reasoning_effort = reasoning_effort,
                preserve_thinking = preserve_thinking,
                continue_final_message = continue_final_message,
            )
            prompt = render_result.prompt
            reasoning_channel_markers = render_result.reasoning_channel_markers

            preserve_native_channels = reasoning_channel_markers is not None
            # An open <think> prefilled by the template lives in the prompt, not the generated tokens; re-emit it so the
            # frontend renders the block.
            # Matches native_token_decoder below: when it runs </think> survives, so re-emit the opener.
            think_close_survives = (
                bool(tools) or tool_protocol_active or reasoning_channel_markers is not None
            ) and decoder_preserves_token(
                self._tokenizer, "</think>", reasoning_control_tokens(reasoning_channel_markers)
            )
            think_prefix = _think_prefix(
                prompt,
                getattr(self._tokenizer, "all_special_tokens", None),
                messages,
                continue_final_message,
                preserves_think_close = think_close_survives,
            )
            constraint = _build_grammar_constraint(
                response_format,
                self._tokenizer,
                prompt,
                reasoning_markers = reasoning_channel_markers,
                reasoning_is_extracted = reasoning_is_extracted,
                # Must match the prefill's answer, or a closer it re-emits would be refused.
                reply_keeps_special_tokens = preserve_native_channels or think_close_survives,
            )
            if constraint is not None and not constraint.allows_reasoning:
                think_prefix = ""
                preserve_native_channels = False
        # Not the request flag: a later tool-loop pass keeps it but renders an ordinary post-tool prompt.
        _resumed_partial = bool(continue_final_message and trailing_assistant_text(messages))
        if seed is None:
            sampler = make_sampler(
                temp = temperature,
                top_p = top_p,
                top_k = int(top_k or 0),
                min_p = float(min_p or 0.0),
                min_tokens_to_keep = 1,
            )
        else:
            sampler = _make_seeded_mlx_sampler(
                seed,
                temp = temperature,
                top_p = top_p,
                top_k = int(top_k or 0),
                min_p = float(min_p or 0.0),
            )
        logits_processors = _mlx_sampling_processors(
            repetition_penalty = repetition_penalty,
            presence_penalty = presence_penalty,
            frequency_penalty = frequency_penalty,
            logit_bias = logit_bias,
            grammar_constraint = constraint,
            penalty_window = penalty_window,
            penalty_prompt = prompt_ids,
        )
        recorded = token_logprobs = None
        if n_probs:
            recorder, recorded = _make_mlx_raw_logprob_recorder()
            logits_processors = [recorder, *(logits_processors or ())]
            token_logprobs = []

        native_token_decoder = (
            NativeToolTokenDecoder(
                self._tokenizer,
                preserved_tokens = reasoning_control_tokens(reasoning_channel_markers),
            )
            if tools or preserve_native_channels or tool_protocol_active
            else None
        )
        if constraint is not None:
            constraint.decoded_dropping(
                native_token_decoder.dropped_ids() if native_token_decoder is not None else None
            )
        # Consulted per token on the reasoning path below, so resolved once here.
        stop_token_ids = (
            _mlx_stop_token_ids(self._tokenizer, self._model)
            if native_token_decoder is not None
            else ()
        )
        streamed_text = (
            _MlxStreamedText(
                _mlx_stream_detokenizer(self._tokenizer),
                native_token_decoder,
                self._tokenizer,
                stop_token_ids,
            )
            if preserve_native_channels
            else None
        )
        token_ids = []
        normalizer = (
            make_reasoning_normalizer(
                reasoning_channel_markers,
                in_reasoning = prompt_opens_reasoning_channel(
                    prompt, reasoning_channel_markers, _resumed_partial
                ),
            )
            if preserve_native_channels
            else None
        )
        # Match sampled text, ahead of prefill and think rewriting: delivered markup would end turns the model never wrote, and never find a requested native marker
        sequences = _mlx_stop_sequences(stop)
        stopped = False
        sampled = ""
        released = 0
        # MLX diffs cumulative snapshots: keep the prefilled <think> prefix in native-protocol snapshots.
        normalized_output = think_prefix
        with (
            self._generation_lock,
            _temporary_mlx_adapter_state(self._model, _adapter_state),
            self._prefill_decision_scopes(),
            self._generation_scopes(),
        ):
            input_embeddings = None
            if images:
                from unsloth_zoo.mlx.generate import image_prompt_embeddings

                # Any two images of one size expand to the same ids, so the token-keyed cache stays out.
                prompt_tokens, input_embeddings = image_prompt_embeddings(
                    self._model, self._tokenizer, prompt, images
                )
                gen_prompt, prompt_cache, cache_key, cached_n = prompt_tokens, None, None, 0
            else:
                (
                    gen_prompt,
                    prompt_cache,
                    cache_key,
                    prompt_tokens,
                    cached_n,
                ) = self._prepare_prompt_cache(prompt, _adapter_state)
            max_new_tokens = self._generation_limit(prompt, max_new_tokens, prompt_tokens)
            logger.info(
                "Generating: prompt_len=%d, cached=%d, max_tokens=%d, model=%s, tokenizer=%s",
                len(prompt),
                cached_n,
                max_new_tokens,
                type(self._model).__name__,
                type(self._tokenizer).__name__,
            )
            final_response = None
            try:
                # Enter request-scoped model state before yielding any response.
                if think_prefix:
                    yield think_prefix
                gen_kwargs = dict(
                    prompt = gen_prompt,
                    max_tokens = max_new_tokens,
                    sampler = sampler,
                )
                gen_kwargs.update(self._kv_window_generate_kwargs())
                gen_kwargs.update(self._prefill_step_kwargs())
                if prompt_cache is not None:
                    gen_kwargs["prompt_cache"] = prompt_cache
                else:
                    gen_kwargs.update(self._kv_quant_generate_kwargs())
                if input_embeddings is not None:
                    gen_kwargs["input_embeddings"] = input_embeddings
                if logits_processors is not None:
                    gen_kwargs["logits_processors"] = logits_processors
                for response in _stream_until_cancelled(
                    stream_generate,
                    cancel_event,
                    self._model,
                    self._tokenizer,
                    **gen_kwargs,
                ):
                    final_response = response
                    token_ids.append(response.token)
                    if recorded:
                        token_logprobs.append(
                            _mlx_token_logprob(
                                recorded.popleft(), response.token, n_probs, self._tokenizer
                            )
                        )
                    if preserve_native_channels:
                        sampled += streamed_text.feed(response)
                        if sequences:
                            cut, stopped = _mlx_stop_cut(sampled, sequences)
                        else:
                            cut = len(sampled)
                        # Cut before normalizing: normalizer-written markers are this layer's own
                        delta = normalizer.feed(sampled[released:cut])
                        released = cut
                        if delta:
                            normalized_output += delta
                            yield normalized_output
                    else:
                        # Re-decoding every id rebuilds rather than extends, so an invalid byte sequence can revise
                        # characters already shown. Predates stop handling and affects plain replies too.
                        if native_token_decoder is not None:
                            sampled = _native_reply_text(
                                native_token_decoder, token_ids, stop_token_ids, think_prefix
                            )
                        else:
                            sampled = self._tokenizer.decode(
                                token_ids,
                                skip_special_tokens = True,
                                # Cleanup would rewrite " ." inside a grammar-approved string,
                                # and llama-server returns a raw completion's pieces as is.
                                **(
                                    {"clean_up_tokenization_spaces": False}
                                    if constraint is not None or prompt_ids is not None
                                    else {}
                                ),
                            )
                        if not sequences:
                            # A character whose bytes span tokens decodes to U+FFFD until its last token lands, and
                            # consumers diff snapshots by length, so yielding that would replace it for good (#8594).
                            yield think_prefix + sampled.rstrip("\ufffd")
                        else:
                            # Matched every step, delivered once at the end: revised snapshots diffed by length would splice text, and a stream cannot unsend
                            cut, stopped = _mlx_stop_cut(sampled, sequences)
                    if stopped:
                        break

                    if cancel_event and cancel_event.is_set():
                        break
                if prompt_cache is not None and prompt_tokens is not None:
                    history = self._prompt_cache_history
                    if history is not None:
                        try:
                            history.insert(cache_key, prompt_tokens + token_ids, prompt_cache)
                        except Exception as exc:
                            logger.debug("MLX prompt cache insert failed: %s", exc)
            except Exception as e:
                import traceback
                logger.error("stream_generate failed:\n%s", traceback.format_exc())
                raise
            finally:
                # Latched here so a later cancel cannot rewrite the finish reason
                if final_response is not None:
                    self.last_generation_stats = _build_generation_stats(
                        getattr(final_response, "prompt_tokens", 0),
                        getattr(final_response, "prompt_tps", 0.0),
                        getattr(final_response, "generation_tokens", 0),
                        getattr(final_response, "generation_tps", 0.0),
                        cached_n,
                        finish_reason = _mlx_finish_reason(
                            final_response,
                            _mlx_stop_token_ids(self._tokenizer, self._model),
                            getattr(final_response, "generation_tokens", 0),
                            max_new_tokens,
                        ),
                    )
                    if token_logprobs is not None:
                        self.last_generation_stats["token_logprobs"] = token_logprobs
        # The turn's settled text: delivered once for the plain path, as the tail for the native-channel one. Every
        # snapshot was matched as it arrived, so a turn no sequence ended owes all of its text, held-back partial
        # included.
        if streamed_text is not None and not stopped:
            # The flush can carry a whole sequence: it was held back unmatched, not cleared.
            sampled += streamed_text.finish(sampled)
            if sequences:
                cut, stopped = _mlx_stop_cut(sampled, sequences)
        if not sequences and streamed_text is None and sampled.endswith("\ufffd"):
            yield think_prefix + sampled
        if sequences or streamed_text is not None:
            if not stopped:
                cut = len(sampled)
            if normalizer is None:
                settled = think_prefix + sampled[:cut]
                # Settled text equal to the prefill needs no second snapshot; the prefill already went out.
                if settled != think_prefix:
                    yield settled
            else:
                delta = normalizer.feed(sampled[released:cut])
                if delta:
                    normalized_output += delta
                    yield normalized_output
        if normalizer is not None:
            # A sequence ends the turn like a stop token, so a reasoning block it cut inside is closed; only a cancelled turn drains, since more was coming
            cancelled = not stopped and cancel_event is not None and cancel_event.is_set()
            tail = normalizer.drain() if cancelled else normalizer.finish()
            if tail:
                normalized_output += tail
                yield normalized_output
        if stopped:
            self._mark_stopped()

    def _plan_text_row(
        self,
        messages,
        *,
        temperature,
        top_p,
        top_k,
        min_p,
        max_new_tokens,
        repetition_penalty,
        tools = None,
        enable_thinking = None,
        reasoning_effort = None,
        preserve_thinking = None,
        continue_final_message = False,
        presence_penalty = 0.0,
        seed = None,
        frequency_penalty = 0.0,
        logit_bias = None,
        stop = None,
        tool_protocol_active = None,
        prompt_ids = None,
        penalty_window = None,
        response_format = None,
        reasoning_is_extracted = False,
        grammar = None,
    ) -> "_TextRowPlan":
        from mlx_lm.sample_utils import make_sampler

        if prompt_ids is not None:
            prompt, reasoning_channel_markers, think_prefix = list(prompt_ids), None, ""
            _resumed_partial = False
            preserve_native_channels = False
            constraint = _raw_prompt_constraint(grammar, self._tokenizer)
        else:
            render_result = self._render_text_prompt(
                messages,
                tools = tools,
                enable_thinking = enable_thinking,
                reasoning_effort = reasoning_effort,
                preserve_thinking = preserve_thinking,
                continue_final_message = continue_final_message,
            )
            prompt = render_result.prompt
            reasoning_channel_markers = render_result.reasoning_channel_markers
            _resumed_partial = bool(continue_final_message and trailing_assistant_text(messages))

            preserve_native_channels = reasoning_channel_markers is not None
            think_close_survives = (
                bool(tools) or tool_protocol_active or preserve_native_channels
            ) and decoder_preserves_token(
                self._tokenizer, "</think>", reasoning_control_tokens(reasoning_channel_markers)
            )
            think_prefix = _think_prefix(
                prompt,
                getattr(self._tokenizer, "all_special_tokens", None),
                messages,
                continue_final_message,
                preserves_think_close = think_close_survives,
            )
            constraint = _build_grammar_constraint(
                response_format,
                self._tokenizer,
                prompt,
                reasoning_markers = reasoning_channel_markers,
                reasoning_is_extracted = reasoning_is_extracted,
                reply_keeps_special_tokens = preserve_native_channels or think_close_survives,
            )
            if constraint is not None and not constraint.allows_reasoning:
                think_prefix = ""
                preserve_native_channels = False
        if seed is None:
            sampler = make_sampler(
                temp = temperature,
                top_p = top_p,
                top_k = int(top_k or 0),
                min_p = float(min_p or 0.0),
                min_tokens_to_keep = 1,
            )
        else:
            sampler = _make_seeded_mlx_sampler(
                seed,
                temp = temperature,
                top_p = top_p,
                top_k = int(top_k or 0),
                min_p = float(min_p or 0.0),
            )
        logits_processors = _mlx_sampling_processors(
            repetition_penalty = repetition_penalty,
            presence_penalty = presence_penalty,
            frequency_penalty = frequency_penalty,
            logit_bias = logit_bias,
            grammar_constraint = constraint,
            penalty_window = penalty_window,
            penalty_prompt = prompt_ids,
        )

        normalizer = (
            make_reasoning_normalizer(
                reasoning_channel_markers,
                in_reasoning = prompt_opens_reasoning_channel(
                    prompt, reasoning_channel_markers, _resumed_partial
                ),
            )
            if preserve_native_channels
            else None
        )
        # As _generate_text decodes the reply, so a batched turn shows the same native controls.
        native_token_decoder = (
            NativeToolTokenDecoder(
                self._tokenizer,
                preserved_tokens = reasoning_control_tokens(reasoning_channel_markers),
            )
            if tools or preserve_native_channels or tool_protocol_active
            else None
        )
        if constraint is not None:
            constraint.decoded_dropping(
                native_token_decoder.dropped_ids() if native_token_decoder is not None else None
            )
        stop_token_ids = (
            _mlx_stop_token_ids(self._tokenizer, self._model)
            if native_token_decoder is not None
            else ()
        )
        streamed = (
            _MlxStreamedText(
                _mlx_stream_detokenizer(self._tokenizer),
                native_token_decoder,
                self._tokenizer,
                stop_token_ids,
            )
            if preserve_native_channels
            else None
        )
        _require_row_detokenizer(streamed)
        if streamed is not None:
            decode = None
        elif native_token_decoder is not None:
            decode = lambda ids: _native_reply_text(
                native_token_decoder, ids, stop_token_ids, think_prefix
            )
        else:
            decode = lambda ids: self._tokenizer.decode(
                ids,
                skip_special_tokens = True,
                **(
                    {"clean_up_tokenization_spaces": False}
                    if constraint is not None or prompt_ids is not None
                    else {}
                ),
            )
        row = _RowStream(
            prefix = think_prefix,
            sequences = _mlx_stop_sequences(stop),
            normalizer = normalizer,
            decode = decode,
        )
        return _TextRowPlan(
            prompt = prompt,
            think_prefix = think_prefix,
            stream = row,
            streamed = streamed,
            sampler = sampler,
            processors = logits_processors,
            max_tokens = (
                self._unset_generation_budget(prompt, prompt_ids)
                if max_new_tokens is None
                else max_new_tokens
            ),
        )

    def _render_vlm_prompt(
        self,
        messages,
        images,
        *,
        videos = None,
        tools = None,
        enable_thinking = None,
        reasoning_effort = None,
        preserve_thinking = None,
        continue_final_message = False,
    ):
        """Render the prompt a vision generation sends, and the target that rendered it. Shared with
        counting, as _render_text_prompt is for text models, so a count cannot price a prompt the
        model never sees. A text-only conversation passes images=None."""
        from core.inference.chat_template_helpers import (
            apply_chat_template_for_generation,
            chat_render_target,
        )

        # Same render target as the healing catalog the route builds (#7066)
        chat_target = (
            _video_render_target(self._processor) if videos else chat_render_target(self._processor)
        )

        attached_images = 0 if images is None else len(images)
        structured_images = sum(
            _count_vlm_images(message.get("content"))
            for message in messages
            if isinstance(message, dict)
        )
        if structured_images != attached_images:
            raise RuntimeError(
                f"VLM conversation contains {structured_images} structured image "
                f"item(s) for {attached_images} attached image(s)."
            )
        attached_videos = 0 if videos is None else len(videos)
        structured_videos = sum(
            _count_vlm_videos(message.get("content"))
            for message in messages
            if isinstance(message, dict)
        )
        if structured_videos != attached_videos:
            raise RuntimeError(
                f"VLM conversation contains {structured_videos} structured video "
                f"item(s) for {attached_videos} attached video(s)."
            )
        prompt = None
        has_tool_history = _vlm_messages_have_tool_history(messages)
        prompt_error = None
        try:
            prompt = apply_chat_template_for_generation(
                chat_target,
                messages,
                tools = tools,
                enable_thinking = enable_thinking,
                reasoning_effort = reasoning_effort,
                preserve_thinking = preserve_thinking,
                continue_final_message = continue_final_message,
            )
        except Exception as exc:
            if images is None or has_tool_history:
                raise
            prompt_error = exc
        prompt_issue = (
            _vlm_prompt_issue(prompt, messages) if prompt_error is None else "a rendering error"
        )
        if prompt_issue and has_tool_history:
            raise RuntimeError(
                f"VLM chat template returned {prompt_issue} and cannot be recovered "
                "without dropping tool-call history."
            ) from prompt_error
        if prompt_issue and attached_videos:
            # mlx-vlm's registered renderers take image and audio counts only.
            raise RuntimeError(
                f"VLM chat template returned {prompt_issue} and cannot be recovered "
                "for a video turn."
            ) from prompt_error

        if images is not None and prompt_issue:
            if tools or any(
                value is not None
                for value in (enable_thinking, reasoning_effort, preserve_thinking)
            ):
                if prompt_error is not None:
                    raise prompt_error
                raise RuntimeError(
                    f"VLM chat template returned {prompt_issue} and cannot be recovered "
                    "without dropping requested tools or reasoning controls."
                )
            try:
                recovered_prompt = _render_registered_vlm_prompt(
                    self._processor,
                    self._model,
                    messages,
                    len(images),
                    continue_final_message = continue_final_message,
                )
            except Exception as recovery_error:
                if prompt_error is not None:
                    raise prompt_error
                raise RuntimeError(
                    f"VLM chat template returned {prompt_issue}; model-aware "
                    f"recovery failed: {recovery_error}"
                ) from recovery_error
            if recovered_prompt is None:
                if prompt_error is not None:
                    raise prompt_error
                raise RuntimeError(
                    f"VLM chat template returned {prompt_issue}, and no registered "
                    "MLX VLM renderer was available for this model."
                )
            recovered_issue = _vlm_prompt_issue(recovered_prompt, messages)
            if recovered_issue:
                if prompt_error is not None:
                    raise prompt_error
                raise RuntimeError(
                    f"Model-aware VLM rendering returned {recovered_issue} for "
                    f"{attached_images} attached image(s)."
                )
            prompt = recovered_prompt
        elif prompt_issue:
            raise RuntimeError(f"VLM chat template returned {prompt_issue}.") from prompt_error
        # mlx-vlm binds pixel_values[i] to the i-th image token: miscounted markers answer wrong.
        if attached_images > 1 and self._multi_image_marker:
            marked = prompt.count(self._multi_image_marker)
            if marked != attached_images:
                raise RuntimeError(
                    f"VLM chat template marked {marked} image(s) for {attached_images} attached."
                )

        markers = detect_reasoning_channel_markers(chat_target, tools = tools)
        if not self._reads_vision_input():
            # No images means no image tokens to place, so skipping this would drop the tool schema.
            from core.inference.chat_template_helpers import (
                render_with_native_template_fallback,
            )

            model_info = self.models.get(self.active_model_name, {})
            recovered = render_with_native_template_fallback(
                formatted_prompt = prompt,
                tokenizer = self._tokenizer,
                model_info = model_info,
                active_model_name = self.active_model_name,
                messages = messages,
                tools = tools,
                enable_thinking = enable_thinking,
                reasoning_effort = reasoning_effort,
                preserve_thinking = preserve_thinking,
                continue_final_message = continue_final_message,
                hf_token = model_info.get("hf_token"),
                return_metadata = True,
            )
            prompt, markers = recovered.prompt, recovered.reasoning_channel_markers
        return prompt, chat_target, markers

    def _media_generation_limit(self, prompt, max_new_tokens, prompt_ids, images, videos):
        # Counted from a temp file (base64 is unreadable to the processor), discarded here.
        counted_clips = []
        try:
            if getattr(self, "_kv_context_budget", None):
                for clip in videos or ():
                    counted_clips.append(_write_video_clip(clip))
            return self._generation_limit(
                prompt,
                max_new_tokens,
                prompt_ids,
                images = images,
                videos = counted_clips or None,
                cap = UNSET_GENERATION_BUDGET if images or videos else None,
            )
        finally:
            for counted_clip in counted_clips:
                _discard_video_clip(counted_clip)

    def _plan_vlm_request(
        self,
        request,
        *,
        decodes_ids = False,
    ) -> "_VLMRowPlan":
        image = request.get("image")
        videos = request.get("videos") or None
        messages = request.get("messages") or []
        conversation = (
            []
            if request.get("prompt_ids") is not None
            else self._conversation_to_render(
                messages, request.get("system_prompt", ""), image, video = len(videos or ())
            )
        )
        return self._plan_vlm_row(
            conversation,
            image,
            tool_protocol_active = request.get("tool_protocol_active"),
            response_format = request.get("response_format"),
            reasoning_is_extracted = request.get("reasoning_is_extracted", False),
            decodes_ids = decodes_ids,
            videos = videos,
            **_row_kwargs(request),
        )

    def _plan_vlm_row(
        self,
        messages,
        image,
        *,
        temperature,
        top_p,
        top_k,
        min_p,
        max_new_tokens,
        repetition_penalty,
        tools = None,
        enable_thinking = None,
        reasoning_effort = None,
        preserve_thinking = None,
        continue_final_message = False,
        presence_penalty = 0.0,
        seed = None,
        frequency_penalty = 0.0,
        logit_bias = None,
        stop = None,
        prompt_ids = None,
        penalty_window = None,
        tool_protocol_active = None,
        response_format = None,
        reasoning_is_extracted = False,
        decodes_ids = False,
        videos = None,
    ) -> "_VLMRowPlan":
        """``decodes_ids``: the batch reports each row's ids, so the reply is decoded from them as
        ``_generate_vlm`` decodes it; otherwise from the batch's text, which keeps no native controls."""
        images = [image] if image is not None else None
        constraint = token_decoder = None
        if prompt_ids is not None:
            prompt, markers, think_prefix = list(prompt_ids), None, ""
        else:
            prompt, chat_target, markers = self._render_vlm_prompt(
                messages,
                images,
                videos = videos,
                tools = tools,
                enable_thinking = enable_thinking,
                reasoning_effort = reasoning_effort,
                preserve_thinking = preserve_thinking,
                continue_final_message = continue_final_message,
            )

            native = decodes_ids and bool(tools or tool_protocol_active or markers is not None)
            think_prefix = _think_prefix(
                prompt,
                getattr(chat_target, "all_special_tokens", None),
                messages,
                continue_final_message,
                preserves_think_close = native
                and decoder_preserves_token(self._tokenizer, "</think>"),
            )
            if decodes_ids:
                constraint = _build_grammar_constraint(
                    response_format,
                    chat_target,
                    prompt,
                    reasoning_markers = markers,
                    tools = tools,
                    reasoning_is_extracted = reasoning_is_extracted,
                    reply_keeps_special_tokens = True,
                )
            if constraint is not None and not constraint.allows_reasoning:
                think_prefix = ""
            elif constraint is not None:
                from core.inference.chat_template_helpers import detect_think_prefill
                think_prefix = detect_think_prefill(prompt, preserves_think_close = True)
            if constraint is not None and _vlm_generation_is_diffusion(self._model):
                from core.inference.grammar_constraint import ResponseFormatError
                raise ResponseFormatError(_DIFFUSION_RESPONSE_FORMAT)
            if native and self._tokenizer:
                token_decoder = NativeToolTokenDecoder(
                    self._tokenizer,
                    preserved_tokens = reasoning_control_tokens(markers),
                )
        normalizer = (
            make_reasoning_normalizer(
                markers,
                in_reasoning = prompt_opens_reasoning_channel(
                    prompt,
                    markers,
                    bool(continue_final_message and trailing_assistant_text(messages)),
                ),
            )
            # The normalizer would rewrite marker text inside the document.
            if markers is not None and (constraint is None or constraint.allows_reasoning)
            else None
        )
        row = _RowStream(
            prefix = think_prefix,
            sequences = _mlx_stop_sequences(stop),
            normalizer = normalizer,
        )
        max_new_tokens = self._media_generation_limit(
            prompt, max_new_tokens, prompt_ids, images, videos
        )
        vlm_kwargs = dict(
            max_tokens = max_new_tokens,
            temperature = temperature,
            top_p = top_p,
            top_k = int(top_k or 0),
            min_p = float(min_p or 0.0),
        )
        vlm_kwargs.update(self._kv_window_generate_kwargs())
        if seed is not None:
            vlm_kwargs["sampler"] = _make_seeded_mlx_sampler(
                seed,
                temp = temperature,
                top_p = top_p,
                top_k = int(top_k or 0),
                min_p = float(min_p or 0.0),
            )
        processors = _mlx_sampling_processors(
            repetition_penalty = repetition_penalty,
            presence_penalty = presence_penalty,
            frequency_penalty = frequency_penalty,
            logit_bias = logit_bias,
            grammar_constraint = constraint,
            penalty_window = penalty_window,
            penalty_prompt = prompt_ids,
        )
        if processors:
            # Not the repetition_penalty shortcut: mlx-vlm would build it outside our narrowing
            vlm_kwargs["logits_processors"] = _row_logits_processors(processors)
        streamed = None
        if constraint is not None:
            # mlx-vlm stops on the config's ids, which some repos set apart from the tokenizer's.
            constraint.stops_on(_mlx_stop_token_ids(self._tokenizer, self._model))
            if token_decoder is not None:
                constraint.decoded_dropping(token_decoder.dropped_ids())
        if token_decoder is not None:
            streamed = _MlxStreamedText(
                _mlx_stream_detokenizer(self._processor)
                or _mlx_stream_detokenizer(self._tokenizer),
                token_decoder,
                self._tokenizer,
                _mlx_stop_token_ids(self._tokenizer, self._model),
            )
            _require_row_detokenizer(streamed)

        return _VLMRowPlan(
            prompt = prompt,
            images = images,
            videos = videos,
            think_prefix = think_prefix,
            stream = row,
            kwargs = vlm_kwargs,
            sampling = {
                "temperature": vlm_kwargs["temperature"],
                "top_p": vlm_kwargs["top_p"],
                "top_k": max(vlm_kwargs["top_k"], 0),
                "min_p": vlm_kwargs["min_p"],
                "seed": seed,
            },
            max_tokens = vlm_kwargs["max_tokens"],
            processors = processors,
            streamed = streamed,
            # As _generate_vlm, which drops the draft for tool turns.
            speculates = not (tools or tool_protocol_active),
        )

    def _generate_vlm(
        self,
        messages,
        images,
        temperature,
        top_p,
        top_k,
        min_p,
        max_new_tokens,
        repetition_penalty,
        cancel_event,
        *,
        tools = None,
        enable_thinking = None,
        reasoning_effort = None,
        preserve_thinking = None,
        continue_final_message = False,
        presence_penalty = 0.0,
        seed = None,
        frequency_penalty = 0.0,
        logit_bias = None,
        _adapter_state = None,
        tool_protocol_active = None,
        stop = None,
        videos = None,
        response_format = None,
        reasoning_is_extracted = False,
        prompt_ids = None,
        penalty_window = None,
        grammar = None,
        n_probs = 0,
    ):
        from mlx_vlm import stream_generate as vlm_stream

        from core.inference.chat_template_helpers import detect_think_prefill

        images = list(images) if images else None
        if prompt_ids is not None:
            prompt, chat_target, vlm_reasoning_markers, prefill = "", None, None, ""
        else:
            prompt, chat_target, vlm_reasoning_markers = self._render_vlm_prompt(
                messages,
                images,
                videos = videos or None,
                tools = tools,
                enable_thinking = enable_thinking,
                reasoning_effort = reasoning_effort,
                preserve_thinking = preserve_thinking,
                continue_final_message = continue_final_message,
            )
            # Detected once: the decoder keeps the delimiters the normalizer below consumes.
            prefill = _think_prefix(
                prompt,
                getattr(chat_target, "all_special_tokens", None),
                messages,
                continue_final_message,
                # The same activation the decoder below uses: in unrestricted mode ``tools`` is
                # empty while the protocol is live, so ``bool(tools)`` said the closer would be
                # stripped, the opener was suppressed, and the stream ran on to an orphan
                # ``</think>``. Mirrors the text path.
                preserves_think_close = (
                    bool(tools) or tool_protocol_active or vlm_reasoning_markers is not None
                )
                and decoder_preserves_token(self._tokenizer, "</think>"),
            )
        vlm_continued = bool(continue_final_message and trailing_assistant_text(messages))
        # Matched on the sampled text, for the reason _generate_text gives.
        sequences = _mlx_stop_sequences(stop)
        stopped = False
        max_new_tokens = self._media_generation_limit(
            prompt, max_new_tokens, prompt_ids, images, videos
        )
        logger.info(
            "VLM generating: prompt_len=%d, images=%d, videos=%d",
            len(prompt),
            len(images or ()),
            len(videos or ()),
        )
        # generate_step expects temperature=; temp= is silently ignored (stuck at greedy)
        vlm_kwargs = dict(
            max_tokens = max_new_tokens,
            temperature = temperature,
            top_p = top_p,
            top_k = int(top_k or 0),
            min_p = float(min_p or 0.0),
        )
        vlm_kwargs.update(self._kv_window_generate_kwargs())
        if seed is not None:
            # generate_step builds its sampler only if sampler is None, so a seeded one must carry the full temp/top_p/min_p/top_k chain or they go silently off
            vlm_kwargs["sampler"] = _make_seeded_mlx_sampler(
                seed,
                temp = temperature,
                top_p = top_p,
                top_k = int(top_k or 0),
                min_p = float(min_p or 0.0),
            )
        _rep_active = repetition_penalty is not None and float(repetition_penalty) not in (
            0.0,
            1.0,
        )
        constraint = (
            _raw_prompt_constraint(grammar, self._tokenizer)
            if prompt_ids is not None
            else _build_grammar_constraint(
                response_format,
                chat_target,
                prompt,
                reasoning_markers = vlm_reasoning_markers,
                tools = tools,
                reasoning_is_extracted = reasoning_is_extracted,
                reply_keeps_special_tokens = True,
            )
        )
        document_only = constraint is not None and not constraint.allows_reasoning
        if document_only:
            prefill = ""
        elif constraint is not None:
            # The grammar lets a kept </think> close the block: re-emit the opener or it all reads as content.
            prefill = detect_think_prefill(prompt, preserves_think_close = True)
        if constraint is not None and _vlm_generation_is_diffusion(self._model):
            from core.inference.grammar_constraint import ResponseFormatError
            raise ResponseFormatError(_DIFFUSION_RESPONSE_FORMAT)
        recorded = token_logprobs = None
        if n_probs:
            if _vlm_generation_is_diffusion(self._model):
                raise LogprobsUnsupported(
                    "logprobs is not supported on this model: it generates by diffusion rather "
                    "than one token at a time."
                )
            recorder, recorded = _make_mlx_raw_logprob_recorder()
            token_logprobs = []
        draft = self._speculative_draft
        if (
            presence_penalty
            or frequency_penalty
            or logit_bias
            or constraint is not None
            or (_rep_active and draft is not None)
            or penalty_window is not None
            or recorded is not None
        ):
            # These need custom processors: pass the full list (repetition + the rest) instead of the
            # repetition_penalty shortcut so all apply.
            vlm_kwargs["logits_processors"] = _mlx_sampling_processors(
                repetition_penalty = repetition_penalty,
                presence_penalty = presence_penalty,
                frequency_penalty = frequency_penalty,
                logit_bias = logit_bias,
                grammar_constraint = constraint,
                penalty_window = penalty_window,
                penalty_prompt = prompt_ids,
            )
            if recorded is not None:
                # Not history_only, so no draft runs: its verify pass would record rejected tokens.
                vlm_kwargs["logits_processors"] = [
                    recorder,
                    *(vlm_kwargs["logits_processors"] or ()),
                ]
        elif _rep_active:
            vlm_kwargs["repetition_penalty"] = float(repetition_penalty)

        # Keep native tool controls and reasoning delimiters, as the text path does, so the parser can tell a wrapped call from markerless prose
        vlm_token_decoder = (
            NativeToolTokenDecoder(
                self._tokenizer,
                preserved_tokens = reasoning_control_tokens(vlm_reasoning_markers),
            )
            if (tools or tool_protocol_active or vlm_reasoning_markers is not None)
            and self._tokenizer
            else None
        )
        if constraint is not None:
            # mlx-vlm stops on the config's ids, which some repos set apart from the tokenizer's.
            constraint.stops_on(_mlx_stop_token_ids(self._tokenizer, self._model))
        if constraint is not None and vlm_token_decoder is not None:
            constraint.decoded_dropping(vlm_token_decoder.dropped_ids())
        # Runtime EOS may be an allowlisted control and would trail each answer
        vlm_stop_ids = (
            _mlx_stop_token_ids(self._tokenizer, self._model)
            if vlm_token_decoder is not None
            else ()
        )
        vlm_detokenizer = (
            _mlx_stream_detokenizer(self._processor) or _mlx_stream_detokenizer(self._tokenizer)
            if vlm_token_decoder is not None
            else None
        )
        vlm_streamed_text = (
            _MlxStreamedText(vlm_detokenizer, vlm_token_decoder, self._tokenizer, vlm_stop_ids)
            if vlm_token_decoder is not None
            else None
        )

        if prompt_ids is not None:
            import mlx.core as mx
            vlm_kwargs["input_ids"] = mx.array([list(prompt_ids)])
            # As the text path and llama-server: control tokens are not reply text.
            vlm_kwargs["skip_special_tokens"] = True
        elif not self._reads_vision_input():
            # mlx-vlm adds special tokens by family, doubling a BOS the template already emitted.
            import mlx.core as mx
            vlm_kwargs["input_ids"] = mx.array([self._encode_prompt(prompt)])
        session = self._vlm_prompt_cache_session(
            _adapter_state, images, prompt, has_video = bool(videos)
        )
        vlm_kwargs.update(self._kv_runtime_quant_kwargs())
        if session is not None:
            vlm_kwargs["prompt_cache"] = session.cache
            vlm_kwargs["prompt_cache_state"] = session
            if session.media_block is not None:
                vlm_kwargs.update(session.media_block.generate_kwargs())
            # Reused and unreused turns must prefill on the same grid to match.
            vlm_kwargs["prefill_step_size"] = session.window
        else:
            vlm_kwargs.update(self._kv_quant_generate_kwargs())
            vlm_kwargs.update(self._prefill_step_kwargs())
        session_scope = session if session is not None else nullcontext()
        processors = vlm_kwargs.get("logits_processors") or ()
        # The draft is told these processors, and takes only ones that read just the history (a grammar's
        # matcher advances per call); tool turns decode as before; a clip is nothing a drafter was built for.
        if (
            draft is None
            or tools
            or tool_protocol_active
            or videos
            or not all(getattr(processor, "history_only", False) for processor in processors)
            or _vlm_generation_is_diffusion(self._model)
        ):
            draft = None
        elif "input_ids" not in vlm_kwargs:
            vlm_kwargs.update(
                _vlm_generate_kwargs(
                    _vlm_prepared_inputs(self._model, self._processor, prompt, images)
                )
            )

        def _stream_vlm_snapshots():
            nonlocal stopped
            sampled = ""
            released = 0
            # Hold the lock and adapter state for the whole stream (Base-vs-LoRA compare)
            with (
                self._generation_lock,
                _temporary_mlx_adapter_state(self._model, _adapter_state),
                self._prefill_decision_scopes(),
                ExitStack() as generation_scope,
                session_scope,
            ):
                if session is None and (images or videos):
                    # Free headroom for the vision pass under the lock; a video turn resumes no snapshot, so retained ones are pure occupancy
                    self._release_vlm_snapshots()
                generation_scope.enter_context(
                    self._generation_scopes("vision", *(() if draft is None else ("speculative",)))
                )
                final_response = None
                clip_paths = []
                if draft is not None:
                    from unsloth_zoo.mlx.generate import SamplingParams

                    # Under the lock: the load's one draft serves one generation at a time.
                    draft.prepare(
                        vlm_kwargs["input_ids"][0].tolist(),
                        SamplingParams(
                            temperature = temperature,
                            top_p = top_p,
                            top_k = max(int(top_k or 0), 0),
                            min_p = float(min_p or 0.0),
                            seed = seed,
                        ),
                        processors,
                    )
                    vlm_kwargs.update(draft_model = draft, draft_kind = draft.draft_kind)
                try:
                    if videos:
                        for clip in videos:
                            clip_paths.append(_write_video_clip(clip))
                        vlm_kwargs["video"] = clip_paths
                        frame_rate = _video_sampling_fps(clip_paths, self._processor)
                        if frame_rate is not None:
                            vlm_kwargs["fps"] = frame_rate
                    # Emit prefilled <think> first; inside the adapter so unsupported requests raise before any output.
                    if prefill:
                        yield prefill
                    with closing(
                        _iter_vlm_responses(
                            vlm_stream(
                                self._model,
                                self._processor,
                                prompt,
                                images,
                                **vlm_kwargs,
                            )
                        )
                    ) as responses:
                        raw_ids = [] if prompt_ids is not None else None
                        for response in responses:
                            final_response = response
                            raw_id = (
                                getattr(response, "token", None) if raw_ids is not None else None
                            )
                            filled = getattr(response, "generation_tokens", None)
                            if raw_id is not None and (
                                filled is None or int(filled) == len(raw_ids) + 1
                            ):
                                # The streaming detokenizer trims the first leading space; decode the ids.
                                raw_ids.append(int(raw_id))
                                if recorded:
                                    token_logprobs.append(
                                        _mlx_token_logprob(
                                            recorded.popleft(),
                                            int(raw_id),
                                            n_probs,
                                            self._tokenizer,
                                        )
                                    )
                                decoded = self._tokenizer.decode(
                                    raw_ids,
                                    skip_special_tokens = True,
                                    clean_up_tokenization_spaces = False,
                                ).rstrip("\ufffd")
                                token_text = (
                                    decoded[len(sampled) :] if decoded.startswith(sampled) else ""
                                )
                            elif (
                                raw_ids is not None
                                and filled is not None
                                and int(filled) <= len(raw_ids)
                            ):
                                token_text = ""
                            elif vlm_streamed_text is not None:
                                token_text = vlm_streamed_text.feed(response)
                            else:
                                # A block naming several ids (a denoised step): the runtime's text stands.
                                raw_ids = None
                                token_text = (
                                    response.text if hasattr(response, "text") else str(response)
                                )
                            sampled += token_text
                            if not sequences:
                                yield prefill + sampled
                            else:
                                cut, stopped = _mlx_stop_cut(sampled, sequences)
                                # These deltas only append, so the cut never moves back over text already released.
                                if cut > released:
                                    released = cut
                                    yield prefill + sampled[:cut]
                                if stopped:
                                    break
                            if cancel_event and cancel_event.is_set():
                                break
                    if vlm_streamed_text is not None and not stopped:
                        tail = vlm_streamed_text.finish(sampled)
                        if tail:
                            sampled += tail
                            if not sequences:
                                yield prefill + sampled
                            else:
                                cut, stopped = _mlx_stop_cut(sampled, sequences)
                                if cut > released:
                                    released = cut
                                    yield prefill + sampled[:cut]
                    # As in _generate_text: what was withheld is ordinary text now.
                    if sequences and not stopped and released < len(sampled):
                        yield prefill + sampled
                finally:
                    for clip_path in clip_paths:
                        _discard_video_clip(clip_path)
                    cached_n = produced_n = 0
                    produced_s = 0.0
                    if session is not None:
                        # Not from mlx-vlm's hook, which a stop or cancel never reaches.
                        try:
                            session.finish()
                        except Exception as exc:
                            logger.debug("MLX VLM prompt cache: snapshot not stored (%s)", exc)
                        # What mlx-vlm reused, less the rows produced here as prefill.
                        cached_n = int(getattr(final_response, "cached_tokens", 0) or 0)
                        produced_n, produced_s = session.produced_tokens, session.produced_seconds
                        cached_n = max(cached_n - produced_n, 0)
                    # mlx_vlm exposes the same stats fields as mlx_lm, minus a finish reason, so that one is derived.
                    if final_response is not None:
                        tokenizer = getattr(self._processor, "tokenizer", self._processor)
                        stop_ids = _mlx_stop_token_ids(tokenizer, self._model)
                        # mlx-vlm rates the full prompt against its prefill, which omits rows the session produced first.
                        prompt_n = int(getattr(final_response, "prompt_tokens", 0) or 0)
                        prefilled_n = max(prompt_n - cached_n, 0)
                        prompt_tps = float(getattr(final_response, "prompt_tps", 0.0) or 0.0)
                        if prompt_tps and (cached_n or produced_s):
                            prompt_tps = prefilled_n / (prompt_n / prompt_tps + produced_s)
                        self.last_generation_stats = _build_generation_stats(
                            prefilled_n,
                            prompt_tps,
                            getattr(final_response, "generation_tokens", 0),
                            getattr(final_response, "generation_tps", 0.0),
                            cached_n,
                            finish_reason = _mlx_finish_reason(
                                final_response,
                                stop_ids,
                                getattr(final_response, "generation_tokens", 0),
                                max_new_tokens,
                            ),
                            drafts = None
                            if draft is None
                            else (draft.draft_n, draft.draft_n_accepted),
                        )
                        if token_logprobs is not None:
                            self.last_generation_stats["token_logprobs"] = token_logprobs

        if document_only or prompt_ids is not None:
            # The normalizer would rewrite marker text inside the document.
            yield from _stream_vlm_snapshots()
        else:
            yield from normalize_reasoning_snapshots(
                _stream_vlm_snapshots(),
                chat_target,
                cancel_event,
                markers = vlm_reasoning_markers,
                tools = tools,
                prompt = prompt,
                continued = vlm_continued,
                ended = lambda: stopped,
            )
        if stopped:
            self._mark_stopped()

    def _kv_policy_batch_reason(self, resident = False):
        if self._kv_quant_bits() is None and not getattr(self, "_kv_context_budget", None):
            return None
        # Only resident vision rows carry their own cache, where the quantization lives.
        if resident and self._is_vlm:
            return _row_quantized_cache_gap()
        return "the load quantizes or budgets its KV cache, which a batch does not carry"

    def batch_unavailable_reason(self, requests):
        if self._model is None:
            return "no model is loaded"
        for request in requests:
            reason = _request_batch_gap(request)
            if reason is not None:
                return reason
        reason = self._video_batch_gap(requests)
        if reason is not None:
            return reason
        reason = self._kv_policy_batch_reason()
        if reason is not None:
            return reason
        if len(requests) < 2:
            return "fewer than two replies were requested"
        if self._is_vlm:
            reason = self._vlm_batch_unavailable_reason(requests)
            if reason is None and self._speculative_draft is not None:
                reason = self._vlm_resident_unavailable_reason({})
            if reason is not None:
                return reason
            if any(_mlx_stop_sequences(request.get("stop")) for request in requests):
                return "a reply asks for stop sequences"
            return None
        return self._text_batch_unavailable_reason()

    def resident_unavailable_reason(self, request):
        if self._model is None:
            return "no model is loaded"
        reason = _request_batch_gap(request)
        if reason is None and _carries_video(request):
            # Whether a clip's row fits the open batch is known only once its frames are decoded.
            reason = "a reply carries a video clip"
        if reason is not None:
            return reason
        reason = self._kv_policy_batch_reason(resident = True)
        if reason is not None:
            return reason
        if self._is_vlm:
            return self._vlm_resident_unavailable_reason(request)
        return self._text_batch_unavailable_reason()

    def _video_batch_gap(self, requests):
        if not any(map(_carries_video, requests)):
            return None
        # The one-at-a-time decode refuses these clips by name.
        if not self._is_vlm or not self._reads_vision_input() or not _mlx_vlm_decodes_video():
            return "a reply carries a video clip this load does not read"
        if self._speculative_draft is not None:
            return "a video clip decodes without the draft"
        try:
            from unsloth_zoo.mlx import generate as engine
        except ImportError:
            return "the installed unsloth-zoo has no MLX generation"
        if "videos" not in _engine_fields(engine, "GenerationRequest"):
            return "the installed unsloth-zoo cannot batch a video clip"
        return None

    def _text_batch_unavailable_reason(self):
        """Why the installed mlx-lm cannot batch this load's replies, or None."""
        try:
            from unsloth_zoo.mlx.generate import stream_unavailable_reason
        except ImportError:
            return "the installed unsloth-zoo cannot keep a batch open"
        return stream_unavailable_reason(self._model, self._tokenizer)

    def _vlm_resident_unavailable_reason(self, request):
        reason = self._vlm_batch_unavailable_reason([request], resident = True)
        if reason is not None:
            return reason
        try:
            from unsloth_zoo.mlx.generate import stream_unavailable_reason
        except ImportError:
            return "the installed unsloth-zoo cannot keep a vision batch open"
        return stream_unavailable_reason(self._model, self._processor)

    def open_resident_batch(
        self,
        *,
        width,
        adapter_state = None,
    ):
        reason = self.resident_unavailable_reason({})
        if reason is not None:
            raise RuntimeError(f"MLX batched generation is unavailable: {reason}")
        session = _VisionBatchSession if self._is_vlm else _TextBatchSession
        return session(
            self,
            width = width,
            adapter_state = adapter_state,
            owns_model = True,
        )

    def _vision_batch_is_available(self) -> bool:
        """Whether this load will decode vision replies together, asked without a request."""
        if not self._is_vlm:
            return False
        return (
            self.batch_unavailable_reason([{}, {}]) is None
            or self.resident_unavailable_reason({}) is None
        )

    def _vlm_batch_unavailable_reason(
        self,
        requests,
        *,
        resident = False,
    ):
        try:
            from unsloth_zoo.mlx import generate as engine
        except ImportError:
            return "the installed unsloth-zoo has no MLX generation"
        gap = _batched_vision_engine_gap(engine)
        if gap is not None:
            return gap
        gap = _row_processor_gap(engine, requests)
        if gap is not None:
            return gap
        # Only the resident session decodes rows from their ids, as the one-at-a-time decode does.
        decodes_ids = resident and _vision_batch_reports_ids(engine, self._speculative_draft)
        if not decodes_ids and any(
            request.get("tools") or request.get("tool_protocol_active") for request in requests
        ):
            return "a reply runs the tool protocol"
        if not decodes_ids and any(_constrained(request) for request in requests):
            return "a reply is constrained to a response format"
        if any(request.get("prompt_ids") is not None for request in requests):
            # The batch detokenizer keeps control tokens and trims the first leading space.
            return "a raw prompt on a vision model decodes on its own"
        if not getattr(self._model, "_is_vlm_model", False):
            return "this model was not loaded as a vision model"
        return None

    def generate_chat_batch(
        self,
        requests,
        *,
        cancel_event = None,
        _adapter_state = None,
    ):
        reason = self.batch_unavailable_reason(requests)
        if reason is not None:
            raise RuntimeError(f"MLX batched generation is unavailable: {reason}")

        self.last_batch_generation_stats = [None] * len(requests)
        self.last_generation_stats = None

        kwargs = {"cancel_event": cancel_event, "_adapter_state": _adapter_state}
        if not self._is_vlm:
            yield from self._generate_session_batch(requests, _TextBatchSession, **kwargs)
        elif self._speculative_draft is not None:  # stream_batch has no speculative mode
            yield from self._generate_session_batch(requests, _VisionBatchSession, **kwargs)
        else:
            yield from self._generate_vlm_batch(requests, **kwargs)

    def _generate_session_batch(
        self,
        requests,
        session_type,
        *,
        cancel_event = None,
        _adapter_state = None,
    ):
        def report(events):
            for row, snapshot in events:
                if isinstance(snapshot, BaseException):
                    # Every row answers one command, which fails as a single reply would.
                    raise snapshot
                if snapshot is None:
                    self.last_batch_generation_stats[row] = session.take_stats(row)
                yield row, snapshot

        with self._generation_lock, _temporary_mlx_adapter_state(self._model, _adapter_state):
            session = session_type(
                self,
                width = len(requests),
                adapter_state = _adapter_state,
            )
            try:
                prefixes = [
                    (row, session.admit(request, row)) for row, request in enumerate(requests)
                ]
                for row, prefix in prefixes:
                    if prefix:
                        yield row, prefix
                while session.rows_in_flight:
                    yield from report(session.step(cancel_event and cancel_event.is_set))
                    if cancel_event is not None and cancel_event.is_set():
                        yield from report(session.withdraw(list(session.handles)))
            finally:
                session.close()

    def _vlm_batch_prompt(self, prompt):
        if isinstance(prompt, list):
            return {"prompt_token_ids": prompt}
        from unsloth_zoo.mlx.generate import vlm_batch_adds_special_tokens

        bos = getattr(self._tokenizer, "bos_token", None)
        if (
            not self._reads_vision_input()
            and bos
            and prompt.startswith(bos)
            and vlm_batch_adds_special_tokens(self._model, self._processor)
        ):
            # The single path encodes a text load's prompt itself, keeping only the template's BOS.
            return {"prompt": prompt[len(bos) :]}
        return {"prompt": prompt}

    def _generate_vlm_batch(
        self,
        requests,
        *,
        cancel_event = None,
        _adapter_state = None,
    ):
        import time

        from unsloth_zoo.mlx.generate import (
            GenerationDefaults,
            GenerationRequest,
            SamplingParams,
            stream_batch,
        )

        plans = [self._plan_vlm_request(request) for request in requests]
        defaults = GenerationDefaults(
            max_tokens = max(plan.max_tokens for plan in plans),
            prefill_batch_size = len(plans),
            completion_batch_size = len(plans),
            **self._prefill_step_kwargs(GenerationDefaults, grid = True),
        )

        with (
            self._generation_lock,
            _temporary_mlx_adapter_state(self._model, _adapter_state),
            self._prefill_decision_scopes(),
            ExitStack() as clips,
        ):
            batch = [
                GenerationRequest(
                    **self._vlm_batch_prompt(plan.prompt),
                    image = plan.images[0] if plan.images else None,
                    **row_clips,
                    max_tokens = plan.max_tokens,
                    sampling = SamplingParams(**plan.sampling),
                    logits_processors = plan.processors,
                )
                for plan, row_clips in zip(plans, _batch_clips(clips, plans, self._processor))
            ]
            if any(plan.images or plan.videos for plan in plans):
                self._release_vlm_snapshots()
            logger.info(
                "Generating %d vision replies as one batch: images=%d, videos=%d, max_tokens=%s, model=%s",
                len(plans),
                sum(1 for plan in plans if plan.images),
                sum(len(plan.videos or ()) for plan in plans),
                [plan.max_tokens for plan in plans],
                type(self._model).__name__,
            )
            for row, plan in enumerate(plans):
                if plan.think_prefix:
                    yield row, plan.think_prefix
            started = time.perf_counter()
            prefilled_at: list[float | None] = [None] * len(plans)
            counted: dict[int, tuple[int, int]] = {}
            pending = set(range(len(plans)))
            events = stream_batch(self._model, self._processor, batch, defaults = defaults)

            def _retire(row, result, *, cancelled):
                stream = plans[row].stream
                yield from ((row, snap) for snap in stream.settle())
                yield from ((row, snap) for snap in stream.drain(cancelled = cancelled))
                ready_at = prefilled_at[row] or time.perf_counter()
                decoded = time.perf_counter() - ready_at
                prefill_s = ready_at - started
                seen_prompt, seen_generated = counted.get(row, (0, 0))
                prompt_n = result.prompt_token_count if result is not None else seen_prompt
                generated = (
                    len(result.token_ids) + (result.finish_reason == "stop")
                    if result is not None
                    else seen_generated
                )
                self.last_batch_generation_stats[row] = _build_generation_stats(
                    prompt_n,
                    (prompt_n / prefill_s) if prefill_s > 0 else 0.0,
                    generated,
                    generated / max(decoded, 1e-9),
                    finish_reason = (result.finish_reason if result is not None else "stop"),
                )
                yield row, None

            try:
                for event in events:
                    row = event.index
                    if row not in pending:
                        continue
                    if prefilled_at[row] is None:
                        prefilled_at[row] = time.perf_counter()
                    counted[row] = (
                        getattr(event, "prompt_tokens", 0),
                        getattr(event, "generated_tokens", 0),
                    )
                    if event.delta:
                        yield from (
                            (row, snapshot) for snapshot in plans[row].stream.feed(text = event.delta)
                        )
                    if event.result is not None:
                        pending.discard(row)
                        yield from _retire(row, event.result, cancelled = False)
                    if cancel_event is not None and cancel_event.is_set():
                        break
                if pending and not (cancel_event is not None and cancel_event.is_set()):
                    raise RuntimeError(
                        f"Batched vision generation ended with {len(pending)} of "
                        f"{len(plans)} replies unfinished."
                    )
                for row in sorted(pending):
                    yield from _retire(row, None, cancelled = True)
            finally:
                close = getattr(events, "close", None)
                if callable(close):
                    close()

    def _batch_prompt_row(self, prompt, adapter_state):
        rest, cache, key, tokens, cached_n = self._prepare_prompt_cache(prompt, adapter_state)
        if cache is None or tokens is None:
            tokens = list(prompt) if isinstance(prompt, list) else self._encode_prompt(prompt)
            rest, cached_n, key = tokens, 0, None
            cache = self._make_row_cache()
        return list(tokens[:cached_n]), list(rest), cache, key, list(tokens)

    def _make_row_cache(self):
        """A row's own cache: mlx-lm applies ``max_kv_size`` only to caches it builds."""
        from mlx_lm.models.cache import make_prompt_cache

        window = getattr(self, "_kv_cache_window", None)
        if window is None:
            return make_prompt_cache(self._model)
        return make_prompt_cache(self._model, max_kv_size = window)

    def _bank_row_cache(self, cache_key, event, withdrawn, prompt_tokens, stream):
        history = self._prompt_cache_history
        if history is None or cache_key is None:
            return
        cache = getattr(event, "prompt_cache", None)
        tokens = getattr(event, "all_tokens", None)
        if cache is None and withdrawn is not None:
            cache, tokens = withdrawn
        if cache is None:
            return
        if not tokens:
            tokens = prompt_tokens + stream.token_ids
        try:
            history.insert(cache_key, list(tokens), cache)
        except Exception as exc:
            logger.debug("MLX prompt cache insert failed: %s", exc)

    def generate_audio_input_response(
        self,
        messages,
        system_prompt,
        audio_array,
        max_new_tokens = 512,
        use_adapter = None,
        cancel_event = None,
        stop = None,
        extra_audio_arrays = None,
        speak = False,
        tools = None,
        **_sampler,
    ):
        """Audio-input chat (omni models): waveform in, incremental text deltas out (the audio route
        forwards deltas, unlike the snapshot-diffing text/vision paths). Greedy, so the worker's
        sampler kwargs go unused. ``speak`` answers in the model's own voice instead: the reply
        text, then one ``{"speech": <base64 wav>, "sample_rate": ...}`` item. ``tools`` (a list,
        empty for tool history alone) renders the catalogue and keeps ``messages`` as sent."""
        entry = self.models.get(self.active_model_name) or {}
        if entry.get("audio_type") != "audio_vlm":
            raise RuntimeError(
                "Audio input is not supported for this model on the MLX backend: "
                "no verified audio-capable tower/processor was detected at load."
            )

        # mlx-vlm silently drops all but the first clip; the route refuses this earlier.
        if extra_audio_arrays:
            raise RuntimeError("This MLX model takes one audio file per message.")

        if speak:
            yield from self._speak_audio_reply(
                messages, system_prompt, audio_array, max_new_tokens, use_adapter, cancel_event
            )
            return

        from mlx_vlm import stream_generate as vlm_stream

        if not system_prompt:
            system_prompt = "You are an assistant that transcribes speech accurately."

        audio_messages = messages_with_attached_image(
            alternating_turns(messages) if tools is None else messages,
            system_prompt = system_prompt,
            fallback_user_text = "Please transcribe this audio.",
            image = 0,
            audio = audio_array,
        )
        prompt = _render_registered_vlm_prompt(
            self._processor,
            self._model,
            audio_messages,
            num_images = 0,
            num_audios = 1,
            **({"tools": tools} if tools else {}),
        )
        if prompt is None:
            raise RuntimeError(
                "mlx-vlm has no registered prompt renderer for this model family; "
                "cannot build an audio prompt."
            )

        max_new_tokens = self._generation_limit(
            prompt,
            max_new_tokens,
            audio = [audio_array],
            cap = UNSET_GENERATION_BUDGET,
        )
        logger.info("MLX audio-input generating: prompt_len=%d", len(prompt))
        markers = detect_reasoning_channel_markers(self._processor)
        normalizer = make_reasoning_normalizer(markers) if markers is not None else None
        # Matched on the sampled deltas, for the reason _generate_text gives.
        sequences = _mlx_stop_sequences(stop)
        sampled = ""
        released = 0
        stopped = False
        # Hold adapter state for the whole stream so Base-vs-LoRA compare stays correct
        with (
            self._generation_lock,
            _temporary_mlx_adapter_state(self._model, use_adapter),
            self._prefill_decision_scopes(),
            ExitStack() as generation_scope,
        ):
            # As on the image path: the tower gets the headroom, under the lock.
            self._release_vlm_snapshots()
            generation_scope.enter_context(self._generation_scopes("vision", "audio"))
            final_response = None
            try:
                with closing(
                    _iter_vlm_responses(
                        vlm_stream(
                            self._model,
                            self._processor,
                            prompt,
                            audio = [audio_array],
                            max_tokens = max_new_tokens,
                            # Greedy; the knobs below are load-time state, not caller kwargs.
                            temperature = 0.0,
                            **self._kv_runtime_quant_kwargs(),
                            **self._kv_quant_generate_kwargs(),
                            **self._kv_window_generate_kwargs(),
                            **self._prefill_step_kwargs(),
                        )
                    )
                ) as responses:
                    for response in responses:
                        final_response = response
                        sampled += response.text if hasattr(response, "text") else str(response)
                        if sequences:
                            cut, stopped = _mlx_stop_cut(sampled, sequences)
                        else:
                            cut = len(sampled)
                        # Cut before normalizing; the normalizer's own markers are unmatched here, as the prefill is.
                        delta = sampled[released:cut]
                        released = cut
                        if normalizer is not None:
                            delta = normalizer.feed(delta)
                        if delta:
                            yield delta
                        if stopped:
                            break
                        if cancel_event and cancel_event.is_set():
                            break
            finally:
                # As the vision path derives it: no finish reason here, and unset reads as a natural end.
                if final_response is not None:
                    tokenizer = getattr(self._processor, "tokenizer", self._processor)
                    self.last_generation_stats = _build_generation_stats(
                        getattr(final_response, "prompt_tokens", 0),
                        getattr(final_response, "prompt_tps", 0.0),
                        getattr(final_response, "generation_tokens", 0),
                        getattr(final_response, "generation_tps", 0.0),
                        finish_reason = _mlx_finish_reason(
                            final_response,
                            _mlx_stop_token_ids(tokenizer, self._model),
                            getattr(final_response, "generation_tokens", 0),
                            max_new_tokens,
                        ),
                    )
        if sequences and not stopped:
            delta = sampled[released:]
            if normalizer is not None:
                delta = normalizer.feed(delta)
            if delta:
                yield delta
        if normalizer is not None:
            # As in _generate_text: a sequence closes the block it cut inside.
            cancelled = not stopped and cancel_event is not None and cancel_event.is_set()
            tail = normalizer.drain() if cancelled else normalizer.finish()
            if tail:
                yield tail
        if stopped:
            self._mark_stopped()

    def _speak_audio_reply(
        self,
        messages,
        system_prompt,
        audio_array,
        max_new_tokens,
        use_adapter,
        cancel_event = None,
    ):
        """One spoken question answered in interleaved text and speech, not streamed."""
        import base64

        from mlx_vlm.generate import generate_audio

        if not (self.models.get(self.active_model_name) or {}).get("speaks"):
            raise RuntimeError("The loaded model cannot answer in speech.")
        # The model card's prompt for speech chat; a user prompt still applies after it.
        system = "\n\n".join(p for p in (_SPEECH_REPLY_SYSTEM_PROMPT, system_prompt) if p)
        prompt = _render_registered_vlm_prompt(
            self._processor,
            self._model,
            messages_with_attached_image(
                alternating_turns(messages),
                system_prompt = system,
                image = 0,
                audio = audio_array,
            ),
            num_images = 0,
            num_audios = 1,
        )
        max_new_tokens = self._generation_limit(
            prompt, max_new_tokens, audio = [audio_array], cap = UNSET_GENERATION_BUDGET
        )
        with (
            self._generation_lock,
            _temporary_mlx_adapter_state(self._model, use_adapter),
            ExitStack() as generation_scope,
        ):
            self._release_vlm_snapshots()
            generation_scope.enter_context(self._generation_scopes("vision", "audio"))
            generation_scope.enter_context(_slots_until_cancelled(self._model, cancel_event))
            result = generate_audio(
                self._model,
                self._processor,
                prompt,
                audio = [audio_array],
                generation_mode = "interleaved",
                max_tokens = max_new_tokens,
            )
            cancelled = cancel_event is not None and cancel_event.is_set()
            # Nothing to encode once stopped, nor for a reply that ended before its first speech frame.
            spoke = not cancelled and result.audio is not None and result.audio.size > 0
            wav = result.to_wav_bytes() if spoke else None
        self.last_generation_stats = _build_generation_stats(
            result.prompt_tokens,
            result.prompt_tps,
            result.generation_tokens,
            result.generation_tps,
            finish_reason = result.finish_reason,
        )
        if cancelled:
            return
        if result.text:
            yield result.text
        if spoke:
            yield {
                "speech": base64.b64encode(wav).decode("ascii"),
                "sample_rate": result.sample_rate,
            }

    def generate_with_adapter_control(
        self,
        use_adapter = None,
        cancel_event = None,
        **gen_kwargs,
    ) -> Generator[str, None, None]:
        yield from self.generate_chat_response(
            cancel_event = cancel_event,
            _adapter_state = use_adapter,
            **gen_kwargs,
        )

    def reset_generation_state(self, caller_cancel_event = None):
        # caller_cancel_event: signature parity with the orchestrator; unused here.
        import mlx.core as mx
        import gc

        gc.collect()
        _drain_generation_streams(mx)
        mx.clear_cache()
