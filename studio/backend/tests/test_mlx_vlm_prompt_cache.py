# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import contextlib
import logging
import sys
import types

import pytest

from core.inference import mlx_inference as snapshots
from core.inference.mlx_inference import (
    VLM_PREFILL_STEP,
    RecordingForward,
    VLMBatchRowCache,
    VLMPromptCacheSession,
    VLMPromptSnapshotStore,
    cache_entries_nbytes,
    cache_entries_offset,
    copy_cache_entries,
    release_cache_entries,
    media_prefix_end,
    shape_stable_prefix,
    vlm_prefill_step,
)

# The tests' grid; production reads mlx-vlm's.
STEP = 256


def Session(*args, **kwargs):
    kwargs.setdefault("step", STEP)
    return VLMPromptCacheSession(*args, **kwargs)


@pytest.fixture(autouse = True)
def _plain_logger(monkeypatch):
    """The backend logs through structlog, which caplog does not see."""
    monkeypatch.setattr(snapshots, "logger", logging.getLogger(snapshots.__name__))


class FakeArray:
    __repr__ = lambda self: repr(self.rows)

    def __init__(self, rows):
        self.rows = list(rows)

    def __add__(self, _other):
        return FakeArray(self.rows)

    @property
    def nbytes(self):
        return 4 * len(self.rows)


@pytest.fixture
def fake_mx(monkeypatch):
    evaluated = []
    mlx_core = types.ModuleType("mlx.core")
    mlx_core.array = FakeArray
    mlx_core.eval = lambda arrays: evaluated.append(list(arrays))
    mlx_core.contiguous = lambda array: FakeArray(array.rows)
    mlx_pkg = types.ModuleType("mlx")
    mlx_pkg.core = mlx_core
    monkeypatch.setitem(sys.modules, "mlx", mlx_pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", mlx_core)
    return evaluated


class FakeKV:
    state = property(lambda self: self.keys)

    def __init__(self):
        self.keys = FakeArray([])
        self.offset = 0

    def advance(self, tokens):
        self.keys = FakeArray(self.keys.rows + list(tokens))
        self.offset += len(tokens)


class FakeState:
    state = property(lambda self: self.cache)

    def __init__(self):
        self.cache = [None, None]

    def advance(self, tokens):
        self.cache[0] = FakeArray(tokens[-1:])
        self.cache[1] = FakeArray([sum(tokens)])


class FakeCacheList:
    state = property(lambda self: [c.state for c in self.caches])

    def __init__(self, *caches):
        self.caches = caches

    def advance(self, tokens):
        for entry in self.caches:
            entry.advance(tokens)


class FakeSimpleKV:
    """Encoder-decoder entry: rows counted in ``cache_length``, not ``offset``."""

    state = property(lambda self: self.keys)

    def __init__(self):
        self.keys = FakeArray([])
        self.cache_length = 0

    def advance(self, tokens):
        self.keys = FakeArray(self.keys.rows + list(tokens))
        self.cache_length += len(tokens)


class FakeLanguageModel:
    def __init__(self):
        self.seen_kwargs = []
        self.outputs = []

    def __call__(
        self,
        inputs,
        cache = None,
        **kwargs,
    ):
        self.seen_kwargs.append(dict(kwargs))
        for entry in cache:
            entry.advance(inputs)
        self.outputs.append(object())
        return self.outputs[-1]


def make_cache():
    return [FakeKV(), FakeState(), FakeKV()]


def run_generation(
    language_model,
    token_ids,
    cache,
    start,
    between = None,
    step = STEP,
    **kwargs,
):
    """mlx-vlm's prefill loop: grid chunks over all but the last token, then it."""
    n = len(token_ids)
    pos = start
    while n - pos > 1:
        take = min(step, n - pos - 1)
        language_model(token_ids[pos : pos + take], cache = cache, **kwargs)
        if between is not None:
            between(cache)
        pos += take
    language_model(token_ids[n - 1 :], cache = cache, **kwargs)


def _snapshot(rows):
    kv = FakeKV()
    kv.advance(rows)
    return [kv]


def _generate(
    store,
    language_model,
    token_ids,
    *,
    honour_reuse = True,
    kwargs = None,
    media_token_ids = (),
    **session_kwargs,
):
    with Session(
        store, "m", language_model, make_cache, media_token_ids = media_token_ids, **session_kwargs
    ) as session:
        prefix = session.find_prefix_length(token_ids)
        cache = session.cache
        if not honour_reuse:
            cache, prefix = make_cache(), 0
        run_generation(language_model, token_ids, cache, prefix, **(kwargs or {}))
        session.update(token_ids, cache)
        stored = session.finish()
    return session, cache, stored


def test_shape_stable_prefix_is_the_last_whole_chunk_from_the_origin():
    grid = [(0, 0, 0), (1, 0, 0), (256, 0, 0), (257, 0, 256), (512, 0, 256), (513, 0, 512)]
    grid += [(2049, 0, 2048), (4018, 0, 3840), (600, 647, 0), (648, 647, 647), (903, 647, 647)]
    grid += [(904, 647, 903), (1160, 647, 1159)]
    for tokens, origin, prefix in grid:
        assert shape_stable_prefix(tokens, origin, STEP) == prefix, (tokens, origin)


def test_copy_and_release_walk_every_array_of_a_composite_layout(fake_mx):
    kv, state = FakeKV(), FakeState()
    kv.advance([1, 2, 3])
    state.advance([1, 2, 3])
    state.extra = (FakeArray([9]), 7)
    nested = FakeCacheList(FakeKV(), FakeState())
    nested.advance([1, 2])
    pair = (FakeSimpleKV(), FakeSimpleKV())
    for entry in pair:
        entry.advance([5])
    entries = [kv, state, nested, pair]
    copies = copy_cache_entries(entries)

    assert copies[0].offset == 3 and copies[0].keys.rows == [1, 2, 3]
    assert copies[0].keys is not kv.keys
    assert copies[1].cache[1].rows == [6] and copies[1].cache is not state.cache
    assert copies[1].cache[0] is not state.cache[0] and copies[1].cache[1] is not state.cache[1]
    assert copies[1].extra[0] is not state.extra[0] and copies[1].extra[1] == 7
    assert copies[2].caches[0].keys is not nested.caches[0].keys
    assert copies[2].caches[1].cache is not nested.caches[1].cache
    assert isinstance(copies[3], tuple) and copies[3][1] is not pair[1]
    kv.advance([4])
    state.advance([4])
    nested.advance([3])
    assert copies[0].keys.rows == [1, 2, 3] and copies[1].cache[0].rows == [3]
    assert copies[2].caches[0].keys.rows == [1, 2] and copies[2].caches[0].offset == 2
    assert len(fake_mx) == 1 and len(fake_mx[0]) == 9

    assert cache_entries_nbytes([nested, pair]) == 4 * (3 + 1 + 1 + 1 + 1)
    assert cache_entries_offset([nested]) == 3 and cache_entries_offset([pair]) is None
    assert cache_entries_offset([state, kv]) == 4 and cache_entries_offset([state]) is None

    release_cache_entries(entries)
    assert not list(snapshots._arrays(entries, sys.modules["mlx.core"]))
    assert nested.caches[0].keys is None and nested.caches[0].offset == 3
    assert nested.caches[1].cache == [None, None] and pair[1].cache_length == 1
    assert len(fake_mx) == 1

    with pytest.raises(TypeError, match = "int cannot be copied"):
        copy_cache_entries([3])
    with pytest.raises(TypeError, match = "SimpleNamespace cannot be copied"):
        copy_cache_entries([types.SimpleNamespace(keys = FakeArray([1]))])

    ids = [1, 2, 9, 9, 9, 3, 4, 9, 5, 7, 6]
    assert media_prefix_end(ids, {9}) == 8
    assert media_prefix_end(ids, [9, 7]) == 10
    assert media_prefix_end(ids, ()) == 0
    assert media_prefix_end([1, 2, 3], {9}) == 0
    assert media_prefix_end([9], {9}) == 1


def test_recording_forward_copies_the_cache_after_the_chosen_forward(fake_mx, caplog):
    language_model = FakeLanguageModel()
    cache = make_cache()
    original_class = type(language_model)
    with RecordingForward(language_model) as recording:
        recording.record.capture_at = 2
        assert type(language_model) is not original_class
        outputs = [
            language_model(chunk, cache = cache, per_layer_inputs = "kept")
            for chunk in ([1, 2], [3, 4], [5, 6])
        ]
        recording.unrecorded([7], cache = cache)
    assert outputs == language_model.outputs[:3] and recording.record.forwards == 3
    snapshot = recording.record.snapshot
    assert snapshot[0].keys.rows == [1, 2, 3, 4] and snapshot[0].offset == 4
    assert snapshot[1].cache[1].rows == [7]
    assert cache[0].offset == 7 and snapshot[0] is not cache[0]
    assert all(kw == {"per_layer_inputs": "kept"} for kw in language_model.seen_kwargs[:3])
    assert type(language_model) is original_class
    assert "_studio_forward_record" not in vars(language_model)

    with RecordingForward(language_model) as recording:
        language_model([1], cache = make_cache())
    assert recording.record.snapshot is None and recording.record.forwards == 1
    with pytest.raises(RuntimeError):
        with RecordingForward(language_model):
            raise RuntimeError("generation failed")
    assert type(language_model) is original_class

    opaque = [types.SimpleNamespace(advance = lambda _t: None)]
    with caplog.at_level("INFO"), RecordingForward(language_model) as recording:
        recording.record.capture_at = 1
        language_model([1], cache = opaque)
        output = language_model([2], cache = opaque)
    assert output is language_model.outputs[-1]
    assert recording.record.snapshot is None
    assert "SimpleNamespace cannot be copied" in caplog.text


def test_recording_forward_copies_what_generation_converted_after_the_boundary_forward(fake_mx):
    """mlx-vlm quantizes the live cache after a chunk; the snapshot must hold that."""

    class FakeQuantizedKV(FakeKV):
        pass

    def quantize(cache):
        if cache[0].offset >= 512:
            converted = FakeQuantizedKV()
            converted.keys, converted.offset = cache[0].keys, cache[0].offset
            cache[0] = converted

    store = VLMPromptSnapshotStore(max_bytes = 10**9)
    ids = list(range(700))
    with Session(store, "m", FakeLanguageModel(), make_cache) as session:
        session.find_prefix_length(ids)
        run_generation(session._forward._language_model, ids, session.cache, 0, between = quantize)
        assert session.finish()
    entries, prefix = store.lookup("m", ids + [-1], limit = 512)
    assert prefix == 512 and type(entries[0]) is FakeQuantizedKV
    assert entries[0].keys.rows == ids[:512]


class Sliceable:
    def __init__(self, rows):
        self.rows = list(rows)
        self.shape = (1, len(self.rows), 8)

    def __getitem__(self, item):
        return Sliceable(self.rows[item[1]])


def test_recording_forward_slices_per_layer_inputs_from_the_resume_offset(fake_mx):
    language_model = FakeLanguageModel()
    embeds = lambda rows: types.SimpleNamespace(shape = (1, rows, 8))
    for resumed in (0, 512):
        suffix = Sliceable(range(resumed, resumed + 300))
        cache = make_cache()
        cache[0].advance(range(resumed))
        with RecordingForward(language_model):
            for rows in (256, 43):
                language_model(
                    inputs = [0] * rows,
                    inputs_embeds = embeds(rows),
                    cache = cache,
                    per_layer_inputs = suffix,
                    other = 1,
                )
            language_model([0], inputs_embeds = embeds(1), cache = cache, per_layer_inputs = suffix)
        seen = [kw.get("per_layer_inputs") for kw in language_model.seen_kwargs[-3:]]
        assert seen[0].rows == list(range(resumed, resumed + 256))
        assert seen[1].rows == list(range(resumed + 256, resumed + 299))
        assert seen[2] is None
    assert language_model.seen_kwargs[0]["other"] == 1

    short = Sliceable(range(100))
    cache = make_cache()
    cache[0].advance(range(512))
    with RecordingForward(language_model):
        language_model(
            types.SimpleNamespace(shape = (1, 43)), cache = [], per_layer_inputs = Sliceable(range(300))
        )
        language_model(
            inputs = [0] * 256, inputs_embeds = embeds(256), cache = cache, per_layer_inputs = short
        )
        language_model(
            inputs = [0] * 256, inputs_embeds = embeds(256), cache = [], per_layer_inputs = short
        )
    seen = [kw["per_layer_inputs"] for kw in language_model.seen_kwargs[-3:]]
    assert seen[0].rows == list(range(43)) and seen[1] is short and seen[2] is short


def test_recording_forward_withholds_prompt_wide_position_ids_only_after_a_resume(fake_mx):
    language_model = FakeLanguageModel()
    shaped = types.SimpleNamespace
    wide = lambda cache: dict(cache = cache, position_ids = shaped(shape = (3, 1, 900)))
    embedded = dict(inputs = [1, 2, 3, 4, 5], inputs_embeds = shaped(shape = (1, 5, 8)))
    with RecordingForward(language_model):
        language_model(shaped(shape = (1, 256)), **wide([]))
        language_model(**embedded, **wide([]))
    resumed = [shaped(offset = 512, advance = lambda _tokens: None)]
    with RecordingForward(language_model):
        language_model(shaped(shape = (1, 256)), **wide(resumed))
        language_model(
            shaped(shape = (1, 256)), cache = resumed, position_ids = shaped(shape = (3, 1, 256))
        )
        language_model(position_ids = shaped(shape = (1, 5)), cache = resumed, **embedded)
        language_model(**embedded, **wide(resumed))
        language_model([1, 2], cache = resumed, position_ids = None)
    kept = ["position_ids" in kw for kw in language_model.seen_kwargs]
    assert kept == [True, True, False, True, True, False, True]


def test_store_serves_the_longest_prefix_and_evicts_to_fit(fake_mx):
    store = VLMPromptSnapshotStore(max_bytes = 10**6)
    ids = list(range(1000))
    store.store("m", ids[:256], _snapshot(ids[:256]))
    store.store("m", ids[:512], _snapshot(ids[:512]))
    store.store("m", [7] * 768, _snapshot([7] * 768))
    store.store("other", ids[:768], _snapshot(ids[:768]))
    entries, prefix = store.lookup("m", ids, limit = 768)
    assert prefix == 512 and entries[0].offset == 512
    assert store.lookup("m", ids, limit = 511) == (store.lookup("m", ids, limit = 256)[0], 256)
    assert store.lookup("m", ids, limit = 255) == (None, 0)
    assert store.lookup("m", [7] * 800, limit = 768)[1] == 768
    assert store.lookup("m", [1] + ids[1:], limit = 768) == (None, 0)

    store = VLMPromptSnapshotStore(max_bytes = 4 * 700)
    a, b, c = [1] * 256, [2] * 256, [3] * 256
    assert store.store("m", a, _snapshot(a)) and store.store("m", b, _snapshot(b))
    store.lookup("m", a + [0], limit = 256)
    assert store.store("m", c, _snapshot(c))
    assert store.lookup("m", b + [0], limit = 256) == (None, 0)
    assert store.lookup("m", a + [0], limit = 256)[1] == 256
    assert store.nbytes == 4 * 512 and len(store) == 2
    assert not store.store("m", [4] * 800, _snapshot([4] * 800), hold = False)
    assert store.store("m", [4] * 800, _snapshot([4] * 800))  # held apart
    assert len(store) == 2 and store.nbytes == 4 * 512

    store = VLMPromptSnapshotStore(max_bytes = 10**9, max_entries = 2)
    store.store("m", [1], _snapshot([1]))
    store.store("m", [1], _snapshot([1, 1]))
    assert len(store) == 1 and store.nbytes == 8
    store.store("m", [2], _snapshot([2]))
    store.store("m", [3], _snapshot([3]))
    assert len(store) == 2 and store.lookup("m", [1, 0], limit = 1) == (None, 0)

    store = VLMPromptSnapshotStore(max_bytes = 10**9)
    store.store("a", [1, 2], _snapshot([1, 2]))
    store.store("b", [1, 2, 3], _snapshot([1, 2, 3]))
    store.store("a", [1, 2, 3, 4], _snapshot([1, 2, 3, 4]))
    store.retain(("a", (1, 2, 3, 4)))
    assert list(store._entries) == [("a", (1, 2, 3, 4))]
    assert store.nbytes == cache_entries_nbytes(_snapshot([1, 2, 3, 4]))
    store.retain(None)
    assert len(store) == 0 and store.nbytes == 0


def _over_budget_store():
    """256 rows fit the budget; one request's 512 boundary and 699 replay are held."""
    store, ids = VLMPromptSnapshotStore(max_bytes = 4 * 300), list(range(1000))
    store.store("m", ids[:256], _snapshot(ids[:256]))
    assert store.store("m", ids[:512], _snapshot(ids[:512]))
    assert store.store("m", ids[:699], _snapshot(ids[:699]), replay = True)
    return store, ids


def test_snapshots_over_the_budget_are_held_for_the_chats_next_lookup_only(fake_mx):
    store, ids = _over_budget_store()
    assert len(store) == 1 and store.nbytes == 4 * 256 and len(store._held) == 2
    assert ("m", tuple(ids[:512])) in store and ("m", tuple(ids[:699])) in store

    # Another key, another prefix, an earlier boundary: served as before, and released.
    for key, prompt, limit, reused in (
        ("t", ids, 768, 0),
        ("m", [-1] + ids[1:], 768, 0),
        ("m", ids, 511, 256),
        ("m", ids[:701], 511, 256),
    ):
        store, ids = _over_budget_store()
        assert store.lookup(key, prompt, limit)[1] == reused and not store._held
    store, ids = _over_budget_store()
    store._entries[("m", tuple(ids[:768]))] = (_snapshot(ids[:768]), 0)
    assert store.lookup("m", ids, 768)[1] == 768 and not store._held

    # The boundary is handed over, uncopied, and its replay goes with it.
    store, ids = _over_budget_store()
    held = store._held._entries[("m", tuple(ids[:512]))][0]
    assert store.lookup("m", ids[:800], limit = 768) == (held, 512)
    assert ("m", tuple(ids[:512])) not in store and not store._held and len(store) == 1
    assert not fake_mx

    # The replay serves the same prompt again and stays, with the boundary reading through it.
    store, ids = _over_budget_store()
    assert store.lookup("m", ids[:700], limit = 512)[1] == 699 and len(store._held) == 2
    assert store.lookup("m", ids[:700], limit = 512)[1] == 699

    # One chat at a time; release follows every way the store frees.
    assert store.store("t", ids[:400], _snapshot(ids[:400]))
    assert list(store._held._entries) == [("t", tuple(ids[:400]))]
    for free in (
        lambda s: s.retain(("m", tuple(ids[:256]))),
        lambda s: s.retain_key("m"),
        lambda s: s.discard(("m", tuple(ids[:512]))),
        lambda s: s.release_held(),
        lambda s: s.clear(),
    ):
        store, ids = _over_budget_store()
        free(store)
        assert ("m", tuple(ids[:512])) not in store


def test_a_long_chat_resumes_from_its_held_snapshot_without_a_copy(fake_mx):
    # 256 rows of make_cache fit; 512 do not.
    store, language_model, ids = (
        VLMPromptSnapshotStore(2400),
        FakeLanguageModel(),
        list(range(1100)),
    )
    _generate(store, language_model, ids[:700])
    assert len(store) == 0 and len(store._held) == 2
    held = store._held._entries[("m", tuple(ids[:512]))][0]
    fake_mx.clear()
    session, cache, stored = _generate(store, language_model, ids[:900])
    assert session.reused_tokens == 512 and cache is held and stored
    assert len(fake_mx) == 2  # the two captures; nothing kept of what it resumed from
    assert [len(item[1]) for item in store._held._entries] == [768, 899]

    fake_mx.clear()
    session, cache, stored = _generate(store, language_model, ids[:900])
    assert session.reused_tokens == 899 and cache[0].offset == 900 and len(fake_mx) == 1
    assert ("m", tuple(ids[:899])) in store and ("m", tuple(ids[:768])) in store
    assert store._held._entries[("m", tuple(ids[:899]))][0][0].offset == 899  # never advanced
    session, _cache, _stored = _generate(store, language_model, ids[:1] + ids[2:900])
    assert session.reused_tokens == 0 and ("m", tuple(ids[:768])) not in store


@pytest.mark.parametrize("served", (512, 699))
def test_a_batched_row_takes_what_is_held_and_holds_nothing(fake_mx, served):
    store, ids = VLMPromptSnapshotStore(2400), list(range(1100))
    _generate(store, FakeLanguageModel(), ids[:700])
    held = store._held._entries[("m", tuple(ids[:served]))][0]
    fake_mx.clear()
    row = VLMBatchRowCache(store, "m", make_cache, step = STEP)
    prefix, cache = _prefill_row(row, ids[: 900 if served == 512 else 700])
    assert prefix == served and cache is held and not store._held and len(store) == 0
    assert len(fake_mx) == (2 if served == 512 else 0)  # its own checkpoints, then dropped


def test_session_captures_the_boundary_and_resumes_leaving_a_copy_behind(fake_mx):
    store = VLMPromptSnapshotStore(max_bytes = 10**9)
    language_model = FakeLanguageModel()
    ids = list(range(700))
    session, cache, stored = _generate(store, language_model, ids)
    assert session.reused_tokens == 0 and stored and cache[0].offset == 700
    stored_entries, prefix = store.lookup("m", ids + [-1], limit = 512)
    assert prefix == 512 and stored_entries[0].keys.rows == ids[:512]
    assert stored_entries[1].cache[0].rows == [511]
    store.store("m", ids[:256], _snapshot(range(9000, 9256)))
    store.store("t", ids[:512], _snapshot(range(9000, 9512)))

    longer = ids + list(range(700, 800))
    fake_mx.clear()
    session, cache, stored = _generate(store, language_model, longer)
    assert session.reused_tokens == 512 and cache is stored_entries
    kept = store.lookup("m", ids, limit = 512)[0]
    assert kept is not stored_entries and kept[0] is not cache[0]
    assert kept[0].offset == 512 and kept[0].keys.rows == ids[:512]
    assert kept[0].keys in fake_mx[0]
    assert cache[0].offset == 800 and cache[0].keys.rows == longer
    assert stored and store.lookup("m", longer + [-1], limit = 768)[1] == 768

    session, _cache, stored = _generate(store, language_model, longer + [1])
    assert session.reused_tokens == 768 and stored and len(store) == 5


class LayeredLanguageModel(FakeLanguageModel):
    capture, skipped = None, ()

    def __call__(
        self,
        inputs,
        cache = None,
        **kwargs,
    ):
        for at in range(0, len(inputs), STEP):
            output = super().__call__(inputs[at : at + STEP], cache, **kwargs)
            if self.capture and min(at + STEP, len(inputs)) == self.capture[0]:
                self.capture[2].update(
                    (id(e), self.capture[1](e))
                    for i, e in enumerate(cache)
                    if i not in self.skipped
                )
        return output


def test_a_window_of_grid_steps_snapshots_the_grid_inside_it(fake_mx, monkeypatch):
    model, zoo, ids = (
        LayeredLanguageModel(),
        types.ModuleType("unsloth_zoo.mlx.offload"),
        list(range(1900)),
    )

    def capture(rows, copy):
        model.capture = (rows, copy, {})
        yield model.capture[2]
        model.capture = None

    zoo.capture_prefill_rows = contextlib.contextmanager(capture)
    monkeypatch.setitem(sys.modules, zoo.__name__, zoo)
    stores = [VLMPromptSnapshotStore(max_bytes = 10**9) for _ in range(2)]
    for tokens in (ids, ids + [1] * 300):
        _generate(stores[0], FakeLanguageModel(), tokens)
        _generate(stores[1], model, tokens, window = 4 * STEP, kwargs = {"step": 4 * STEP})
    held = [
        {key: [entry.state for entry in entries] for key, (entries, _) in store._entries.items()}
        for store in stores
    ]
    assert [len(key[1]) for key in held[1]] == [1792, 2048, 2199] and repr(held[0]) == repr(held[1])
    model.skipped, store = (1,), VLMPromptSnapshotStore(max_bytes = 10**9)
    _generate(store, model, ids, window = 4 * STEP, kwargs = {"step": 4 * STEP})
    assert [len(key[1]) for key in store._entries] == [1899]


def test_session_replays_a_prompt_from_where_its_prefill_ended(fake_mx):
    store = VLMPromptSnapshotStore(max_bytes = 10**9)
    language_model = FakeLanguageModel()
    ids = list(range(700))
    _generate(store, language_model, ids)
    assert [len(item[1]) for item in store._entries] == [512, 699]
    assert store.lookup("m", ids + [1] * 300, limit = 768)[1] == 512
    session, cache, stored = _generate(store, language_model, ids)
    assert session.reused_tokens == 699 and cache[0].offset == 700 and not stored
    assert session._forward.record.snapshot is None and session._forward.record.tail is None
    assert store.lookup("m", ids + [1] * 300, limit = 768)[1] == 512
    _generate(store, language_model, ids + [1])
    assert [len(item[1]) for item in store._entries] == [512, 700]


def test_session_stores_what_the_snapshot_holds_when_reuse_was_declined(fake_mx, monkeypatch):
    store = VLMPromptSnapshotStore(max_bytes = 10**9)
    language_model = FakeLanguageModel()
    ids = list(range(1100))
    _generate(store, language_model, ids)
    original = store.lookup("m", ids + [-1], limit = 1024)[0]
    store.store("t", ids[:512], _snapshot(range(9000, 9512)))
    fake_mx.clear()
    session, _cache, stored = _generate(store, language_model, ids + [1] * 300, honour_reuse = False)
    assert session.reused_tokens == 1024 and stored
    assert store.lookup("m", ids + [-1], limit = 256)[1] == 256
    # The declined offer is released, not copied: it would be declined again.
    assert len(store) == 3 and store.lookup("m", ids + [-1], limit = 1024)[1] == 256
    assert store.lookup("t", ids, limit = 512)[1] == 512
    assert not list(snapshots._arrays(original, sys.modules["mlx.core"]))
    assert original[0].offset == 1024 and len(fake_mx) == 2

    store, ids = VLMPromptSnapshotStore(max_bytes = 10**9), list(range(700))
    _generate(store, language_model, ids)

    def _fail(_arrays):
        raise MemoryError("no room for a copy")

    monkeypatch.setattr(sys.modules["mlx.core"], "eval", _fail)
    with Session(store, "m", language_model, make_cache) as session:
        assert session.find_prefix_length(ids + [1]) == 512
        run_generation(language_model, ids + [1], session.cache, 512)
        assert session.cache[0].offset == 701 and list(store._entries) == [("m", tuple(ids[:699]))]
        assert not session.finish()


def test_session_stores_only_snapshots_that_sit_on_the_grid(fake_mx):
    store = VLMPromptSnapshotStore(max_bytes = 10**9)
    language_model = FakeLanguageModel()
    ids = list(range(600))
    with Session(store, "m", language_model, make_cache) as session:
        session.find_prefix_length(ids)
        language_model(ids[:150], cache = session.cache)
        language_model(ids[150:300], cache = session.cache)
        language_model(ids[300:], cache = session.cache)
        language_model([0], cache = session.cache)
        assert session._forward.record.snapshot is not None and session._forward.record.tail
        assert not session.finish()
    with Session(store, "m", language_model, lambda: [FakeState()]) as session:
        session.find_prefix_length(ids[:300])
        run_generation(session._forward._language_model, ids[:300], session.cache, 0)
        assert session._forward.record.snapshot is not None and not session.finish()
    with Session(store, "m", language_model, make_cache) as session:
        assert not session.finish() and session.cache[0].offset == 0
    assert len(store) == 0

    layout = lambda: [FakeCacheList(FakeKV(), FakeState()), FakeCacheList(FakeKV(), FakeState())]
    with Session(store, "m", language_model, layout) as session:
        session.find_prefix_length(ids)
        run_generation(language_model, ids, session.cache, 0)
        assert session.finish()
    entries, prefix = store.lookup("m", ids + [-1], limit = 512)
    assert prefix == 512 and entries[1].caches[0].offset == 512
    assert store.nbytes == 2 * 4 * (512 + 1 + 1) + 2 * 4 * (599 + 1 + 1)


def _media_prompt(n):
    ids = list(range(1000, 1000 + n))
    ids[500:601] = [9] * 101
    return ids


def test_session_serves_and_stores_only_prefixes_past_the_last_media_token(fake_mx):
    store = VLMPromptSnapshotStore(max_bytes = 10**9)
    language_model = FakeLanguageModel()
    session, _cache, stored = _generate(
        store, language_model, _media_prompt(700), media_token_ids = (9,)
    )
    assert not stored and len(store) == 0 and session.reused_tokens == 0 and not fake_mx
    _generate(store, language_model, _media_prompt(900), media_token_ids = (9,))
    assert store.lookup("m", _media_prompt(900) + [-1], limit = 768)[1] == 768
    session, _cache, _stored = _generate(
        store, language_model, _media_prompt(1000), media_token_ids = (9,)
    )
    assert session.reused_tokens == 768
    session, _cache, stored = _generate(
        store, language_model, _media_prompt(1100), honour_reuse = False, media_token_ids = (9,)
    )
    assert session.reused_tokens == 768 and not stored and len(store) == 1

    store.clear()
    prompt = _media_prompt(1000)
    _generate(store, language_model, prompt, media_token_ids = (9,))
    prompt.append(5)
    store.store("t", prompt[:768], _snapshot(prompt[:768]))
    store.store("m", [5] + prompt[1:768], _snapshot(range(768)))
    session = lambda **kw: Session(
        store, "m", language_model, make_cache, media_token_ids = (9,), **kw
    )
    assert session().find_prefix_length(prompt) == 768 and len(store) == 4
    assert session(releases_unserved = True).find_prefix_length(prompt) == 768
    assert list(store._entries) == [("m", tuple(prompt[:768]))]
    assert session(releases_unserved = True).find_prefix_length(_media_prompt(700)) == 0
    assert len(store) == 0 and store.nbytes == 0


@pytest.mark.parametrize("ring", (False, True))
def test_a_lone_batched_row_holds_its_snapshots_from_the_cache_it_hands_back(ring):
    mx = pytest.importorskip("mlx.core")
    cache = pytest.importorskip("mlx_vlm.models.cache")

    def positions(start, n):
        return mx.arange(start, start + n, dtype = mx.float32).reshape(1, 1, n, 1) * mx.ones(
            (1, 2, 1, 64)
        )

    def advance(entries, start, n):
        for entry in entries:
            if isinstance(entry, cache.ArraysCache):
                entry[0] = mx.full((1, 64), float(start + n))
            else:
                entry.update_and_fetch(positions(start, n), -positions(start, n))
        mx.eval([entry.state for entry in entries])

    def prefill(
        row,
        ids,
        decoded = 6,
    ):
        entries, lengths = row.open(ids)
        done = cache_entries_offset(entries)
        while done < len(ids) - 1:
            take = min(STEP - done % STEP, len(ids) - 1 - done)
            advance(entries, done, take)
            done += take
            if done in lengths:
                row.checkpoint(done, entries)
        for step in range(decoded):
            advance(entries, done + step, 1)
        # What unsloth-zoo hands back for a lone row: its own entries.
        return lambda: entries

    ids, alone = list(range(1000)), [True]
    kv = (lambda: cache.RotatingKVCache(max_size = 4096)) if ring else cache.KVCache
    plain = lambda: [kv(), cache.ArraysCache(size = 1), kv()]
    window = lambda: cache.RotatingKVCache(max_size = 256)
    wrapped = lambda: [window(), cache.ArraysCache(size = 1), window()]
    store = VLMPromptSnapshotStore(max_bytes = 200_000)

    def holds(lengths):
        assert [len(item[1]) for item in store._held._entries] == lengths
        for (_key, prefix), (entries, _nbytes) in store._held._entries.items():
            n = len(prefix)
            assert [entry.offset for entry in entries[::2]] == [n, n]
            assert entries[1][0][0, 0].item() == n
            for entry in entries[::2]:
                assert mx.array_equal(entry.keys, positions(0, n)).item()
                assert mx.array_equal(entry.values, -positions(0, n)).item()

    row = VLMBatchRowCache(store, "m", plain, step = STEP, alone = lambda: alone[0])
    take = prefill(row, ids[:700])
    assert not store._held and len(store) == 0
    assert [e.keys for _n, skeleton, _r in row._skeletons for e in skeleton[::2]] == [None] * 4
    row.retire(take)
    holds([512, 699])
    # A regenerate banks nothing, so it holds again what served it and the boundary it extends.
    row = VLMBatchRowCache(store, "m", plain, step = STEP, alone = lambda: alone[0])
    take = prefill(row, ids[:700])
    assert not store._held and cache_entries_offset(take()) == 705
    row.retire(take)
    holds([512, 699])
    # A prompt one past the boundary resumes it whole and holds it again, not the longer replay.
    row = VLMBatchRowCache(store, "m", plain, step = STEP, alone = lambda: alone[0])
    row.retire(prefill(row, ids[:513]))
    holds([512])
    # The next turn resumes the boundary and holds only what it banks itself.
    row = VLMBatchRowCache(store, "m", plain, step = STEP, alone = lambda: alone[0])
    take = prefill(row, ids[:900])
    assert [n for n, _skeleton, _replay in row._skeletons] == [768, 899]
    row.retire(take)
    holds([768, 899])
    # A short turn inside that boundary's chunk banks only its replay and holds the boundary again.
    row = VLMBatchRowCache(store, "m", plain, step = STEP, alone = lambda: alone[0])
    row.retire(prefill(row, ids[:950]))
    holds([768, 949])

    # Held only by a row alone in the batch, from a cache holding its rows in order, and only
    # where unsloth-zoo hands that cache back.
    short = prefill(VLMBatchRowCache(None, "m", plain), ids[:600], decoded = 0)
    slid = prefill(VLMBatchRowCache(None, "m", wrapped), ids[:700], decoded = 0)
    for layout, alone[0], handed in (
        (plain, False, lambda: pytest.fail("taken")),
        (plain, True, short),
        (plain, True, slid),
        (wrapped, True, lambda: wrapped()),
    ):
        row = VLMBatchRowCache(store, "m", layout, step = STEP, alone = lambda: alone[0])
        prefill(row, ids[:700])
        assert bool(row._skeletons) is (layout is plain)
        row.retire(handed)
        assert not store._held and len(store) == 0 and not row._skeletons
    prefill(VLMBatchRowCache(store, "m", plain, step = STEP), ids[:700])
    assert not store._held and len(store) == 0
    # Within the budget a row banks copies as it prefills.
    store = VLMPromptSnapshotStore(max_bytes = 10**9)
    row = VLMBatchRowCache(store, "m", plain, step = STEP, alone = lambda: True)
    prefill(row, ids[:700])
    assert len(store) == 2 and not row._skeletons


def _prefill_row(state, token_ids):
    """unsloth-zoo's batched row: resume the opened cache, checkpoint where a grid chunk ends."""
    cache, lengths = state.open(token_ids)
    done = prefix = cache_entries_offset(cache)
    while len(token_ids) - done > 1:
        take = min(STEP, len(token_ids) - done - 1)
        for entry in cache:
            entry.advance(token_ids[done : done + take])
        done += take
        if done in lengths:
            state.checkpoint(done, cache)
    return prefix, cache


@pytest.mark.parametrize("media", (False, True))
def test_a_batched_row_resumes_and_banks_what_the_single_path_does(fake_mx, media):
    single, batched = (VLMPromptSnapshotStore(max_bytes = 10**9) for _ in range(2))
    media_ids = (9,) if media else ()
    prompts = [_media_prompt(n) if media else list(range(n)) for n in (700, 700, 900, 1000)]
    prompts.append(prompts[-1] + [1])
    for prompt in prompts:
        for store in (single, batched):
            store.store("m", [-1], _snapshot([-1]))
        session, _cache, _stored = _generate(
            single, FakeLanguageModel(), prompt, media_token_ids = media_ids, releases_unserved = media
        )
        row = VLMBatchRowCache(batched, "m", make_cache, media_ids, media, step = STEP)
        prefix, cache = _prefill_row(row, prompt)
        assert prefix == session.reused_tokens and cache[0].keys.rows == prompt[:-1]
        assert set(batched._entries) == set(single._entries)
        assert batched._replays == single._replays
    entries, prefix = batched.lookup("m", prompts[-1] + [0], limit = 768)
    assert prefix == 768 and entries[0].keys.rows == prompts[-1][:768]
    cache, lengths = VLMBatchRowCache(None, "m", make_cache).open(prompts[-1])
    assert cache_entries_offset(cache) == 0 and not lengths


class FakeMediaBlock:
    """A prompt's first ``rows`` rows are its image; prefilled outside the count."""

    fail = False
    prefilled = 0
    store = None

    def __init__(self, rows):
        self.block_rows = rows

    def rows(self, token_ids):
        return self.block_rows if len(token_ids) > self.block_rows else 0

    def prefill(self, forward, rows):
        if self.fail:
            raise RuntimeError("no block")
        self.prefilled += 1
        self.entries_at_prefill = len(self.store)
        entries = make_cache()
        forward.unrecorded(list(range(rows)), cache = entries)
        return entries


def _block_prompt(n):
    return [9] * 100 + list(range(1000, 900 + n))


def test_session_prefills_the_media_block_and_chains_from_it(fake_mx, caplog):
    store = VLMPromptSnapshotStore(max_bytes = 10**9)
    language_model = FakeLanguageModel()
    block = FakeMediaBlock(100)
    block.store = store
    store.store("t", [1, 2, 3], _snapshot(range(3)))
    kwargs = dict(media_token_ids = (9,), media_block = block, releases_unserved = True)
    generate = lambda n, **kw: _generate(store, language_model, _block_prompt(n), **kwargs, **kw)
    session, cache, stored = generate(500)
    assert session.reused_tokens == 100 and block.prefilled == 1 and stored
    assert block.entries_at_prefill == 0 and session.produced_seconds > 0
    assert session.produced_tokens == 100 and cache[0].offset == 500 and len(fake_mx) == 3
    assert [len(item[1]) for item in store._entries] == [100, 356, 499]
    session, _cache, stored = generate(700)
    assert session.reused_tokens == 356 and block.prefilled == 1 and stored
    assert session.produced_tokens == 0
    assert [len(item[1]) for item in store._entries] == [356, 612, 699]
    store.clear()
    session, _cache, stored = generate(300)
    assert session.reused_tokens == 100 and block.prefilled == 2 and stored
    assert [len(item[1]) for item in store._entries] == [100, 299]
    session, _cache, stored = generate(500, honour_reuse = False)
    assert session.reused_tokens == 100 and block.prefilled == 2
    assert not stored and len(store) == 0
    session, _cache, stored = generate(200, honour_reuse = False)
    assert session.reused_tokens == 100 and not stored

    # Over the budget the block is not held beside the generation that produced it.
    small, prompt = VLMPromptSnapshotStore(max_bytes = 100), _block_prompt(500)
    block.store = small
    with Session(small, "m", language_model, make_cache, **kwargs) as session:
        assert session.find_prefix_length(prompt) == 100
        run_generation(language_model, prompt, session.cache, 100)
        assert not small._held
        assert session.finish() and [len(item[1]) for item in small._held._entries] == [356, 499]

    store, failing = VLMPromptSnapshotStore(max_bytes = 10**9), FakeMediaBlock(100)
    failing.fail = True
    fake_mx.clear()
    with caplog.at_level("INFO"):
        session, cache, stored = _generate(
            store,
            FakeLanguageModel(),
            _block_prompt(500),
            media_token_ids = (9,),
            media_block = failing,
        )
    assert session.reused_tokens == 0 and cache[0].offset == 500
    assert not stored and len(store) == 0 and not fake_mx
    assert session.produced_tokens == 0 and session.produced_seconds > 0
    assert "media block not prefilled" in caplog.text


class FakeTypeIds:
    def __init__(self, n, start):
        self.shape = (1, n)
        self.start = start

    def __getitem__(self, item):
        offset = item[1].start or 0
        return FakeTypeIds(self.shape[1] - offset, self.start + offset)


class FakeHost:
    def __init__(self):
        self.seen = []

    def chunked_prefill_policy(self, **kwargs):
        self.seen.append(kwargs["prefill_kwargs"]["mm_token_type_ids"].start)
        return True


def test_session_feeds_the_chunking_policy_only_the_rows_past_the_cache(fake_mx):
    store = VLMPromptSnapshotStore(max_bytes = 10**9)
    host, plain = FakeHost(), object()
    resumed = make_cache()
    resumed[0].advance(range(100))
    ask = lambda cache, n: host.chunked_prefill_policy(
        prompt_cache = cache, prefill_kwargs = {"mm_token_type_ids": FakeTypeIds(n, 0)}
    )
    with Session(store, "m", FakeLanguageModel(), make_cache, policy_hosts = (host, plain)):
        assert ask(resumed, 300) is True
        ask(make_cache(), 300)
        ask(resumed, 50)
    assert host.seen == [100, 0, 0]
    assert "chunked_prefill_policy" not in vars(host)
    ask(resumed, 300)
    assert host.seen[-1] == 0


def test_snapshot_module_imports_mlx_only_when_copying(monkeypatch):
    monkeypatch.delitem(sys.modules, "mlx.core", raising = False)
    monkeypatch.setitem(sys.modules, "mlx", None)
    monkeypatch.setitem(sys.modules, "mlx.core", None)
    assert shape_stable_prefix(300, step = STEP) == 256
    with pytest.raises(ImportError):
        copy_cache_entries([FakeKV()])
    assert snapshots.VLM_PROMPT_CACHE_ENTRIES == 6


def test_session_defaults_to_mlx_vlm_prefill_step(fake_mx):
    store = VLMPromptSnapshotStore(max_bytes = 10**9)
    language_model = FakeLanguageModel()
    ids = list(range(VLM_PREFILL_STEP + 60))
    with VLMPromptCacheSession(store, "m", language_model, make_cache) as session:
        assert session.step == vlm_prefill_step() == VLM_PREFILL_STEP == 2048
        assert session.find_prefix_length(ids) == 0
        run_generation(language_model, ids, session.cache, 0, step = session.step)
        assert session.finish()
    assert store.lookup("m", ids + [-1], limit = len(ids))[1] == 2048
    assert shape_stable_prefix(2048) == 0 and shape_stable_prefix(2049) == 2048


@pytest.mark.parametrize(
    "default",
    [None, 0, -1, "2048", object()],
    ids = ["none", "zero", "negative", "string", "unreadable"],
)
def test_a_step_no_grid_can_be_built_on_falls_back_instead_of_raising(monkeypatch, default):
    """The step divides every boundary; one that cannot must fall back, not fail generation."""
    module = types.ModuleType("mlx_vlm.generate.common")
    module.DEFAULT_PREFILL_STEP_SIZE = default
    monkeypatch.setitem(sys.modules, "mlx_vlm", types.ModuleType("mlx_vlm"))
    monkeypatch.setitem(sys.modules, "mlx_vlm.generate", types.ModuleType("mlx_vlm.generate"))
    monkeypatch.setitem(sys.modules, "mlx_vlm.generate.common", module)

    assert vlm_prefill_step() == VLM_PREFILL_STEP == 2048
    assert shape_stable_prefix(3000, step = vlm_prefill_step()) == 2048


def test_a_step_mlx_vlm_names_is_used_even_when_it_is_not_the_default(monkeypatch):
    module = types.ModuleType("mlx_vlm.generate.common")
    module.DEFAULT_PREFILL_STEP_SIZE = 512
    monkeypatch.setitem(sys.modules, "mlx_vlm", types.ModuleType("mlx_vlm"))
    monkeypatch.setitem(sys.modules, "mlx_vlm.generate", types.ModuleType("mlx_vlm.generate"))
    monkeypatch.setitem(sys.modules, "mlx_vlm.generate.common", module)

    assert vlm_prefill_step() == 512


def test_a_module_that_names_no_step_falls_back(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlx_vlm", types.ModuleType("mlx_vlm"))
    monkeypatch.setitem(sys.modules, "mlx_vlm.generate", types.ModuleType("mlx_vlm.generate"))
    monkeypatch.setitem(
        sys.modules, "mlx_vlm.generate.common", types.ModuleType("mlx_vlm.generate.common")
    )

    assert vlm_prefill_step() == VLM_PREFILL_STEP


class FakeSlottedKV:
    """A cache whose fields live in ``__slots__``: ``vars()`` cannot reach them."""

    __slots__ = ("keys", "offset")
    state = property(lambda self: self.keys)

    def __init__(self):
        self.keys = FakeArray([1, 2, 3])
        self.offset = 3


class FakeDictKV:
    """A cache holding its arrays in a dict, as a hybrid layout would."""

    state = property(lambda self: self.tensors)

    def __init__(self):
        self.tensors = {"keys": FakeArray([1, 2, 3])}
        self.offset = 3


def test_copy_walks_a_dict_held_state(fake_mx):
    """A dict of arrays must be copied, or the next generation writes into the stored copy."""
    live = [FakeCacheList(FakeDictKV())]
    saved = copy_cache_entries(live)
    assert saved[0].caches[0].tensors is not live[0].caches[0].tensors
    assert saved[0].caches[0].tensors["keys"] is not live[0].caches[0].tensors["keys"]
    assert cache_entries_nbytes(saved) == 12
    release_cache_entries(saved)
    assert saved[0].caches[0].tensors["keys"] is None


def test_copy_refuses_a_cache_it_cannot_walk_rather_than_sharing_it(fake_mx):
    """Nested unrecognised caches are refused, not shared: sharing lets a turn advance a live cache
    and costs the next turn its answer, while refusing only costs reuse."""
    with pytest.raises(TypeError):
        copy_cache_entries([FakeCacheList(FakeSlottedKV())])


def test_copy_shares_buffers_yet_keeps_what_the_live_cache_overwrites():
    mx = pytest.importorskip("mlx.core")
    cache = pytest.importorskip("mlx_vlm.models.cache")
    np = pytest.importorskip("numpy")

    def rows(start, n):
        return mx.arange(start * 128, (start + n) * 128, dtype = mx.float32).reshape(1, 2, n, 64)

    kv, ring = cache.KVCache(), cache.RotatingKVCache(max_size = 16)
    for entry in (kv, ring):
        entry.update_and_fetch(rows(0, 16), -rows(0, 16))
    live = [kv, ring]
    mx.eval([entry.state for entry in live])
    held = [np.array(entry.keys) for entry in live]

    mx.synchronize()
    before = mx.get_active_memory()
    saved = copy_cache_entries(live)
    mx.synchronize()
    assert mx.get_active_memory() - before < cache_entries_nbytes(live) // 2

    for entry in live:
        entry.update_and_fetch(rows(16, 1), -rows(16, 1))
    mx.eval([entry.state for entry in live])
    assert ring.keys[0, 0, 0, 0].item() == 16 * 128
    for entry, keys in zip(saved, held):
        assert entry.offset == 16 and np.array_equal(np.array(entry.keys), keys)

    mx.synchronize()
    before = mx.get_active_memory()
    conv = cache.ArraysCache(size = 1)
    conv[0] = mx.arange(4096 * 256, dtype = mx.float32).reshape(1, 4096, 256)[:, -3:, :]
    mx.eval(conv[0])
    window = np.array(conv[0])
    (saved,) = copy_cache_entries([conv])
    del conv
    mx.synchronize()
    assert mx.get_active_memory() - before < 4096 * 256 * 4 // 2
    assert np.array_equal(np.array(saved[0]), window)


def test_session_restores_the_model_when_one_host_is_named_twice(fake_mx):
    """policy_hosts may list one object twice (the language model falls back to the model), so
    unpatching twice must not strand the recording class."""
    store = VLMPromptSnapshotStore(max_bytes = 10**9)
    host = FakeHost()
    before = type(host)
    with Session(store, "m", host, make_cache, policy_hosts = (host, host)):
        pass
    assert type(host) is before
    assert "chunked_prefill_policy" not in vars(host)


def test_session_restores_the_model_even_if_unpatching_fails(fake_mx):
    """A failed unpatch must still restore the model, or later requests use a stale forward."""
    store = VLMPromptSnapshotStore(max_bytes = 10**9)
    host = FakeHost()
    before = type(host)
    session = Session(store, "m", host, make_cache, policy_hosts = (host,))
    session.__enter__()
    del host.chunked_prefill_policy
    session.__exit__(None, None, None)
    assert type(host) is before


def test_store_holds_the_kv_rows_nested_snapshots_share_once():
    mx = pytest.importorskip("mlx.core")
    cache = pytest.importorskip("mlx_vlm.models.cache")

    def advance(entries, start, n):
        for entry in entries[:2]:
            rows = mx.arange(start * 128, (start + n) * 128, dtype = mx.float32).reshape(1, 2, n, 64)
            entry.update_and_fetch(rows, -rows)
        entries[2][0] = mx.full((1, 64), float(start + n))
        mx.eval([entry.state for entry in entries])

    short = [cache.KVCache(), cache.RotatingKVCache(max_size = 4096, keep = 4)]
    short.append(cache.ArraysCache(size = 1))
    advance(short, 0, 256)
    served = [entry.state[0] * 1 for entry in short[:2]]
    mx.eval(served)
    extended = copy_cache_entries(short)
    advance(extended, 256, 256)
    ids = list(range(1000, 1512))
    shared = sum(entry.keys.nbytes + entry.values.nbytes for entry in short[:2])

    store = VLMPromptSnapshotStore(1 << 30)
    store.store("m", ids[:256], short)
    mx.synchronize()
    before = mx.get_active_memory()
    store.store("m", ids, extended)
    mx.eval([entry.state for entry in short])
    mx.synchronize()
    assert before - mx.get_active_memory() >= shared
    assert store.nbytes == cache_entries_nbytes(extended) + cache_entries_nbytes(short) - shared

    entries, reused = store.lookup("m", ids[:300], 300)
    assert reused == 256 and entries is short
    for entry, rows in zip(entries, served):
        assert mx.array_equal(entry.state[0], rows).item()
    assert entries[2][0][0, 0].item() == 256.0
    base = ("m", tuple(ids))
    assert list(store._entries)[-1] == base
    store.retain(("m", tuple(ids[:256])))
    assert len(store) == 2
    store.discard(base)
    assert len(store) == 0 and store.nbytes == 0

    store.store("m", ids[:256], short)
    store.store("m", ids, extended)
    store.store("m", ids, copy_cache_entries(extended))
    assert len(store) == 2
    lone = [cache.ArraysCache(size = 1)]
    lone[0][0] = mx.zeros((1, 64))
    store.store("m", ids, lone)
    assert len(store) == 1 and store.nbytes == cache_entries_nbytes(lone)

    windows = [[cache.RotatingKVCache(max_size = 16)] for _ in range(2)]
    for entries, n in zip(windows, (32, 64)):
        entries.append(cache.KVCache())
        entries.append(cache.ArraysCache(size = 1))
        advance(entries, 0, n // 2)
        advance(entries, n // 2, n // 2)
    assert snapshots.share_kv_rows(*windows) == windows[0][1].keys.nbytes * 2
    nested = [[cache.KVCache(), cache.RotatingKVCache(max_size = 16)] for _ in range(3)]
    for entries, n in zip(nested, (8, 12, 32)):
        entries.append(cache.ArraysCache(size = 1))
        advance(entries, 0, min(n, 16))
        if n > 16:
            advance(entries, 16, n - 16)
        store.store("w", ids[:n], entries)
    store.discard(("w", tuple(ids[:12])))
    assert len(store) == 2
    assert store.nbytes == cache_entries_nbytes(lone) + cache_entries_nbytes(nested[2])


def test_a_replay_reads_through_no_snapshot_yet_serves_as_a_base():
    mx = pytest.importorskip("mlx.core")
    cache = pytest.importorskip("mlx_vlm.models.cache")

    def entries(n, last_chunk):
        rows = mx.arange(n * 64, dtype = mx.float32).reshape(1, 1, n, 64)
        rows = rows + last_chunk * (mx.arange(n) >= 512).reshape(1, 1, n, 1)
        kv = cache.KVCache()
        kv.update_and_fetch(rows, -rows)
        mx.eval(kv.keys, kv.values)
        return [kv]

    ids = list(range(1000))
    for order in ((600, 768), (768, 600)):
        store = VLMPromptSnapshotStore(1 << 30)
        for n in order:
            store.store("m", ids[:n], entries(n, 0.5 if n == 600 else 0.0), replay = n == 600)
        replay, n = store.lookup("m", ids[:601], limit = 512)
        assert n == 600 and replay[0].keys[0, 0, 599, 0].item() == 599 * 64 + 0.5
    grid = ("m", tuple(ids[:512]))
    store.store("m", ids[:512], entries(512, 0.0))
    assert store._bases[grid] == ("m", tuple(ids[:600])) and store._entries[grid][1] == 0
    store.store("m", ids[:580], entries(580, 0.25), replay = True)
    assert ("m", tuple(ids[:600])) not in store._entries
    assert store._bases[grid] == ("m", tuple(ids[:580])) and store._entries[grid][1] == 0
    divergent = ids[:300] + [-1] * 700
    store.store("m", divergent[:590], entries(590, 0.75), replay = True)
    longer = ("m", tuple(ids[:768]))
    assert set(store._entries) == {grid, longer, ("m", tuple(divergent[:590]))}
    assert store._bases[grid] == longer and store._entries[grid][1] == 0
    assert store._entries[grid][0][0].keys[0, 0, 511, 0].item() == 511 * 64
    assert store.nbytes == cache_entries_nbytes(store._entries[longer][0]) + cache_entries_nbytes(
        store._entries[("m", tuple(divergent[:590]))][0]
    )
    store.discard(("m", tuple(divergent[:590])))
    assert not store._replays


def test_an_edit_keeps_the_old_branch_to_return_to():
    mx = pytest.importorskip("mlx.core")
    cache = pytest.importorskip("mlx_vlm.models.cache")

    def entries(ids):
        rows = mx.array(ids, dtype = mx.float32).reshape(1, 1, len(ids), 1)
        kv = cache.KVCache()
        kv.update_and_fetch(rows, -rows)
        mx.eval(kv.keys, kv.values)
        return [kv]

    def request(ids):
        _, reused = store.lookup("m", ids, len(ids) - 1)
        for n in range(256, len(ids), 256):
            if n > reused:
                store.store("m", ids[:n], entries(ids[:n]))
        store.store("m", ids[:-1], entries(ids[:-1]), replay = True)
        return reused

    store = VLMPromptSnapshotStore(1 << 30)
    turns = [list(range(n)) for n in (557, 931, 1311, 1674)]
    for ids in turns:
        request(ids)
    assert request(turns[0] + [-1] * 393) == 512
    kept, reused = store.lookup("m", turns[3], 1673)
    assert reused == 1536
    assert mx.array_equal(kept[0].keys[0, 0, :, 0], mx.arange(1536, dtype = mx.float32)).item()
    assert store.nbytes == sum(nbytes for _, nbytes in store._entries.values())
    assert all(base in store._entries for base in store._bases.values())
    store.clear()


@pytest.mark.parametrize("keep", [0, 4])
@pytest.mark.parametrize("next_rows", [1, 5])
def test_a_compacted_window_updates_exactly_as_the_full_one(keep, next_rows):
    mx = pytest.importorskip("mlx.core")
    cache = pytest.importorskip("mlx_vlm.models.cache")

    def rows(start, n):
        return mx.arange(start * 8, (start + n) * 8, dtype = mx.float32).reshape(1, 1, n, 8)

    def window():
        ring = cache.RotatingKVCache(max_size = 16, keep = keep)
        for start in (0, 24):
            ring.update_and_fetch(rows(start, 24), -rows(start, 24))
        return ring

    full, compact = window(), window()
    VLMPromptSnapshotStore(1 << 30).store("m", [1], [[compact]])
    assert compact.keys.shape[2] == 16 < full.keys.shape[2]
    for start, n in ((48, next_rows), (48 + next_rows, 1)):
        got = compact.update_and_fetch(rows(start, n), -rows(start, n))
        want = full.update_and_fetch(rows(start, n), -rows(start, n))
        for a, b in zip(got, want):
            assert a.shape == b.shape and mx.array_equal(a, b).item()


def test_a_short_turn_holds_again_the_boundary_it_resumed_from():
    mx = pytest.importorskip("mlx.core")
    cache = pytest.importorskip("mlx_vlm.models.cache")

    def positions(start, n):
        return mx.arange(start, start + n, dtype = mx.float32).reshape(1, 1, n, 1) * mx.ones(
            (1, 2, 1, 8)
        )

    class Model:
        def __call__(
            self,
            inputs,
            cache = None,
            **_kwargs,
        ):
            start = cache_entries_offset(cache)
            for entry in cache:
                entry.update_and_fetch(
                    positions(start, len(inputs)), -positions(start, len(inputs))
                )
            mx.eval([entry.state for entry in cache])

    model, ids = Model(), list(range(1000))
    make = lambda: [cache.KVCache(), cache.KVCache()]

    def turn(store, n):
        with Session(store, "m", model, make) as session:
            prefix = session.find_prefix_length(ids[:n])
            run_generation(model, ids[:n], session.cache, prefix)
            for step in range(4):
                model([0], cache = session.cache)
            session.finish()
        return prefix

    def held(store):
        for (_key, prefix), (entries, _nbytes) in store._held._entries.items():
            n = len(prefix)
            assert [entry.offset for entry in entries] == [n, n]
            assert mx.array_equal(entries[0].keys[..., :n, :], positions(0, n)).item()
        return [len(item[1]) for item in store._held._entries]

    store = VLMPromptSnapshotStore(max_bytes = 1)
    assert turn(store, 700) == 0 and held(store) == [512, 699]
    # Each short turn stays inside the 512 boundary's chunk, so it banks only its replay.
    for n in (720, 740, 760):
        assert turn(store, n) == 512
        assert held(store) == [512, n - 1]
    assert turn(store, 800) == 512 and turn(store, 800) == 799
