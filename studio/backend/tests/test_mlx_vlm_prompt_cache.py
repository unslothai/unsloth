import logging
import sys
import types

import pytest

from core.inference import mlx_inference as snapshots
from core.inference.mlx_inference import (
    VLM_PROMPT_CACHE_PREFILL_STEP as STEP,
    RecordingForward,
    VLMPromptCacheSession,
    VLMPromptSnapshotStore,
    cache_entries_nbytes,
    cache_entries_offset,
    copy_cache_entries,
    release_cache_entries,
    media_prefix_end,
    shape_stable_prefix,
)


@pytest.fixture(autouse = True)
def _plain_logger(monkeypatch):
    """The backend logs through structlog, which caplog does not see."""
    monkeypatch.setattr(snapshots, "logger", logging.getLogger(snapshots.__name__))


class FakeArray:
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
    mlx_pkg = types.ModuleType("mlx")
    mlx_pkg.core = mlx_core
    monkeypatch.setitem(sys.modules, "mlx", mlx_pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", mlx_core)
    return evaluated


class FakeKV:
    """Rows appended per forward, like KVCache; ``offset`` counts them."""

    state = property(lambda self: self.keys)

    def __init__(self):
        self.keys = FakeArray([])
        self.offset = 0

    def advance(self, tokens):
        self.keys = FakeArray(self.keys.rows + list(tokens))
        self.offset += len(tokens)


class FakeState:
    """Two arrays rewritten per forward, like an ArraysCache of recurrent state."""

    state = property(lambda self: self.cache)

    def __init__(self):
        self.cache = [None, None]

    def advance(self, tokens):
        self.cache[0] = FakeArray(tokens[-1:])
        self.cache[1] = FakeArray([sum(tokens)])


class FakeCacheList:
    """Nested caches per layer, like CacheList; no offset of its own."""

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
    **kwargs,
):
    """mlx-vlm's prefill loop: grid chunks over all but the last token, then it.

    ``between`` runs on the cache after each chunk, where generate_step
    quantizes it.
    """
    n = len(token_ids)
    pos = start
    while n - pos > 1:
        take = min(STEP, n - pos - 1)
        language_model(token_ids[pos : pos + take], cache = cache, **kwargs)
        if between is not None:
            between(cache)
        pos += take
    language_model(token_ids[n - 1 :], cache = cache, **kwargs)


@pytest.mark.parametrize(
    ("tokens", "prefix"),
    [(0, 0), (1, 0), (256, 0), (257, 256), (512, 256), (513, 512), (2049, 2048), (4018, 3840)],
)
def test_shape_stable_prefix_is_the_last_whole_chunk_before_the_held_token(tokens, prefix):
    assert shape_stable_prefix(tokens) == prefix


def test_copy_cache_entries_duplicates_every_array_and_keeps_scalars(fake_mx):
    kv, state = FakeKV(), FakeState()
    kv.advance([1, 2, 3])
    state.advance([1, 2, 3])
    state.extra = (FakeArray([9]), 7)
    copies = copy_cache_entries([kv, state])

    assert copies[0].offset == 3 and copies[0].keys.rows == [1, 2, 3]
    assert copies[0].keys is not kv.keys
    assert copies[1].cache[1].rows == [6] and copies[1].cache is not state.cache
    assert copies[1].cache[0] is not state.cache[0] and copies[1].cache[1] is not state.cache[1]
    assert copies[1].extra[0] is not state.extra[0] and copies[1].extra[1] == 7
    kv.advance([4])
    state.advance([4])
    assert copies[0].keys.rows == [1, 2, 3] and copies[1].cache[0].rows == [3]
    # Evaluated as one batch so the copies own their data before anything moves on.
    assert len(fake_mx) == 1 and len(fake_mx[0]) == 4


def test_copy_cache_entries_refuses_an_entry_it_cannot_see_into(fake_mx):
    with pytest.raises(TypeError, match = "int cannot be copied"):
        copy_cache_entries([3])
    with pytest.raises(TypeError, match = "SimpleNamespace cannot be copied"):
        copy_cache_entries([types.SimpleNamespace(keys = FakeArray([1]))])


def test_copy_cache_entries_descends_into_composite_entries(fake_mx):
    nested = FakeCacheList(FakeKV(), FakeState())
    nested.advance([1, 2])
    pair = (FakeSimpleKV(), FakeSimpleKV())
    for entry in pair:
        entry.advance([5])
    copies = copy_cache_entries([nested, pair])

    assert copies[0].caches[0] is not nested.caches[0]
    assert copies[0].caches[0].keys is not nested.caches[0].keys
    assert copies[0].caches[1].cache is not nested.caches[1].cache
    assert isinstance(copies[1], tuple) and copies[1][1] is not pair[1]
    nested.advance([3])
    for entry in pair:
        entry.advance([6])
    assert copies[0].caches[0].keys.rows == [1, 2] and copies[0].caches[0].offset == 2
    assert copies[0].caches[1].cache[0].rows == [2]
    assert copies[1][1].keys.rows == [5]
    assert len(fake_mx[0]) == 5
    assert cache_entries_nbytes([nested, pair]) == 4 * (3 + 1 + 1 + 2 + 2)
    assert cache_entries_offset([pair]) is None
    assert cache_entries_offset([nested]) == 3


def test_nbytes_and_offset_read_the_entries(fake_mx):
    kv, state = FakeKV(), FakeState()
    kv.advance([1, 2])
    state.advance([1, 2])
    assert cache_entries_nbytes([kv, state]) == 4 * (2 + 1 + 1)
    assert cache_entries_offset([state, kv]) == 2
    assert cache_entries_offset([state]) is None


def test_media_prefix_end_is_the_row_after_the_last_media_token():
    ids = [1, 2, 9, 9, 9, 3, 4, 9, 5, 7, 6]
    assert media_prefix_end(ids, {9}) == 8
    assert media_prefix_end(ids, [9, 7]) == 10
    assert media_prefix_end(ids, ()) == 0
    assert media_prefix_end([1, 2, 3], {9}) == 0
    assert media_prefix_end([9], {9}) == 1


def test_recording_forward_copies_the_cache_after_the_chosen_forward(fake_mx):
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
    assert outputs == language_model.outputs
    snapshot = recording.record.snapshot
    assert snapshot[0].keys.rows == [1, 2, 3, 4] and snapshot[0].offset == 4
    assert snapshot[1].cache[1].rows == [7]
    assert cache[0].offset == 6 and snapshot[0] is not cache[0]
    # A forward without per-layer inputs to withhold passes them through.
    assert all(kw == {"per_layer_inputs": "kept"} for kw in language_model.seen_kwargs)
    assert type(language_model) is original_class
    assert "_studio_forward_record" not in vars(language_model)


def test_recording_forward_keeps_the_answer_when_the_cache_cannot_be_copied(fake_mx, caplog):
    language_model = FakeLanguageModel()
    opaque = [types.SimpleNamespace(advance = lambda _t: None)]
    with caplog.at_level("INFO"), RecordingForward(language_model) as recording:
        recording.record.capture_at = 1
        language_model([1], cache = opaque)
        output = language_model([2], cache = opaque)
    assert output is language_model.outputs[-1]
    assert recording.record.snapshot is None
    assert "SimpleNamespace cannot be copied" in caplog.text


def test_session_drops_a_served_snapshot_it_cannot_copy(fake_mx, monkeypatch):
    store = VLMPromptSnapshotStore(max_bytes = 10**9)
    language_model = FakeLanguageModel()
    ids = list(range(700))
    _generate(store, language_model, ids)

    def _fail(_arrays):
        raise MemoryError("no room for a copy")

    monkeypatch.setattr(sys.modules["mlx.core"], "eval", _fail)
    with VLMPromptCacheSession(store, "m", language_model, make_cache) as session:
        assert session.find_prefix_length(ids + [1]) == 512
        run_generation(language_model, ids + [1], session.cache, 512)
        # The request keeps its reuse; the consumed snapshot leaves the store.
        assert session.cache[0].offset == 701 and len(store) == 0
        assert not session.finish()


def test_session_stores_nothing_when_the_capture_failed(fake_mx):
    store = VLMPromptSnapshotStore(max_bytes = 10**9)
    ids = list(range(300))
    opaque = lambda: [types.SimpleNamespace(advance = lambda _t: None)]
    with VLMPromptCacheSession(store, "m", FakeLanguageModel(), opaque) as session:
        session.find_prefix_length(ids)
        run_generation(session._forward._language_model, ids, session.cache, 0)
        assert not session.finish()
    assert len(store) == 0


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
    with VLMPromptCacheSession(store, "m", FakeLanguageModel(), make_cache) as session:
        session.find_prefix_length(ids)
        run_generation(session._forward._language_model, ids, session.cache, 0, between = quantize)
        assert session.finish()
    entries, prefix = store.lookup("m", ids, limit = 512)
    assert prefix == 512 and type(entries[0]) is FakeQuantizedKV
    assert entries[0].keys.rows == ids[:512]


def test_recording_forward_captures_nothing_when_no_forward_is_chosen(fake_mx):
    language_model = FakeLanguageModel()
    with RecordingForward(language_model) as recording:
        language_model([1], cache = make_cache())
    assert recording.record.snapshot is None
    assert recording.record.forwards == 1


class Sliceable:
    """A per-layer-inputs array: rows labelled by prompt position."""

    def __init__(self, rows):
        self.rows = list(rows)
        self.shape = (1, len(self.rows), 8)

    def __getitem__(self, item):
        return Sliceable(self.rows[item[1]])


def test_recording_forward_slices_per_layer_inputs_from_the_resume_offset(fake_mx):
    language_model = FakeLanguageModel()
    original_class = type(language_model)
    embeds = lambda rows: types.SimpleNamespace(shape = (1, rows, 8))
    for resumed in (0, 512):
        # mlx-vlm computes the array for the uncached rows only.
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
            # A single token: the model computes its own from the id.
            language_model([0], inputs_embeds = embeds(1), cache = cache, per_layer_inputs = suffix)
        assert type(language_model) is original_class
        seen = [kw.get("per_layer_inputs") for kw in language_model.seen_kwargs[-3:]]
        assert seen[0].rows == list(range(resumed, resumed + 256))
        assert seen[1].rows == list(range(resumed + 256, resumed + 299))
        assert seen[2] is None
    assert language_model.seen_kwargs[0]["other"] == 1

    # Ids alone say how long the chunk is when no embeddings are passed.
    with RecordingForward(language_model):
        language_model(
            types.SimpleNamespace(shape = (1, 43)), cache = [], per_layer_inputs = Sliceable(range(300))
        )
    assert language_model.seen_kwargs[-1]["per_layer_inputs"].rows == list(range(43))


def test_recording_forward_leaves_per_layer_inputs_it_cannot_place(fake_mx):
    language_model = FakeLanguageModel()
    embeds = types.SimpleNamespace(shape = (1, 256, 8))
    short = Sliceable(range(100))
    cache = make_cache()
    cache[0].advance(range(512))
    with RecordingForward(language_model):
        language_model(inputs = [0] * 256, inputs_embeds = embeds, cache = cache, per_layer_inputs = short)
        language_model(inputs = [0] * 256, inputs_embeds = embeds, cache = [], per_layer_inputs = short)
    assert all(kw["per_layer_inputs"] is short for kw in language_model.seen_kwargs)


def test_recording_forward_restores_the_class_after_an_error(fake_mx):
    language_model = FakeLanguageModel()
    original_class = type(language_model)
    with pytest.raises(RuntimeError):
        with RecordingForward(language_model):
            raise RuntimeError("generation failed")
    assert type(language_model) is original_class


def test_release_cache_entries_drops_every_array_in_place(fake_mx):
    nested = FakeCacheList(FakeKV(), FakeState())
    nested.advance([1, 2])
    pair = (FakeSimpleKV(), FakeSimpleKV())
    for entry in pair:
        entry.advance([5])
    entries = [nested, pair]
    release_cache_entries(entries)
    assert not list(snapshots._arrays(entries, sys.modules["mlx.core"]))
    assert nested.caches[0].keys is None and nested.caches[0].offset == 2
    assert nested.caches[1].cache == [None, None] and pair[1].cache_length == 1
    assert not fake_mx


def _snapshot(rows):
    kv = FakeKV()
    kv.advance(rows)
    return [kv]


def test_store_serves_the_longest_matching_prefix_within_the_limit(fake_mx):
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


def test_store_evicts_least_recently_used_to_fit_the_budget(fake_mx):
    store = VLMPromptSnapshotStore(max_bytes = 4 * 700)
    a, b, c = [1] * 256, [2] * 256, [3] * 256
    assert store.store("m", a, _snapshot(a)) and store.store("m", b, _snapshot(b))
    store.lookup("m", a + [0], limit = 256)  # a is now the most recent
    assert store.store("m", c, _snapshot(c))
    assert store.lookup("m", b + [0], limit = 256) == (None, 0)
    assert store.lookup("m", a + [0], limit = 256)[1] == 256
    assert store.lookup("m", c + [0], limit = 256)[1] == 256
    assert store.nbytes == 4 * 512 and len(store) == 2


def test_store_refuses_a_snapshot_over_the_whole_budget(fake_mx):
    store = VLMPromptSnapshotStore(max_bytes = 4 * 255)
    assert not store.store("m", [1] * 256, _snapshot([1] * 256))
    assert len(store) == 0 and store.nbytes == 0


def test_store_retains_one_snapshot_and_drops_the_rest(fake_mx):
    store = VLMPromptSnapshotStore(max_bytes = 10**9)
    store.store("a", [1, 2], _snapshot([1, 2]))
    store.store("b", [1, 2, 3], _snapshot([1, 2, 3]))
    store.store("a", [1, 2, 3, 4], _snapshot([1, 2, 3, 4]))
    store.retain(("a", (1, 2, 3, 4)))
    assert list(store._entries) == [("a", (1, 2, 3, 4))]
    assert store.nbytes == cache_entries_nbytes(_snapshot([1, 2, 3, 4]))
    store.retain(None)
    assert len(store) == 0 and store.nbytes == 0


def test_store_replaces_a_prefix_and_caps_its_entry_count(fake_mx):
    store = VLMPromptSnapshotStore(max_bytes = 10**9, max_entries = 2)
    store.store("m", [1], _snapshot([1]))
    store.store("m", [1], _snapshot([1, 1]))
    assert len(store) == 1 and store.nbytes == 8
    store.store("m", [2], _snapshot([2]))
    store.store("m", [3], _snapshot([3]))
    assert len(store) == 2 and store.lookup("m", [1, 0], limit = 1) == (None, 0)
    store.clear()
    assert len(store) == 0 and store.nbytes == 0


def _generate(
    store,
    language_model,
    token_ids,
    *,
    honour_reuse = True,
    kwargs = None,
    media_token_ids = (),
):
    """One request the way mlx-vlm drives the duck type."""
    with VLMPromptCacheSession(
        store, "m", language_model, make_cache, media_token_ids = media_token_ids
    ) as session:
        prefix = session.find_prefix_length(token_ids)
        cache = session.cache
        if not honour_reuse:
            cache, prefix = make_cache(), 0
        run_generation(language_model, token_ids, cache, prefix, **(kwargs or {}))
        session.update(token_ids, cache)
        stored = session.finish()
    return session, cache, stored


def test_session_stores_a_composite_layout(fake_mx):
    store = VLMPromptSnapshotStore(max_bytes = 10**9)
    language_model = FakeLanguageModel()
    ids = list(range(600))
    layout = lambda: [FakeCacheList(FakeKV(), FakeState()), FakeCacheList(FakeKV(), FakeState())]
    with VLMPromptCacheSession(store, "m", language_model, layout) as session:
        session.find_prefix_length(ids)
        run_generation(language_model, ids, session.cache, 0)
        assert session.finish()
    entries, prefix = store.lookup("m", ids, limit = 512)
    assert prefix == 512 and entries[1].caches[0].offset == 512
    assert store.nbytes == 2 * 4 * (512 + 1 + 1)


def test_session_captures_the_boundary_on_a_cold_request(fake_mx):
    store = VLMPromptSnapshotStore(max_bytes = 10**9)
    ids = list(range(700))
    session, cache, stored = _generate(store, FakeLanguageModel(), ids)

    assert session.reused_tokens == 0 and stored
    assert cache[0].offset == 700
    entries, prefix = store.lookup("m", ids, limit = 512)
    assert prefix == 512 and entries[0].keys.rows == ids[:512]
    assert entries[1].cache[0].rows == [511]


def test_session_resumes_from_the_snapshot_and_leaves_a_copy_behind(fake_mx):
    store = VLMPromptSnapshotStore(max_bytes = 10**9)
    language_model = FakeLanguageModel()
    ids = list(range(700))
    _generate(store, language_model, ids)
    stored_entries = store.lookup("m", ids, limit = 512)[0]
    # Decoys a copy taken from the wrong entries would leave behind.
    store.store("m", ids[:256], _snapshot(range(9000, 9256)))
    store.store("t", ids[:512], _snapshot(range(9000, 9512)))

    longer = ids + list(range(700, 800))
    fake_mx.clear()
    session, cache, stored = _generate(store, language_model, longer)
    assert session.reused_tokens == 512 and cache is stored_entries
    kept = store.lookup("m", ids, limit = 512)[0]
    assert kept is not stored_entries and kept[0] is not cache[0]
    assert kept[0].offset == 512 and kept[0].keys.rows == ids[:512]  # the copy never moved
    assert kept[0].keys in fake_mx[0]  # the copy, evaluated at the first forward
    assert cache[0].offset == 800 and cache[0].keys.rows == longer
    assert stored and store.lookup("m", longer, limit = 768)[1] == 768

    # The same boundary again: served, nothing new captured.
    session, _cache, stored = _generate(store, language_model, longer + [1])
    assert session.reused_tokens == 768 and not stored
    assert len(store) == 4


def test_session_stores_what_the_snapshot_holds_when_reuse_was_declined(fake_mx):
    store = VLMPromptSnapshotStore(max_bytes = 10**9)
    language_model = FakeLanguageModel()
    ids = list(range(1100))
    _generate(store, language_model, ids)  # boundary 1024
    original = store.lookup("m", ids, limit = 1024)[0]
    store.store("t", ids[:512], _snapshot(range(9000, 9512)))  # a decoy
    fake_mx.clear()
    session, _cache, stored = _generate(
        store,
        language_model,
        ids + [1] * 300,
        honour_reuse = False,
    )
    # Offered 1024, ran from zero: the capture landed one chunk in, at 256.
    assert session.reused_tokens == 1024 and stored
    assert store.lookup("m", ids, limit = 256)[1] == 256
    # The declined offer is dropped and its arrays released, not copied (the
    # one copy evaluated is the capture): mlx-vlm keeps referencing it and
    # would decline it again.
    assert len(store) == 2 and store.lookup("m", ids, limit = 1024)[1] == 256
    assert store.lookup("t", ids, limit = 512)[1] == 512
    assert not list(snapshots._arrays(original, sys.modules["mlx.core"]))
    assert original[0].offset == 1024 and len(fake_mx) == 1


def test_session_drops_a_snapshot_that_is_not_on_the_grid(fake_mx):
    store = VLMPromptSnapshotStore(max_bytes = 10**9)
    language_model = FakeLanguageModel()
    ids = list(range(600))
    with VLMPromptCacheSession(store, "m", language_model, make_cache) as session:
        session.find_prefix_length(ids)
        # Prefilled on another grid: the capture forward ends at 300 rows, not 512.
        language_model(ids[:150], cache = session.cache)
        language_model(ids[150:300], cache = session.cache)
        language_model(ids[300:], cache = session.cache)
        assert session._forward.record.snapshot is not None
        assert not session.finish()
    assert len(store) == 0


def test_session_drops_a_snapshot_whose_rows_nothing_counts(fake_mx):
    store = VLMPromptSnapshotStore(max_bytes = 10**9)
    ids = list(range(300))
    with VLMPromptCacheSession(store, "m", FakeLanguageModel(), lambda: [FakeState()]) as session:
        session.find_prefix_length(ids)
        run_generation(session._forward._language_model, ids, session.cache, 0)
        assert session._forward.record.snapshot is not None
        assert not session.finish()
    assert len(store) == 0


def test_session_stores_nothing_before_a_prompt_is_seen(fake_mx):
    store = VLMPromptSnapshotStore(max_bytes = 10**9)
    with VLMPromptCacheSession(store, "m", FakeLanguageModel(), make_cache) as session:
        assert not session.finish()
    assert session.cache[0].offset == 0


def test_recording_forward_withholds_only_prompt_wide_position_ids(fake_mx):
    language_model = FakeLanguageModel()
    shaped = types.SimpleNamespace
    with RecordingForward(language_model):
        language_model(shaped(shape = (1, 256)), cache = [], position_ids = shaped(shape = (3, 1, 900)))
        language_model(shaped(shape = (1, 256)), cache = [], position_ids = shaped(shape = (3, 1, 256)))
        # Ids without a shape: the embeddings say how long the chunk is.
        language_model(
            inputs = [1, 2, 3, 4, 5],
            inputs_embeds = shaped(shape = (1, 5, 8)),
            cache = [],
            position_ids = shaped(shape = (1, 5)),
        )
        language_model(
            inputs = [1, 2, 3, 4, 5],
            inputs_embeds = shaped(shape = (1, 5, 8)),
            cache = [],
            position_ids = shaped(shape = (3, 1, 900)),
        )
        language_model([1, 2], cache = [], position_ids = None)
    kept = ["position_ids" in kw for kw in language_model.seen_kwargs]
    assert kept == [False, True, True, False, True]


def _media_prompt(n):
    """A prompt whose image placeholders occupy rows 500-600."""
    ids = list(range(1000, 1000 + n))
    ids[500:601] = [9] * 101
    return ids


def test_session_serves_and_stores_only_prefixes_past_the_last_media_token(fake_mx):
    store = VLMPromptSnapshotStore(max_bytes = 10**9)
    language_model = FakeLanguageModel()
    # Boundary 512 lies inside the image: nothing to serve, nothing worth
    # copying (no snapshot was evaluated), nothing kept.
    session, _cache, stored = _generate(
        store, language_model, _media_prompt(700), media_token_ids = (9,)
    )
    assert not stored and len(store) == 0 and session.reused_tokens == 0 and not fake_mx
    # Boundary 768 covers it, so the suffix mlx-vlm embeds is text.
    _session, _cache, stored = _generate(
        store, language_model, _media_prompt(900), media_token_ids = (9,)
    )
    assert stored and store.lookup("m", _media_prompt(900), limit = 768)[1] == 768
    session, _cache, _stored = _generate(
        store, language_model, _media_prompt(1000), media_token_ids = (9,)
    )
    assert session.reused_tokens == 768
    # Declined and run from zero: the capture lands at 256, inside the image,
    # and the declined offer is dropped.
    session, _cache, stored = _generate(
        store, language_model, _media_prompt(1100), honour_reuse = False, media_token_ids = (9,)
    )
    assert session.reused_tokens == 768 and not stored and len(store) == 0


def test_media_session_keeps_only_the_snapshot_it_serves(fake_mx):
    store = VLMPromptSnapshotStore(max_bytes = 10**9)
    language_model = FakeLanguageModel()
    prompt = _media_prompt(1000)
    _generate(store, language_model, prompt, media_token_ids = (9,))
    store.store("t", prompt[:768], _snapshot(prompt[:768]))
    store.store("m", [5] + prompt[1:768], _snapshot(range(768)))
    session = lambda **kw: VLMPromptCacheSession(
        store, "m", language_model, make_cache, media_token_ids = (9,), **kw
    )
    # A text request leaves the store alone.
    assert session().find_prefix_length(prompt) == 768 and len(store) == 3
    # A media request keeps what serves it; another conversation, another
    # image, and a boundary still inside the image leave nothing allocated.
    assert session(releases_unserved = True).find_prefix_length(prompt) == 768
    assert list(store._entries) == [("m", tuple(prompt[:768]))]
    assert session(releases_unserved = True).find_prefix_length(_media_prompt(700)) == 0
    assert len(store) == 0 and store.nbytes == 0


def test_snapshot_module_imports_mlx_only_when_copying(monkeypatch):
    monkeypatch.delitem(sys.modules, "mlx.core", raising = False)
    monkeypatch.setitem(sys.modules, "mlx", None)
    monkeypatch.setitem(sys.modules, "mlx.core", None)
    assert shape_stable_prefix(300) == 256
    with pytest.raises(ImportError):
        copy_cache_entries([FakeKV()])
    assert snapshots.VLM_PROMPT_CACHE_ENTRIES == 6
