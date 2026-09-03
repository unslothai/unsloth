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
    **kwargs,
):
    """mlx-vlm's prefill loop: grid chunks over all but the last token, then it."""
    n = len(token_ids)
    pos = start
    while n - pos > 1:
        take = min(STEP, n - pos - 1)
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
    with VLMPromptCacheSession(
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
        assert shape_stable_prefix(tokens, origin) == prefix, (tokens, origin)


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

    # A cache that cannot be copied still answers.
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
    with VLMPromptCacheSession(store, "m", FakeLanguageModel(), make_cache) as session:
        session.find_prefix_length(ids)
        run_generation(session._forward._language_model, ids, session.cache, 0, between = quantize)
        assert session.finish()
    entries, prefix = store.lookup("m", ids, limit = 512)
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
            # A single token: the model computes its own from the id.
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


def test_recording_forward_withholds_only_prompt_wide_position_ids(fake_mx):
    language_model = FakeLanguageModel()
    shaped = types.SimpleNamespace
    embedded = dict(inputs = [1, 2, 3, 4, 5], inputs_embeds = shaped(shape = (1, 5, 8)), cache = [])
    with RecordingForward(language_model):
        language_model(shaped(shape = (1, 256)), cache = [], position_ids = shaped(shape = (3, 1, 900)))
        language_model(shaped(shape = (1, 256)), cache = [], position_ids = shaped(shape = (3, 1, 256)))
        language_model(position_ids = shaped(shape = (1, 5)), **embedded)
        language_model(position_ids = shaped(shape = (3, 1, 900)), **embedded)
        language_model([1, 2], cache = [], position_ids = None)
    kept = ["position_ids" in kw for kw in language_model.seen_kwargs]
    assert kept == [False, True, True, False, True]


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
    store.lookup("m", a + [0], limit = 256)  # a is now the most recent
    assert store.store("m", c, _snapshot(c))
    assert store.lookup("m", b + [0], limit = 256) == (None, 0)
    assert store.lookup("m", a + [0], limit = 256)[1] == 256
    assert store.nbytes == 4 * 512 and len(store) == 2
    assert not store.store("m", [4] * 800, _snapshot([4] * 800))
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


def test_session_captures_the_boundary_and_resumes_leaving_a_copy_behind(fake_mx):
    store = VLMPromptSnapshotStore(max_bytes = 10**9)
    language_model = FakeLanguageModel()
    ids = list(range(700))
    session, cache, stored = _generate(store, language_model, ids)
    assert session.reused_tokens == 0 and stored and cache[0].offset == 700
    stored_entries, prefix = store.lookup("m", ids, limit = 512)
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
    assert kept[0].offset == 512 and kept[0].keys.rows == ids[:512]  # the copy never moved
    assert kept[0].keys in fake_mx[0]  # the copy, evaluated at the first forward
    assert cache[0].offset == 800 and cache[0].keys.rows == longer
    assert stored and store.lookup("m", longer, limit = 768)[1] == 768

    session, _cache, stored = _generate(store, language_model, longer + [1])
    assert session.reused_tokens == 768 and not stored and len(store) == 4


def test_session_stores_what_the_snapshot_holds_when_reuse_was_declined(fake_mx, monkeypatch):
    store = VLMPromptSnapshotStore(max_bytes = 10**9)
    language_model = FakeLanguageModel()
    ids = list(range(1100))
    _generate(store, language_model, ids)  # boundary 1024
    original = store.lookup("m", ids, limit = 1024)[0]
    store.store("t", ids[:512], _snapshot(range(9000, 9512)))  # a decoy
    fake_mx.clear()
    session, _cache, stored = _generate(store, language_model, ids + [1] * 300, honour_reuse = False)
    assert session.reused_tokens == 1024 and stored
    assert store.lookup("m", ids, limit = 256)[1] == 256
    # The declined offer is released, not copied: it would be declined again.
    assert len(store) == 2 and store.lookup("m", ids, limit = 1024)[1] == 256
    assert store.lookup("t", ids, limit = 512)[1] == 512
    assert not list(snapshots._arrays(original, sys.modules["mlx.core"]))
    assert original[0].offset == 1024 and len(fake_mx) == 1

    # A copy that cannot be made drops the snapshot it was taken from, too.
    store, ids = VLMPromptSnapshotStore(max_bytes = 10**9), list(range(700))
    _generate(store, language_model, ids)

    def _fail(_arrays):
        raise MemoryError("no room for a copy")

    monkeypatch.setattr(sys.modules["mlx.core"], "eval", _fail)
    with VLMPromptCacheSession(store, "m", language_model, make_cache) as session:
        assert session.find_prefix_length(ids + [1]) == 512
        run_generation(language_model, ids + [1], session.cache, 512)
        assert session.cache[0].offset == 701 and len(store) == 0
        assert not session.finish()


def test_session_stores_only_snapshots_that_sit_on_the_grid(fake_mx):
    store = VLMPromptSnapshotStore(max_bytes = 10**9)
    language_model = FakeLanguageModel()
    ids = list(range(600))
    with VLMPromptCacheSession(store, "m", language_model, make_cache) as session:
        session.find_prefix_length(ids)
        language_model(ids[:150], cache = session.cache)
        language_model(ids[150:300], cache = session.cache)
        language_model(ids[300:], cache = session.cache)
        assert session._forward.record.snapshot is not None
        assert not session.finish()
    with VLMPromptCacheSession(store, "m", language_model, lambda: [FakeState()]) as session:
        session.find_prefix_length(ids[:300])
        run_generation(session._forward._language_model, ids[:300], session.cache, 0)
        assert session._forward.record.snapshot is not None and not session.finish()
    with VLMPromptCacheSession(store, "m", language_model, make_cache) as session:
        assert not session.finish() and session.cache[0].offset == 0
    assert len(store) == 0

    layout = lambda: [FakeCacheList(FakeKV(), FakeState()), FakeCacheList(FakeKV(), FakeState())]
    with VLMPromptCacheSession(store, "m", language_model, layout) as session:
        session.find_prefix_length(ids)
        run_generation(language_model, ids, session.cache, 0)
        assert session.finish()
    entries, prefix = store.lookup("m", ids, limit = 512)
    assert prefix == 512 and entries[1].caches[0].offset == 512
    assert store.nbytes == 2 * 4 * (512 + 1 + 1)


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
    assert store.lookup("m", _media_prompt(900), limit = 768)[1] == 768
    session, _cache, _stored = _generate(
        store, language_model, _media_prompt(1000), media_token_ids = (9,)
    )
    assert session.reused_tokens == 768
    session, _cache, stored = _generate(
        store, language_model, _media_prompt(1100), honour_reuse = False, media_token_ids = (9,)
    )
    assert session.reused_tokens == 768 and not stored and len(store) == 0

    prompt = _media_prompt(1000)
    _generate(store, language_model, prompt, media_token_ids = (9,))
    store.store("t", prompt[:768], _snapshot(prompt[:768]))
    store.store("m", [5] + prompt[1:768], _snapshot(range(768)))
    session = lambda **kw: VLMPromptCacheSession(
        store, "m", language_model, make_cache, media_token_ids = (9,), **kw
    )
    assert session().find_prefix_length(prompt) == 768 and len(store) == 3
    # A media request keeps only what serves it, and these three serve nothing.
    assert session(releases_unserved = True).find_prefix_length(prompt) == 768
    assert list(store._entries) == [("m", tuple(prompt[:768]))]
    assert session(releases_unserved = True).find_prefix_length(_media_prompt(700)) == 0
    assert len(store) == 0 and store.nbytes == 0


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
    assert session.produced_tokens == 100 and cache[0].offset == 500 and len(fake_mx) == 2
    assert [len(item[1]) for item in store._entries] == [100, 356]
    session, _cache, stored = generate(700)
    assert session.reused_tokens == 356 and block.prefilled == 1 and stored
    assert session.produced_tokens == 0
    assert [len(item[1]) for item in store._entries] == [356, 612]
    store.clear()
    session, _cache, stored = generate(300)  # no whole chunk past the block
    assert session.reused_tokens == 100 and block.prefilled == 2 and not stored
    assert [len(item[1]) for item in store._entries] == [100]
    # Declined and run from zero: the capture lands at 256, off the grid; block dropped.
    session, _cache, stored = generate(500, honour_reuse = False)
    assert session.reused_tokens == 100 and block.prefilled == 2
    assert not stored and len(store) == 0

    # A block that fails: prefilled by the caller as before, nothing captured.
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
    with VLMPromptCacheSession(
        store, "m", FakeLanguageModel(), make_cache, policy_hosts = (host, plain)
    ):
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
    assert shape_stable_prefix(300) == 256
    with pytest.raises(ImportError):
        copy_cache_entries([FakeKV()])
    assert snapshots.VLM_PROMPT_CACHE_ENTRIES == 6
