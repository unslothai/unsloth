# SPDX-License-Identifier: AGPL-3.0-only
import gc
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import transformers
from transformers.utils import sentencepiece_model_pb2

from unsloth.tokenizer_utils import fix_sentencepiece_tokenizer


NORMAL, CONTROL = 1, 3


def _spm_bytes(pieces):
    m = sentencepiece_model_pb2.ModelProto()
    for piece, score, typ in pieces:
        p = m.pieces.add()
        p.piece = piece
        p.score = score
        p.type = typ
    return m.SerializeToString()


def _read_pieces(path):
    m = sentencepiece_model_pb2.ModelProto()
    with open(path, "rb") as f:
        m.ParseFromString(f.read())
    return [p.piece for p in m.pieces]


class _FakeTokenizer:
    """Minimal stand-in for a sentencepiece-backed slow tokenizer.

    ``save_pretrained`` writes a tokenizer.model, which is what the real slow
    tokenizers do and what fix_sentencepiece_tokenizer reads back.
    """

    def __init__(
        self,
        name,
        spm_bytes = None,
        vocab = None,
    ):
        self.name = name
        self.eos_token = "</s>"
        self.pad_token = "<pad>"
        self._spm_bytes = spm_bytes
        self._vocab = vocab or {}
        self.saved_to = []

    def save_pretrained(self, location):
        self.saved_to.append(location)
        os.makedirs(location, exist_ok = True)
        if self._spm_bytes is not None:
            with open(os.path.join(location, "tokenizer.model"), "wb") as f:
                f.write(self._spm_bytes)

    def __call__(
        self,
        texts,
        add_special_tokens = False,
    ):
        class _Encoded:
            pass

        encoded = _Encoded()
        encoded.input_ids = [[self._vocab[text]] for text in texts]
        return encoded


def _tokenizers():
    pieces = [("<s>", 0.0, CONTROL), ("a", -1.0, NORMAL), ("</s>", 0.0, CONTROL)]
    old = _FakeTokenizer("old", spm_bytes = _spm_bytes(pieces), vocab = {"</s>": 2})
    new = _FakeTokenizer("new")
    return old, new


class _ReloadedTokenizer:
    """Weakref-able stand-in for the tokenizer AutoTokenizer.from_pretrained returns."""

    def __init__(self, location):
        self.location = location


def _stub_auto_tokenizer(monkeypatch):
    """fix_sentencepiece_tokenizer reloads the patched directory through
    AutoTokenizer at the end; that needs a full tokenizer on disk, which is
    out of scope here. Record the reload location and hand back a sentinel.
    """
    loaded = []

    class _StubAutoTokenizer:
        @staticmethod
        def from_pretrained(location, **kwargs):
            loaded.append(location)
            return _ReloadedTokenizer(location)

    monkeypatch.setattr(transformers, "AutoTokenizer", _StubAutoTokenizer)
    return loaded


def test_old_tokenizer_is_saved_so_its_model_can_be_read(tmp_path, monkeypatch):
    """The guard must not skip the body on a fresh temporary directory.

    fix_sentencepiece_tokenizer creates its scratch directory itself and then
    checks for a tokenizer.model inside it, but that file only appears once
    old_tokenizer.save_pretrained() has run.
    """
    _stub_auto_tokenizer(monkeypatch)
    old, new = _tokenizers()
    location = str(tmp_path / "_unsloth_sentencepiece_temp")

    fix_sentencepiece_tokenizer(old, new, {"</s>": "<|im_end|>"}, temporary_location = location)

    assert old.saved_to, "old tokenizer was never saved: the body did not run"


def test_token_mapping_is_applied_to_the_sentencepiece_model(tmp_path, monkeypatch):
    loaded = _stub_auto_tokenizer(monkeypatch)
    old, new = _tokenizers()
    location = str(tmp_path / "_unsloth_sentencepiece_temp")

    # Hold the returned tokenizer so its scratch dir survives until we read it.
    tok = fix_sentencepiece_tokenizer(old, new, {"</s>": "<|im_end|>"}, temporary_location = location)

    assert "<|im_end|>" in _read_pieces(f"{loaded[-1]}/tokenizer.model")
    assert tok is not None


def test_tokenizer_without_a_sentencepiece_model_is_returned_untouched(tmp_path, monkeypatch):
    """A fast-only tokenizer writes no tokenizer.model, so the guard still
    short-circuits and the caller gets new_tokenizer back unchanged. Its scratch
    dir is unreferenced and reclaimed immediately.
    """
    _stub_auto_tokenizer(monkeypatch)
    old = _FakeTokenizer("old", spm_bytes = None)
    new = _FakeTokenizer("new")
    location = str(tmp_path / "_unsloth_sentencepiece_temp")

    result = fix_sentencepiece_tokenizer(
        old, new, {"</s>": "<|im_end|>"}, temporary_location = location
    )

    assert result is new
    assert not any(name.startswith("tokenizer_") for name in os.listdir(location)), (
        "the fast-only scratch dir was not reclaimed"
    )


def test_each_call_uses_a_fresh_isolated_subdirectory(tmp_path, monkeypatch):
    """Each call must work in its own unique subdirectory, so concurrent or
    repeated calls never share scratch files, stale artifacts never leak into
    the reload, and nothing the caller left in the scratch location is deleted.
    """
    loaded = _stub_auto_tokenizer(monkeypatch)
    location = str(tmp_path / "_unsloth_sentencepiece_temp")
    os.makedirs(location, exist_ok = True)

    # A pre-existing artifact in the shared scratch location.
    marker = os.path.join(location, "leftover.json")
    with open(marker, "w") as f:
        f.write("{}")

    old1, new1 = _tokenizers()
    old2, new2 = _tokenizers()
    # Hold both returned tokenizers so their scratch dirs stay alive.
    tok1 = fix_sentencepiece_tokenizer(old1, new1, {"</s>": "<|im_end|>"}, temporary_location = location)
    tok2 = fix_sentencepiece_tokenizer(old2, new2, {"</s>": "<|im_end|>"}, temporary_location = location)

    work1, work2 = loaded[0], loaded[1]
    assert work1 != work2, "two calls reused the same directory"
    assert os.path.dirname(work1) == location and os.path.dirname(work2) == location
    assert os.path.isdir(work1) and os.path.isdir(work2)
    # Nothing the caller left behind is deleted, and it never leaks into a work dir.
    assert os.path.isfile(marker), "a pre-existing scratch file was deleted"
    assert not os.path.isfile(os.path.join(work1, "leftover.json"))
    assert not os.path.isfile(os.path.join(work2, "leftover.json"))
    assert tok1 is not None and tok2 is not None


def test_sentencepiece_scratch_dir_is_reclaimed_once_the_tokenizer_is_gone(tmp_path, monkeypatch):
    """The scratch dir must live as long as the returned tokenizer (its vocab_file
    points there), then be reclaimed when the tokenizer is garbage collected.
    """
    loaded = _stub_auto_tokenizer(monkeypatch)
    old, new = _tokenizers()
    location = str(tmp_path / "_unsloth_sentencepiece_temp")

    tok = fix_sentencepiece_tokenizer(old, new, {"</s>": "<|im_end|>"}, temporary_location = location)
    work = loaded[-1]
    assert os.path.isdir(work), "scratch dir vanished while the tokenizer was alive"

    del tok
    gc.collect()
    assert not os.path.isdir(work), "scratch dir was not reclaimed after the tokenizer was freed"


class _CopyFromSubdirTokenizer:
    """A slow tokenizer whose sentencepiece source lives elsewhere (like the
    tokenizers convert_to_fast_tokenizer produces under {location}/{name}).
    save_pretrained copies that source into the destination, as HF slow
    tokenizers copy their vocab_file.
    """

    def __init__(self, source_model_path):
        self.eos_token = "</s>"
        self.pad_token = "<pad>"
        self._source_model_path = source_model_path

    def save_pretrained(self, location):
        os.makedirs(location, exist_ok = True)
        if os.path.isfile(self._source_model_path):
            with open(self._source_model_path, "rb") as src:
                data = src.read()
            with open(os.path.join(location, "tokenizer.model"), "wb") as dst:
                dst.write(data)

    def __call__(
        self,
        texts,
        add_special_tokens = False,
    ):
        class _Encoded:
            pass

        encoded = _Encoded()
        encoded.input_ids = [[2] for _ in texts]
        return encoded


def test_source_vocab_outside_the_work_directory_is_not_disturbed(tmp_path, monkeypatch):
    """A tokenizer whose sentencepiece source lives elsewhere (e.g. the subtree
    convert_to_fast_tokenizer created) is copied into the fresh work directory
    and patched there; the original source is left untouched.
    """
    loaded = _stub_auto_tokenizer(monkeypatch)
    location = str(tmp_path / "_unsloth_sentencepiece_temp")
    subdir = os.path.join(location, "some_model")
    os.makedirs(subdir, exist_ok = True)

    pieces = [("<s>", 0.0, CONTROL), ("a", -1.0, NORMAL), ("</s>", 0.0, CONTROL)]
    source_model = os.path.join(subdir, "tokenizer.model")
    with open(source_model, "wb") as f:
        f.write(_spm_bytes(pieces))

    old = _CopyFromSubdirTokenizer(source_model)
    new = _FakeTokenizer("new")
    tok = fix_sentencepiece_tokenizer(old, new, {"</s>": "<|im_end|>"}, temporary_location = location)

    assert _read_pieces(source_model) == [
        "<s>",
        "a",
        "</s>",
    ], "the original source vocab was modified"
    assert "<|im_end|>" in _read_pieces(f"{loaded[-1]}/tokenizer.model")
    assert tok is not None


def test_swap_mapping_swaps_both_pieces_without_duplicating(tmp_path, monkeypatch):
    """When the caller swaps eos and stop_word in the fast JSON it must pass both
    directions here; a one-way mapping would leave two stop_word pieces and no eos.
    """
    loaded = _stub_auto_tokenizer(monkeypatch)
    location = str(tmp_path / "_unsloth_sentencepiece_temp")

    pieces = [("<s>", 0.0, CONTROL), ("<|im_end|>", -1.0, NORMAL), ("</s>", 0.0, CONTROL)]
    old = _FakeTokenizer("old", spm_bytes = _spm_bytes(pieces), vocab = {"</s>": 2, "<|im_end|>": 1})
    new = _FakeTokenizer("new")

    tok = fix_sentencepiece_tokenizer(
        old, new, {"</s>": "<|im_end|>", "<|im_end|>": "</s>"}, temporary_location = location
    )

    result = _read_pieces(f"{loaded[-1]}/tokenizer.model")
    assert result.count("<|im_end|>") == 1 and result.count("</s>") == 1, result
    assert tok is not None


def test_only_applied_mappings_are_patched(tmp_path, monkeypatch):
    """When the caller skips a mapping whose target already exists, it must not
    pass that mapping here, or the skipped source token gets renamed anyway and
    duplicates the existing target in the model.
    """
    loaded = _stub_auto_tokenizer(monkeypatch)
    location = str(tmp_path / "_unsloth_sentencepiece_temp")

    pieces = [("<s>", 0.0, CONTROL), ("aa", -1.0, NORMAL), ("bb", -1.0, NORMAL), ("X", -1.0, NORMAL)]
    old = _FakeTokenizer("old", spm_bytes = _spm_bytes(pieces), vocab = {"aa": 1, "bb": 2})
    new = _FakeTokenizer("new")

    # Caller skipped aa->X (X already exists) and applied bb->Y, so only bb->Y is passed.
    tok = fix_sentencepiece_tokenizer(old, new, {"bb": "Y"}, temporary_location = location)

    result = _read_pieces(f"{loaded[-1]}/tokenizer.model")
    assert result.count("X") == 1 and "Y" in result and "aa" in result, result
    assert tok is not None
