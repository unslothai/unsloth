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


def _stub_auto_tokenizer(monkeypatch):
    """fix_sentencepiece_tokenizer reloads the patched directory through
    AutoTokenizer at the end; that needs a full tokenizer on disk, which is
    out of scope here. Record the call and hand back a sentinel.
    """
    loaded = []

    class _StubAutoTokenizer:
        @staticmethod
        def from_pretrained(location, **kwargs):
            loaded.append(location)
            return "reloaded-tokenizer"

    monkeypatch.setattr(transformers, "AutoTokenizer", _StubAutoTokenizer)
    return loaded


def test_old_tokenizer_is_saved_so_its_model_can_be_read(tmp_path, monkeypatch):
    """The guard must not skip the body on a fresh temporary directory.

    fix_sentencepiece_tokenizer creates the temporary directory itself and then
    checks for a tokenizer.model inside it, but that file only appears once
    old_tokenizer.save_pretrained() has run.
    """
    _stub_auto_tokenizer(monkeypatch)
    old, new = _tokenizers()
    location = str(tmp_path / "_unsloth_sentencepiece_temp")

    fix_sentencepiece_tokenizer(old, new, {"</s>": "<|im_end|>"}, temporary_location = location)

    assert old.saved_to, "old tokenizer was never saved: the body did not run"


def test_token_mapping_is_applied_to_the_sentencepiece_model(tmp_path, monkeypatch):
    _stub_auto_tokenizer(monkeypatch)
    old, new = _tokenizers()
    location = str(tmp_path / "_unsloth_sentencepiece_temp")

    fix_sentencepiece_tokenizer(old, new, {"</s>": "<|im_end|>"}, temporary_location = location)

    assert "<|im_end|>" in _read_pieces(f"{location}/tokenizer.model")


def test_tokenizer_without_a_sentencepiece_model_is_returned_untouched(tmp_path, monkeypatch):
    """A fast-only tokenizer writes no tokenizer.model, so the guard still
    short-circuits and the caller gets new_tokenizer back unchanged.
    """
    _stub_auto_tokenizer(monkeypatch)
    old = _FakeTokenizer("old", spm_bytes = None)
    new = _FakeTokenizer("new")
    location = str(tmp_path / "_unsloth_sentencepiece_temp")

    result = fix_sentencepiece_tokenizer(
        old, new, {"</s>": "<|im_end|>"}, temporary_location = location
    )

    assert result is new


def test_stale_model_from_a_previous_call_does_not_poison_a_fast_only_call(tmp_path, monkeypatch):
    """The temporary directory defaults to a fixed, reusable location. A prior
    sentencepiece call leaves a tokenizer.model there; a fast-only tokenizer
    saved afterwards writes none, so without clearing the stale file the guard
    would pass on the previous model and patch the wrong tokenizer.
    """
    _stub_auto_tokenizer(monkeypatch)
    location = str(tmp_path / "_unsloth_sentencepiece_temp")

    # Call 1: a real sentencepiece tokenizer, writes a tokenizer.model.
    old_sp, new_sp = _tokenizers()
    fix_sentencepiece_tokenizer(old_sp, new_sp, {"</s>": "<|im_end|>"}, temporary_location = location)
    assert os.path.isfile(f"{location}/tokenizer.model")

    # Call 2: a fast-only tokenizer reusing the SAME directory must be returned
    # untouched, not reloaded from call 1's stale model.
    old_fast = _FakeTokenizer("fast", spm_bytes = None, vocab = {"</s>": 2})
    new_fast = _FakeTokenizer("fast_new")
    result = fix_sentencepiece_tokenizer(
        old_fast, new_fast, {"</s>": "<|special|>"}, temporary_location = location
    )

    assert result is new_fast


def test_reused_directory_is_emptied_so_stale_artifacts_do_not_leak(tmp_path, monkeypatch):
    """The reload reads the whole reusable directory, so a leftover file from a
    previous tokenizer that the next save does not overwrite must not survive.
    """
    _stub_auto_tokenizer(monkeypatch)
    location = str(tmp_path / "_unsloth_sentencepiece_temp")
    os.makedirs(location, exist_ok = True)

    # An artifact from a previous tokenizer that the next save does not regenerate.
    stale = os.path.join(location, "added_tokens.json")
    with open(stale, "w") as f:
        f.write('{"<stale>": 999}')

    old, new = _tokenizers()
    fix_sentencepiece_tokenizer(old, new, {"</s>": "<|im_end|>"}, temporary_location = location)

    assert not os.path.isfile(stale), "stale artifact from a previous call was not cleared"


class _CopyFromSubdirTokenizer:
    """A slow tokenizer whose sentencepiece source lives in a subdirectory, like
    the tokenizers convert_to_fast_tokenizer produces under {location}/{name}.
    save_pretrained copies that source up to the destination, as HF slow
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


def test_clearing_keeps_the_converted_tokenizer_source_subdirectory(tmp_path, monkeypatch):
    """convert_to_fast_tokenizer stores a converted tokenizer's source vocab under a
    {location}/{name} subdirectory. Clearing the reusable directory must not delete
    that subtree, or old_tokenizer.save_pretrained cannot copy tokenizer.model and the
    sentencepiece rename is silently lost.
    """
    _stub_auto_tokenizer(monkeypatch)
    location = str(tmp_path / "_unsloth_sentencepiece_temp")
    subdir = os.path.join(location, "some_model")
    os.makedirs(subdir, exist_ok = True)

    pieces = [("<s>", 0.0, CONTROL), ("a", -1.0, NORMAL), ("</s>", 0.0, CONTROL)]
    source_model = os.path.join(subdir, "tokenizer.model")
    with open(source_model, "wb") as f:
        f.write(_spm_bytes(pieces))

    old = _CopyFromSubdirTokenizer(source_model)
    new = _FakeTokenizer("new")
    fix_sentencepiece_tokenizer(old, new, {"</s>": "<|im_end|>"}, temporary_location = location)

    assert os.path.isfile(source_model), "converted tokenizer source subdirectory was deleted"
    assert "<|im_end|>" in _read_pieces(f"{location}/tokenizer.model")
