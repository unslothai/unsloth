# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""HarmonyTextStreamer decodes and parses incrementally yet emits exactly what re-decoding and
re-parsing the whole reply on every token emits."""

import queue
import random
import re

import pytest

inf = pytest.importorskip("core.inference.inference")
torch = pytest.importorskip("torch")
tokenizers = pytest.importorskip("tokenizers")
transformers = pytest.importorskip("transformers")

_SPECIAL = ["<|channel|>", "<|message|>", "<|end|>", "<|start|>", "<|return|>"]
_RE = re.compile(r"<\|channel\|>(\w+)<\|message\|>(.*?)(?=<\|end\|>|<\|channel\|>|\Z)", re.DOTALL)


class _Reference:
    """The whole-reply algorithm: decode every generated id and re-match every channel per token."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.out = []
        self.ids = []
        self.open = self.close = False
        self.analysis = self.final = 0

    def put(self, ids):
        self.ids.extend(ids)
        self.parse(self.tokenizer.decode(self.ids, skip_special_tokens = False).rstrip("\ufffd"))

    def end(self):
        if self.ids:
            self.parse(self.tokenizer.decode(self.ids, skip_special_tokens = False))
        if self.open and not self.close:
            self.out.append("</think>")
        return self.out

    def parse(self, raw):
        for m in _RE.finditer(raw):
            channel, content = m.group(1).lower(), m.group(2)
            if channel == "analysis":
                if not self.open:
                    self.out.append("<think>")
                    self.open = True
                if content[self.analysis :]:
                    self.out.append(content[self.analysis :])
                    self.analysis = len(content)
            elif channel in ("final", "assistant"):
                if self.open and not self.close:
                    self.out.append("</think>")
                    self.close = True
                if content[self.final :]:
                    self.out.append(content[self.final :])
                    self.final = len(content)


def _byte_level_tokenizer():
    """One token per byte, harmony markers as special tokens: splits every multibyte character."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers.convert_slow_tokenizer import bytes_to_unicode

    vocab = {ch: i for i, ch in enumerate(bytes_to_unicode().values())}
    tok = Tokenizer(models.BPE(vocab = vocab, merges = []))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space = False)
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens(_SPECIAL)
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object = tok, clean_up_tokenization_spaces = False
    )


def _harmony_text(rng):
    words = ["the", " quick", " 饺子", " 🦥", " é", "\n", " <|end", " ü", " 中文", " ok", "\ufffd"]
    parts = []
    for _ in range(rng.randint(0, 5)):
        channel = rng.choice(["analysis", "final", "commentary", "assistant", "Final"])
        parts.append(f"<|channel|>{channel}<|message|>" if rng.random() < 0.9 else "<|channel|>ana")
        parts.append("".join(rng.choice(words) for _ in range(rng.randint(0, 40))))
        if rng.random() < 0.5:
            parts.append(rng.choice(["<|end|><|start|>assistant", "<|end|>"]))
    return "".join(parts)


def _drain(streamer):
    out = []
    while True:
        try:
            out.append(next(streamer))
        except StopIteration:
            return out
        except queue.Empty:
            raise AssertionError("streamer did not finish")


def test_byte_level_tokenizer_takes_the_incremental_decode():
    assert inf._decodes_by_concatenation(_byte_level_tokenizer())

    class Cleaning:
        clean_up_tokenization_spaces = True

    assert not inf._decodes_by_concatenation(Cleaning())
    assert not inf._decodes_by_concatenation(object())


def _stripping_tokenizers():
    """decode / _decode overrides that post-process the whole text, which per-suffix decoding would change."""
    base = _byte_level_tokenizer()

    class StripDecode(transformers.PreTrainedTokenizerFast):
        def decode(self, *args, **kwargs):
            return super().decode(*args, **kwargs).lstrip()

    class StripPrivateDecode(transformers.PreTrainedTokenizerFast):
        def _decode(self, *args, **kwargs):
            return super()._decode(*args, **kwargs).lstrip()

    out = []
    for cls in (StripDecode, StripPrivateDecode):
        out.append(cls(tokenizer_object = base.backend_tokenizer, clean_up_tokenization_spaces = False))
    instance = _byte_level_tokenizer()
    plain = instance.decode
    instance.decode = lambda *args, **kwargs: plain(*args, **kwargs).lstrip()
    out.append(instance)
    return out


def test_overridden_decode_takes_the_whole_reply_path():
    rng = random.Random(11)
    for tokenizer in _stripping_tokenizers():
        assert not inf._decodes_by_concatenation(tokenizer), type(tokenizer).__name__
        for _ in range(40):
            ids = tokenizer.encode(_harmony_text(rng), add_special_tokens = False)
            reference = _Reference(tokenizer)
            streamer = inf.HarmonyTextStreamer(tokenizer, skip_prompt = False)
            for token in ids:
                reference.put([token])
                streamer.put(torch.tensor([token]))
            streamer.end()
            assert _drain(streamer) == reference.end()


@pytest.mark.parametrize("seed", range(4))
def test_stream_matches_whole_reply_decode(seed):
    tokenizer = _byte_level_tokenizer()
    vocab = len(tokenizer)
    rng = random.Random(seed)
    for _ in range(60):
        ids = tokenizer.encode(_harmony_text(rng), add_special_tokens = False)
        for _ in range(rng.randint(0, 4)):
            ids.insert(rng.randint(0, len(ids)), rng.randrange(vocab))
        reference = _Reference(tokenizer)
        streamer = inf.HarmonyTextStreamer(tokenizer, skip_prompt = True)
        streamer.put(torch.tensor([[1, 2]]))
        i = 0
        while i < len(ids):
            step = ids[i : i + (1 if rng.random() < 0.8 else rng.randint(1, 4))]
            i += len(step)
            reference.put(step)
            streamer.put(torch.tensor(step))
        streamer.end()
        assert _drain(streamer) == reference.end()


def test_parse_matches_reference_when_text_is_rewritten():
    tokenizer = _byte_level_tokenizer()
    rng = random.Random(7)
    for _ in range(300):
        text = _harmony_text(rng)
        seen = []
        for _ in range(rng.randint(1, 20)):
            if seen and rng.random() < 0.2:
                seen.append(seen[-1][: rng.randint(0, len(seen[-1]))])
            else:
                seen.append(text[: rng.randint(len(seen[-1]) if seen else 0, len(text))])
        reference = _Reference(tokenizer)
        streamer = inf.HarmonyTextStreamer(tokenizer, skip_prompt = False)
        for raw in seen:
            reference.parse(raw)
            streamer._process_incremental(raw)
        streamer.end()
        assert _drain(streamer) == reference.end()
