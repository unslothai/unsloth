"""The Gemma body brace scan resumes a saved walk of a prefix; every answer must match a fresh walk."""

import random

import pytest

from core.inference import tool_call_parser as tcp
from core.inference import llama_cpp
from core.inference import safetensors_agentic


def _fresh_scan(text, brace_pos):
    """The walk before the resume cache, kept verbatim as the oracle."""
    if brace_pos >= len(text) or text[brace_pos] != "{":
        return None
    depth = 0
    quote = ""
    prev = ""
    prev_raw = ""
    i = brace_pos
    n = len(text)
    while i < n:
        ch = text[i]
        if quote:
            if ch == "\\" and i + 1 < n:
                i += 2
                continue
            if ch == quote:
                quote = ""
        elif ch in "\"'" and (prev in ":{[(,=" or (ch == '"' and prev_raw.isspace())):
            quote = ch
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        if not ch.isspace():
            prev = ch
        prev_raw = ch
        i += 1
    return None


@pytest.fixture(autouse = True)
def _empty_cache():
    tcp._gemma_body_scans[:] = [None] * tcp._GEMMA_BODY_SCAN_SLOTS
    yield
    tcp._gemma_body_scans[:] = [None] * tcp._GEMMA_BODY_SCAN_SLOTS


_PIECES = [
    "{",
    "}",
    '"',
    "'",
    "\\",
    ":",
    ",",
    " ",
    "\n",
    "x" * 300,
    "abc",
    "é",
    "😀",
    "(",
    "=",
    "`",
]


def _streams(seed, count):
    rng = random.Random(seed)
    for _ in range(count):
        yield "call:write_file{" + "".join(rng.choice(_PIECES) for _ in range(rng.randint(1, 120)))


def test_every_prefix_matches_a_fresh_walk():
    for text in _streams(0, 100):
        for end in range(len("call:write_file{"), len(text) + 1):
            prefix = text[:end]
            assert tcp._gemma_body_brace_end(prefix, 15) == _fresh_scan(prefix, 15)


def test_interleaved_and_rewritten_buffers_match_a_fresh_walk():
    rng = random.Random(1)
    texts = list(_streams(2, 40))
    for _ in range(4000):
        text = rng.choice(texts)
        end = rng.randint(16, len(text))
        probe = text[:end]
        if rng.random() < 0.3:
            k = rng.randrange(16, len(probe)) if len(probe) > 16 else 16
            probe = probe[:k] + rng.choice(_PIECES) + probe[k + 1 :]
        assert tcp._gemma_body_brace_end(probe, 15) == _fresh_scan(probe, 15)


def test_trailing_backslash_inside_a_quote_is_not_saved_past():
    head = 'call:f{content:"' + "y" * 400
    assert tcp._gemma_body_brace_end(head + "\\", 6) is None
    # The backslash escapes the quote that follows, so the body is still open.
    assert tcp._gemma_body_brace_end(head + '\\"}', 6) is None
    assert tcp._gemma_body_brace_end(head + '\\""}', 6) == len(head) + 3


def test_open_body_scan_is_saved_at_the_end_of_the_text():
    text = 'call:write_file{content:<|"|>' + "a {b " * 200
    assert tcp._gemma_body_brace_end(text, 15) is None
    saved = [e for e in tcp._gemma_body_scans if e is not None and e[1] == 15]
    assert saved and saved[0][2] == len(text)
    grown = text + "c {d "
    assert tcp._gemma_body_brace_end(grown, 15) is None
    saved = [e for e in tcp._gemma_body_scans if e is not None and e[1] == 15]
    assert len(saved) == 1 and saved[0][2] == len(grown)


def test_held_tail_lengths_unchanged_for_an_open_gemma_body():
    tools = [{"type": "function", "function": {"name": "write_file"}}]
    text = 'Sure. call:write_file{content:<|"|>'
    for i in range(400):
        text += "{" if i % 50 == 0 else "word "
        expect = len(text) - text.index("call:")
        assert llama_cpp._held_rehearsal_tail_len(text, tools) == expect
        assert safetensors_agentic._held_rehearsal_tail_len(text, tools) == expect


def test_rehearsal_prefix_still_skips_unpromotable_names():
    tools = [
        {"type": "function", "function": {"name": "terminal"}},
        {"type": "function", "function": {"name": "mcp__fs__read"}},
        {"type": "function", "function": {"name": "web_search"}},
    ]
    assert not llama_cpp._is_rehearsal_prefix("termin", tools)
    assert not llama_cpp._is_rehearsal_prefix("mcp__fs", tools)
    assert llama_cpp._is_rehearsal_prefix("web_search[AR", tools)
    assert not safetensors_agentic._is_rehearsal_prefix("terminal", tools)
    assert safetensors_agentic._is_rehearsal_prefix("web_se", tools)
    assert llama_cpp._held_rehearsal_tail_len("try web_se", tools) == len("web_se")
    assert llama_cpp._held_rehearsal_tail_len("try termin", tools) == 0
