# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""The drafterless retry must survive the arch gate narrowing the argv (#7670).

`_spec_start` is captured as a positional index just before the speculative
flags are appended. The arch gate added in #7670 rebinds `cmd` with tokens
REMOVED from the prefix (`--split-mode` / `--tensor-split`), which slides the
spec block left and leaves the stored index pointing past it.

Before the fix the fallback then kept `spec_flags[:N]`, re-running the drafter
that had just failed, and sliced N tokens out of the middle of the tail. With
the four-token narrowing that included `--api-key`, so the retry would have
launched llama-server unauthenticated. There is no AMD hardware on any runner,
so this drives the real helpers over a fabricated argv.
"""

from core.inference.llama_cpp import _subsequence_index, LlamaCppBackend

_SPEC_FLAGS = ["--spec-type", "mtp", "--draft-max", "16"]

_CMD = [
    "llama-server",
    "-m",
    "model.gguf",
    "--gpu-layers",
    "99",
    "--fit",
    "off",
    "--tensor-split",
    "3,1",
    "--split-mode",
    "tensor",
    *_SPEC_FLAGS,
    "--chat-template-file",
    "t.jinja",
    "--api-key",
    "SECRET",
    "--cache-ram",
    "0",
]
_HINT = _CMD.index("--spec-type")


def _fallback(cmd, hint = _HINT):
    """The shipped reconstruction, verbatim, so a drift here fails loudly."""
    at = _subsequence_index(cmd, _SPEC_FLAGS, hint)
    return cmd[:at] + ["--spec-default"] + cmd[at + len(_SPEC_FLAGS) :]


def _narrowed(*flags):
    return LlamaCppBackend._without_flags(_CMD, flags)


def test_the_hint_still_works_when_nothing_narrowed():
    assert _fallback(_CMD) == [
        "llama-server",
        "-m",
        "model.gguf",
        "--gpu-layers",
        "99",
        "--fit",
        "off",
        "--tensor-split",
        "3,1",
        "--split-mode",
        "tensor",
        "--spec-default",
        "--chat-template-file",
        "t.jinja",
        "--api-key",
        "SECRET",
        "--cache-ram",
        "0",
    ]


def test_the_api_key_survives_the_four_token_narrowing():
    """The forced-CPU arm drops 4 tokens. A stale index ate --api-key SECRET,
    so the retry would have served an unauthenticated llama-server."""
    fb = _fallback(_narrowed("--split-mode", "-sm", "--tensor-split", "-ts"))
    assert fb[fb.index("--api-key") + 1] == "SECRET"
    assert "--chat-template-file" in fb and "t.jinja" in fb


def test_the_failed_drafter_is_not_re_run_after_narrowing():
    """llama.cpp accumulates spec types, so a surviving --spec-type would retry
    the very mode that just failed and --spec-default could not undo it."""
    for flags in (
        ("--tensor-split", "-ts"),
        ("--split-mode", "-sm"),
        ("--split-mode", "-sm", "--tensor-split", "-ts"),
    ):
        fb = _fallback(_narrowed(*flags))
        assert "--spec-type" not in fb, flags
        assert "--draft-max" not in fb, flags
        assert fb.count("--spec-default") == 1, flags


def test_no_token_is_lost_by_any_narrowing():
    for flags in (
        ("--tensor-split", "-ts"),
        ("--split-mode", "-sm"),
        ("--split-mode", "-sm", "--tensor-split", "-ts"),
    ):
        narrowed = _narrowed(*flags)
        fb = _fallback(narrowed)
        # Everything except the spec block itself must still be present, in order.
        kept = [t for t in narrowed if t not in _SPEC_FLAGS]
        assert [t for t in fb if t != "--spec-default"] == kept, flags


def test_the_block_is_found_even_if_it_moved_right():
    """Defensive: the hint is a hint. A future site that INSERTS ahead of the
    block must not corrupt the slice either."""
    padded = ["--threads", "8", *_CMD]
    fb = _fallback(padded)
    # The whole block goes and nothing else does: asserting only that --spec-type
    # left would pass on a slice cut two tokens early, which eats --split-mode.
    assert [t for t in fb if t != "--spec-default"] == [t for t in padded if t not in _SPEC_FLAGS]
    assert fb.count("--spec-default") == 1
    assert fb[fb.index("--api-key") + 1] == "SECRET"


def test_an_absent_block_does_not_raise():
    stripped = [t for t in _CMD if t not in _SPEC_FLAGS]
    at = _subsequence_index(stripped, _SPEC_FLAGS, _HINT)
    assert 0 <= at <= len(stripped)
