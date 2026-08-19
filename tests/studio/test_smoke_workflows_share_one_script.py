# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The same smoke test on three operating systems has to be the same test.

The multi-turn chat check ran inline in studio-inference-smoke.yml,
studio-mac-inference-smoke.yml and studio-windows-inference-smoke.yml as three copies of
one script. On 2026-05-22 an unrelated event-loop fix (#5669) turned the Linux copy's
determinism assertion into a printed warning. macOS and Windows kept it and are otherwise
identical in logic. Nothing compared them, so for three months the leg that runs on every
pull request was the one not checking, and the two that still checked run rarely.

So the copies are gone, and these tests are about keeping them gone.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / ".github" / "scripts" / "studio_smoke" / "multi_turn_chat.py"
LEGS = (
    "studio-inference-smoke.yml",
    "studio-mac-ui-smoke.yml",
    "studio-windows-inference-smoke.yml",
)


def _workflow(name: str) -> str:
    return (REPO / ".github" / "workflows" / name).read_text(encoding = "utf-8")


@pytest.fixture(scope = "module")
def script():
    """The shared script, imported. It reads no environment and imports no SDK at module
    level precisely so this is possible."""
    assert SCRIPT.is_file(), f"{SCRIPT} is gone; the three legs have nothing to share"
    spec = importlib.util.spec_from_file_location("multi_turn_chat", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["multi_turn_chat"] = module
    spec.loader.exec_module(module)
    return module


def test_every_leg_runs_the_shared_script():
    missing = [name for name in LEGS if "studio_smoke/multi_turn_chat.py" not in _workflow(name)]
    assert not missing, (
        f"{missing} no longer run the shared multi-turn check. Three copies of it is how "
        f"one of them stopped asserting determinism without anyone noticing."
    )


def test_no_leg_has_grown_its_own_copy_back():
    """Reverting one leg to an inline block is the regression, and it looks additive."""
    offenders = [name for name in LEGS if "def run_anthropic" in _workflow(name)]
    assert not offenders, (
        f"{offenders} carry an inline copy of the multi-turn check again. Change "
        f"{SCRIPT.relative_to(REPO)} instead, so the other legs get the change too."
    )


def test_a_divergent_second_run_is_a_failure_not_a_warning(script):
    """The assertion #5669 removed on Linux, pinned by running it.

    Asserted through behaviour rather than the text of the check, so rewriting it is
    fine and weakening it is not.
    """
    clean = ["1 is 2", "you asked about 1+1", "paris", "paris"]
    script.check("ok", clean, list(clean))  # the baseline passes, or nothing below means anything

    with pytest.raises(AssertionError, match = "non-deterministic"):
        script.check("drift", clean, ["1 is 2", "you asked about 1+1", "paris", "london"])


def test_trailing_whitespace_alone_is_still_tolerated(script):
    """The reason the comparison is on stripped text, kept honest.

    llama-server varies a final newline between identical greedy runs depending on where
    the stream is closed. Tightening this to an exact match would fail on that.
    """
    clean = ["1 is 2", "you asked about 1+1", "paris", "paris"]
    script.check("whitespace", clean, [t + "\n" for t in clean])


def test_an_empty_reply_is_a_failure_in_either_run(script):
    """A server answering nothing at all is deterministic, and the worst outcome.

    Both runs, because the stripped comparison cannot tell them apart: a second run
    returning "" against a first returning "\n" compares EQUAL, so checking only the
    first would print OK for a server that had stopped answering halfway through. The
    Linux copy asserted both before this was consolidated onto the macOS one, which
    asserted only the first.
    """
    clean = ["1 is 2", "you asked about 1+1", "paris", "paris"]
    with pytest.raises(AssertionError, match = "empty turn"):
        script.check("first", ["", "b", "paris", "paris"], ["", "b", "paris", "paris"])
    with pytest.raises(AssertionError, match = "empty turn"):
        script.check("second", clean, ["", "b", "paris", "paris"])
    # The exact pair the stripped comparison is blind to.
    with pytest.raises(AssertionError, match = "empty turn"):
        script.check(
            "whitespace vs nothing", ["\n", "b", "paris", "paris"], ["", "b", "paris", "paris"]
        )


def test_history_grounding_is_still_checked(script):
    """Two of the four turns are answerable only from the earlier ones. That is what the
    'paris' check is for: it fails when history is dropped, rather than when the model is
    wrong about France."""
    with pytest.raises(AssertionError, match = "paris"):
        script.check("nohistory", ["1 is 2", "b", "c", "d"], ["1 is 2", "b", "c", "d"])


def test_the_script_needs_no_environment_to_import(script):
    """What lets every test above exist.

    Reading BASE_URL at module level, or importing the SDKs there, would make the
    checking half unreachable from a test and put it back where it was: only ever
    exercised by a full smoke run on three operating systems.
    """
    source = SCRIPT.read_text(encoding = "utf-8")
    head = source.split("def _server", 1)[0]
    for forbidden in ("os.environ[", "from openai", "from anthropic"):
        assert forbidden not in head, (
            f"{forbidden} moved to module level in {SCRIPT.name}, so importing it now "
            f"needs a running server or the SDKs installed, and these tests cannot run"
        )
