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
    clean = ["1 is 2", "you asked about 2", "paris", "paris"]
    script.check("ok", clean, list(clean))  # the baseline passes, or nothing below means anything

    with pytest.raises(AssertionError, match = "non-deterministic"):
        script.check("drift", clean, ["1 is 2", "you asked about 2", "paris", "london"])


def test_trailing_whitespace_alone_is_still_tolerated(script):
    """The reason the comparison is on stripped text, kept honest.

    llama-server varies a final newline between identical greedy runs depending on where
    the stream is closed. Tightening this to an exact match would fail on that.
    """
    clean = ["1 is 2", "you asked about 2", "paris", "paris"]
    script.check("whitespace", clean, [t + "\n" for t in clean])


def test_an_empty_reply_is_a_failure_in_either_run(script):
    """A server answering nothing at all is deterministic, and the worst outcome.

    Both runs, because the stripped comparison cannot tell them apart: a second run
    returning "" against a first returning "\n" compares EQUAL, so checking only the
    first would print OK for a server that had stopped answering halfway through. The
    Linux copy asserted both before this was consolidated onto the macOS one, which
    asserted only the first.
    """
    clean = ["1 is 2", "you asked about 2", "paris", "paris"]
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
    """Two of the four turns are answerable only from the earlier ones. Those checks fail
    when history is dropped, rather than when the model is wrong about France."""
    good2 = "you asked about 2"
    with pytest.raises(AssertionError, match = "history reached the model"):
        script.check("nohistory", ["1 is 2", good2, "c", "d"], ["1 is 2", good2, "c", "d"])

    # The gap #10009 found: 'paris' in the JOINED transcript proves nothing, because
    # turn 3 supplies it on its own. Checked per turn, a server that answers turn 3 and
    # then loses the history still fails.
    lost_after_3 = ["1 is 2", "b", "paris", "Okay, I'm ready."]
    with pytest.raises(AssertionError, match = "history reached the model"):
        script.check("joined-is-not-enough", lost_after_3, list(lost_after_3))


def test_grounding_is_asserted_on_a_turn_a_270m_model_can_carry(script):
    """Which turn holds the grounding assertion is itself the regression risk.

    #10009 put it on turn 2, requiring the reply to restate turn 1's number. On macOS
    that failed against a server whose history demonstrably arrived: the same run
    answered the last turn 'The capital of France is Paris.', which its own prompt
    ("Repeat the city name") cannot produce without turn 3. So an unhelpfully worded
    turn 2 must not fail, while a last turn that cannot name the city must.
    """
    # Exactly the macOS transcript, which is a healthy server.
    macos = ["58 + 27 = 95", "You haven't provided the previous question.", "paris", "paris"]
    script.check("macos", macos, list(macos))

    # And the measured reply of a server sent the last prompt with no history at all.
    no_history = ["58 + 27 = 95", "the answer was 95", "paris", "Okay, I'm ready."]
    with pytest.raises(AssertionError, match = "history reached the model"):
        script.check("dropped", no_history, list(no_history))


def test_turn_1_must_still_answer_with_a_number(script):
    """Turn 1 is what makes the conversation multi-turn; a server that cannot do
    arithmetic at all is not exercising history."""
    broken = ["I cannot do arithmetic.", "b", "paris", "paris"]
    with pytest.raises(AssertionError, match = "should contain a number"):
        script.check("nonumber", broken, list(broken))


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


def test_the_replay_retry_cannot_pass_a_truly_nondeterministic_server(script, monkeypatch):
    """A retry is a way to hide a real fault, so it is pinned by running it.

    The near-tie this exists for flips occasionally; a server that samples disagrees
    every time. The first must pass and the second must still fail, and the retry must
    be bounded rather than looping until it gets lucky.
    """
    clean = ["58 + 27 = 95", "the answer was 95", "paris", "paris"]

    calls = {"n": 0}

    def always_divergent():
        calls["n"] += 1
        # A different last turn every call, so no two replays ever agree.
        return ["58 + 27 = 95", "the answer was 95", "paris", f"paris {calls['n']}"]

    monkeypatch.setattr(script, "assert_reproducible_backend", lambda: None)
    monkeypatch.setattr(script, "run_openai", always_divergent)
    monkeypatch.setattr(script, "run_anthropic", always_divergent)
    with pytest.raises(AssertionError, match = "non-deterministic"):
        script.main()
    # Bounded: two runners per attempt, and it must not have looped past ATTEMPTS.
    assert calls["n"] == 2 * script.ATTEMPTS, calls["n"]

    # And a single flip, followed by agreement, passes.
    calls["n"] = 0

    def flips_once():
        calls["n"] += 1
        if calls["n"] == 2:  # the second replay of the first attempt
            return ["58 + 27 = 95", "the answer was 95", "paris", "paris!"]
        return list(clean)

    monkeypatch.setattr(script, "run_openai", flips_once)
    monkeypatch.setattr(script, "run_anthropic", flips_once)
    assert script.main() == 0


def test_a_non_divergence_failure_is_not_retried(script, monkeypatch):
    """Retrying an empty reply or a dropped history would just cost three runs and
    still fail, and would let a broken server look intermittent."""
    calls = {"n": 0}

    def no_history():
        calls["n"] += 1
        return ["58 + 27 = 95", "b", "paris", "Okay, I'm ready."]

    monkeypatch.setattr(script, "assert_reproducible_backend", lambda: None)
    monkeypatch.setattr(script, "run_openai", no_history)
    monkeypatch.setattr(script, "run_anthropic", no_history)
    with pytest.raises(AssertionError, match = "history reached the model"):
        script.main()
    assert calls["n"] == 2, f"retried a non-divergence failure {calls['n']} times"


def test_every_leg_pins_the_probe_load_to_a_reproducible_backend():
    """The determinism assertion is only answerable against a pinned load.

    A non-MTP GGUF like gemma-3-270m-it takes llama.cpp's `--spec-default` branch,
    which is live n-gram drafting, not the absence of a drafter: the server logs
    `draft acceptance = 0.46875 (30 accepted / 64 generated)`. The draft pool is
    built from text the server has already seen, so the first turn-1 request after a
    load drafts nothing and decodes one token at a time while every later one
    verifies a ~32-token draft in one batch, and llama-server's README is explicit
    that logits are not bit-for-bit identical across batch shapes. Since
    check(runner(), runner()) always compares the cold replay against a warm one,
    that made attempt 1 structurally unable to agree, and the whole check a coin
    flip the retry usually won.

    Only the load that feeds the multi-turn probe is pinned. The tool-calling and
    vision phases deliberately keep the defaults, so the shipped path stays covered.
    """
    for name in LEGS:
        text = _workflow(name)
        # rsplit, not split: two of the legs also name the script in their `paths:`
        # filter, long before the step that runs it.
        probe = text.rsplit("studio_smoke/multi_turn_chat.py", 1)[0]
        assert '\\"speculative_type\\":\\"off\\"' in probe, (
            f"{name} loads the multi-turn probe's model without pinning "
            f"speculative_type=off, so a drafted batch can replace sequential decode "
            f"between the two replays and greedy output stops being reproducible"
        )
        assert '\\"n_parallel\\":1' in probe, (
            f"{name} loads the multi-turn probe's model without pinning n_parallel=1, "
            f"so --kv-unified shares one KV pool across slots and its occupancy is "
            f"another input to the batch"
        )


def test_the_probe_refuses_a_backend_that_cannot_be_reproducible(script, monkeypatch):
    """The workflow pin is worth nothing if a load can silently drop it.

    A divergence below is only evidence of a fault if the server was in a
    configuration that could have avoided one, so main() checks before it generates
    and names what is wrong instead of reporting a mystery disagreement.
    """
    clean = ["58 + 27 = 95", "the answer was 95", "paris", "paris"]
    monkeypatch.setattr(script, "run_openai", lambda: list(clean))
    monkeypatch.setattr(script, "run_anthropic", lambda: list(clean))

    def _with(status):
        monkeypatch.setattr(script, "_read_backend_status", lambda: status)

    _with({"speculative_type": "default", "parallel_slots": 1})
    with pytest.raises(AssertionError, match = "speculative decoding off"):
        script.main()

    _with({"speculative_type": "off", "parallel_slots": 4})
    with pytest.raises(AssertionError, match = "one decode slot"):
        script.main()

    _with({"speculative_type": "off", "parallel_slots": 1})
    assert script.main() == 0
