# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Four turns through both SDKs, twice, against a running Unsloth server.

Two properties at once. The conversation is built so that turns 2 and 4 are only
answerable from the earlier turns, which exercises the history wiring; and the whole
conversation is run twice at temperature 0.0 with a fixed seed, which is the only check
anywhere that greedy decoding is reproducible.

This lived inline in three workflows, and being three copies is what let one of them
stop checking. On 2026-05-22 an unrelated event-loop fix (#5669) relaxed the Linux copy
to print a warning instead of failing; the macOS and Windows copies, which are otherwise
byte-identical in logic, kept the assertion. Linux is the leg that runs on every pull
request, so the check was effectively off where it mattered most, for three months, with
nothing to notice it. One file cannot drift from itself.

Reads BASE_URL and TOKEN from the environment, which is the only thing that differs
between the three callers: each boots its server on its own port.
"""

from __future__ import annotations

import os
import re
import sys

SEED = 3407
MAX_TOKENS = 80
# How many times the two-replay comparison may be re-run before a disagreement is called a real fault.
ATTEMPTS = 3


class Nondeterministic(AssertionError):
    """Greedy decoding disagreed between two replays of the same conversation.

    A subclass of AssertionError so `pytest.raises(AssertionError)` in
    tests/studio/test_smoke_workflows_share_one_script.py keeps holding, and so
    main() can tell this one failure apart from the others and retry only it.
    """


# Turn 2 cannot be answered without turn 1, and turn 4 without turn 3, so a server that drops history fails here rather
# than returning something plausible.
# Turn 2 asks for the ANSWER, not the question.
PROMPTS = [
    "What is 58+27?",
    "What was the answer to my previous question?",
    "What is the capital of France?",
    "Repeat the city name",
]


def _server() -> tuple[str, str]:
    """Where to talk and what to send. The only thing that differs per caller: each
    workflow boots its server on its own port. Read here rather than at import, so the
    checking half of this file can be exercised without a server or the SDKs."""
    return os.environ["BASE_URL"], os.environ["TOKEN"]


def run_openai() -> list[str]:
    from openai import OpenAI

    BASE, KEY = _server()
    client = OpenAI(base_url = f"{BASE}/v1", api_key = KEY)
    history: list[dict] = []
    replies = []
    for prompt in PROMPTS:
        history.append({"role": "user", "content": prompt})
        resp = client.chat.completions.create(
            model = "default",
            messages = history,
            temperature = 0.0,
            max_tokens = MAX_TOKENS,
            seed = SEED,
            extra_body = {"enable_thinking": False},
        )
        text = resp.choices[0].message.content or ""
        replies.append(text)
        history.append({"role": "assistant", "content": text})
    return replies


def run_anthropic() -> list[str]:
    from anthropic import Anthropic

    BASE, KEY = _server()
    # Two SDK quirks against Unsloth: 1.
    # the SDK appends /v1/messages itself, and a base_url that already has it hits /v1/v1/messages and 405s.
    # The SDK sends x-api-key by default, but Unsloth's auth layer is HTTPBearer only, so Authorization has to be set
    # through default_headers instead.
    client = Anthropic(
        base_url = BASE,
        api_key = "unused",
        default_headers = {"Authorization": f"Bearer {KEY}"},
    )
    history: list[dict] = []
    replies = []
    for prompt in PROMPTS:
        history.append({"role": "user", "content": prompt})
        msg = client.messages.create(
            model = "default",
            max_tokens = MAX_TOKENS,
            messages = history,
            temperature = 0.0,
            extra_body = {"seed": SEED, "enable_thinking": False},
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
        replies.append(text)
        history.append({"role": "assistant", "content": text})
    return replies


def check(label: str, first: list[str], second: list[str]) -> None:
    for i, (a, b) in enumerate(zip(first, second), start = 1):
        print(f"[{label} turn {i}] {a!r}")
        # BOTH runs, not just the first.
        assert a, f"{label}: empty turn {i} response in the first run"
        assert b, f"{label}: empty turn {i} response in the second run"
        # Compared stripped: llama-server varies trailing whitespace (a final newline) between otherwise identical
        if a.strip() != b.strip():
            raise Nondeterministic(
                f"{label} non-deterministic at turn {i} with temperature=0.0:\n"
                f"  run1: {a!r}\n  run2: {b!r}"
            )
    numbers = re.findall(r"\d+", first[0])
    assert numbers, f"{label}: turn-1 answer should contain a number, got {first[0]!r}"

    # History grounding is asserted on the LAST turn, per turn.
    # measured against llama-server b10360 on the UD-Q4_K_XL file the workflow loads.
    # that looking for 'paris' in the JOINED transcript proves nothing, because turn 3 supplies it on its own
    # 10009 asserted this on turn 2 instead, requiring the reply to restate turn 1's number, and that is a false failure
    # on macOS: in the same run that turn 2 came back "You haven't provided the previous question.", turn 4 answered
    # "The capital of France is Paris.", which is only possible with history attached. #10009's actual finding
    assert len(first) == len(PROMPTS), f"{label}: expected {len(PROMPTS)} replies, got {len(first)}"
    assert "paris" in first[-1].lower(), (
        f"{label}: the last turn must name the city from turn 3, so history reached the "
        f"model. Its own prompt contains no city name. Got {first[-1]!r}"
    )
    print(f"[{label}] OK -- 4 turns, run1 == run2, history grounded on the last turn")


def main() -> int:
    for label, runner in (("openai", run_openai), ("anthropic", run_anthropic)):
        # NOT done by softening check():
        # Deliberately NOT done by softening check(): #5669 turned this assertion into a
        # Only a replay disagreement is retried, and only here.
        for attempt in range(1, ATTEMPTS + 1):
            try:
                check(label, runner(), runner())
                break
            except Nondeterministic as divergence:
                if attempt == ATTEMPTS:
                    print(f"[{label}] divergent on all {ATTEMPTS} attempts", file = sys.stderr)
                    raise
                print(
                    f"[{label}] attempt {attempt}/{ATTEMPTS} disagreed between replays, "
                    f"retrying:\n{divergence}"
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
