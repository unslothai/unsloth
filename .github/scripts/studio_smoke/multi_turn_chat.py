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
import sys

SEED = 3407
MAX_TOKENS = 80

# Turn 2 cannot be answered without turn 1, and turn 4 without turn 3, so a server that
# drops history fails here rather than returning something plausible.
PROMPTS = [
    "What is 1+1?",
    "What did I ask before?",
    "What is the capital of France?",
    "Repeat the city name",
]


def _server() -> tuple[str, str]:
    """Where to talk and what to send. The only thing that differs per caller: each
    workflow boots its server on its own port. Read here rather than at import, so the
    checking half of this file can be exercised without a server or the SDKs."""
    return os.environ["BASE_URL"], os.environ["TOKEN"]  # a JWT is accepted as Bearer


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
    # Two SDK quirks against Unsloth:
    #   1. base_url must NOT include /v1 -- the SDK appends /v1/messages itself, and a
    #      base_url that already has it hits /v1/v1/messages and 405s.
    #   2. The SDK sends x-api-key by default, but Unsloth's auth layer is HTTPBearer
    #      only, so Authorization has to be set through default_headers instead.
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
        # BOTH runs, not just the first. Stripping makes the comparison below blind to
        # the difference between "\n" and "": a second run that returned nothing at all
        # would compare equal to a first that returned only tolerated whitespace, and the
        # smoke test would print OK for a server that had stopped answering. The Linux
        # copy asserted both before this was consolidated; the macOS one it was taken
        # from asserted only the first.
        assert a, f"{label}: empty turn {i} response in the first run"
        assert b, f"{label}: empty turn {i} response in the second run"
        # Compared stripped: llama-server varies trailing whitespace (a final newline)
        # between otherwise identical greedy runs, depending on the batch-flush boundary
        # at which the stream is closed. The generated tokens are the same; only that
        # whitespace differs. The raw repr stays in the message so a real divergence is
        # still legible.
        assert a.strip() == b.strip(), (
            f"{label} non-deterministic at turn {i} with temperature=0.0:\n"
            f"  run1: {a!r}\n  run2: {b!r}"
        )
    # Turn 2 should mention the earlier question and turn 4 the city turn 3 produced.
    # Lower-cased substring checks, so formatting jitter is not a failure.
    joined = " ".join(first).lower()
    assert "1" in first[0], f"{label}: turn-1 answer should contain '1', got {first[0]!r}"
    assert (
        "paris" in joined
    ), f"{label}: expected 'paris' somewhere in the four-turn transcript: {first}"
    print(f"[{label}] OK -- 4 turns, run1 == run2, history grounded")


def main() -> int:
    for label, runner in (("openai", run_openai), ("anthropic", run_anthropic)):
        check(label, runner(), runner())
    return 0


if __name__ == "__main__":
    sys.exit(main())
