# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Four turns through both SDKs, twice, against a running Unsloth server.

Two properties at once. The conversation is built so that turns 2 and 4 are only
answerable from the earlier turns, which exercises the history wiring; and the whole
conversation is run twice at temperature 0.0 against a slot with speculative decoding
off, which is the only check anywhere that greedy decoding is reproducible.

The seed is sent but does no sampling work: at temperature 0 llama.cpp collapses to
argmax and never reaches an RNG. What it does do is make the Studio send
cache_prompt=False, which is one of the two things the comparison below needs. The
other is the load pin that assert_reproducible_backend() checks.

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

import json
import os
import re
import sys
import urllib.request

SEED = 3407
MAX_TOKENS = 80
# How many times the two-replay comparison may be re-run before a disagreement is called a real fault. See main():
# the retry lives there, NOT in check(), so check()'s contract is unchanged and a divergence handed to it is still a
# hard failure every time.
ATTEMPTS = 3


class Nondeterministic(AssertionError):
    """Greedy decoding disagreed between two replays of the same conversation.

    A subclass of AssertionError so `pytest.raises(AssertionError)` in
    tests/studio/test_smoke_workflows_share_one_script.py keeps holding, and so
    main() can tell this one failure apart from the others and retry only it.
    """


# Turn 2 cannot be answered without turn 1, and turn 4 without turn 3, so a server that drops history fails here rather
# than returning something plausible.
# Turn 2 asks for the ANSWER, not the question: gemma-3-270m-it answers "What did I ask before?" with "I am doing
# well! How can I help you today?" whether or not the history is attached, so that phrasing probes nothing.
# 58+27 rather than 1+1 so the answer cannot be guessed: deriving the expected value from turn 1 is only worth
# anything if turn 1 is unpredictable.
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
    return os.environ["BASE_URL"], os.environ["TOKEN"]  # a JWT is accepted as Bearer


def _read_backend_status() -> dict:
    """What the server says it loaded. Split out so the assertion below is reachable
    from a test without standing a server up, the same reason _server() exists."""
    base, token = _server()
    request = urllib.request.Request(
        f"{base}/api/inference/status", headers = {"Authorization": f"Bearer {token}"}
    )
    with urllib.request.urlopen(request, timeout = 30) as response:
        return json.load(response)


def assert_reproducible_backend() -> None:
    """Refuse to assert determinism against a server that cannot provide it.

    llama-server's own README says it, about cache_prompt: "the logits are not
    guaranteed to be bit-for-bit identical for different batch" sizes. Two things in
    the launch the Studio derives change the batch between one replay and the next,
    and both are per-LOAD settings, so the three workflows pin them and this checks
    the pin took.

    Speculative decoding is the one that actually bites. A non-MTP GGUF like
    gemma-3-270m-it takes the `--spec-default` branch, which llama.cpp's arg parser
    turns into n-gram-mod drafting, and the server logs prove it is live:

        slot print_timing: id 3 | task 100 | draft acceptance = 0.46875
            (30 accepted / 64 generated), mean len = 31.00

    The draft pool is built from text the server has already seen, so the FIRST turn-1
    request after a load drafts nothing and decodes one token at a time, while every
    later one verifies a ~32-token draft in a single batch. Same tokens, different
    arithmetic. That is not a near-tie a retry rides out: `check(runner(), runner())`
    means attempt 1 always compares the one cold replay against a warm one, and the
    divergences it produces are nested prefixes of each other, which is what a 270M
    model near-tied on whether to stop looks like, not a broken sampler.

    --parallel > 1 is the second: it also brings --kv-unified, one shared KV pool
    whose occupancy is another input to the batch.

    Checked, not assumed. A load that quietly dropped the pin would leave this
    printing OK for a server it never pinned, which is the same shape of hole #5669
    left on the Linux leg for three months.
    """
    status = _read_backend_status()
    speculative, slots = status.get("speculative_type"), status.get("parallel_slots")
    assert speculative in ("off", "none"), (
        f"this probe needs speculative decoding off and the server reports "
        f'{speculative!r}. Load with "speculative_type": "off": a drafted batch '
        f"replacing sequential decode is not bit-identical, so the greedy output "
        f"below would not be reproducible however many times it is retried."
    )
    assert slots == 1, (
        f"this probe needs one decode slot and the server reports "
        f'parallel_slots={slots!r}. Load with "n_parallel": 1: more slots bring '
        f"--kv-unified, whose shared-pool occupancy is another input to the batch."
    )


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
    #   1. base_url must NOT include /v1 -- the SDK appends /v1/messages itself, and a base_url that already has it
    #      hits /v1/v1/messages and 405s.
    #   2. The SDK sends x-api-key by default, but Unsloth's auth layer is HTTPBearer only, so Authorization has to be
    #      set through default_headers instead.
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
        # greedy runs, depending on the batch-flush boundary at which the stream is closed. The generated tokens are the
        # same; only that whitespace differs. The raw repr stays in the message so a real divergence is still legible.
        if a.strip() != b.strip():
            raise Nondeterministic(
                f"{label} non-deterministic at turn {i} with temperature=0.0:\n"
                f"  run1: {a!r}\n  run2: {b!r}"
            )
    numbers = re.findall(r"\d+", first[0])
    assert numbers, f"{label}: turn-1 answer should contain a number, got {first[0]!r}"

    # History grounding is asserted on the LAST turn, per turn. "Repeat the city name" names no city, so 'Paris' can
    # only have come from turn 3, and a server that kept only the latest turn answers "Okay, I'm ready." -- measured
    # against llama-server b10360 on the UD-Q4_K_XL file the workflow loads.
    #
    # #10009 asserted this on turn 2 instead, requiring the reply to restate turn 1's number, and that is a false
    # failure on macOS: in the same run that turn 2 came back "You haven't provided the previous question.", turn 4
    # answered "The capital of France is Paris.", which is only possible with history attached. Whether a 270M model
    # phrases a recalled number is a property of the model, not of the server's history wiring, so it cannot carry
    # this assertion. #10009's actual finding -- that looking for 'paris' in the JOINED transcript proves nothing,
    # because turn 3 supplies it on its own -- is what the per-turn check below keeps.
    assert len(first) == len(PROMPTS), f"{label}: expected {len(PROMPTS)} replies, got {len(first)}"
    assert "paris" in first[-1].lower(), (
        f"{label}: the last turn must name the city from turn 3, so history reached the "
        f"model. Its own prompt contains no city name. Got {first[-1]!r}"
    )
    print(f"[{label}] OK -- 4 turns, run1 == run2, history grounded on the last turn")


def main() -> int:
    # Before any generation: a divergence below is only evidence of a fault if the
    # server was in a configuration that could have avoided one.
    assert_reproducible_backend()
    for label, runner in (("openai", run_openai), ("anthropic", run_anthropic)):
        # Only a replay disagreement is retried, and only here. Two replays are not two identical requests: each turn
        # is built from the reply before it, and the second replay meets the server holding cache and slot state the
        # first one left behind, so a token the model is near-tied on can land differently without decoding being
        # broken. A server that is actually non-deterministic disagrees on every attempt and still fails.
        #
        # Deliberately NOT done by softening check(): #5669 turned this assertion into a printed warning on the Linux
        # leg and it went unnoticed for three months, so every divergence handed to check() is still a hard failure,
        # and every other fault -- an empty reply, history not reaching the model -- fails on the first attempt.
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
