# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Every checkpoint a wired leg loads must be on the prefetch lane.

The prefetch runs in the driver's own interpreter from t=0, on no card and no
virtualenv, and it measured **~203 MB/s** on kernel
unsloth-probe-prefetch-verify-9568-7a0bdd -- 12.5 GB of gpt-oss in 61.7s. A repo
it does not name is downloaded by the leg instead, ON the card, while that card
is allocated and idle.

The failure is silent in the way this directory keeps being caught by: nothing
is red, the leg simply takes longer, and a schedule built on the assumption that
downloads are hidden is quietly wrong. It had already happened. `PREFETCH_REPOS`
listed Qwen2.5-0.5B-Instruct and gpt-oss, while `vision_fla_compile` --
the leg that SETS the makespan -- fetched its own 4.58 GB Qwen3.5-2B inline, and
`default` fetched Qwen3-0.6B inline.

Two details the rule has to respect, both learned the hard way:

* a leg with no ``--model`` takes its payload's argparse default, so reading the
  args alone reports the wrong checkpoint for every such leg;
* ``load_in_4bit=True`` sends unsloth through FLOAT_TO_INT_MAPPER to a
  ``-unsloth-bnb-4bit`` sibling, so warming the name in the args can warm a
  cache the leg never reads. ``LOAD_REDIRECTS`` records the ones that are known
  to differ.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAYLOAD_DIR = ROOT / "tests" / "kaggle" / "t4_smoke"
sys.path.insert(0, str(ROOT / ".github" / "scripts"))

from kaggle_t4_ci import legs  # noqa: E402


def _payload_default(entry: str) -> str | None:
    """The `--model` default the payload itself carries."""
    source = (PAYLOAD_DIR / entry).read_text(encoding = "utf-8")
    if 'ap.add_argument("--model", default = DEFAULT_MODEL)' not in source:
        return None
    match = re.search(r'^DEFAULT_MODEL = "([^"]+)"', source, re.MULTILINE)
    return match.group(1) if match else None


def models_for(leg) -> set[str]:
    args = list(leg.args)
    if "--model" in args:
        named = args[args.index("--model") + 1]
    else:
        named = _payload_default(leg.entry)
    if not named:
        return set()
    # The declared name AND whatever it really resolves to on an sm_75 card.
    return {named, legs.LOAD_REDIRECTS.get(named, named)}


def test_every_wired_leg_loads_a_prefetched_checkpoint():
    wired = [name for kernel in legs.KERNELS for name in kernel]
    missing = {}
    for name in wired:
        wanted = models_for(legs.LEGS[name])
        # The redirect target is what gets read, so satisfying either name is
        # enough only when they are the same repo.
        if wanted and not (wanted & set(legs.PREFETCH_REPOS)):
            missing[name] = sorted(wanted)
    assert not missing, (
        "these wired legs download their checkpoint on an allocated card "
        f"instead of on the free prefetch lane: {missing}. Add them to "
        "PREFETCH_REPOS."
    )


def test_the_model_walk_reads_the_payload_default_and_not_only_the_args():
    """Three of the five wired legs carry no --model at all. A rule that read
    the args alone would report them as having no checkpoint and pass by
    finding nothing, which is the shape of a guard that guards nothing."""
    assert models_for(legs.LEGS["canary"]) == {"unsloth/Qwen2.5-0.5B-Instruct"}
    assert "unsloth/gpt-oss-20b-unsloth-bnb-4bit" in models_for(legs.LEGS["gptoss"]), (
        "the gpt-oss redirect is not being applied, so the prefetch would warm "
        "the 16-bit repo that no sm_75 run ever loads"
    )


def test_the_critical_path_leg_is_fetched_before_the_one_with_slack():
    """Order is not decoration: the lane fetches in the order given.

    vision_fla_compile starts at t~21 and sets the makespan; gptoss is admitted
    only when a card empties, around t~500 on the measured schedule. The
    original order put gpt-oss first for margin when D was unknown; D is
    measured now, so the leg with the least slack goes first.
    """
    order = list(legs.PREFETCH_REPOS)
    assert order.index("unsloth/Qwen3.5-2B") < order.index(
        "unsloth/gpt-oss-20b-unsloth-bnb-4bit"
    ), "the leg that sets the makespan is queued behind the leg with 500s of slack"


def test_nothing_is_prefetched_that_no_leg_reads():
    """The other direction, and it costs bandwidth rather than time: a repo
    nobody loads is a download the session pays for and never uses. Checked
    against every leg, not only the wired ones, so a leg parked in UNWIRED can
    keep its entry."""
    loaded = set()
    for leg in legs.LEGS.values():
        loaded |= models_for(leg)
    # Studio's own models are fetched by the Studio builder under its own
    # HF_HOME, so they are deliberately not in this list.
    stray = [repo for repo in legs.PREFETCH_REPOS if repo not in loaded]
    assert not stray, f"prefetched but never loaded by any leg: {stray}"


def test_every_redirect_target_is_prefetched_under_its_EXACT_name():
    """The HF cache keys on the literal repo string.

    `models--unsloth--qwen3-0.6b-unsloth-bnb-4bit` and
    `models--unsloth--Qwen3-0.6B-unsloth-bnb-4bit` are different directories, so
    prefetching the pretty spelling of a repo the loader asks for in lower case
    warms a cache nobody reads and the session downloads it twice -- at full
    cost, with no error and nothing red. Two hardware reports give the exact
    strings; this asserts the lists agree with them character for character.
    """
    prefetch = set(legs.PREFETCH_REPOS)
    wired = {name for kernel in legs.KERNELS for name in kernel}
    for leg_name in wired:
        leg = legs.LEGS[leg_name]
        args = list(leg.args)
        named = (
            args[args.index("--model") + 1] if "--model" in args else _payload_default(leg.entry)
        )
        if named is None or named not in legs.LOAD_REDIRECTS:
            continue
        target = legs.LOAD_REDIRECTS[named]
        assert target in prefetch, (
            f"{leg_name} loads {target!r}, which is not prefetched under that "
            f"exact string; a case-different entry does not warm the same cache"
        )


def test_the_lower_case_qwen3_redirect_is_not_tidied_away():
    """It looks like a typo and it is a measurement. Written down so the next
    reader corrects the capitals in the report, not in this file."""
    assert legs.LOAD_REDIRECTS["unsloth/Qwen3-0.6B"] == "unsloth/qwen3-0.6b-unsloth-bnb-4bit"
