# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What the picker offers has to be what /training/start accepts.

``local_options.py`` starts the cached-dataset subset and split options the picker shows,
and ``TrainingStartRequest`` decides what a start request may carry. They were written with
different grammars, which fails in both directions:

- ``_SPLIT_RE`` was ``\\w+(?:\\.\\w+)*``, so it dropped the hyphen. A real split name like
  ``train-clean`` (LibriSpeech) never reached the picker, and an offline user had to type it
  by hand into a field that would have accepted it.
- ``\\w`` is Unicode-aware in Python, so ``tréin`` was offered and then rejected by the
  ASCII-only split validator. ``_CONFIG_RE`` excluded only filesystem-hostile characters, so
  a config name with a space was offered and then rejected by the subset validator.

Both directions are silent: one hides a usable option, the other turns a click into a 422.
"""

import re

import pytest

from hub.services.datasets.local_options import _CONFIG_RE, _SPLIT_RE, _valid_option
from models.training import TrainingStartRequest


def _accepted_by(field: str, value: str) -> bool:
    """Run the real TrainingStartRequest validator for *field*, so the test cannot drift
    from the model by restating its regex here."""
    for validator in TrainingStartRequest.__pydantic_decorators__.field_validators.values():
        if field in validator.info.fields:
            try:
                validator.func(value)  # already a bound classmethod
            except ValueError:
                return False
            return True
    raise AssertionError(f"no validator found for {field}")


def _subset_accepted(value: str) -> bool:
    return _accepted_by("subset", value)


def _split_accepted(value: str) -> bool:
    return _accepted_by("train_split", value)


def _offered_split(value: str):
    return _valid_option(value, _SPLIT_RE, reject_dotdot = True)


# (value, offered_by, accepted_by)
_SPLITS = ["train", "validation", "test", "train-clean", "train.clean", "train[:10%]"]
_CONFIGS = ["default", "cfg-1", "en.simple", "wikitext-103-raw-v1", "v1..v2"]
_REJECTED = ["tréin", "my config", "träin", "tr..in"]


@pytest.mark.parametrize("value", _SPLITS)
def test_a_split_the_backend_accepts_is_offered(value):
    """The hiding direction: a start request would take it, so the picker must show it."""
    assert _split_accepted(value), f"fixture wrong: {value!r} is not accepted by the backend"
    assert _offered_split(value) == value, (
        f"{value!r} is a valid split for /training/start but the picker filters it out, "
        "so an offline user has to type it by hand"
    )


@pytest.mark.parametrize("value", _CONFIGS)
def test_a_subset_the_backend_accepts_is_offered(value):
    assert _subset_accepted(value), f"fixture wrong: {value!r} is not accepted by the backend"
    assert _valid_option(value, _CONFIG_RE) == value


@pytest.mark.parametrize("value", _REJECTED)
def test_nothing_the_backend_rejects_is_offered(value):
    """The 422 direction: offering it turns a click into a rejected start."""
    offered_split = _offered_split(value)
    assert offered_split is None or _split_accepted(offered_split), (
        f"{value!r} is offered as split {offered_split!r} but /training/start rejects it, so "
        "selecting the option the picker showed returns 422"
    )
    offered_subset = _valid_option(value, _CONFIG_RE)
    assert offered_subset is None or _subset_accepted(
        offered_subset
    ), f"{value!r} is offered as subset {offered_subset!r} but /training/start rejects it"


def test_the_two_grammars_agree_over_a_generated_alphabet():
    """Neither side may drift: every string built from the union charset agrees."""
    alphabet = "abZ09_-.[]:%+ é/\\"
    mismatches = []
    for a in alphabet:
        for b in ("", "x", ".x"):
            value = f"tr{a}{b}"
            # _valid_option normalizes (it strips), and the normalized string is what the
            # picker offers, so that is what has to survive the start validator.
            offered_split = _offered_split(value)
            if offered_split is not None and not _split_accepted(offered_split):
                mismatches.append(("split offered, start rejects", value, offered_split))
            offered_subset = _valid_option(value, _CONFIG_RE)
            if offered_subset is not None and not _subset_accepted(offered_subset):
                mismatches.append(("subset offered, start rejects", value, offered_subset))
    assert not mismatches, mismatches
