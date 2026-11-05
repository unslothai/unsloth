# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""formatted_int must not print 1,000.0 of a unit when the next unit applies."""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("huggingface_hub")

_PATH = Path(__file__).resolve().parents[1] / "unsloth" / "utils" / "hf_hub.py"
_spec = importlib.util.spec_from_file_location("_unsloth_hf_hub_under_test", _PATH)
hf_hub = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hf_hub)


@pytest.mark.parametrize(
    "value, expected",
    [
        (0, "0"),
        (999, "999"),
        (1_000, "1.0K"),
        (12_345, "12.3K"),
        (999_949, "999.9K"),
        (999_999, "1.0M"),
        (1_000_000, "1.0M"),
        (999_999_999, "1.0B"),
        (1_500_000_000, "1.5B"),
        (2_500_000_000_000, "2,500.0B"),
    ],
)
def test_formatted_int(value, expected):
    assert hf_hub.formatted_int(value) == expected
