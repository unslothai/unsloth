# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The picker filters in ``core/inference/providers.py`` are pattern-based, so
a new model family silently disappears when a pattern is not widened. These
tests pin live ids against the filters."""

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from core.inference.providers import PROVIDER_REGISTRY  # noqa: E402


def test_current_generation_model_ids_reach_the_picker():
    """Live ids from each provider's list endpoint must survive the
    allow/deny filters, otherwise the newest models are invisible."""
    live = {
        "openai": ["gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.5", "gpt-5.5-pro"],
        "gemini": [
            "gemini-3.6-flash",
            "gemini-3.5-flash",
            "gemini-3.5-flash-lite",
            "gemini-3.1-pro-preview",
            "gemini-3.1-flash-image",
            "gemini-3-pro-image",
            "gemini-2.5-flash",
        ],
    }
    for provider, model_ids in live.items():
        registry = PROVIDER_REGISTRY[provider]
        allow = registry.get("model_id_allowlist")
        deny = registry.get("model_id_denylist")
        deny_exact = registry.get("model_id_deny_exact") or ()
        for model_id in model_ids:
            if allow is not None:
                assert allow.match(model_id), f"{provider}: {model_id} not allowlisted"
            assert not (deny and deny.search(model_id)), f"{provider}: {model_id} denylisted"
            assert model_id not in deny_exact, f"{provider}: {model_id} denied exactly"


def test_dated_snapshots_and_retired_ids_stay_out_of_the_picker():
    openai = PROVIDER_REGISTRY["openai"]
    for model_id in ("gpt-5.6-sol-2026-07-09", "gpt-5.5-2026-04-23", "gpt-5.3"):
        assert openai["model_id_denylist"].search(model_id), model_id
    gemini = PROVIDER_REGISTRY["gemini"]
    assert "gemini-3-pro-preview" in gemini["model_id_deny_exact"]
