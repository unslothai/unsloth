# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Custom-code generate that rejects cache fields (LiquidAI LFM3, unslothai/unsloth#16054).

LFM3-Omni's own generate lists cache_implementation in _generation_fields_unsupported and raises
when it is set, so FastModel generate must not force a static cache onto it.
"""

import types

from unsloth.models.vision import _generate_rejects_cache_fields


def test_model_listing_cache_implementation_is_skipped():
    model = types.SimpleNamespace(
        _generation_fields_unsupported = {"cache_implementation": None, "use_cache": True}
    )
    assert _generate_rejects_cache_fields(model) is True


def test_regular_model_keeps_static_cache():
    assert _generate_rejects_cache_fields(types.SimpleNamespace()) is False
    model = types.SimpleNamespace(_generation_fields_unsupported = {"max_time": None})
    assert _generate_rejects_cache_fields(model) is False
