# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""LFM3's generate rejects forced cache fields (#16054)."""

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
