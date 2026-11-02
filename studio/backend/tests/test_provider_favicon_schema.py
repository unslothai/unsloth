# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pydantic validation for the persisted provider site icon field."""

import pytest
from pydantic import ValidationError

from models.providers import (
    MAX_PROVIDER_ICON_DATA_URL_LENGTH,
    ProviderCreate,
    ProviderUpdate,
)

_PNG_DATA_URL = "data:image/png;base64,AAAA"
_SVG_DATA_URL = "data:image/svg+xml;base64,PHN2Zz48L3N2Zz4="


def test_provider_create_accepts_png_data_url():
    created = ProviderCreate(
        provider_type = "custom",
        display_name = "Custom",
        base_url = "https://example.com/v1",
        favicon_data_url = _PNG_DATA_URL,
    )
    assert created.favicon_data_url == _PNG_DATA_URL


def test_provider_create_rejects_svg_data_url():
    with pytest.raises(ValidationError):
        ProviderCreate(
            provider_type = "custom",
            display_name = "Custom",
            base_url = "https://example.com/v1",
            favicon_data_url = _SVG_DATA_URL,
        )


def test_provider_create_rejects_oversized_data_url():
    big = "data:image/png;base64," + "A" * (MAX_PROVIDER_ICON_DATA_URL_LENGTH + 1)
    with pytest.raises(ValidationError):
        ProviderCreate(
            provider_type = "custom",
            display_name = "Custom",
            base_url = "https://example.com/v1",
            favicon_data_url = big,
        )


def test_provider_update_accepts_png_data_url_and_null():
    assert ProviderUpdate(favicon_data_url = _PNG_DATA_URL).favicon_data_url == _PNG_DATA_URL
    assert ProviderUpdate(favicon_data_url = None).favicon_data_url is None
    assert ProviderUpdate().favicon_data_url is None


def test_provider_update_rejects_svg_data_url():
    with pytest.raises(ValidationError):
        ProviderUpdate(favicon_data_url = _SVG_DATA_URL)
