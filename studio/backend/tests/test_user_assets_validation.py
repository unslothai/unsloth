# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest

from core.user_assets_validation import UserAssetValidationError, validate_recipe_payload


def test_proxy_authorization_is_rejected_and_redacted_by_the_same_policy():
    payload = {"headers": {"Proxy-Authorization": "Basic secret"}}
    with pytest.raises(UserAssetValidationError, match = "secret fields"):
        validate_recipe_payload(payload)
    clean, paths = validate_recipe_payload(payload, legacy = True)
    assert clean == {"headers": {}}
    assert paths == ['$.headers["Proxy-Authorization"]']
