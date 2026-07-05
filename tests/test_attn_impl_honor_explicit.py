"""An explicit non-flash attention request must survive the flash disable path.

When flash attention is disabled for a model, a caller who explicitly asked for
"sdpa" or "flex_attention" should keep that choice instead of being downgraded
to whatever the conservative supports_* fallback would pick.
"""

import pytest

from unsloth.models._utils import _disable_flash_attention_if_needed


def test_explicit_sdpa_is_honored_even_when_not_marked_supported():
    config = {}
    result = _disable_flash_attention_if_needed(
        config,
        attn_implementation = "sdpa",
        supports_sdpa = False,  # conservative flag would have skipped sdpa
        supports_flex_attention = False,
        would_use_flash_attention = True,
        disable_reason = "unit test forces flash disabled",
    )
    assert result == "sdpa"
    assert config.get("_attn_implementation") == "sdpa"


def test_explicit_flex_is_honored_when_supported():
    config = {}
    result = _disable_flash_attention_if_needed(
        config,
        attn_implementation = "flex_attention",
        supports_sdpa = True,
        supports_flex_attention = True,
        would_use_flash_attention = True,
        disable_reason = "unit test forces flash disabled",
    )
    assert result == "flex_attention"
    assert config.get("_attn_implementation") == "flex_attention"


def test_explicit_flex_falls_back_when_not_supported():
    # flex_attention is False for known-broken/excluded configs (e.g. gpt_oss),
    # so an explicit flex request must not select that backend - it falls back.
    config = {}
    result = _disable_flash_attention_if_needed(
        config,
        attn_implementation = "flex_attention",
        supports_sdpa = True,
        supports_flex_attention = False,
        would_use_flash_attention = True,
        disable_reason = "unit test forces flash disabled",
    )
    assert result == "sdpa"


def test_synthesized_config_sdpa_is_not_treated_as_explicit():
    # The language loader seeds the config with attn_implementation="sdpa"; when the
    # caller passes nothing, that synthesized value must not override the flex fallback
    # for a model that supports flex but not sdpa.
    config = {"attn_implementation": "sdpa"}
    result = _disable_flash_attention_if_needed(
        config,
        attn_implementation = None,
        supports_sdpa = False,
        supports_flex_attention = True,
        would_use_flash_attention = False,
        disable_reason = "unit test forces flash disabled",
    )
    assert result == "flex_attention"


def test_no_disable_reason_returns_request_untouched():
    result = _disable_flash_attention_if_needed(
        {},
        attn_implementation = "flash_attention_2",
        disable_reason = None,
    )
    assert result == "flash_attention_2"


def test_flash_request_still_falls_back_when_disabled():
    config = {}
    result = _disable_flash_attention_if_needed(
        config,
        attn_implementation = "flash_attention_2",
        supports_sdpa = True,
        would_use_flash_attention = True,
        disable_reason = "unit test forces flash disabled",
    )
    assert result == "sdpa"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
