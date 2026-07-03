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


def test_explicit_flex_is_honored_even_when_not_marked_supported():
    config = {}
    result = _disable_flash_attention_if_needed(
        config,
        attn_implementation = "flex_attention",
        supports_sdpa = True,
        supports_flex_attention = False,
        would_use_flash_attention = True,
        disable_reason = "unit test forces flash disabled",
    )
    assert result == "flex_attention"


def test_no_disable_reason_returns_request_untouched():
    # When nothing disables flash, the function returns the request as-is.
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
