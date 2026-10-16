"""An explicit sdpa or flex_attention request must survive when flash attention is disabled."""

import pytest

from unsloth.models._utils import (
    _disable_flash_attention_if_needed,
    resolve_attention_implementation,
)


def test_explicit_sdpa_is_honored_even_when_not_marked_supported():
    config = {}
    result = _disable_flash_attention_if_needed(
        config,
        attn_implementation = "sdpa",
        supports_sdpa = False,
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
    # flex_attention is False for excluded configs (e.g. gpt_oss), so an explicit flex request falls back.
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
    # The loader seeds attn_implementation="sdpa"; that synthesized value must not override the fallback.
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


def test_resolver_honors_explicit_sdpa_when_not_supported_and_flash_disabled():
    config = {"model_type": "test", "head_dim": 512}  # head_dim > 256 disables flash
    result = resolve_attention_implementation(
        model_class = None,
        config = config,
        requested_attn_implementation = "sdpa",
        supports_sdpa = False,
    )
    assert result == "sdpa"
    assert config.get("_attn_implementation") == "sdpa"


def test_resolver_downgrades_non_explicit_sdpa_when_not_supported():
    config = {"model_type": "test", "attn_implementation": "sdpa"}
    result = resolve_attention_implementation(
        model_class = None,
        config = config,
        requested_attn_implementation = None,
        supports_sdpa = False,
    )
    assert result == "eager"


def test_resolver_downgrades_explicit_sdpa_for_sdpa_excluded_model():
    # gpt_oss sdpa is known-broken: an explicit sdpa request must still downgrade to eager.
    config = {"model_type": "gpt_oss"}
    result = resolve_attention_implementation(
        model_class = None,
        config = config,
        requested_attn_implementation = "sdpa",
        supports_sdpa = True,
    )
    assert result == "eager"
    assert config.get("_attn_implementation") == "eager"


@pytest.mark.parametrize("model_type", ["gemma3", "gemma3_text"])
def test_resolver_downgrades_explicit_sdpa_for_disable_sdpa_model(model_type):
    # gemma3's bundled SDPA modules are wrong, so an explicit sdpa request downgrades to eager.
    config = {"model_type": model_type, "head_dim": 512}
    result = resolve_attention_implementation(
        model_class = None,
        config = config,
        requested_attn_implementation = "sdpa",
        supports_sdpa = False,
    )
    assert result == "eager"
    assert config.get("_attn_implementation") == "eager"


def test_resolver_does_not_overmatch_gemma3n_for_explicit_sdpa():
    # The "gemma3," guard must not match gemma3n, which is not in DISABLE_SDPA_MODEL_NAMES.
    config = {"model_type": "gemma3n", "head_dim": 512}
    result = resolve_attention_implementation(
        model_class = None,
        config = config,
        requested_attn_implementation = "sdpa",
        supports_sdpa = False,
    )
    assert result == "sdpa"
    assert config.get("_attn_implementation") == "sdpa"


def test_resolver_downgrades_synthesized_sdpa_for_disable_sdpa_model():
    config = {"model_type": "gemma3", "attn_implementation": "sdpa"}
    result = resolve_attention_implementation(
        model_class = None,
        config = config,
        requested_attn_implementation = None,
        supports_sdpa = False,
    )
    assert result == "eager"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
