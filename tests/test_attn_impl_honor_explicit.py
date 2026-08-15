"""An explicit non-flash attention request must survive the flash disable path.

When flash attention is disabled for a model, a caller who explicitly asked for
"sdpa" or "flex_attention" should keep that choice instead of being downgraded
to whatever the conservative supports_* fallback would pick.
"""

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


def test_resolver_honors_explicit_sdpa_when_not_supported_and_flash_disabled():
    # End-to-end through the public resolver: an explicit sdpa request with a
    # flash-disabled config (oversized head dim) and supports_sdpa=False must not be
    # rewritten to eager by the resolver's own not-supports_sdpa guard.
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
    # No explicit request: the model resolution seeds sdpa/eager and the guard must
    # still downgrade a synthesized sdpa to eager for a model that cannot run it.
    config = {"model_type": "test", "attn_implementation": "sdpa"}
    result = resolve_attention_implementation(
        model_class = None,
        config = config,
        requested_attn_implementation = None,
        supports_sdpa = False,
    )
    assert result == "eager"


def test_resolver_downgrades_explicit_sdpa_for_sdpa_excluded_model():
    # gpt_oss is in _SDPA_EXCLUDED_MODELS (sdpa is known-broken) and _FLASH_EXCLUDED_MODELS
    # (flash disabled). Honoring an explicit sdpa request must not re-enable that broken
    # backend: it downgrades to eager, mirroring how an explicit flex request falls back
    # for _FLEX_EXCLUDED_MODELS. supports_sdpa=True proves the exclusion overrides even a
    # model that otherwise advertises SDPA support.
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
    # gemma3 / gemma3_text are in DISABLE_SDPA_MODEL_NAMES: the loader forces
    # supports_sdpa=False because their bundled SDPA modules are wrong. An explicit
    # sdpa request with flash disabled must NOT re-enable that known-wrong path - it
    # downgrades to eager, exactly like _SDPA_EXCLUDED_MODELS (gpt_oss). head_dim>256
    # disables flash to mirror the real flash-disabled scenario.
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
    # The "gemma3," trailing-comma guard must not match gemma3n: gemma3n is not in
    # DISABLE_SDPA_MODEL_NAMES, so it stays a conservative (not known-wrong) model and an
    # explicit sdpa request is still honored. Proves the substring match neither over- nor
    # under-matches.
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
    # A synthesized/default sdpa (requested is None; the value came from config) on a
    # DISABLE_SDPA_MODEL_NAMES model must still downgrade to eager.
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
