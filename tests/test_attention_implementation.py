from types import SimpleNamespace

import unsloth  # noqa: F401
from transformers.utils import import_utils

from unsloth.models import _utils


class SupportsAllAttention:
    _supports_flash_attn_2 = True
    _supports_flex_attn = True
    _supports_sdpa = True


class SupportsFlexAndSdpa:
    _supports_flash_attn_2 = False
    _supports_flex_attn = True
    _supports_sdpa = True


def _config(model_type, **kwargs):
    values = {"model_type": model_type, "attention_dropout": 0}
    values.update(kwargs)
    return SimpleNamespace(**values)


def _enable_attention_backends(monkeypatch, flex_available=True):
    monkeypatch.setattr(_utils, "HAS_FLASH_ATTENTION", True)
    monkeypatch.setattr(
        import_utils,
        "is_torch_flex_attn_available",
        lambda: flex_available,
        raising=False,
    )
    monkeypatch.setenv("UNSLOTH_ENABLE_FLEX_ATTENTION", "1")


def test_gemma3_family_prefers_flex_attention_over_flash_attention(monkeypatch):
    _enable_attention_backends(monkeypatch)

    for model_type in ("gemma3", "gemma3_text", "shieldgemma2"):
        config = _config(model_type)

        impl = _utils.resolve_attention_implementation(SupportsAllAttention, config)

        assert impl == "flex_attention"
        assert config._attn_implementation == "flex_attention"


def test_nested_gemma3_text_prefers_flex_attention(monkeypatch):
    _enable_attention_backends(monkeypatch)
    text_config = _config("gemma3_text")
    config = _config("gemma3", text_config=text_config)

    impl = _utils.resolve_attention_implementation(SupportsAllAttention, config)

    assert impl == "flex_attention"
    assert config._attn_implementation == "flex_attention"


def test_non_gemma3_keeps_flash_attention_priority(monkeypatch):
    _enable_attention_backends(monkeypatch)
    config = _config("llama")

    impl = _utils.resolve_attention_implementation(SupportsAllAttention, config)

    assert impl == "flash_attention_2"
    assert config._attn_implementation == "flash_attention_2"


def test_generic_model_uses_flex_when_flash_attention_is_not_safe(monkeypatch):
    _enable_attention_backends(monkeypatch)
    config = _config("llama", head_dim=320)

    impl = _utils.resolve_attention_implementation(SupportsAllAttention, config)

    assert impl == "flex_attention"
    assert config._attn_implementation == "flex_attention"


def test_gemma3n_defaults_to_eager_unless_explicitly_overridden(monkeypatch):
    _enable_attention_backends(monkeypatch)
    config = _config("gemma3n")

    impl = _utils.resolve_attention_implementation(SupportsAllAttention, config)

    assert impl == "eager"
    assert config._attn_implementation == "eager"

    impl = _utils.resolve_attention_implementation(
        SupportsAllAttention,
        config,
        requested_attn_implementation="sdpa",
    )

    assert impl == "sdpa"
    assert config._attn_implementation == "sdpa"


def test_flex_excluded_model_does_not_auto_select_flex(monkeypatch):
    _enable_attention_backends(monkeypatch)
    config = _config("gpt_oss")

    impl = _utils.resolve_attention_implementation(SupportsFlexAndSdpa, config)

    assert impl == "sdpa"
    assert config._attn_implementation == "sdpa"


def test_explicit_gemma3_flash_attention_request_is_preserved(monkeypatch):
    _enable_attention_backends(monkeypatch)
    config = _config("gemma3")

    impl = _utils.resolve_attention_implementation(
        SupportsAllAttention,
        config,
        requested_attn_implementation="flash_attention_2",
    )

    assert impl == "flash_attention_2"
    assert config._attn_implementation == "flash_attention_2"


def test_explicit_gemma3_sdpa_and_eager_requests_are_preserved(monkeypatch):
    _enable_attention_backends(monkeypatch)

    for requested in ("sdpa", "eager"):
        config = _config("gemma3")

        impl = _utils.resolve_attention_implementation(
            SupportsAllAttention,
            config,
            requested_attn_implementation=requested,
        )

        assert impl == requested
        assert config._attn_implementation == requested


def test_explicit_flash_attention_request_falls_back_on_unsupported_head_dim(
    monkeypatch,
):
    _enable_attention_backends(monkeypatch)
    config = _config(
        "gemma3",
        head_dim=320,
    )

    impl = _utils.resolve_attention_implementation(
        SupportsAllAttention,
        config,
        requested_attn_implementation="flash_attention_2",
    )

    assert impl == "sdpa"
    assert config._attn_implementation == "sdpa"


def test_gemma3_uses_flash_attention_when_flex_is_disabled(monkeypatch):
    _enable_attention_backends(monkeypatch)
    monkeypatch.setenv("UNSLOTH_ENABLE_FLEX_ATTENTION", "0")
    config = _config("gemma3")

    impl = _utils.resolve_attention_implementation(SupportsAllAttention, config)

    assert impl == "flash_attention_2"
    assert config._attn_implementation == "flash_attention_2"


def test_flex_attention_requires_zero_dropout_in_nested_config(monkeypatch):
    _enable_attention_backends(monkeypatch)
    text_config = _config("gemma3_text", attention_dropout=0.1)
    config = _config("gemma3", text_config=text_config)

    impl = _utils.resolve_attention_implementation(SupportsAllAttention, config)

    assert impl == "flash_attention_2"
    assert config._attn_implementation == "flash_attention_2"
