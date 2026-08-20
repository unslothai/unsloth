from types import SimpleNamespace

import torch
import unsloth  # noqa: F401
from transformers.utils import import_utils

from unsloth.models import _utils


class SupportsFlexAndSdpa:
    _supports_flash_attn_2 = True
    _supports_flex_attn = True
    _supports_sdpa = True


class SupportsFlashAndSdpa:
    _supports_flash_attn_2 = True
    _supports_flex_attn = False
    _supports_sdpa = True


def _config(model_type, **kwargs):
    values = {"model_type": model_type, "attention_dropout": 0}
    values.update(kwargs)
    return SimpleNamespace(**values)


def _set_flex_available(monkeypatch, available):
    monkeypatch.setenv("UNSLOTH_ENABLE_FLEX_ATTENTION", "1")
    monkeypatch.setattr(
        import_utils,
        "is_torch_flex_attn_available",
        lambda: available,
        raising = False,
    )


def test_gpt_oss_uses_eager_instead_of_flash_flex_or_sdpa(monkeypatch):
    _set_flex_available(monkeypatch, True)
    config = _config("gpt_oss")

    impl = _utils.resolve_attention_implementation(
        SupportsFlexAndSdpa,
        config,
        supports_sdpa = True,
    )

    assert impl == "eager"
    assert config._attn_implementation == "eager"


def test_gpt_oss_falls_back_to_eager_when_flex_unavailable(monkeypatch):
    _set_flex_available(monkeypatch, False)
    config = _config("gpt_oss")

    impl = _utils.resolve_attention_implementation(
        SupportsFlexAndSdpa,
        config,
        supports_sdpa = True,
    )

    assert impl == "eager"
    assert config._attn_implementation == "eager"


def test_float32_downgrades_flash_to_sdpa(monkeypatch):
    monkeypatch.setattr(_utils, "HAS_FLASH_ATTENTION", True)
    config = _config("qwen2")

    impl = _utils.resolve_attention_implementation(
        SupportsFlashAndSdpa,
        config,
        supports_sdpa = True,
        dtype = torch.float32,
    )

    assert impl == "sdpa"
    assert config._attn_implementation == "sdpa"


def test_float32_downgrades_explicit_flash_request(monkeypatch):
    monkeypatch.setattr(_utils, "HAS_FLASH_ATTENTION", True)
    config = _config("qwen2")

    impl = _utils.resolve_attention_implementation(
        SupportsFlashAndSdpa,
        config,
        requested_attn_implementation = "flash_attention_2",
        supports_sdpa = True,
        dtype = torch.float32,
    )

    assert impl == "sdpa"


def test_half_dtypes_keep_flash(monkeypatch):
    monkeypatch.setattr(_utils, "HAS_FLASH_ATTENTION", True)
    for dtype in (torch.float16, torch.bfloat16, None):
        config = _config("qwen2")

        impl = _utils.resolve_attention_implementation(
            SupportsFlashAndSdpa,
            config,
            supports_sdpa = True,
            dtype = dtype,
        )

        assert impl == "flash_attention_2"


def test_float32_does_not_reroute_an_explicit_non_flash_request(monkeypatch):
    """A float32 load must only veto flash, not re-answer other requests.

    gemma3's SDPA is known-broken, so an explicit "sdpa" resolves to eager, but the flash
    fallback ladder prefers gemma3's flex_attention. Routing fp32 through the ladder would
    answer the same request differently for float32 than for bfloat16.
    """
    _set_flex_available(monkeypatch, True)
    monkeypatch.setattr(_utils, "HAS_FLASH_ATTENTION", True)

    answers = {}
    for dtype in (torch.bfloat16, torch.float32):
        config = _config("gemma3")
        answers[dtype] = _utils.resolve_attention_implementation(
            SupportsFlexAndSdpa,
            config,
            requested_attn_implementation = "sdpa",
            supports_sdpa = False,
            dtype = dtype,
        )

    assert answers[torch.float32] == answers[torch.bfloat16] == "eager"


def test_float32_still_downgrades_an_explicit_flash_request(monkeypatch):
    """The narrowing above must not let a flash request through on float32."""
    _set_flex_available(monkeypatch, True)
    monkeypatch.setattr(_utils, "HAS_FLASH_ATTENTION", True)
    config = _config("qwen2")

    impl = _utils.resolve_attention_implementation(
        SupportsFlashAndSdpa,
        config,
        requested_attn_implementation = "flash_attention_2",
        supports_sdpa = True,
        dtype = torch.float32,
    )

    assert impl == "sdpa"


def test_config_disable_reason_still_reroutes_a_non_flash_request(monkeypatch):
    """Only the float32 reason is narrowed - a config-driven one keeps its old behaviour."""
    _set_flex_available(monkeypatch, True)
    monkeypatch.setattr(_utils, "HAS_FLASH_ATTENTION", True)
    config = _config("gemma3", head_dim = 512)

    impl = _utils.resolve_attention_implementation(
        SupportsFlexAndSdpa,
        config,
        requested_attn_implementation = "sdpa",
        supports_sdpa = False,
        dtype = torch.bfloat16,
    )

    assert impl == "flex_attention"


def test_float32_does_not_let_a_config_seeded_eager_win(monkeypatch):
    """A checkpoint shipping `"attn_implementation": "eager"` must not drag fp32 to eager.

    With no flash-specific reason the config value never steered the choice, so float32 must
    leave the checkpoint on sdpa exactly as bfloat16 does.
    """
    _set_flex_available(monkeypatch, True)
    monkeypatch.setattr(_utils, "HAS_FLASH_ATTENTION", False)

    answers = {}
    for dtype in (torch.bfloat16, torch.float32):
        config = _config("qwen2", _attn_implementation = "eager")
        answers[dtype] = _utils.resolve_attention_implementation(
            SupportsFlashAndSdpa,
            config,
            supports_sdpa = True,
            dtype = dtype,
        )

    assert answers[torch.float32] == answers[torch.bfloat16] == "sdpa"


def test_float32_with_flash_available_and_config_seeded_eager_lands_on_sdpa(monkeypatch):
    """Same shape, but flash was genuinely on the table - the fallback is sdpa, not eager."""
    _set_flex_available(monkeypatch, True)
    monkeypatch.setattr(_utils, "HAS_FLASH_ATTENTION", True)
    config = _config("qwen2", _attn_implementation = "eager")

    impl = _utils.resolve_attention_implementation(
        SupportsFlashAndSdpa,
        config,
        supports_sdpa = True,
        dtype = torch.float32,
    )

    assert impl == "sdpa"


def test_config_disable_reason_still_honors_a_config_seeded_eager(monkeypatch):
    """The narrowing is float32-only: a real flash exclusion keeps reading the config."""
    _set_flex_available(monkeypatch, True)
    monkeypatch.setattr(_utils, "HAS_FLASH_ATTENTION", True)
    config = _config("qwen2", head_dim = 512, _attn_implementation = "eager")

    impl = _utils.resolve_attention_implementation(
        SupportsFlashAndSdpa,
        config,
        supports_sdpa = True,
        dtype = torch.bfloat16,
    )

    assert impl == "eager"
