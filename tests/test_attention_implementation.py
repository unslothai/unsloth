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
