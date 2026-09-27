# SPDX-License-Identifier: AGPL-3.0-only
"""A class that lists its only compatible flash kernels without flash_attention_2 must not get flash_attention_2.

transformers 5 rewrites a flash_attention_2 request on such a class to its first compatible kernel
(MiMoV2FlashPreTrainedModel._compatible_flash_implementations = ["flash_attention_4"]), so FastModel on
MiMo-V2-Flash / MiMo-V2.6 died with "FlashAttention4 has been toggled on, but ... doesn't seem to be installed".
"""

from types import SimpleNamespace

import pytest
import unsloth  # noqa: F401

from unsloth.models import _utils


class _Base:
    _supports_flash_attn_2 = True
    _supports_flash_attn = True
    _supports_flex_attn = False
    _supports_sdpa = True


class Fa4Only(_Base):
    _compatible_flash_implementations = ["flash_attention_4"]


class Fa4OnlyNoSdpa(Fa4Only):
    _supports_sdpa = False


class ListsFa2(_Base):
    _compatible_flash_implementations = ["flash_attention_4", "flash_attention_2"]


@pytest.fixture(autouse = True)
def _no_fa3_fa4_or_hub_kernels(monkeypatch):
    # The fallback only applies when the kernel transformers would switch to is missing.
    monkeypatch.setattr(_utils, "_flash_implementation_available", lambda name: False)


def _config(model_type):
    return SimpleNamespace(model_type = model_type, attention_dropout = 0)


def test_class_without_fa2_in_its_compatible_list_is_not_flash():
    assert _utils._model_class_supports_flash_attention(Fa4Only) is False
    assert _utils._model_class_supports_flash_attention(ListsFa2) is True
    assert _utils._model_class_supports_flash_attention(_Base) is True


def test_resolver_picks_a_non_flash_backend(monkeypatch):
    monkeypatch.setattr(_utils, "HAS_FLASH_ATTENTION", True, raising = False)
    impl = _utils.resolve_attention_implementation(
        Fa4Only, _config("fa4_only_test"), supports_sdpa = True
    )
    assert impl == "sdpa"
    impl = _utils.resolve_attention_implementation(
        Fa4OnlyNoSdpa, _config("fa4_only_test"), supports_sdpa = False
    )
    assert impl == "eager"


def test_real_mimo_v2_flash_class_is_not_given_flash_attention_2():
    modeling = pytest.importorskip("transformers.models.mimo_v2_flash.modeling_mimo_v2_flash")
    cls = modeling.MiMoV2FlashForCausalLM
    if "flash_attention_2" in (
        getattr(cls, "_compatible_flash_implementations", None) or ["flash_attention_2"]
    ):
        pytest.skip("this transformers lets mimo_v2_flash use flash_attention_2")
    assert _utils._model_class_supports_flash_attention(cls) is False


def test_flash_is_kept_when_the_switched_to_kernel_is_installed(monkeypatch):
    monkeypatch.setattr(
        _utils, "_flash_implementation_available", lambda name: name == "flash_attention_4"
    )
    assert _utils._model_class_supports_flash_attention(Fa4Only) is True

    class HubOnly(_Base):
        _compatible_flash_implementations = ["kernels-community/some-flash"]

    assert _utils._model_class_supports_flash_attention(HubOnly) is False
