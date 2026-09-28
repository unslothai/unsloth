# SPDX-License-Identifier: AGPL-3.0-only
"""A class declaring `_supports_sdpa = False` (MiMo-V2-Flash) resolves to eager, not sdpa."""

from types import SimpleNamespace

import pytest
import unsloth  # noqa: F401

from transformers import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

from unsloth.models import _utils


class _Pre(PreTrainedModel):
    config_class = PretrainedConfig
    _supports_sdpa = True


class DeclaresNoSdpa(_Pre):
    _supports_sdpa = False


class Inherits(_Pre):
    pass


class BarePreTrained(PreTrainedModel):
    config_class = PretrainedConfig


def _config():
    return SimpleNamespace(model_type = "declared_no_sdpa_test", attention_dropout = 0)


@pytest.fixture(autouse = True)
def _no_flash(monkeypatch):
    monkeypatch.setattr(_utils, "HAS_FLASH_ATTENTION", False, raising = False)


def test_declared_false_wins_over_a_source_level_sdpa_guess():
    assert (
        _utils.resolve_attention_implementation(DeclaresNoSdpa, _config(), supports_sdpa = True)
        == "eager"
    )


def test_classes_that_do_not_opt_out_keep_sdpa():
    assert (
        _utils.resolve_attention_implementation(Inherits, _config(), supports_sdpa = True) == "sdpa"
    )
    assert (
        _utils.resolve_attention_implementation(BarePreTrained, _config(), supports_sdpa = True)
        == "sdpa"
    )


def test_explicit_sdpa_request_is_still_honoured():
    impl = _utils.resolve_attention_implementation(
        DeclaresNoSdpa, _config(), requested_attn_implementation = "sdpa", supports_sdpa = True
    )
    assert impl == "sdpa"


def test_real_mimo_v2_flash_resolves_to_eager():
    modeling = pytest.importorskip("transformers.models.mimo_v2_flash.modeling_mimo_v2_flash")
    cls = modeling.MiMoV2FlashForCausalLM
    if cls._supports_sdpa:
        pytest.skip(reason = "this transformers gives mimo_v2_flash an sdpa path, nothing to resolve away")
    config = SimpleNamespace(model_type = "mimo_v2_flash", attention_dropout = 0)
    assert _utils.resolve_attention_implementation(cls, config, supports_sdpa = True) == "eager"
