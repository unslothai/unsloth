# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A config that refuses to answer is not a config that has no answer.

transformers 5.x gave heterogeneous models (Gemma 3n, Gemma 4, anything with
`per_layer_config`) a `__getattribute__` that raises
AmbiguousGlobalPerLayerAttributeError on a global read of a per-layer field.
That is not an AttributeError, so a `getattr` default does not cover it, and it
escaped the Flash Attention head-dim probe as a hard failure at model load
(`Gemma4_(E2B)_Reinforcement_Learning_Sudoku_Game` on transformers 5.15.0 /
trl 1.9.2, L4, first cell).

Asserted below: the probe survives a refusal, and it is still right. Turning a
refusal into a default would report no head dim, which reads to
`_get_flash_attention_disable_reason` as "nothing exceeds the limit" on exactly
the models whose layers may differ, so the per-layer values must be read.

The exception type is rebuilt here rather than imported: transformers 4.57.6 is
still supported and has no such class.
"""

from types import SimpleNamespace

import pytest

import unsloth  # noqa: F401

from unsloth.models import _utils


class AmbiguousGlobalPerLayerAttributeError(Exception):
    """Like the transformers 5.x one: an Exception, NOT AttributeError.

    That single fact is the whole bug: inheriting from AttributeError would
    make every test below pass without the fix.
    """


class HeterogeneousConfig:
    """Refuses global reads of `per_layer_attributes`, like the real one."""

    def __init__(
        self,
        per_layer_head_dims,
        model_type = "gemma4",
        **kwargs,
    ):
        self.model_type = model_type
        self.attention_dropout = 0
        self.per_layer_attributes = {"head_dim"}
        self.per_layer_config = tuple(
            SimpleNamespace(head_dim = dim, attention_dropout = 0) for dim in per_layer_head_dims
        )
        self._global_head_dim = per_layer_head_dims[0] if per_layer_head_dims else None
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getattribute__(self, key):
        per_layer = object.__getattribute__(self, "__dict__").get("per_layer_attributes", ())
        if key in per_layer:
            raise AmbiguousGlobalPerLayerAttributeError(
                f"'{key}' is a per-layer attribute and may vary across layers."
            )
        return object.__getattribute__(self, key)


class SequenceView:
    """A `Sequence` over per-layer configs that is not a list or a tuple.

    transformers hands back `_PerLayerConfigView`, a `collections.abc.Sequence`
    subclass, so an `isinstance(..., (list, tuple))` guard would silently skip
    the whole per-layer path.
    """

    def __init__(self, items):
        self._items = list(items)

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        return self._items[index]


def test_a_global_read_that_raises_is_treated_as_no_answer():
    config = HeterogeneousConfig([128, 128])
    assert _utils._config_get(config, "head_dim", "fallback") == "fallback"


def test_a_field_that_is_not_per_layer_still_reads_normally():
    """The guard must not turn every read into the default."""
    config = HeterogeneousConfig([128], num_attention_heads = 8)
    assert _utils._config_get(config, "num_attention_heads", None) == 8
    assert _utils._config_get(config, "model_type", None) == "gemma4"


def test_a_config_that_raises_something_else_entirely_is_also_survived():
    """A config is third-party code. The fix is behavioural, not by type."""

    class Hostile:
        def __getattribute__(self, key):
            raise RuntimeError("no")

    assert _utils._config_get(Hostile(), "head_dim", 7) == 7


def test_the_head_dim_probe_reads_the_per_layer_values():
    """Not merely "does not crash". The numbers have to arrive."""
    config = HeterogeneousConfig([64, 128, 256])
    assert sorted(_utils._get_per_layer_values(config, "head_dim")) == [64, 128, 256]


def test_the_per_layer_view_does_not_have_to_be_a_list_or_tuple():
    config = HeterogeneousConfig([64, 128])
    config.__dict__["per_layer_config"] = SequenceView(config.__dict__["per_layer_config"])
    assert sorted(_utils._get_per_layer_values(config, "head_dim")) == [64, 128]


def test_the_max_head_dim_is_the_largest_layer_not_none():
    """`_get_flash_attention_disable_reason` compares this against Flash
    Attention's 256 ceiling, and `None` means "no reason to disable", so a
    swallowed refusal would leave FA2 on for a 512-wide layer.
    """
    config = HeterogeneousConfig([128, 512, 128])
    assert _utils._get_max_attention_head_dim(config) == 512


def test_an_oversized_heterogeneous_layer_still_disables_flash_attention():
    config = HeterogeneousConfig([128, 512])
    reason = _utils._get_flash_attention_disable_reason(config)
    assert reason is not None and "512" in reason


def test_a_heterogeneous_config_within_the_limit_is_left_alone():
    config = HeterogeneousConfig([128, 128])
    assert _utils._get_flash_attention_disable_reason(config) is None


def test_resolving_the_attention_implementation_no_longer_raises():
    """The end-to-end shape of the reported failure: it died here, at load."""

    class Supports:
        _supports_flash_attn_2 = True
        _supports_flex_attn = False
        _supports_sdpa = True

    config = HeterogeneousConfig([128, 128])
    impl = _utils.resolve_attention_implementation(Supports, config, supports_sdpa = True)
    assert isinstance(impl, str) and impl


def _saved_gemma4_text_config():
    """What transformers 5.15 writes to config.json for a saved Gemma 4.

    `Gemma4TextConfig` synthesizes `per_layer_config` with `head_dim = 512` on
    every full-attention layer, and `to_dict` serializes it as a mapping of
    zero-padded layer index to overrides, not the `_PerLayerConfigView` sequence
    a live config hands back. Verbatim from
    `AutoConfig.from_pretrained("google/gemma-4-E2B-it").save_pretrained(...)`.
    """
    return {
        "model_type": "gemma4_text",
        "attention_dropout": 0,
        "head_dim": 256,
        "per_layer_config": {"04": {"head_dim": 512}, "09": {"head_dim": 512}},
    }


def _to_namespace(value):
    """Studio's `_load_config_for_gpu_estimate`, verbatim: it never builds a
    transformers config, it reads config.json and recursively wraps every dict
    in a SimpleNamespace, so the per-layer mapping arrives as an object whose
    attribute names are the layer indices.
    """
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _to_namespace(item) for key, item in value.items()})
    return value


def test_a_serialized_per_layer_config_is_read_from_a_dict():
    """Same checkpoint, same answer, whichever form of the config arrives.

    The object form reports 512 and disables Flash Attention; before this, the
    dict form reported the global 256 and left FA2 on for the same model.
    """
    config = _saved_gemma4_text_config()
    assert sorted(_utils._get_per_layer_values(config, "head_dim")) == [512, 512]
    assert _utils._get_max_attention_head_dim(config) == 512
    assert _utils._get_flash_attention_disable_reason(config) is not None


def test_a_serialized_per_layer_config_is_read_from_a_namespace():
    """Studio's VRAM estimate reads config.json, so this is the shape it sees."""
    config = _to_namespace({"model_type": "gemma4", "text_config": _saved_gemma4_text_config()})
    assert _utils._get_max_attention_head_dim(config) == 512
    assert _utils._get_flash_attention_disable_reason(config) is not None


@pytest.mark.parametrize("per_layer", [None, (), "not-a-sequence"])
def test_configs_without_per_layer_values_are_unaffected(per_layer):
    """transformers 4.57.6 has no per-layer concept at all, and a homogeneous
    5.x config has an empty one. Neither may change behaviour."""
    config = SimpleNamespace(model_type = "llama", attention_dropout = 0, head_dim = 128)
    if per_layer is not None:
        config.per_layer_config = per_layer
    assert _utils._get_per_layer_values(config, "head_dim") == []
    assert _utils._get_max_attention_head_dim(config) == 128
