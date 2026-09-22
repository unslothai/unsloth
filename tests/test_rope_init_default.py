# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Remote modeling code written for transformers 4.x indexes ``ROPE_INIT_FUNCTIONS["default"]``.

transformers 5 removed that key (plain RoPE moved into each model), so inclusionAI/Ling-2.6-flash
stopped in ``BailingMoeV2_5RotaryEmbedding.__init__`` with ``KeyError: 'default'``. Unsloth puts
the 4.x function back under the key, and only when it is missing.
"""

import types

import pytest
import torch
import transformers

import unsloth  # noqa: F401
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

# On 4.x the key and the None rope_scaling are native, and both shims are no-ops.
only_v5 = pytest.mark.skipif(
    int(transformers.__version__.split(".")[0]) < 5, reason = "transformers 4.x needs no shim"
)


def test_default_key_is_available():
    assert "default" in ROPE_INIT_FUNCTIONS


def test_default_matches_the_4x_formula_with_partial_rotary():
    # Ling-2.6-flash: head_dim 128, partial_rotary_factor 0.5, rope_theta 6e6.
    config = types.SimpleNamespace(
        rope_theta = 6_000_000,
        partial_rotary_factor = 0.5,
        head_dim = 128,
        hidden_size = 4096,
        num_attention_heads = 32,
    )
    inv_freq, scaling = ROPE_INIT_FUNCTIONS["default"](config, "cpu")
    dim = 64
    expected = 1.0 / (6_000_000 ** (torch.arange(0, dim, 2, dtype = torch.float) / dim))
    assert scaling == 1.0
    torch.testing.assert_close(inv_freq, expected)


@only_v5
def test_default_reads_rope_parameters_when_theta_moved():
    config = types.SimpleNamespace(
        rope_parameters = {"rope_type": "default", "rope_theta": 1_000_000.0},
        hidden_size = 256,
        num_attention_heads = 4,
    )
    inv_freq, _ = ROPE_INIT_FUNCTIONS["default"](config, None, seq_len = 10, layer_type = None)
    expected = 1.0 / (1_000_000.0 ** (torch.arange(0, 64, 2, dtype = torch.float) / 64))
    torch.testing.assert_close(inv_freq, expected)


def test_existing_default_is_left_alone(monkeypatch):
    from unsloth.import_fixes import fix_transformers_rope_init_default

    sentinel = object()
    monkeypatch.setitem(ROPE_INIT_FUNCTIONS, "default", sentinel)
    fix_transformers_rope_init_default()
    assert ROPE_INIT_FUNCTIONS["default"] is sentinel


def _config_class(
    module_name,
    rope_parameters,
    legacy = True,
):
    from transformers import PretrainedConfig

    if legacy:  # written for 4.x: takes rope_scaling

        def __init__(
            self,
            rope_scaling = None,
            **kwargs,
        ):
            PretrainedConfig.__init__(self, **kwargs)

    else:  # written for 5.x: takes rope_parameters

        def __init__(
            self,
            rope_parameters = None,
            **kwargs,
        ):
            PretrainedConfig.__init__(self, **kwargs)

    cls = type(
        "RemoteConfig",
        (PretrainedConfig,),
        {"model_type": "remote_plain_rope_test", "__init__": __init__},
    )
    cls.__module__ = module_name
    config = cls()
    config.rope_parameters = rope_parameters
    return config


@only_v5
def test_remote_config_with_plain_rope_reads_rope_scaling_none():
    plain = {"rope_type": "default", "rope_theta": 6_000_000}
    remote = _config_class(
        "transformers_modules.inclusionAI.Ling.configuration_bailing", dict(plain)
    )
    assert remote.rope_scaling is None
    assert remote.rope_parameters == plain  # the real dict is still there

    yarn = {"rope_type": "yarn", "factor": 4.0, "rope_theta": 1e6}
    remote_yarn = _config_class("transformers_modules.x.configuration_x", dict(yarn))
    assert remote_yarn.rope_scaling == yarn

    native = _config_class("transformers.models.llama.configuration_llama", dict(plain))
    assert native.rope_scaling == plain

    # A remote config written for 5.x keeps the alias.
    v5 = _config_class("transformers_modules.x.configuration_x", dict(plain), legacy = False)
    assert v5.rope_scaling == plain
