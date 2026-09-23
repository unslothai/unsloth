# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Remote modeling code written for transformers 4.x reads ``config.rope_scaling is None`` as plain RoPE.

transformers 5 made ``rope_scaling`` an alias of ``rope_parameters``, which is a dict even for plain
RoPE, so inclusionAI/Ling-2.6-flash's attention read scaling keys it never has. Remote configs with
plain RoPE read ``None`` again; native configs and real scaling dicts are unchanged.
"""

import pytest
import torch
import transformers

import unsloth  # noqa: F401
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

# On 4.x the None rope_scaling is native and the shim is a no-op.
only_v5 = pytest.mark.skipif(
    int(transformers.__version__.split(".")[0]) < 5, reason = "transformers 4.x needs no shim"
)


@only_v5
def test_the_shared_rope_table_is_not_given_a_default():
    """transformers 5.17's `_init_weights` spreads ROPE_INIT_FUNCTIONS over a rotary's own default,
    so a global "default" entry would replace every native model's own. Remote modules get theirs
    from their own namespace instead."""
    assert "default" not in ROPE_INIT_FUNCTIONS


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


@only_v5
def test_a_native_default_rope_is_not_replaced_on_load(tmp_path):
    """ERNIE-4.5-VL pre-rotates its default inv_freq; loading must keep the model's own."""
    ernie = pytest.importorskip("transformers.models.ernie4_5_vl_moe.modeling_ernie4_5_vl_moe")
    from transformers import AutoConfig

    try:
        config = AutoConfig.for_model(
            "ernie4_5_vl_moe",
            text_config = dict(
                hidden_size = 64,
                intermediate_size = 64,
                moe_intermediate_size = [32, 32],
                num_hidden_layers = 2,
                num_attention_heads = 4,
                num_key_value_heads = 2,
                moe_num_experts = 4,
                moe_k = 2,
                vocab_size = 128,
                moe_num_shared_experts = 1,
                mlp_layer_types = ["dense", "sparse"],
                rope_parameters = {
                    "rope_type": "default",
                    "rope_theta": 500000.0,
                    "mrope_section": [2, 2, 4],
                },
            ),
            vision_config = dict(depth = 1, hidden_size = 32, intermediate_size = 32, num_heads = 2),
        )
    except Exception as error:  # 5.4's strict text config rejects its own use_bias default
        pytest.skip(f"this transformers cannot build the ERNIE-4.5-VL config: {error}")
    model = ernie.Ernie4_5_VLMoeModel(config)
    model.save_pretrained(tmp_path)
    loaded = ernie.Ernie4_5_VLMoeModel.from_pretrained(tmp_path)
    rotary = loaded.language_model.rotary_emb
    expected, _ = type(rotary).compute_default_rope_parameters(rotary.config)
    torch.testing.assert_close(rotary.inv_freq.float(), expected.float(), rtol = 0, atol = 0)
