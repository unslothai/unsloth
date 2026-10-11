"""An explicit `rope_scaling=` reaches the model config and vLLM (issue #3083): it used to be
dropped, so fast_inference ran the checkpoint's RoPE past its window and hit a device assert."""

import pytest
import torch

YARN = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}
ROPE_THETA = 1000000.0


def _qwen2_config():
    from transformers import Qwen2Config
    return Qwen2Config(
        hidden_size = 256,
        intermediate_size = 512,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        max_position_embeddings = 32768,
        rope_theta = ROPE_THETA,
    )


def test_user_rope_scaling_keeps_rope_theta():
    # Assigning rope_scaling on transformers v5 replaces all of rope_parameters, so without the
    # carried rope_theta the base falls back to 10000.
    from unsloth.models.llama import _get_rope_theta, _user_rope_scaling

    config = _qwen2_config()
    caller = dict(YARN)
    scaling = _user_rope_scaling(config, caller)
    assert caller == YARN, "the caller's dict must not be mutated"
    # vLLM applies it as an hf_override, outside any patched config setter.
    if hasattr(config, "rope_parameters"):
        assert scaling["rope_theta"] == ROPE_THETA
    config.rope_scaling = scaling
    assert _get_rope_theta(config) == ROPE_THETA
    rope = getattr(config, "rope_parameters", None) or config.rope_scaling
    assert rope["rope_type"] == "yarn" and rope["factor"] == 4.0


def test_user_rope_scaling_fills_rope_type_from_legacy_type():
    from unsloth.models.llama import _user_rope_scaling

    scaling = _user_rope_scaling(_qwen2_config(), {"type": "yarn", "factor": 4.0})
    assert scaling["rope_type"] == "yarn"
    # The checkpoint window, not the widened max_seq_length transformers would fall back to.
    assert scaling["original_max_position_embeddings"] == 32768


def test_user_rope_scaling_inv_freq_matches_transformers_yarn():
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    from unsloth.models.llama import _compute_config_rope_inv_freq, _user_rope_scaling

    config = _qwen2_config()
    config.rope_scaling = _user_rope_scaling(config, YARN)
    rope = getattr(config, "rope_parameters", None) or config.rope_scaling
    inv_freq, attention_scaling = _compute_config_rope_inv_freq(config, rope)
    ref_inv_freq, ref_scaling = ROPE_INIT_FUNCTIONS["yarn"](config, "cpu")
    torch.testing.assert_close(inv_freq.float().cpu(), ref_inv_freq.float(), rtol = 1e-5, atol = 0)
    assert attention_scaling == pytest.approx(ref_scaling) and attention_scaling > 1.0


def test_yarn_inv_freq_under_meta_device_context():
    # transformers builds the model under a meta device context; yarn's helpers then made meta
    # tensors and the rotary fell back to unscaled RoPE with a warning.
    from unsloth.models.llama import _compute_config_rope_inv_freq, _user_rope_scaling

    config = _qwen2_config()
    config.rope_scaling = _user_rope_scaling(config, YARN)
    rope = getattr(config, "rope_parameters", None) or config.rope_scaling
    with torch.device("meta"):
        inv_freq, attention_scaling = _compute_config_rope_inv_freq(config, rope)
    assert inv_freq is not None and inv_freq.device.type == "cpu"
    assert attention_scaling > 1.0
