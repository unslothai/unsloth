# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A plain transformers from_pretrained after Unsloth patched Llama (TRL loading a reward model by name, #1494) must get a valid RoPE."""

import inspect
import subprocess
import sys

import pytest
import torch
from packaging.version import Version
import transformers


def _has_real_gpu():
    try:
        torch.zeros(1).to("cuda")
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.skipif(
        not _has_real_gpu(), reason = "Unsloth's rotary classes build CUDA caches in __init__"
    ),
    pytest.mark.skipif(
        Version(transformers.__version__) < Version("5.0.0"),
        reason = "only transformers v5 blanks non-persistent buffers on load",
    ),
]


# Subprocess: pre_patch swaps Llama classes process-wide, which would leak into later tests.
_CHECK = """
import sys, torch
import unsloth
from transformers import AutoModelForSequenceClassification, LlamaConfig, LlamaForSequenceClassification
from unsloth.models.llama import FastLlamaModel, LlamaRotaryEmbedding

FastLlamaModel.pre_patch()
config = LlamaConfig(
    vocab_size = 128, hidden_size = 64, intermediate_size = 128, num_hidden_layers = 2,
    num_attention_heads = 4, num_key_value_heads = 2, max_position_embeddings = 256,
    rope_theta = 500000.0, num_labels = 1, pad_token_id = 0,
)
torch.manual_seed(0)
LlamaForSequenceClassification(config).save_pretrained(sys.argv[1])
model = AutoModelForSequenceClassification.from_pretrained(
    sys.argv[1], num_labels = 1, output_loading_info = sys.argv[2] == "1"
)
if sys.argv[2] == "1":
    model, info = model
    assert isinstance(info, dict)
rotary = model.model.rotary_emb
assert isinstance(rotary, LlamaRotaryEmbedding), type(rotary)
expected = rotary._unsloth_recompute_inv_freq().float()
torch.testing.assert_close(rotary.inv_freq.detach().cpu().float(), expected)
cos = rotary.multi_gpu_cos_cached[torch.cuda.current_device()].float().cpu()
freqs = torch.outer(torch.arange(cos.shape[0], dtype = torch.float32), expected)
torch.testing.assert_close(cos, torch.cat((freqs, freqs), -1).cos(), atol = 1e-3, rtol = 0)
"""


@pytest.mark.parametrize("output_loading_info", ["0", "1"])
def test_plain_from_pretrained_after_pre_patch_gets_valid_rope(tmp_path, output_loading_info):
    result = subprocess.run(
        [sys.executable, "-c", _CHECK, str(tmp_path / "tiny_rm"), output_loading_info],
        capture_output = True,
        text = True,
        timeout = 600,
    )
    assert result.returncode == 0, result.stderr[-4000:]


def test_wrap_is_idempotent_and_keeps_transformers_source():
    from transformers import PreTrainedModel

    from unsloth.models import loader
    from unsloth.models.modelopt_fp8 import _from_pretrained_own_kwargs

    first = PreTrainedModel.from_pretrained.__func__
    loader._patch_from_pretrained_rope_fix()
    assert PreTrainedModel.from_pretrained.__func__ is first
    assert first._unsloth_rope_fix
    assert not getattr(first.__wrapped__, "_unsloth_rope_fix", False)
    # Names transformers only reads via kwargs.pop; a scan of the wrapper's source would miss them.
    assert {"dtype", "device_map", "subfolder"} <= _from_pretrained_own_kwargs()
    assert "kwargs.pop" in inspect.getsource(inspect.unwrap(PreTrainedModel.from_pretrained))
