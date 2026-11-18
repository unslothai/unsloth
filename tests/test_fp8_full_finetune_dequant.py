# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Full finetuning a block-fp8 checkpoint dequantizes it on load (the 16bit weights are what trains)."""

import inspect
import json
import os
import subprocess
import sys

import pytest
import torch
from real_accelerator import has_real_cuda

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fp8_16bit_text_loader_dequant import _write_tiny_block_fp8_llama  # noqa: E402

try:
    from transformers import FineGrainedFP8Config
    _HAS_DEQUANTIZE = "dequantize" in inspect.signature(FineGrainedFP8Config).parameters
except Exception:
    _HAS_DEQUANTIZE = False

# Records the quantization config FastModel hands transformers, then stops before any weight is read.
_CAPTURE = """
import json, sys, torch
sys.path.insert(0, {tests_dir!r})
if not torch.cuda.is_available():
    # No weight is read, so a GPU-less box runs this too, under the CI spoof.
    import _zoo_aggressive_cuda_spoof
    _zoo_aggressive_cuda_spoof.apply()
import unsloth, transformers
from unsloth import FastModel
if {old_transformers}:
    import transformers.quantizers.auto as auto
    base = auto.AUTO_QUANTIZATION_CONFIG_MAPPING["fp8"]
    class OldFineGrainedFP8Config(base):
        def __init__(self, activation_scheme = "dynamic", weight_block_size = (128, 128),
                     modules_to_not_convert = None, **kwargs):
            super().__init__(activation_scheme = activation_scheme, weight_block_size = weight_block_size,
                             modules_to_not_convert = modules_to_not_convert, **kwargs)
    auto.AUTO_QUANTIZATION_CONFIG_MAPPING["fp8"] = OldFineGrainedFP8Config
class Stop(Exception):
    pass
def capture(cls, *args, **kwargs):
    qc = kwargs.get("quantization_config")
    print("CAPTURE " + json.dumps(dict(cls = type(qc).__name__, dequantize = getattr(qc, "dequantize", None))), flush = True)
    raise Stop
transformers.AutoModelForCausalLM.from_pretrained = classmethod(capture)
try:
    FastModel.from_pretrained({path!r}, max_seq_length = 64, dtype = torch.bfloat16, device_map = "cpu", **{kwargs!r})
except Stop:
    pass
except RuntimeError as e:
    print("REFUSED " + json.dumps(str(e)), flush = True)
"""


def _run(
    tmp_path,
    kwargs,
    old_transformers = False,
):
    _write_tiny_block_fp8_llama(str(tmp_path))
    code = _CAPTURE.format(
        tests_dir = os.path.dirname(os.path.abspath(__file__)),
        path = str(tmp_path),
        kwargs = kwargs,
        old_transformers = old_transformers,
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output = True, text = True, timeout = 600)
    for line in out.stdout.splitlines():
        if line.startswith("CAPTURE "):
            return "load", json.loads(line[len("CAPTURE ") :])
        if line.startswith("REFUSED "):
            return "refused", json.loads(line[len("REFUSED ") :])
    raise AssertionError((out.stdout[-3000:], out.stderr[-3000:]))


@pytest.mark.skipif(not _HAS_DEQUANTIZE, reason = "transformers has no fp8 dequantize flag")
@pytest.mark.parametrize(
    "kwargs",
    [
        dict(full_finetuning = True, load_in_4bit = False),
        dict(
            full_finetuning = True
        ),  # load_in_4bit defaults on and is turned off for full finetuning
        dict(full_finetuning = True, load_in_4bit = False, load_in_16bit = True),
    ],
)
def test_full_finetuning_requests_fp8_dequantize(tmp_path, kwargs):
    kind, got = _run(tmp_path, kwargs)
    assert kind == "load" and got == {"cls": "FineGrainedFP8Config", "dequantize": True}, got


@pytest.mark.skipif(not _HAS_DEQUANTIZE, reason = "transformers has no fp8 dequantize flag")
def test_lora_load_keeps_fp8(tmp_path):
    kind, got = _run(tmp_path, dict(load_in_4bit = False))
    assert kind == "load" and got == {"cls": "FineGrainedFP8Config", "dequantize": False}, got


def test_full_finetuning_refused_without_dequantize_flag(tmp_path):
    kind, got = _run(
        tmp_path, dict(full_finetuning = True, load_in_4bit = False), old_transformers = True
    )
    assert kind == "refused" and "full finetuning an FP8 checkpoint" in got, got


# Real 4.57.x trips the fp8 capability check on the spoofed sm80; only the simulated old config is tested.
@pytest.mark.skipif(not _HAS_DEQUANTIZE, reason = "transformers has no fp8 dequantize flag")
def test_lora_load_not_refused_without_dequantize_flag(tmp_path):
    kind, got = _run(tmp_path, dict(load_in_4bit = False), old_transformers = True)
    assert kind == "load", got


_FULL_FT = """
import json, sys, torch
import unsloth
from unsloth import FastModel
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM
src, dst = sys.argv[1], sys.argv[2]
state = load_file(src + "/model.safetensors")
model, _ = FastModel.from_pretrained(src, max_seq_length = 64, dtype = torch.bfloat16,
    full_finetuning = True, load_in_4bit = False, device_map = "cuda:0")
loaded = {k: v.detach().cpu() for k, v in model.state_dict().items() if k.endswith("proj.weight")}
bad = []
for k, w in loaded.items():
    q, s = state[k], state[k.replace(".weight", ".weight_scale_inv")]
    ref = (q.float() * s.repeat_interleave(128, 0).repeat_interleave(128, 1)).to(torch.bfloat16)
    err = ((w.float() - ref.float()).norm() / ref.float().norm()).item()
    if w.dtype != torch.bfloat16 or err > 1e-2:
        bad.append((k, str(w.dtype), err))
fp8 = sum(p.dtype == torch.float8_e4m3fn for p in model.parameters())
model.save_pretrained(dst)
saved_qc = json.load(open(dst + "/config.json")).get("quantization_config")
after = AutoModelForCausalLM.from_pretrained(dst, dtype = torch.bfloat16).state_dict()
changed = [k for k, w in loaded.items() if not torch.equal(after[k].cpu(), w)]
print("RESULT " + json.dumps(dict(n = len(loaded), bad = bad, fp8 = int(fp8), saved_qc = saved_qc, changed = changed)), flush = True)
"""


def _write_tiny_tokenizer(path):
    """FastModel.from_pretrained loads a tokenizer before it returns: a word level one, offline."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, "<pad>": 3, **{f"t{i}": i + 4 for i in range(60)}}
    raw = Tokenizer(models.WordLevel(vocab = vocab, unk_token = "<unk>"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object = raw,
        unk_token = "<unk>",
        bos_token = "<s>",
        eos_token = "</s>",
        pad_token = "<pad>",
    ).save_pretrained(path)


@pytest.mark.skipif(not _HAS_DEQUANTIZE, reason = "transformers has no fp8 dequantize flag")
@pytest.mark.skipif(not has_real_cuda(), reason = "fp8 checkpoints load as fp8 only on CUDA")
def test_full_finetuning_loads_and_saves_dequantized_weights(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "saved"
    src.mkdir()
    _write_tiny_block_fp8_llama(str(src))
    _write_tiny_tokenizer(str(src))
    out = subprocess.run(
        [sys.executable, "-c", _FULL_FT, str(src), str(dst)],
        capture_output = True,
        text = True,
        timeout = 900,
    )
    lines = [l for l in out.stdout.splitlines() if l.startswith("RESULT ")]
    assert lines, (out.stdout[-3000:], out.stderr[-3000:])
    got = json.loads(lines[-1][len("RESULT ") :])
    assert got["n"] == 14 and got["bad"] == [] and got["fp8"] == 0, got
    assert got["saved_qc"] is None and got["changed"] == [], got
