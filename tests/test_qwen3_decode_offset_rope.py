# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Qwen3 decode with an explicit position past the rows the prefill built.

The RoPE cache only builds the rows a forward asks for, so a prefill without position_ids on a
model whose max_position_embeddings exceeds the 32768-row starting cache leaves later rows unbuilt.
A decode step at such a position must still read a valid row, matching the decode after a prefill
that built every row. Runs in a subprocess because an out-of-range index is a CUDA device assert.
"""

import os
import subprocess
import sys
import textwrap

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")

MODEL = "trl-internal-testing/tiny-Qwen3ForCausalLM"  # max_position_embeddings 40960

SCRIPT = textwrap.dedent(
    """
    import sys
    import unsloth
    from unsloth import FastLanguageModel
    import torch

    pos = int(sys.argv[2])
    model, _ = FastLanguageModel.from_pretrained(
        sys.argv[1], max_seq_length = 2048, load_in_4bit = False, dtype = torch.float16
    )
    FastLanguageModel.for_inference(model)
    ids = torch.tensor([[1, 2, 3, 4]], device = "cuda")

    def logits(out):
        return out.logits if hasattr(out, "logits") else out[0]

    def decode(prefill_position_ids):
        with torch.no_grad():
            out = model(input_ids = ids, position_ids = prefill_position_ids, use_cache = True)
            out = model(
                input_ids = ids[:, :1],
                past_key_values = out.past_key_values,
                position_ids = torch.tensor([[pos]], device = "cuda"),
                use_cache = True,
            )
        torch.cuda.synchronize()
        return logits(out).float()

    lazy = decode(None)
    full = decode(torch.arange(4, device = "cuda")[None])
    assert torch.isfinite(lazy).all()
    torch.testing.assert_close(lazy, full, rtol = 0, atol = 0)
    print("OK")
    """
)


def test_qwen3_decode_reads_rows_past_the_lazy_prefill_cache(tmp_path):
    script = tmp_path / "decode_offset.py"
    script.write_text(SCRIPT)
    env = dict(os.environ)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("UNSLOTH_COMPILE_LOCATION", str(tmp_path / "unsloth_compiled_cache"))
    result = subprocess.run(
        [sys.executable, str(script), MODEL, "35000"],
        capture_output = True,
        text = True,
        env = env,
        timeout = 900,
    )
    assert result.returncode == 0 and "OK" in result.stdout, (
        result.stdout[-2000:] + result.stderr[-4000:]
    )
