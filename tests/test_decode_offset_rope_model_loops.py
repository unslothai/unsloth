# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Decode with an explicit position past the rows the prefill built, for the models with their own
decode loop (Gemma2, Cohere, Granite, FalconH1).

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
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")

TOKENIZER = "trl-internal-testing/tiny-Qwen3ForCausalLM"
MODELS = {
    "gemma2": "trl-internal-testing/tiny-Gemma2ForCausalLM",
    "cohere": "trl-internal-testing/tiny-CohereForCausalLM",
    "granite": "hf-internal-testing/tiny-random-GraniteForCausalLM",
    "falcon_h1": None,  # No tiny hub repo: built from a shrunk config below.
}

SCRIPT = textwrap.dedent(
    """
    import sys
    import unsloth
    from unsloth import FastLanguageModel
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    arch, repo, out_dir, tok_repo, pos = sys.argv[1:6]
    pos = int(pos)
    torch.manual_seed(0)
    if repo == "None":
        config = AutoConfig.for_model(
            "falcon_h1", hidden_size = 64, intermediate_size = 128, num_hidden_layers = 2,
            num_attention_heads = 4, num_key_value_heads = 2, head_dim = 16, mamba_d_ssm = 128,
            mamba_n_heads = 8, mamba_d_head = 16, mamba_d_state = 16, mamba_chunk_size = 16,
            vocab_size = 32784,
        )
        model = AutoModelForCausalLM.from_config(config)
    elif arch == "cohere":
        # The hub tiny has head_dim 2, which no xformers prefill kernel takes.
        config = AutoConfig.from_pretrained(repo)
        config.update(dict(
            hidden_size = 128, intermediate_size = 256, num_attention_heads = 4, num_key_value_heads = 2
        ))
        model = AutoModelForCausalLM.from_config(config)
    else:
        model = AutoModelForCausalLM.from_pretrained(repo, dtype = torch.float32)
    # Past the 32768-row starting cache, so the prefill below leaves the later rows unbuilt.
    model.config.max_position_embeddings = 40960
    model.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(tok_repo if repo == "None" else repo).save_pretrained(out_dir)
    del model

    load = dict(max_seq_length = 2048, load_in_4bit = False, dtype = torch.float16)
    if arch in ("cohere", "granite", "falcon_h1"):
        # FastLanguageModel routes these to the generic path; their own decode loop is
        # reached through their patcher.
        from unsloth.models.llama import FastLlamaModel
        from unsloth.models.cohere import FastCohereModel
        from unsloth.models.granite import FastGraniteModel
        from unsloth.models.falcon_h1 import FastFalconH1Model

        patcher = {
            "cohere": FastCohereModel,
            "granite": FastGraniteModel,
            "falcon_h1": FastFalconH1Model,
        }[arch]
        model, _ = FastLlamaModel.from_pretrained(out_dir, model_patcher = patcher, **load)
    else:
        model, _ = FastLanguageModel.from_pretrained(out_dir, **load)
    FastLanguageModel.for_inference(model)
    if arch == "falcon_h1":
        # The decode loop hands Mamba the legacy tuple cache, which it cannot read; Mamba does not
        # touch RoPE, so a zero branch keeps the check on the attention rows.
        for layer in model.model.layers:
            layer.mamba.forward = lambda hidden_states, **kwargs: torch.zeros_like(hidden_states)
    ids = torch.tensor([[1, 2, 3, 4]], device = "cuda")

    def logits(out):
        return out.logits if hasattr(out, "logits") else out[0]

    def decode(prefill_position_ids):
        for layer in model.model.layers:
            if hasattr(layer.self_attn, "paged_attention"):
                del layer.self_attn.paged_attention
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


@needs_cuda
@pytest.mark.parametrize("arch", list(MODELS))
def test_decode_reads_rows_past_the_lazy_prefill_cache(arch, tmp_path):
    script = tmp_path / "decode_offset.py"
    script.write_text(SCRIPT)
    env = dict(os.environ)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("UNSLOTH_COMPILE_LOCATION", str(tmp_path / "unsloth_compiled_cache"))
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            arch,
            str(MODELS[arch]),
            str(tmp_path / arch),
            TOKENIZER,
            "35000",
        ],
        capture_output = True,
        text = True,
        env = env,
        timeout = 900,
    )
    assert result.returncode == 0 and "OK" in result.stdout, (
        result.stdout[-2000:] + result.stderr[-4000:]
    )


class _NoSync:
    def max(self):
        raise AssertionError("host sync on a cache that already holds every position")


class _Rope:
    def __init__(self, floor, max_position_embeddings):
        self._rope_cache_floor = floor
        self.max_position_embeddings = max_position_embeddings


def test_decode_rotary_seq_len_syncs_only_when_rows_can_be_missing():
    from unsloth.models.llama import _decode_rotary_seq_len

    # Every supported position is in the eager cache (max_position_embeddings <= 32768): no sync.
    assert _decode_rotary_seq_len(_Rope(8192, 8192), 9, _NoSync()) == 9
    assert _decode_rotary_seq_len(_Rope(8192, 8192), 9, None) == 9
    # Lazy rows past the floor: one sync, covering the explicit position.
    position_ids = torch.tensor([[35000]])
    assert _decode_rotary_seq_len(_Rope(32768, 40960), 9, position_ids) == 35001
    assert _decode_rotary_seq_len(_Rope(32768, 40960), 9, torch.tensor([[3]])) == 9
