# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A model split over GPUs (device_map) hands each decoder layer its own card's cos / sin tables.

Before, every layer on a second card received the first card's tables and accelerate's hook copied
them across on each call (forward and checkpoint recompute). The rotary cache already builds one
table per card, so `_split_layer_rope_tables` picks that one instead, built once per card per forward.
"""

import os
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest
import torch
import unsloth  # noqa: F401

from real_accelerator import has_real_cuda
from unsloth.models.llama import _split_layer_rope_tables


class _Layer(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        # What unsloth_zoo publishes at load; per_layer_device reads it first.
        self._per_layer_device = device


class _Rotary:
    def __init__(self, rows = 8):
        self.calls = []
        self.rows = rows

    def get_cached(
        self,
        seq_len = None,
        device_index = None,
    ):
        self.calls.append((seq_len, device_index))
        return (torch.full((self.rows, 2), float(device_index)), torch.full((self.rows, 2), -1.0))


def _model(*devices, rows = 8):
    return SimpleNamespace(
        layers = torch.nn.ModuleList([_Layer(torch.device(d)) for d in devices]),
        rotary_emb = _Rotary(rows),
    )


BASE = (torch.zeros(8, 2), torch.zeros(8, 2))
CUDA0 = torch.device("cuda:0")


def test_single_card_keeps_one_table():
    model = _model("cuda:0", "cuda:0", "cuda:0")
    assert _split_layer_rope_tables(model, CUDA0, BASE, 8, 8) is None
    assert model.rotary_emb.calls == []


def test_split_layers_get_their_card_table_once():
    model = _model("cuda:0", "cuda:0", "cuda:1", "cuda:1", "cuda:2")
    out = _split_layer_rope_tables(model, CUDA0, BASE, 8, 8)
    assert out[0] is BASE and out[1] is BASE
    assert out[2] is out[3] and float(out[2][0][0, 0]) == 1.0
    assert float(out[4][0][0, 0]) == 2.0
    # One build per extra card, not one per layer, with the same row count as the first card's.
    assert model.rotary_emb.calls == [(8, 1), (8, 2)]


def test_secondary_card_cache_is_refreshed_before_reading():
    model = _model("cuda:0", "cuda:1")
    ensured = []
    model.rotary_emb._ensure_rope_cache = ensured.append
    _split_layer_rope_tables(model, CUDA0, BASE, 8, 8)
    assert ensured == [1]


def test_short_packed_tables_get_the_same_padding():
    # rope_len < rows (packed positions restart): the first card's tables are padded to rows.
    model = _model("cuda:0", "cuda:1", rows = 4)
    out = _split_layer_rope_tables(model, CUDA0, BASE, 4, 8)
    cos, sin = out[1]
    assert cos.shape == (8, 2) and sin.shape == (8, 2)
    assert torch.equal(cos[4:], torch.zeros(4, 2)) and float(cos[0, 0]) == 1.0


def test_model_on_another_card_than_hidden_states():
    model = _model("cuda:1", "cuda:1")
    out = _split_layer_rope_tables(model, CUDA0, BASE, 8, 8)
    assert out[0] is out[1] and float(out[0][0][0, 0]) == 1.0
    assert model.rotary_emb.calls == [(8, 1)]


def test_cpu_or_block_swap_layers_fall_back():
    # A CPU layer (offload) keeps the old behaviour: accelerate moves the tables as before.
    model = _model("cuda:0", "cpu")
    assert _split_layer_rope_tables(model, CUDA0, BASE, 8, 8) == [BASE, BASE]
    swapped = _model("cuda:0", "cuda:1")
    swapped.layers._unsloth_block_swap = object()
    assert _split_layer_rope_tables(swapped, CUDA0, BASE, 8, 8) is None
    assert _split_layer_rope_tables(_model("cuda:0"), CUDA0, BASE, 8, 8) is None
    assert model.rotary_emb.calls == [] and swapped.rotary_emb.calls == []


SCRIPT = textwrap.dedent(
    """
    import json, sys
    import unsloth
    from unsloth import FastLanguageModel
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode

    devmap = {"model.embed_tokens": 0, "model.rotary_emb": 0, "model.layers.0": 0,
              "model.layers.1": 1, "model.norm": 1, "lm_head": 1}
    model, tok = FastLanguageModel.from_pretrained(
        "trl-internal-testing/tiny-Qwen3ForCausalLM", max_seq_length = 256, load_in_4bit = False,
        device_map = devmap,
    )
    model = FastLanguageModel.get_peft_model(
        model, r = 4, lora_alpha = 4, target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
        use_gradient_checkpointing = "unsloth", random_state = 3407,
    )
    rope = model.base_model.model.model.rotary_emb

    class Count(TorchDispatchMode):
        n = 0
        def __torch_dispatch__(self, func, types, args = (), kwargs = None):
            kwargs = kwargs or {}
            if str(func).startswith("aten._to_copy") and isinstance(args[0], torch.Tensor):
                dst = kwargs.get("device")
                if dst is not None and torch.device(dst) != args[0].device and args[0].dim() == 2 \\
                        and args[0].shape[0] >= 1024:
                    Count.n += 1
            return func(*args, **kwargs)

    ids = torch.randint(100, 2000, (1, 24), device = "cuda:0")
    batch = {"input_ids": ids, "labels": ids, "attention_mask": torch.ones_like(ids)}
    model.train()
    model(**batch).loss.backward()
    with Count():
        out = model(**batch)
        out.loss.backward()
    cos0, cos1 = rope.multi_gpu_cos_cached[0], rope.multi_gpu_cos_cached[1]
    same = cos1 is not None and torch.equal(cos0[: cos1.shape[0]].cpu(), cos1.cpu())
    print("RESULT " + json.dumps({"rope_copies": Count.n, "same_table": bool(same)}))
    """
)


GROW_SCRIPT = textwrap.dedent(
    """
    import json, sys
    import unsloth
    from unsloth import FastLanguageModel
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = sys.argv[1]
    repo = "trl-internal-testing/tiny-Qwen3ForCausalLM"
    AutoTokenizer.from_pretrained(repo).save_pretrained(out_dir)
    base = AutoModelForCausalLM.from_pretrained(repo)
    # Past the 32768-row starting cache, so explicit positions can need more rows later.
    base.config.max_position_embeddings = 131072
    base.save_pretrained(out_dir)
    devmap = {"model.embed_tokens": 0, "model.rotary_emb": 0, "model.layers.0": 0,
              "model.layers.1": 1, "model.norm": 1, "lm_head": 1}
    model, tok = FastLanguageModel.from_pretrained(
        out_dir, max_seq_length = 256, load_in_4bit = False, device_map = devmap,
    )
    from unsloth.models import llama
    rope = model.model.rotary_emb
    ids = torch.randint(100, 2000, (1, 24), device = "cuda:0")
    model.eval()
    with torch.no_grad():
        model(input_ids = ids)  # short forward without positions: 32768 rows on both cards
        far = torch.arange(40000, 40024, device = "cuda:0")[None]
        got = model(input_ids = ids, position_ids = far).logits.float().cpu()
        rows = [rope.multi_gpu_cos_cached[i].shape[0] for i in (0, 1)]
        original = llama._split_layer_rope_tables
        llama._split_layer_rope_tables = lambda *a, **k: None  # every layer reads card 0's table
        ref = model(input_ids = ids, position_ids = far).logits.float().cpu()
        llama._split_layer_rope_tables = original
    print("RESULT " + json.dumps({"rows": rows, "same": bool(torch.equal(got, ref))}))
    """
)


# has_real_cuda(): the zoo CUDA spoof patches torch.cuda.is_available() True on CPU runners.
@pytest.mark.skipif(not has_real_cuda(), reason = "needs CUDA")
def test_split_model_copies_no_rope_table(tmp_path):
    count = subprocess.run(
        [sys.executable, "-c", "import torch; print(torch.cuda.device_count())"],
        capture_output = True,
        text = True,
    )
    if int((count.stdout.strip() or "0").splitlines()[-1]) < 2:
        pytest.skip("needs 2 CUDA GPUs")
    env = {**os.environ, "UNSLOTH_COMPILE_LOCATION": str(tmp_path / "cache")}
    proc = subprocess.run(
        [sys.executable, "-c", SCRIPT], capture_output = True, text = True, env = env, timeout = 900
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    line = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")][-1]
    import json

    res = json.loads(line[len("RESULT ") :])
    assert res["rope_copies"] == 0
    assert res["same_table"]


@pytest.mark.skipif(not has_real_cuda(), reason = "needs CUDA")
def test_split_model_far_positions_after_a_short_forward(tmp_path):
    count = subprocess.run(
        [sys.executable, "-c", "import torch; print(torch.cuda.device_count())"],
        capture_output = True,
        text = True,
    )
    if int((count.stdout.strip() or "0").splitlines()[-1]) < 2:
        pytest.skip("needs 2 CUDA GPUs")
    env = {**os.environ, "UNSLOTH_COMPILE_LOCATION": str(tmp_path / "cache")}
    proc = subprocess.run(
        [sys.executable, "-c", GROW_SCRIPT, str(tmp_path / "model")],
        capture_output = True,
        text = True,
        env = env,
        timeout = 900,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    import json

    line = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")][-1]
    res = json.loads(line[len("RESULT ") :])
    # The second card's table covers the rows the far positions index, and matches card 0's output.
    assert res["rows"][1] >= 40024 and res["rows"][1] == res["rows"][0]
    assert res["same"]
