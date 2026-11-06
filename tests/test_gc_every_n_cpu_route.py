# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""every_n_layers on a CPU-only host, where `import unsloth` loads _cpu_init.py instead of models/_utils.py."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]

_PROBE = r"""
import json, sys
import unsloth
import torch
from transformers import PreTrainedModel, Qwen3Config, Qwen3ForCausalLM

def model():
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size = 64, hidden_size = 16, intermediate_size = 32, num_hidden_layers = 4,
                      num_attention_heads = 2, num_key_value_heads = 1, head_dim = 8, max_position_embeddings = 64)
    m = Qwen3ForCausalLM(cfg).train()
    m.config.use_cache = False
    return m

def layers(m):
    return [x for x in m.modules() if type(x).__name__ == "Qwen3DecoderLayer"]

def run(m):
    calls = [0] * len(layers(m))
    for i, layer in enumerate(layers(m)):
        f = layer.__dict__.get("_gradient_checkpointing_func")
        if f is not None:
            def w(*a, _f = f, _i = i, **k):
                calls[_i] += 1
                return _f(*a, **k)
            layer._gradient_checkpointing_func = w
    ids = torch.randint(0, 64, (2, 8))
    m(input_ids = ids, labels = ids).loss.backward()
    return calls

out = {"device_type": unsloth.DEVICE_TYPE,
       "models_loaded": sorted(n for n in sys.modules if n == "unsloth.models" or n.startswith("unsloth.models.")),
       "patched": bool(getattr(PreTrainedModel.gradient_checkpointing_enable, "_unsloth_gc_kwargs", False))}
res = {}
# dict form (TrainingArguments on transformers < 5.17, PEFT prepare_model_for_kbit_training, direct calls)
m = model()
try:
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": False, "every_n_layers": 2})
    res["dict"] = run(m)
except Exception as e:
    res["dict"] = repr(e)[:200]
# keyword form, then every flag rewritten to True (what for_training / the trainer do): skipped layers stay out
m = model()
m.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": False}, every_n_layers = 2)
for x in m.modules():
    if hasattr(x, "gradient_checkpointing"):
        x.gradient_checkpointing = True
res["flag_rewrite"] = run(m)
# plain enable stays every layer
m = model()
m.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": False})
res["plain"] = run(m)
out["calls"] = res
print("PROBE " + json.dumps(out))
"""


def _zoo_has_cpu_backend():
    import importlib.util
    spec = importlib.util.find_spec("unsloth_zoo")
    return bool(spec and spec.submodule_search_locations) and any(
        (Path(p) / "cpu" / "loader.py").is_file() for p in spec.submodule_search_locations
    )


@pytest.mark.skipif(not _zoo_has_cpu_backend(), reason = "unsloth_zoo without the CPU backend")
def test_every_n_layers_on_the_cpu_route():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    for name in ("xpu", "npu"):
        try:
            if getattr(torch, name).is_available():
                pytest.skip(f"{name} host: hiding CUDA leaves that accelerator selected")
        except Exception:
            pass
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("UNSLOTH_ALLOW_CPU", "UNSLOTH_FORCE_CPU", "UNSLOTH_DISABLE_CPU_FALLBACK")
    }
    env.update({"CUDA_VISIBLE_DEVICES": "", "UNSLOTH_FORCE_GPU_PATH": "1"})
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(_ROOT), env.get("PYTHONPATH")]))
    out = subprocess.run(
        [sys.executable, "-c", _PROBE], capture_output = True, text = True, env = env, timeout = 1200
    )
    probe = next(
        (json.loads(x[6:]) for x in out.stdout.splitlines() if x.startswith("PROBE ")), None
    )
    if probe is None and "ROCm" in out.stderr:
        pytest.skip("ROCm runtime host without a ROCm torch keeps its install hint")
    assert probe is not None, out.stderr[-3000:]
    assert probe["device_type"] == "cpu"
    assert probe["models_loaded"] == [], "the CPU route must not import the GPU model stack"
    assert probe["patched"]
    calls = probe["calls"]
    assert calls["plain"] == [1, 1, 1, 1]
    # transformers' rule: layers 0, N, 2N... stay checkpointed
    assert calls["dict"] == [1, 0, 1, 0], calls["dict"]
    assert calls["flag_rewrite"] == [1, 0, 1, 0], calls["flag_rewrite"]
