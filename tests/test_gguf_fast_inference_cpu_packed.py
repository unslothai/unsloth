# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Two GGUF requests that used to be dropped silently are now refused up front.

1. `fast_inference = True` with `gguf_file`: vLLM is handed only the repo or folder name (`load_vllm` has no
   `gguf_file` argument), so it loaded the safetensors beside the GGUF or failed on a GGUF-only repo, and the
   training model shared those weights instead of the GGUF's.
2. `gguf_packed` on the CPU backend: the keyword reached transformers' model constructor and failed there with
   `__init__() got an unexpected keyword argument 'gguf_packed'` (also for `gguf_packed = False`) after the GGUF
   was read.
"""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_gguf_load_fixes import write_tiny_gguf_repo

ROOT = Path(__file__).resolve().parents[1]


def _gguf_load():
    spec = importlib.util.spec_from_file_location(
        "_unsloth_gguf_load_g153", ROOT / "unsloth" / "_gguf_load.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- 1. fast_inference with a GGUF


@pytest.mark.parametrize(
    "model_name, gguf_file",
    [
        pytest.param("unsloth/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q4_K_M.gguf", id = "gguf-repo"),
        pytest.param("unsloth/Qwen3-0.6B", "Qwen3-0.6B-Q4_K_M.gguf", id = "safetensors-repo"),
        pytest.param("/models/Qwen3-0.6B-Q4_K_M.GGUF", None, id = "local-gguf-path"),
    ],
)
def test_fast_inference_refuses_a_gguf(model_name, gguf_file):
    module = _gguf_load()
    with pytest.raises(NotImplementedError) as error:
        module.refuse_gguf_fast_inference(model_name, gguf_file, True)
    message = str(error.value)
    assert "fast_inference" in message and "vLLM" in message
    assert (repr(gguf_file) if gguf_file else repr(model_name)) in message
    assert "save_pretrained_merged" in message and "gguf_packed = True" in message


def test_fast_inference_without_gguf_and_gguf_without_fast_inference_pass():
    module = _gguf_load()
    module.refuse_gguf_fast_inference("unsloth/Qwen3-0.6B", None, True)
    module.refuse_gguf_fast_inference("unsloth/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q4_K_M.gguf", False)
    module.refuse_gguf_fast_inference("/models/x.gguf", None, False)


def test_both_gpu_loaders_refuse_before_vllm():
    loader = (ROOT / "unsloth" / "models" / "loader.py").read_text(encoding = "utf-8")
    call = 'refuse_gguf_fast_inference(model_name, kwargs.get("gguf_file"), fast_inference)'
    assert loader.count(call) == 2  # FastLanguageModel, FastModel
    first, second = loader.index(call), loader.rindex(call)
    # Each before its loader's vLLM setup, and after the gguf_packed check (which names its own refusals).
    assert (
        loader.index("check_gguf_packed_request(\n", 0)
        < first
        < loader.index("fast_inference_setup(model_name")
    )
    assert loader.rindex("check_gguf_packed_request(\n") < second
    assert second > loader.index("def from_pretrained(", first)


# ---------------------------------------------------------------- 2. gguf_packed on the CPU backend


_CPU_PROBE = r"""
import json, sys
import unsloth
from unsloth import FastLanguageModel
out = {"device_type": unsloth.DEVICE_TYPE}
for packed in (True, False):
    try:
        model, _ = FastLanguageModel.from_pretrained(
            sys.argv[1], gguf_file = "tiny.gguf", max_seq_length = 64, load_in_4bit = False, gguf_packed = packed,
        )
        out[str(packed)] = {"error": None, "q_proj": str(model.model.layers[0].self_attn.q_proj.weight.dtype)}
    except Exception as e:
        out[str(packed)] = {"error": f"{type(e).__name__}: {e}"}
print("PROBE " + json.dumps(out))
"""


def test_cpu_backend_refuses_gguf_packed_and_drops_false(tmp_path):
    pytest.importorskip("gguf")
    folder = tmp_path / "repo"
    write_tiny_gguf_repo(folder, "qwen3")
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("UNSLOTH_ALLOW_CPU", "UNSLOTH_DISABLE_CPU_FALLBACK")
    }
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "UNSLOTH_FORCE_GPU_PATH": "1",
            "UNSLOTH_FORCE_CPU": "1",
            "HF_HUB_OFFLINE": "1",
        }
    )
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), env.get("PYTHONPATH")]))
    out = subprocess.run(
        [sys.executable, "-c", _CPU_PROBE, str(folder)],
        capture_output = True,
        text = True,
        env = env,
        timeout = 600,
    )
    probe = next(
        (json.loads(l[6:]) for l in out.stdout.splitlines() if l.startswith("PROBE ")), None
    )
    if probe is None:
        pytest.skip("the CPU backend does not import here: " + out.stderr[-500:])
    if probe["device_type"] != "cpu":
        pytest.skip(f"this host does not select the CPU backend ({probe['device_type']})")
    refused = probe["True"]["error"]
    assert refused and refused.startswith(
        "NotImplementedError: Unsloth: `gguf_packed = True`"
    ), probe
    assert "CPU backend" in refused and "Drop `gguf_packed`" in refused and "'tiny.gguf'" in refused
    # gguf_packed = False is the default: the dense CPU load runs as without the keyword.
    assert probe["False"]["error"] is None, probe
