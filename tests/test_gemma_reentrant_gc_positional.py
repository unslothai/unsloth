# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Gemma 3n / Gemma 4 force reentrant checkpointing by wrapping gradient_checkpointing_enable. TRL's
unwrap_model_for_generation (HF generation in GRPO, e.g. with tools=) re-enables it on exit with
gradient_checkpointing_enable(gradient_checkpointing_kwargs) positionally, so the wrapper must take it positionally."""

import ast
import json
import os
import pathlib
import subprocess
import sys

import pytest

_VISION = pathlib.Path(__file__).resolve().parents[1] / "unsloth" / "models" / "vision.py"


def _load_wrapper(original):
    tree = ast.parse(_VISION.read_text(encoding = "utf-8"))
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_gc_enable_reentrant"
    )
    ns = {"_original_gc_enable": original}
    exec(compile(ast.Module(body = [fn], type_ignores = []), str(_VISION), "exec"), ns)
    return ns["_gc_enable_reentrant"]


def _recorder():
    calls = []

    def original(
        gradient_checkpointing_kwargs = None,
        every_n_layers = 1,
        offload = False,
    ):
        calls.append((gradient_checkpointing_kwargs, every_n_layers, offload))

    return calls, original


@pytest.mark.parametrize(
    "call",
    [
        lambda f, kw: f(kw),  # TRL unwrap_model_for_generation / disable_gradient_checkpointing
        lambda f, kw: f(gradient_checkpointing_kwargs = kw),  # Trainer
        lambda f, kw: f(gradient_checkpointing_kwargs = kw, every_n_layers = 1, offload = False),
        lambda f, kw: f(kw, 1, False),
    ],
)
def test_wrapper_accepts_positional_and_keyword(call):
    calls, original = _recorder()
    wrapper = _load_wrapper(original)
    caller_kwargs = {"use_reentrant": False}
    call(wrapper, caller_kwargs)
    assert calls[-1][0] == {"use_reentrant": True}
    # The caller's dict (often args.gradient_checkpointing_kwargs) is not mutated.
    assert caller_kwargs == {"use_reentrant": False}


def test_wrapper_no_args():
    calls, original = _recorder()
    _load_wrapper(original)()
    assert calls == [({"use_reentrant": True}, 1, False)]


_CHILD = r"""
import json
import unsloth  # noqa: F401
import torch
from accelerate import Accelerator
from transformers import LlamaConfig, LlamaForCausalLM
from unsloth.models.vision import FastBaseModel
from trl.models import unwrap_model_for_generation

config = LlamaConfig(
    vocab_size = 128, hidden_size = 64, intermediate_size = 128, num_hidden_layers = 2,
    num_attention_heads = 4, num_key_value_heads = 2, max_position_embeddings = 64,
)
model = LlamaForCausalLM(config).cuda()
# post_patch_model keys the Gemma reentrant wrapper on config.model_type.
model.config.model_type = "gemma4"
model = FastBaseModel.post_patch_model(model, use_gradient_checkpointing = "unsloth", trust_remote_code = True)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": False})
was = bool(model.is_gradient_checkpointing)
with unwrap_model_for_generation(model, Accelerator(), gradient_checkpointing_kwargs = {"use_reentrant": False}):
    pass
funcs = [
    m._gradient_checkpointing_func for m in model.modules()
    if getattr(m, "_gradient_checkpointing_func", None) is not None
]
print("@@@" + json.dumps({
    "was_gc": was,
    "is_gc": bool(model.is_gradient_checkpointing),
    "n_funcs": len(funcs),
    "use_reentrant": sorted({str(getattr(f, "keywords", {}).get("use_reentrant")) for f in funcs}),
}))
"""


def test_trl_unwrap_reenables_gemma4(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("importing unsloth needs a GPU")
    pytest.importorskip("trl")
    env = dict(os.environ)
    repo_root = str(_VISION.parents[2])
    env["PYTHONPATH"] = os.pathsep.join(
        [repo_root] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    env.setdefault("UNSLOTH_COMPILE_LOCATION", str(tmp_path / "unsloth_compiled_cache"))
    result = subprocess.run(
        [sys.executable, "-c", _CHILD],
        cwd = str(tmp_path),
        env = env,
        capture_output = True,
        text = True,
        timeout = 900,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    out = json.loads(next(l for l in result.stdout.splitlines() if l.startswith("@@@"))[3:])
    assert out["was_gc"] and out["is_gc"], out
    assert out["n_funcs"] > 0, out
    # The wrapper forces reentrant: the caller's use_reentrant=False must not reach the layers.
    assert "False" not in out["use_reentrant"], out
