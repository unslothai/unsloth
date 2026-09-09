# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""A float32 model on a GPU without bf16 must not be wrapped in fp16 autocast.

Spark_TTS_(0_5B) loads with `dtype = torch.float32` and sets `fp16 = False,
bf16 = False`. On a T4 it logged [nan] x 7 and then died at inference inside
torch.multinomial, which refuses a distribution containing NaN.

The cause is upstream of the sampler: rl.py reads "neither flag set" as "user
did not choose" and picks the autocast dtype itself, which on a T4 is float16.
float16 carries five exponent bits against float32's eight, so a value the
model was loaded wide enough to hold overflows to inf and then NaN. bf16 GPUs
keep the autocast, since bf16 has float32's exponent range; only float16 is
unsafe and only that case changes.

The block lives in rl.py as a string compiled into the generated trainer, so
these tests pull the literal out and execute it against fake `args` / `model`
objects. No GPU, no network, no trl import.
"""

import ast
import types
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
RL_PY = REPO_ROOT / "unsloth" / "models" / "rl.py"


def _mixed_precision_source() -> str:
    """Extract the `mixed_precision = (...)` string literal from rl.py."""
    src = RL_PY.read_text(encoding = "utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if "mixed_precision" not in targets:
            continue
        if isinstance(node.value, (ast.Constant, ast.JoinedStr, ast.BinOp)):
            pass
        try:
            return ast.literal_eval(node.value)
        except ValueError:
            continue
    raise AssertionError("mixed_precision block not found in rl.py")


MP_SRC = _mixed_precision_source()


def _get_dtype(dtype):
    """Stand-in for unsloth_zoo.utils._get_dtype: accept a dtype or its name."""
    if isinstance(dtype, torch.dtype):
        return dtype
    return getattr(torch, str(dtype).replace("torch.", ""))


class _Args:
    def __init__(
        self,
        fp16 = False,
        bf16 = False,
    ):
        self.fp16 = fp16
        self.bf16 = bf16


def _run(
    model_dtype,
    bf16_supported,
    fp16 = False,
    bf16 = False,
    force_float32 = "0",
    full_finetuning = "0",
    mixed_precision = "float32",
    user_float32 = None,
):
    """Execute the block and report what it decided."""
    config = types.SimpleNamespace(dtype = model_dtype, torch_dtype = model_dtype)
    # from_pretrained records this only for an explicit dtype = torch.float32.
    model = types.SimpleNamespace(
        config = config,
        _unsloth_user_float32 = (
            (model_dtype is torch.float32) if user_float32 is None else user_float32 == "1"
        ),
    )
    args = _Args(fp16 = fp16, bf16 = bf16)
    env = {
        "UNSLOTH_FORCE_FLOAT32": force_float32,
        "UNSLOTH_ENABLE_FULL_FINETUNING": full_finetuning,
        "UNSLOTH_MIXED_PRECISION": mixed_precision,
    }
    fake_os = types.SimpleNamespace(environ = env)

    ns = {
        "torch": torch,
        "os": fake_os,
        "args": args,
        "model": model,
        "print": lambda *a, **k: None,
    }
    # The block imports device_is_bf16_supported and falls back to torch.cuda.is_bf16_supported; make both answer the
    # same way.
    real_cuda = torch.cuda
    torch.cuda = types.SimpleNamespace(is_bf16_supported = lambda: bf16_supported)
    import sys

    # Stub the PARENT too: `from unsloth_zoo.device_type import x` imports unsloth_zoo first, and a raising package
    # __init__ would silently route the block through the torch.cuda fallback instead of the branch under test.
    mod = types.ModuleType("unsloth_zoo.device_type")
    mod.device_is_bf16_supported = lambda: bf16_supported
    utils = types.ModuleType("unsloth_zoo.utils")
    utils._get_dtype = _get_dtype
    parent = types.ModuleType("unsloth_zoo")
    parent.__path__ = []  # make it a package, not a plain module
    parent.device_type = mod
    parent.utils = utils
    names = ("unsloth_zoo", "unsloth_zoo.device_type", "unsloth_zoo.utils")
    saved = {k: sys.modules.get(k) for k in names}
    sys.modules["unsloth_zoo"] = parent
    sys.modules["unsloth_zoo.device_type"] = mod
    sys.modules["unsloth_zoo.utils"] = utils
    try:
        exec(MP_SRC, ns)
    finally:
        torch.cuda = real_cuda
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    # The fallback would mask a broken branch, so prove the stub was used.
    assert ns["_bf16_supported"] is mod.device_is_bf16_supported
    return args, env


# ---- the bug -------------------------------------------------------------


def test_float32_model_on_t4_stays_float32():
    args, env = _run(torch.float32, bf16_supported = False)
    assert args.fp16 is False, "float32 model must not get float16 autocast"
    assert args.bf16 is False
    assert env["ACCELERATE_MIXED_PRECISION"] == "no"


def test_float32_full_finetuning_on_t4_stays_float32():
    # Spark_TTS exactly: full_finetuning = True, both flags off, no bf16.
    args, env = _run(torch.float32, bf16_supported = False, full_finetuning = "1")
    assert (args.fp16, args.bf16) == (False, False)
    assert env["ACCELERATE_MIXED_PRECISION"] == "no"


# ---- everything that must NOT change -------------------------------------


def test_float32_model_on_bf16_gpu_still_autocasts():
    # bf16 shares float32's exponent range, so this stays safe and cheap.
    args, env = _run(torch.float32, bf16_supported = True)
    assert args.bf16 is True and args.fp16 is False
    assert env["ACCELERATE_MIXED_PRECISION"] == "bf16"


def test_float16_model_on_t4_still_gets_fp16_autocast():
    args, env = _run(torch.float16, bf16_supported = False)
    assert args.fp16 is True and args.bf16 is False
    assert env["ACCELERATE_MIXED_PRECISION"] == "fp16"


def test_bfloat16_model_on_bf16_gpu_unchanged():
    args, env = _run(torch.bfloat16, bf16_supported = True)
    assert args.bf16 is True and args.fp16 is False
    assert env["ACCELERATE_MIXED_PRECISION"] == "bf16"


def test_explicit_fp16_on_a_float32_model_is_obeyed():
    # An explicit request is a choice, not a default; leave it alone.
    args, env = _run(torch.float32, bf16_supported = False, fp16 = True)
    assert args.fp16 is True
    assert env["ACCELERATE_MIXED_PRECISION"] == "fp16"


def test_explicit_bf16_on_a_float32_model_is_obeyed():
    args, env = _run(torch.float32, bf16_supported = True, bf16 = True)
    assert args.bf16 is True
    assert env["ACCELERATE_MIXED_PRECISION"] == "bf16"


def test_force_float32_models_take_the_earlier_branch():
    # Gemma3 / gpt-oss on a T4: force_float32 wins before the new branch and already lands on pure float32, so the
    # outcome is identical either way.
    args, env = _run(torch.float32, bf16_supported = False, force_float32 = "1")
    assert (args.fp16, args.bf16) == (False, False)
    assert env["ACCELERATE_MIXED_PRECISION"] == "no"


def test_force_float32_full_finetuning_on_bf16_gpu_keeps_bf16_autocast():
    # The documented fast path: master weights stay float32, autocast is bf16.
    args, env = _run(torch.float32, bf16_supported = True, force_float32 = "1", full_finetuning = "1")
    assert args.bf16 is True and args.fp16 is False
    assert env["ACCELERATE_MIXED_PRECISION"] == "bf16"


def test_bfloat16_mixed_precision_mode_unchanged():
    # UNSLOTH_MIXED_PRECISION = bfloat16 does no autocasting at all.
    args, env = _run(torch.bfloat16, bf16_supported = True, mixed_precision = "bfloat16")
    assert (args.fp16, args.bf16) == (False, False)
    assert env["ACCELERATE_MIXED_PRECISION"] == "no"


def test_upcast_float32_on_a_v100_still_gets_fp16_autocast():
    """The float32 the model was UPCAST to is not a request for float32.

    Full finetuning upcasts trainable weights to float32 by itself, and
    float16 autocast over float32 master weights is the ordinary V100/T4
    mixed-precision recipe (issue #4082). Only an explicit
    `dtype = torch.float32` at load time may suppress it, which is why the
    new branch is gated on the recorded request rather than on the dtype.
    """
    args, env = _run(torch.float32, bf16_supported = False, full_finetuning = "1", user_float32 = "0")
    assert (args.fp16, args.bf16) == (True, False)
    assert env["ACCELERATE_MIXED_PRECISION"] == "fp16"


def test_loaders_record_the_explicit_request():
    """Every public entry point, since only the outermost one sees the
    argument as the caller wrote it."""
    for rel in ("unsloth/models/loader.py", "unsloth/models/vision.py"):
        src = (REPO_ROOT / rel).read_text(encoding = "utf-8")
        assert "_requested_float32(dtype)" in src, rel
        assert "_mark_requested_float32(" in src, rel


def test_the_legacy_language_model_path_records_it_too():
    """llama, mistral, gemma, gemma2, qwen2 and qwen3 LoRA/QLoRA loads go
    through dispatch_model.from_pretrained, which is neither of the two loaders
    that used to record this. Those are most of the notebooks."""
    src = (REPO_ROOT / "unsloth" / "models" / "loader.py").read_text(encoding = "utf-8")
    tree = ast.parse(src)
    cls = next(
        n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "FastLanguageModel"
    )
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "from_pretrained")
    body = ast.unparse(fn)
    assert "_requested_float32(dtype)" in body
    # Every exit, including the two that hand off to FastModel: it would otherwise record the dtype we derived from a
    # 4bit compute dtype.
    returns = [
        ast.unparse(n) for n in ast.walk(fn) if isinstance(n, ast.Return) and n.value is not None
    ]
    assert returns, "expected the loader to return a model"
    for statement in returns:
        assert "_mark_requested_float32(" in statement, statement


def test_the_text_diffusion_path_records_it_too():
    """DiffusionGemma leaves FastModel through _dispatch_diffusion, which returns
    before the stamping at the end of from_pretrained. A `dtype = torch.float32`
    load on a T4 would otherwise reach the trainer unmarked and autocast to
    float16, which is the overflow this whole branch exists to avoid."""
    src = (REPO_ROOT / "unsloth" / "models" / "loader.py").read_text(encoding = "utf-8")
    tree = ast.parse(src)
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_dispatch_diffusion"
    )
    returns = [ast.unparse(n) for n in ast.walk(fn) if isinstance(n, ast.Return)]
    assert returns, "expected the diffusion dispatch to return a model"
    for statement in returns:
        assert "_mark_requested_float32(model, user_float32)" in statement, statement


def test_the_request_is_read_from_the_model_not_the_environment():
    """A process-global would describe whichever model loaded last, so a
    program that loads two before building a trainer would train the first
    with the second's precision."""
    assert "_unsloth_user_float32" in MP_SRC
    assert "UNSLOTH_USER_FLOAT32" not in MP_SRC


def test_a_model_without_the_marker_keeps_the_old_behaviour():
    """Anything the loaders did not touch must not opt into the new branch."""
    args, _ = _run(torch.float32, bf16_supported = False, user_float32 = "0")
    assert (args.fp16, args.bf16) == (True, False)


def test_block_still_compiles():
    compile(MP_SRC, "mixed_precision", "exec")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
