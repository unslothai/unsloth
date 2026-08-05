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

"""The GRPO autocast belongs to a trainer, not to the process.

rl.py decides the precision inside the generated trainer's __init__ and
records it twice: on `args`, which is that trainer's own object, and in
ACCELERATE_MIXED_PRECISION, which every trainer in the process shares.

A program that builds two trainers before running either one therefore has
one env var describing two different answers. Build a float32 trainer for a
T4 first (rl.py writes 'no' so nothing autocasts into float16) and a normal
float16 trainer second (rl.py writes 'fp16'), and the first trainer's
generation loop, which read the env var afresh on every batch, would enter a
float16 autocast it was explicitly kept out of. That is the same overflow to
inf and then NaN that test_float32_no_fp16_autocast.py exists to prevent.

Both halves are executed here rather than described: the rl.py __init__ block
and the _prepare_inputs autocast header are pulled out of the sources as
strings and run against fake args / model / trainer objects, sharing one
dict as the process environment. No GPU, no model download, no trl import.
There is no T4 on the machine this runs on, so the hardware is simulated:
torch.cuda is made to answer "available, no bfloat16", which is all the code
under test ever asks it.
"""

import ast
import re
import sys
import types
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

RL_PY = REPO_ROOT / "unsloth" / "models" / "rl.py"
RL_REPLACEMENTS = REPO_ROOT / "unsloth" / "models" / "rl_replacements.py"
RL_SRC = RL_PY.read_text(encoding = "utf-8")
REPL_SRC = RL_REPLACEMENTS.read_text(encoding = "utf-8")


# ---- the two pieces of source under test ---------------------------------


def _mixed_precision_source() -> str:
    """The `mixed_precision = (...)` literal rl.py compiles into __init__."""
    for node in ast.walk(ast.parse(RL_SRC)):
        if not isinstance(node, ast.Assign):
            continue
        if "mixed_precision" not in [t.id for t in node.targets if isinstance(t, ast.Name)]:
            continue
        try:
            return ast.literal_eval(node.value)
        except ValueError:
            continue
    raise AssertionError("mixed_precision block not found in rl.py")


def _prepare_inputs_snippet() -> str:
    """The `with` header grpo_trainer__prepare_inputs splices into TRL."""
    start = REPL_SRC.index('"with torch.inference_mode(), "')
    end = REPL_SRC.index('",\n', start)
    return ast.literal_eval("(" + REPL_SRC[start : end + 1] + ")")


def _autocast_helper_source() -> str:
    """Module level helpers the header may call, mirrored into the generated
    trainer through RL_PRE_ITEMS. Empty before those helpers existed."""
    parts = [
        ast.get_source_segment(REPL_SRC, node)
        for node in ast.parse(REPL_SRC).body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_unsloth_grpo_autocast")
    ]
    return "\n\n".join(parts)


MP_SRC = _mixed_precision_source()


class _pretend_cuda:
    """torch.cuda answering as a card without bfloat16, or with it."""

    def __init__(self, has_bf16):
        self.has_bf16 = has_bf16

    def __enter__(self):
        self._saved = (torch.cuda.is_available, torch.cuda.is_bf16_supported)
        torch.cuda.is_available = lambda *args, **kwargs: True
        torch.cuda.is_bf16_supported = lambda *args, **kwargs: self.has_bf16
        return self

    def __exit__(self, *exc):
        torch.cuda.is_available, torch.cuda.is_bf16_supported = self._saved


# ---- a trainer, as far as any of this code can tell ----------------------


class _Args:
    """The fields of TrainingArguments that rl.py writes and the header reads.

    transformers < 5 has no `mixed_precision`, and rl.py only assigns it under
    hasattr, so `has_mixed_precision` picks which of the two worlds we are in.
    """

    def __init__(
        self,
        fp16 = False,
        bf16 = False,
        has_mixed_precision = True,
    ):
        self.fp16 = fp16
        self.bf16 = bf16
        if has_mixed_precision:
            self.mixed_precision = "no"


def _build_trainer(
    env,
    model_dtype,
    bf16_supported,
    fp16 = False,
    bf16 = False,
    user_float32 = None,
    has_mixed_precision = True,
):
    """Run rl.py's __init__ block for one trainer against the shared env."""
    args = _Args(fp16 = fp16, bf16 = bf16, has_mixed_precision = has_mixed_precision)
    model = types.SimpleNamespace(
        config = types.SimpleNamespace(dtype = model_dtype, torch_dtype = model_dtype),
        _unsloth_user_float32 = (
            (model_dtype is torch.float32) if user_float32 is None else user_float32
        ),
    )
    env.setdefault("UNSLOTH_FORCE_FLOAT32", "0")
    env.setdefault("UNSLOTH_ENABLE_FULL_FINETUNING", "0")
    env.setdefault("UNSLOTH_MIXED_PRECISION", "float32")

    def _get_dtype(dtype):
        return dtype if isinstance(dtype, torch.dtype) else getattr(torch, str(dtype))

    device_type = types.ModuleType("unsloth_zoo.device_type")
    device_type.device_is_bf16_supported = lambda: bf16_supported
    utils = types.ModuleType("unsloth_zoo.utils")
    utils._get_dtype = _get_dtype
    parent = types.ModuleType("unsloth_zoo")
    parent.__path__ = []
    names = ("unsloth_zoo", "unsloth_zoo.device_type", "unsloth_zoo.utils")
    saved = {k: sys.modules.get(k) for k in names}
    sys.modules["unsloth_zoo"] = parent
    sys.modules["unsloth_zoo.device_type"] = device_type
    sys.modules["unsloth_zoo.utils"] = utils
    scope = {
        "torch": torch,
        "os": types.SimpleNamespace(environ = env),
        "args": args,
        "model": model,
        "print": lambda *a, **k: None,
    }
    try:
        with _pretend_cuda(has_bf16 = bf16_supported):
            exec(MP_SRC, scope)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    assert scope["_bf16_supported"] is device_type.device_is_bf16_supported
    # The trainer, as much of one as the autocast header ever touches.
    return types.SimpleNamespace(args = args)


def _generate(trainer, env, has_bf16):
    """Enter the injected _prepare_inputs header once and report the autocast."""
    scope = {
        "torch": torch,
        "os": types.SimpleNamespace(environ = env),
        "nullcontext": nullcontext,
        "self": trainer,
        "seen": [],
    }
    helpers = _autocast_helper_source()
    if helpers:
        exec(helpers, scope)
    body = (
        "\n    seen.append((torch.is_autocast_enabled('cuda'), "
        "torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled('cuda') else None))\n"
    )
    with _pretend_cuda(has_bf16 = has_bf16):
        exec(_prepare_inputs_snippet() + body, scope)
    return scope["seen"][0]


# ---- the bug -------------------------------------------------------------


@pytest.mark.parametrize("has_mixed_precision", [True, False])
def test_a_later_trainer_cannot_re_enable_this_trainers_autocast(has_mixed_precision):
    """First trainer: float32 on a card without bfloat16, so rl.py writes 'no'.
    Second trainer: an ordinary float16 model, so rl.py writes 'fp16' over it.
    The first trainer has not generated a single batch yet."""
    env = {}
    first = _build_trainer(
        env, torch.float32, bf16_supported = False, has_mixed_precision = has_mixed_precision
    )
    assert env["ACCELERATE_MIXED_PRECISION"] == "no"

    _build_trainer(
        env, torch.float16, bf16_supported = False, has_mixed_precision = has_mixed_precision
    )
    assert env["ACCELERATE_MIXED_PRECISION"] == "fp16", "the second trainer owns the env now"

    enabled, dtype = _generate(first, env, has_bf16 = False)
    assert enabled is False, "float32 trainer was pulled into the other trainer's fp16 autocast"
    assert dtype is None


def test_the_decision_survives_a_trainer_built_after_the_first_batch():
    """Latching on first use is not enough on its own; check the other order
    too, where the first trainer has already generated once."""
    env = {}
    first = _build_trainer(env, torch.float32, bf16_supported = False)
    assert _generate(first, env, has_bf16 = False) == (False, None)

    _build_trainer(env, torch.float16, bf16_supported = False)
    assert _generate(first, env, has_bf16 = False) == (False, None)


def test_two_trainers_in_one_process_each_keep_their_own_answer():
    env = {}
    first = _build_trainer(env, torch.float32, bf16_supported = False)
    second = _build_trainer(env, torch.float16, bf16_supported = False)

    assert _generate(first, env, has_bf16 = False) == (False, None)
    assert _generate(second, env, has_bf16 = False) == (True, torch.float16)


# ---- everything that must NOT change -------------------------------------


def test_a_float16_trainer_alone_still_autocasts():
    env = {}
    trainer = _build_trainer(env, torch.float16, bf16_supported = False)
    assert env["ACCELERATE_MIXED_PRECISION"] == "fp16"
    assert _generate(trainer, env, has_bf16 = False) == (True, torch.float16)


def test_a_bfloat16_trainer_alone_still_autocasts_in_bfloat16():
    env = {}
    trainer = _build_trainer(env, torch.bfloat16, bf16_supported = True)
    assert env["ACCELERATE_MIXED_PRECISION"] == "bf16"
    assert _generate(trainer, env, has_bf16 = True) == (True, torch.bfloat16)


def test_pure_bfloat16_full_finetuning_still_does_not_autocast():
    env = {"UNSLOTH_MIXED_PRECISION": "bfloat16", "UNSLOTH_ENABLE_FULL_FINETUNING": "1"}
    trainer = _build_trainer(env, torch.bfloat16, bf16_supported = True)
    assert env["ACCELERATE_MIXED_PRECISION"] == "no"
    assert _generate(trainer, env, has_bf16 = True) == (False, None)


def test_force_float32_still_autocasts_in_float16():
    """Gemma3 and gpt-oss set 'no' as well, and still want float16 autocast."""
    env = {"UNSLOTH_FORCE_FLOAT32": "1"}
    trainer = _build_trainer(env, torch.float32, bf16_supported = False)
    assert env["ACCELERATE_MIXED_PRECISION"] == "no"
    assert _generate(trainer, env, has_bf16 = False) == (True, torch.float16)


def test_an_upcast_float32_trainer_still_gets_float16_autocast():
    """Only an explicit float32 load suppresses it, not the float32 that full
    finetuning upcasts to by itself (issue #4082)."""
    env = {"UNSLOTH_ENABLE_FULL_FINETUNING": "1"}
    trainer = _build_trainer(env, torch.float32, bf16_supported = False, user_float32 = False)
    assert env["ACCELERATE_MIXED_PRECISION"] == "fp16"
    assert _generate(trainer, env, has_bf16 = False) == (True, torch.float16)


def test_a_trainer_rl_py_never_touched_falls_back_to_the_environment():
    """Nothing on args to read: an object with no fp16 / bf16 / mixed_precision
    must still get the old environment answer rather than silently 'no'."""
    env = {"ACCELERATE_MIXED_PRECISION": "fp16", "UNSLOTH_FORCE_FLOAT32": "0"}
    trainer = types.SimpleNamespace()
    assert _generate(trainer, env, has_bf16 = False) == (True, torch.float16)


def test_only_one_place_reads_the_shared_environment():
    """One fallback read, in the helper that latches. Any other reader, in code
    or in an injected string, would be a way back to the process wide answer."""
    reads = re.findall(r"environ\.get\(\s*['\"]ACCELERATE_MIXED_PRECISION", REPL_SRC)
    assert len(reads) == 1, reads


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
