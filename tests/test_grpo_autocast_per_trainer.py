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

"""Autocast precision is per trainer, but ACCELERATE_MIXED_PRECISION is shared by the whole process."""

import ast
import re
import sys
import textwrap
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


class _Args:
    """Fake TrainingArguments fields; mixed_precision exists only on transformers 5 and newer."""

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
    mark_forced_float32 = True,
    forced_float32 = None,
    mark_full_finetuning = True,
    full_finetuning = None,
    env_override = None,
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
    if mark_forced_float32:
        # Stamped by from_pretrained; `forced_float32` differs from the env an earlier load leaves.
        model._unsloth_forced_float32 = (
            (env["UNSLOTH_FORCE_FLOAT32"] == "1") if forced_float32 is None else forced_float32
        )
    env.setdefault("UNSLOTH_ENABLE_FULL_FINETUNING", "0")
    if mark_full_finetuning:
        model._unsloth_full_finetuning = (
            (env["UNSLOTH_ENABLE_FULL_FINETUNING"] == "1")
            if full_finetuning is None
            else full_finetuning
        )
    env.setdefault("UNSLOTH_MIXED_PRECISION", "float32")
    if env_override:
        env.update(env_override)

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
    return types.SimpleNamespace(args = args, model = model)


def _generate(trainer, env, has_bf16):
    """Enter the injected _prepare_inputs header once and report the autocast."""
    scope = {
        "torch": torch,
        "os": types.SimpleNamespace(environ = env),
        "nullcontext": nullcontext,
        "self": trainer,
        "seen": [],
        # The generated trainer gets this from rl.py's preamble, so the header resolves it there too.
        "DEVICE_TYPE_TORCH": "cuda",
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


def test_a_later_load_cannot_take_this_trainers_float16_autocast_away():
    """UNSLOTH_FORCE_FLOAT32 is reset by every load, so the trainer must keep its own stamp."""
    env = {"UNSLOTH_FORCE_FLOAT32": "1"}
    first = _build_trainer(env, torch.float32, bf16_supported = False)
    assert env["ACCELERATE_MIXED_PRECISION"] == "no"

    env["UNSLOTH_FORCE_FLOAT32"] = "0"
    _build_trainer(env, torch.float16, bf16_supported = False)

    assert _generate(first, env, has_bf16 = False) == (True, torch.float16)


def test_a_forced_float32_model_keeps_the_bfloat16_the_trainer_chose():
    """A forced float32 stamp must not pull generation back to float16 when the trainer picked bf16."""
    env = {"UNSLOTH_FORCE_FLOAT32": "1", "UNSLOTH_ENABLE_FULL_FINETUNING": "1"}
    trainer = _build_trainer(env, torch.bfloat16, bf16_supported = True)
    assert env["ACCELERATE_MIXED_PRECISION"] == "bf16"
    assert _generate(trainer, env, has_bf16 = True) == (True, torch.bfloat16)


def test_a_later_load_cannot_take_full_finetunings_bfloat16_away():
    """UNSLOTH_ENABLE_FULL_FINETUNING is rewritten by every load; the trainer must keep its own copy."""
    env = {"UNSLOTH_FORCE_FLOAT32": "1", "UNSLOTH_ENABLE_FULL_FINETUNING": "1"}
    trainer = _build_trainer(
        env,
        torch.bfloat16,
        bf16_supported = True,
        full_finetuning = True,
        env_override = {"UNSLOTH_ENABLE_FULL_FINETUNING": "0"},
    )
    assert env["ACCELERATE_MIXED_PRECISION"] == "bf16"
    assert _generate(trainer, env, has_bf16 = True) == (True, torch.bfloat16)


def test_an_unstamped_model_still_takes_the_environments_finetuning_mode():
    """The fallback, for a model loaded before this stamp existed."""
    env = {"UNSLOTH_FORCE_FLOAT32": "1", "UNSLOTH_ENABLE_FULL_FINETUNING": "1"}
    trainer = _build_trainer(env, torch.bfloat16, bf16_supported = True, mark_full_finetuning = False)
    assert not hasattr(trainer.model, "_unsloth_full_finetuning")
    assert env["ACCELERATE_MIXED_PRECISION"] == "bf16"


def test_the_loaders_stamp_the_full_finetuning_answer_on_the_model():
    for rel in (
        "unsloth/models/loader.py",
        "unsloth/models/vision.py",
        "unsloth/models/sentence_transformer.py",
    ):
        src = (REPO_ROOT / rel).read_text(encoding = "utf-8")
        assert "_mark_full_finetuning(" in src, rel


def test_the_trainer_init_prefers_the_finetuning_stamp_over_the_shared_flag():
    """One fallback read, as with the forced float32 flag."""
    assert "_unsloth_full_finetuning" in MP_SRC
    reads = re.findall(r"environ\.get\(\s*['\"]UNSLOTH_ENABLE_FULL_FINETUNING", MP_SRC)
    assert len(reads) == 1, reads


def _fast_generate_autocast_source() -> str:
    """The autocast unsloth_base_fast_generate builds around _old_generate."""
    src = (REPO_ROOT / "unsloth" / "models" / "vision.py").read_text(encoding = "utf-8")
    start = src.index("    # Mixed precision autocast")
    end = src.index("    # Prepare LoRA\n", start)
    return textwrap.dedent(src[start:end])


class _RecordingTorch:
    """Records autocast arguments instead of applying them, since real autocast is off without CUDA."""

    def __init__(self, calls):
        self._calls = calls

    def __getattr__(self, name):
        return getattr(torch, name)

    def autocast(
        self,
        device_type = None,
        dtype = None,
        enabled = True,
        **kwargs,
    ):
        self._calls.append((enabled, dtype))
        return nullcontext()


def _fast_generate(model, env, dtype):
    """Build that autocast once and report what it would enter."""
    calls = []
    scope = {
        "torch": _RecordingTorch(calls),
        "os": types.SimpleNamespace(environ = env),
        "self": model,
        "dtype": dtype,
        "DEVICE_TYPE_TORCH": "cuda",
    }
    exec(_fast_generate_autocast_source(), scope)
    assert len(calls) == 1, calls
    enabled, fast_dtype = calls[0]
    return enabled, (fast_dtype if enabled else None)


def test_a_forced_load_cannot_pull_generation_into_float16():
    """Generation reads the model's own stamp, so a later forced load cannot move it into float16."""
    model = types.SimpleNamespace(_unsloth_forced_float32 = False)
    env = {"UNSLOTH_FORCE_FLOAT32": "1"}
    assert _fast_generate(model, env, torch.float32) == (False, None)


def test_a_forced_model_still_autocasts_after_a_plain_load():
    """The mirror: the stamp has to keep the float16 as well as refuse it."""
    model = types.SimpleNamespace(_unsloth_forced_float32 = True)
    env = {"UNSLOTH_FORCE_FLOAT32": "0"}
    assert _fast_generate(model, env, torch.float32) == (True, torch.float16)


def test_generation_falls_back_to_the_environment_without_a_stamp():
    model = types.SimpleNamespace()
    assert _fast_generate(model, {"UNSLOTH_FORCE_FLOAT32": "1"}, torch.float32) == (
        True,
        torch.float16,
    )
    assert _fast_generate(model, {"UNSLOTH_FORCE_FLOAT32": "0"}, torch.float32) == (False, None)


@pytest.mark.parametrize(
    "dtype, expected",
    [
        (torch.float16, (True, torch.float16)),
        (torch.bfloat16, (True, torch.bfloat16)),
        (torch.float32, (False, None)),
    ],
)
def test_generation_without_a_forced_family_is_unchanged(dtype, expected):
    model = types.SimpleNamespace(_unsloth_forced_float32 = False)
    assert _fast_generate(model, {}, dtype) == expected


def test_the_loaders_stamp_the_forced_float32_answer_on_the_model():
    """The stamp has to exist for the trainer to read, on both loaders that
    consult the forced list."""
    for rel in ("unsloth/models/loader.py", "unsloth/models/vision.py"):
        src = (REPO_ROOT / rel).read_text(encoding = "utf-8")
        assert "_mark_forced_float32(" in src, rel


def test_only_one_place_reads_the_shared_forced_float32_flag():
    """One fallback read, for models loaded before the stamp existed."""
    reads = re.findall(r"environ\.get\(\s*['\"]UNSLOTH_FORCE_FLOAT32", REPL_SRC)
    assert len(reads) == 1, reads


def test_a_model_without_the_stamp_keeps_the_old_environment_answer():
    env = {"UNSLOTH_FORCE_FLOAT32": "1"}
    trainer = _build_trainer(env, torch.float32, bf16_supported = False, mark_forced_float32 = False)
    assert not hasattr(trainer.model, "_unsloth_forced_float32")
    assert _generate(trainer, env, has_bf16 = False) == (True, torch.float16)


def test_a_forced_float32_load_cannot_force_an_unforced_trainer():
    """An unforced float32 model must store its own False stamp, or a later forced load overrides it."""
    env = {"UNSLOTH_FORCE_FLOAT32": "0"}
    trainer = _build_trainer(env, torch.float32, bf16_supported = False)
    assert trainer.model._unsloth_forced_float32 is False
    assert env["ACCELERATE_MIXED_PRECISION"] == "no"

    env["UNSLOTH_FORCE_FLOAT32"] = "1"

    assert _generate(trainer, env, has_bf16 = False) == (False, None)


@pytest.mark.parametrize(
    "model_dtype, bf16_supported, precision, autocast",
    [
        (torch.bfloat16, True, "bf16", (True, torch.bfloat16)),
        (torch.float16, False, "fp16", (True, torch.float16)),
    ],
)
def test_a_forced_load_earlier_in_the_process_cannot_force_this_trainer(
    model_dtype, bf16_supported, precision, autocast
):
    """Trainer init must read the model stamp, not the env var, which an earlier forced load leaves at 1"""
    env = {"UNSLOTH_FORCE_FLOAT32": "1"}
    trainer = _build_trainer(env, model_dtype, bf16_supported = bf16_supported, forced_float32 = False)
    assert env["ACCELERATE_MIXED_PRECISION"] == precision
    assert _generate(trainer, env, has_bf16 = bf16_supported) == autocast


def test_an_unstamped_model_still_takes_the_environment_answer_in_init():
    """The fallback, for a model loaded before the stamp existed."""
    env = {"UNSLOTH_FORCE_FLOAT32": "1"}
    trainer = _build_trainer(env, torch.float32, bf16_supported = False, mark_forced_float32 = False)
    assert env["ACCELERATE_MIXED_PRECISION"] == "no"
    assert _generate(trainer, env, has_bf16 = False) == (True, torch.float16)


def test_the_trainer_init_prefers_the_stamp_over_the_shared_flag():
    """One fallback read in the __init__ block, as in the generation helper."""
    assert "_unsloth_forced_float32" in MP_SRC
    reads = re.findall(r"environ\.get\(\s*['\"]UNSLOTH_FORCE_FLOAT32", MP_SRC)
    assert len(reads) == 1, reads


def _own_returns(node):
    """Returns belonging to `node` itself, not to a function nested inside it."""
    found = []
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        if isinstance(child, ast.Return) and child.value is not None:
            found.append(child)
        found.extend(_own_returns(child))
    return found


def _exit_scopes(fn):
    """Each (scope, return) that hands a model back, following one level of local helper calls."""
    helpers = {n.name: n for n in ast.walk(fn) if isinstance(n, ast.FunctionDef) and n is not fn}
    own, scopes = [], []
    for ret in _own_returns(fn):
        call = ret.value
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id in helpers
        ):
            helper = helpers[call.func.id]
            scopes.append((helper, _own_returns(helper)))
        else:
            own.append(ret)
    if own:
        scopes.append((fn, own))
    return scopes


def test_every_loader_return_path_stamps_the_forced_float32_answer():
    """Each loader path that returns a model must stamp FORCE_FLOAT32 or the trainer falls back to env."""
    seen = 0
    for rel in ("unsloth/models/loader.py", "unsloth/models/vision.py"):
        tree = ast.parse((REPO_ROOT / rel).read_text(encoding = "utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.FunctionDef) and node.name == "from_pretrained"):
                continue
            for scope, returns in _exit_scopes(node):
                calls = [
                    (c.func.id, c.args[0].id)
                    for c in ast.walk(scope)
                    if isinstance(c, ast.Call)
                    and isinstance(c.func, ast.Name)
                    and c.func.id in ("_mark_requested_float32", "_mark_forced_float32")
                    and c.args
                    and isinstance(c.args[0], ast.Name)
                ]
                requested = {arg for name, arg in calls if name == "_mark_requested_float32"}
                forced = {arg for name, arg in calls if name == "_mark_forced_float32"}
                # `delegated` came from another from_pretrained, already stamped there.
                assert (requested - forced) <= {"delegated"}, (rel, requested, forced)
                for ret in returns:
                    seen += 1
                    statement = ast.unparse(ret)
                    assert "_mark_requested_float32(" in statement, (rel, statement)
    assert seen >= 6, seen


def test_the_diffusion_dispatch_stamps_both_answers():
    """A DiffusionGemma load leaves FastModel through _dispatch_diffusion, which
    predates this stamp: unstamped, a T4 float32 load autocasts to float16."""
    tree = ast.parse((REPO_ROOT / "unsloth/models/loader.py").read_text(encoding = "utf-8"))
    helper = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_dispatch_diffusion"
    )
    body = ast.unparse(helper)
    assert "_mark_forced_float32(model, False)" in body
    assert "_mark_requested_float32(model, user_float32)" in body


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


def test_an_outer_autocast_is_inherited_rather_than_overridden():
    """Omit the dtype key rather than passing a nullcontext sentinel: autocast only accepts a
    torch.dtype."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA: torch.is_autocast_enabled('cuda') is the branch")
    from unsloth.models.rl_replacements import _unsloth_grpo_autocast_kwargs

    class _Trainer:
        pass

    trainer = _Trainer()
    trainer._autocast_enabled = True
    trainer._autocast_dtype = torch.float16
    trainer._autocast_force_float32 = False

    outside = _unsloth_grpo_autocast_kwargs(trainer)
    assert outside == {"enabled": True, "dtype": torch.float16}, outside

    with torch.amp.autocast(device_type = "cuda", dtype = torch.bfloat16):
        inside = _unsloth_grpo_autocast_kwargs(trainer)
        assert "dtype" not in inside, inside
        with torch.amp.autocast(device_type = "cuda", **inside):
            x = torch.randn(4, 4, device = "cuda")
            assert (x @ x).dtype is torch.bfloat16

    # Forcing float32 keeps naming float16 even inside an outer autocast.
    trainer._autocast_force_float32 = True
    with torch.amp.autocast(device_type = "cuda", dtype = torch.bfloat16):
        forced = _unsloth_grpo_autocast_kwargs(trainer)
    assert forced == {"enabled": True, "dtype": torch.float16}, forced


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
