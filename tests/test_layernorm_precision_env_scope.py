# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""UNSLOTH_HIGH_PRECISION_LAYERNORM is per load: a leak gave the next family float32 norms
("float != BFloat16"), and a cached recompile must replay the family's own decision."""

import os
import subprocess
import sys
import textwrap

import pytest

ENV = "UNSLOTH_HIGH_PRECISION_LAYERNORM"


@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.delenv(ENV, raising = False)


def test_load_scope_restores_unset_value(clean_env):
    from unsloth.models.loader_utils import _restore_load_scoped_env

    @_restore_load_scoped_env
    def load():
        os.environ[ENV] = "1"
        return "model"

    assert load() == "model"
    assert ENV not in os.environ


def test_load_scope_restores_user_value_on_error(monkeypatch):
    from unsloth.models.loader_utils import _restore_load_scoped_env

    monkeypatch.setenv(ENV, "1")

    @_restore_load_scoped_env
    def load():
        os.environ[ENV] = "0"
        raise ValueError("boom")

    with pytest.raises(ValueError):
        load()
    assert os.environ[ENV] == "1"


def test_fast_model_from_pretrained_is_load_scoped():
    from unsloth.models.loader import FastModel

    fn, names = FastModel.from_pretrained, []
    while fn is not None:
        names.append(getattr(fn.__code__, "co_qualname", fn.__code__.co_name))
        fn = getattr(fn, "__wrapped__", None)
    assert "_restore_load_scoped_env.<locals>._wrapper" in names, names


def _fake_compiler(detects):
    """Mimic zoo: check norms on a type's first compile only, OR-ing into the env."""
    patched = set()

    def compile_one(model_type, **kwargs):
        if model_type in patched:
            return
        patched.add(model_type)
        high = os.environ.get(ENV, "0") == "1" or model_type in detects
        os.environ[ENV] = "1" if high else "0"

    return compile_one


@pytest.fixture
def compile_fn(monkeypatch, clean_env):
    import unsloth.models._utils as U

    monkeypatch.setattr(U, "_HIGH_PRECISION_LAYERNORM_MODEL_TYPES", set())
    monkeypatch.setattr(U, "_run_temporary_patches", lambda *a, **k: None)
    monkeypatch.setattr(U, "_unsloth_compile_transformers", _fake_compiler({"qwen3_5"}))

    def run(model_types):
        return U.unsloth_compile_transformers(None, "m", list(model_types))

    return run


def test_cached_compile_replays_high_precision(compile_fn):
    compile_fn(["siglip", "qwen3_5", "qwen3_5_text"])
    assert os.environ[ENV] == "1"
    os.environ.pop(ENV)  # what the load scope does after the first load
    compile_fn(["siglip", "qwen3_5", "qwen3_5_text"])  # compiler returns early now
    assert os.environ[ENV] == "1"


def test_other_family_is_not_upcast(compile_fn):
    compile_fn(["siglip", "qwen3_5"])
    os.environ.pop(ENV)
    compile_fn(["siglip", "qwen3"])
    assert os.environ.get(ENV, "0") == "0"


def test_inherited_value_is_kept_but_not_attributed(compile_fn):
    import unsloth.models._utils as U

    os.environ[ENV] = "1"  # e.g. the loader's Gemma 3 branch
    compile_fn(["siglip", "gemma3"])
    assert os.environ[ENV] == "1"
    assert U._HIGH_PRECISION_LAYERNORM_MODEL_TYPES == set()


_SEQUENCE = textwrap.dedent(
    """
    import os, sys, json, torch
    from unsloth import FastModel
    for name in sys.argv[1:]:
        m, _ = FastModel.from_pretrained(name, max_seq_length = 256, load_in_4bit = False)
        norms = sorted({str(p.dtype) for n, p in m.named_parameters() if "norm" in n.split(".")[-2].lower()})
        torch.manual_seed(0)
        with torch.no_grad():
            m(input_ids = torch.randint(0, 100, (1, 8), device = m.device))
        print("RESULT", json.dumps({"model": name, "norms": norms}), flush = True)
        del m
    """
)

Q35 = "trl-internal-testing/tiny-Qwen3_5ForConditionalGeneration"
Q3 = "trl-internal-testing/tiny-Qwen3ForCausalLM"
G3 = "hf-internal-testing/tiny-random-Gemma3ForCausalLM"
LLAMA = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _run_sequence(tmp_path, *models):
    env = dict(os.environ, UNSLOTH_COMPILE_LOCATION = str(tmp_path / "cache"))
    env.pop(ENV, None)
    proc = subprocess.run(
        [sys.executable, "-c", _SEQUENCE, *models],
        env = env,
        capture_output = True,
        text = True,
        timeout = 1800,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    import json

    return [
        json.loads(l[len("RESULT ") :]) for l in proc.stdout.splitlines() if l.startswith("RESULT ")
    ]


cuda = pytest.mark.skipif(
    not (os.environ.get("UNSLOTH_RUN_GPU_TESTS") == "1"),
    reason = "loads tiny checkpoints on a GPU; set UNSLOTH_RUN_GPU_TESTS=1",
)


@cuda
@pytest.mark.parametrize(
    "first, second, first_fp32, second_fp32",
    [
        (Q35, Q3, True, False),
        (Q35, Q35, True, True),
        (G3, LLAMA, True, False),
        (G3, Q3, True, False),
    ],
)
def test_load_sequence_keeps_each_models_norm_dtype(
    tmp_path, first, second, first_fp32, second_fp32
):
    a, b = _run_sequence(tmp_path, first, second)
    assert ("torch.float32" in a["norms"]) == first_fp32, a
    assert ("torch.float32" in b["norms"]) == second_fp32, b
