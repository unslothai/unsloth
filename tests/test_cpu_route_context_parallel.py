# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""CPU backend: a trainer asking for CP gets it or an error before __init__, never plain DDP.
Fresh interpreters since the backend is chosen at import; 2-rank cells are gloo on a tiny Qwen3."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_REPO = "trl-internal-testing/tiny-Qwen3ForCausalLM"


def _env(extra = None):
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("UNSLOTH_ALLOW_CPU", "UNSLOTH_FORCE_CPU", "UNSLOTH_DISABLE_CPU_FALLBACK")
        and not k.startswith("UNSLOTH_CP_")
        and k != "UNSLOTH_PARALLEL"
    }
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["UNSLOTH_FORCE_CPU"] = "1"
    env["OMP_NUM_THREADS"] = "2"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(_ROOT), env.get("PYTHONPATH")]))
    env.update(extra or {})
    return env


def _run(
    code,
    extra = None,
    timeout = 900,
):
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output = True,
        text = True,
        env = _env(extra),
        timeout = timeout,
        cwd = tempfile.gettempdir(),
    )


def _result(out):
    for line in out.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    pytest.fail(f"no RESULT line\nstdout:\n{out.stdout[-3000:]}\nstderr:\n{out.stderr[-5000:]}")


_ROUTE = "import unsloth\nassert unsloth.DEVICE_TYPE == 'cpu', unsloth.DEVICE_TYPE\n"


@pytest.mark.parametrize(
    "statement",
    [
        "from unsloth.utils import attention_dispatch",
        "import unsloth.context_parallel",
        "import unsloth.utils.packing",
        "from unsloth.utils import ulysses_hybrid",
    ],
)
def test_cpu_route_imports_cp_and_attention_modules(statement):
    code = (
        _ROUTE
        + statement
        + (
            "\nfrom unsloth.utils import attention_dispatch as ad\n"
            "from unsloth import context_parallel as cp\n"
            "assert ad.get_cp_manager is cp.get_cp_manager and cp.get_cp_manager() is None\n"
            "print('RESULT ' + __import__('json').dumps({'ok': True}))\n"
        )
    )
    out = _run(code)
    assert out.returncode == 0, out.stderr[-5000:]
    assert _result(out)["ok"]


def test_cpu_route_plain_trainer_skips_gpu_stack():
    # No CP asked for: the gate must not load the GPU model stack.
    code = _ROUTE + (
        "import sys, json, trl\n"
        "from unsloth import FastLanguageModel\n"
        f"model, tok = FastLanguageModel.from_pretrained({_REPO!r}, max_seq_length = 32, load_in_4bit = False)\n"
        "from datasets import Dataset\n"
        "ds = Dataset.from_dict({'text': ['hello world'] * 4})\n"
        "cfg = trl.SFTConfig(output_dir = 'o_plain', max_steps = 1, report_to = 'none', dataset_num_proc = 1)\n"
        "trainer = trl.SFTTrainer(model = model, args = cfg, train_dataset = ds, processing_class = tok)\n"
        "print('RESULT ' + json.dumps({'models': 'unsloth.models' in sys.modules,\n"
        "    'cp': 'unsloth.context_parallel' in sys.modules}))\n"
    )
    out = _run(code)
    assert out.returncode == 0, out.stderr[-5000:]
    assert _result(out) == {"models": False, "cp": False}


@pytest.mark.parametrize("forwarded", [1, None])
def test_cpu_route_forwarded_cp_size_of_one_opts_out(forwarded):
    # SFTTrainer(args = config with cp 2, context_parallel_size = 1 / None): the keyword wins, so no CP and no error.
    code = _ROUTE + (
        "import sys, json, trl\n"
        "from unsloth import FastLanguageModel\n"
        f"model, tok = FastLanguageModel.from_pretrained({_REPO!r}, max_seq_length = 32, load_in_4bit = False)\n"
        "from datasets import Dataset\n"
        "ds = Dataset.from_dict({'text': ['hello world'] * 4})\n"
        "cfg = trl.SFTConfig(output_dir = 'o_opt_out', max_steps = 1, report_to = 'none', dataset_num_proc = 1)\n"
        "cfg.context_parallel_size = 2\n"
        "trainer = trl.SFTTrainer(model = model, args = cfg, train_dataset = ds, processing_class = tok,\n"
        f"    context_parallel_size = {forwarded!r})\n"
        "print('RESULT ' + json.dumps({'cp': 'unsloth.context_parallel' in sys.modules,\n"
        "    'manager': getattr(trainer, '_context_parallel_manager', None) is not None}))\n"
    )
    out = _run(code)
    assert out.returncode == 0, out.stderr[-5000:]
    assert _result(out) == {"cp": False, "manager": False}


def test_cpu_route_cp_refused_before_init_when_unavailable():
    # The CP machinery cannot load (simulated): the trainer must raise before the TRL __init__ runs.
    code = _ROUTE + (
        "import sys, json, types, trl\n"
        "sys.modules['unsloth.context_parallel'] = None\n"
        "calls = []\n"
        "base = trl.SFTTrainer.__mro__[1]\n"
        "real = base.__init__\n"
        "base.__init__ = lambda self, *a, **k: calls.append(1)\n"
        "cfg = types.SimpleNamespace(context_parallel_size = 2)\n"
        "try:\n"
        "    trl.SFTTrainer(model = None, args = cfg)\n"
        "    err = None\n"
        "except Exception as e:\n"
        "    err = f'{type(e).__name__}: {e}'\n"
        "print('RESULT ' + json.dumps({'error': err, 'init_ran': bool(calls)}))\n"
    )
    out = _run(code)
    res = _result(out)
    assert res["error"] is not None and "context parallelism" in res["error"], res
    assert res["init_ran"] is False, res


_WORKER = r"""
import json, os, sys
import unsloth
assert unsloth.DEVICE_TYPE == "cpu", unsloth.DEVICE_TYPE
import torch, trl
from datasets import Dataset
from unsloth import FastLanguageModel

torch.manual_seed(0)
model, tok = FastLanguageModel.from_pretrained(REPO, max_seq_length = 32, load_in_4bit = False)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
ds = Dataset.from_dict({"text": ["the quick brown fox jumps over the lazy dog " * 3] * 8})
cfg = trl.SFTConfig(
    output_dir = OUT, max_steps = 2, per_device_train_batch_size = 2, learning_rate = 1e-3,
    report_to = "none", logging_steps = 1, save_strategy = "no", dataset_num_proc = 1,
    max_length = 32, padding_free = False, ddp_find_unused_parameters = False,
)
if MODE == "config":
    cfg.context_parallel_size = 2
    cfg.context_parallel_backend = "ulysses"
trainer = trl.SFTTrainer(model = model, args = cfg, train_dataset = ds, processing_class = tok)
manager = getattr(trainer, "_context_parallel_manager", None)
trainer.train()
losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
print("RESULT " + json.dumps({
    "rank": int(os.environ["RANK"]),
    "manager": None if manager is None else {"size": manager.size, "backend": str(manager.backend)},
    "losses": losses,
}), flush = True)
"""


def _free_port():
    from tests._rendezvous_port import free_port
    return free_port()


@pytest.mark.parametrize("mode", ["config", "env"])
def test_cpu_route_two_rank_cp_engages(mode, tmp_path):
    pytest.importorskip("trl")
    port = _free_port()
    code = f"REPO = {_REPO!r}\nOUT = {str(tmp_path / 'out')!r}\nMODE = {mode!r}\n" + _WORKER
    extra = {"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port), "WORLD_SIZE": "2"}
    if mode == "env":
        extra["UNSLOTH_CP_SIZE"] = "2"
        extra["UNSLOTH_CP_BACKEND"] = "ulysses"
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", code],
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
            text = True,
            env = _env({**extra, "RANK": str(r), "LOCAL_RANK": str(r)}),
            cwd = str(tmp_path),
        )
        for r in range(2)
    ]
    outs = []
    for p in procs:
        try:
            stdout, stderr = p.communicate(timeout = 900)
        except subprocess.TimeoutExpired:
            for q in procs:
                q.kill()
            raise
        outs.append(subprocess.CompletedProcess(p.args, p.returncode, stdout, stderr))
    for out in outs:
        assert out.returncode == 0, out.stderr[-6000:]
    results = [_result(out) for out in outs]
    for res in results:
        assert res["manager"] is not None, f"CP silently fell back to DDP: {res}"
        assert res["manager"]["size"] == 2, res
        assert "ulysses" in res["manager"]["backend"], res
    # CP ranks train one sequence split across them: the same loss on both.
    assert results[0]["losses"] and results[0]["losses"] == pytest.approx(
        results[1]["losses"], rel = 1e-5
    )
