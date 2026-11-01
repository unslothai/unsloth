# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""fast_inference under torchrun: the vLLM-or-HF decision is the same on every rank.

The capability gates read each rank's own card. On a mixed host one rank kept vLLM (an
external_launcher engine in the job's group) while the other loaded HF and went on to the
Trainer's collectives, and both waited out the group timeout.
"""

import ast
import json
import os
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LOADER = ROOT / "unsloth" / "models" / "loader.py"
_NAMES = (
    "_agree_fast_inference_across_ranks",
    "_drop_vllm_engine_kwargs",
    "_vllm_needs_newer_gpu",
    "_vllm_engine_device",
)
_ASSIGNS = ("_VLLM_ENGINE_KWARGS", "VLLM_NEEDS_SM80_MODEL_TYPES")


def _extract_source():
    tree = ast.parse(LOADER.read_text(encoding = "utf-8"))
    wanted = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in _NAMES)
        or (
            isinstance(node, ast.Assign)
            and any(getattr(t, "id", None) in _ASSIGNS for t in node.targets)
        )
    ]
    found = {getattr(n, "name", None) for n in wanted}
    missing = [n for n in _NAMES if n not in found]
    assert not missing, f"loader.py lacks {missing}"
    return ast.unparse(ast.Module(body = wanted, type_ignores = []))


def _helpers(monkeypatch, agree):
    zoo_vllm = types.ModuleType("unsloth_zoo.vllm_utils")
    calls = []
    if agree is not None:

        def torchrun_ranks_agree(local_ok):
            calls.append(local_ok)
            return agree(local_ok)

        zoo_vllm.torchrun_ranks_agree = torchrun_ranks_agree
    monkeypatch.setitem(sys.modules, "unsloth_zoo", types.ModuleType("unsloth_zoo"))
    monkeypatch.setitem(sys.modules, "unsloth_zoo.vllm_utils", zoo_vllm)
    namespace = {"DEVICE_TYPE": "cuda"}
    exec(compile(_extract_source(), "loader.py", "exec"), namespace)
    return namespace["_agree_fast_inference_across_ranks"], calls


def test_a_peer_without_vllm_turns_it_off_here(monkeypatch, capsys):
    agree, calls = _helpers(monkeypatch, lambda ok: False)
    kwargs = {"max_num_batched_tokens": 4096, "token": "x"}
    assert agree(True, True, kwargs) is False
    assert kwargs == {"token": "x"}
    assert calls == [True]
    assert "another torchrun rank" in capsys.readouterr().out


def test_a_rank_that_dropped_vllm_still_takes_part(monkeypatch):
    agree, calls = _helpers(monkeypatch, lambda ok: False)
    assert agree(True, False, {}) is False
    assert calls == [False]


def test_every_rank_keeping_vllm_keeps_it(monkeypatch):
    agree, calls = _helpers(monkeypatch, lambda ok: ok)
    kwargs = {"max_num_batched_tokens": 4096}
    assert agree(True, True, kwargs) is True
    assert kwargs == {"max_num_batched_tokens": 4096}


def test_not_requested_never_communicates(monkeypatch):
    agree, calls = _helpers(
        monkeypatch, lambda ok: pytest.fail("no collective without fast_inference")
    )
    assert agree(False, False, {}) is False
    assert calls == []


def test_older_zoo_keeps_the_local_answer(monkeypatch):
    agree, _ = _helpers(monkeypatch, None)
    assert agree(True, True, {}) is True
    assert agree(True, False, {}) is False


def _body(source, start, end):
    i = source.index(start)
    return source[i : source.index(end, i)]


def test_agreement_follows_the_last_per_card_gate():
    src = LOADER.read_text(encoding = "utf-8")
    # Prefix only: the formatter wraps the call's arguments onto their own lines.
    call = "fast_inference = _agree_fast_inference_across_ranks("
    flm = _body(src, "class FastLanguageModel(", "class FastModel(")
    i_req = flm.index("fast_inference_requested = fast_inference")
    i_cc = flm.index("torch.cuda.get_device_capability()[0] < 7")
    i_gb10 = flm.index('"NVIDIA GB10"')
    i_call = flm.index(call)
    assert (
        i_req
        < i_cc
        < i_gb10
        < i_call
        < flm.index("raise_if_fast_inference_under_fsdp2(fast_inference)")
    )
    fm = src[src.index("class FastModel(") :]
    i_req = fm.index("fast_inference_requested = fast_inference")
    i_cc = fm.index("torch.cuda.get_device_capability()[0] < 7")
    i_old = fm.index("_vllm_old_gpu = _vllm_needs_newer_gpu(model_types)")
    i_call = fm.index(call)
    assert i_req < i_cc < i_old < i_call
    # Nothing turns vLLM off per rank between the gate and the agreement.
    assert (
        "fast_inference = False"
        not in fm[fm.index("_drop_vllm_engine_kwargs(kwargs)", i_old) + 1 : i_call]
    )


def _free_port():
    from tests._rendezvous_port import free_port
    return free_port()


def test_mixed_gpu_torchrun_ranks_take_the_same_path(tmp_path):
    """Two gloo ranks, rank 0 on an sm80 card and rank 1 on a T4, loading Gemma 4: one decision."""
    try:
        from unsloth_zoo.vllm_utils import torchrun_ranks_agree  # noqa: F401
    except Exception as error:
        pytest.skip(f"unsloth_zoo without torchrun_ranks_agree: {error}")
    (tmp_path / "loader_helpers.py").write_text(_extract_source())
    script = tmp_path / "worker.py"
    script.write_text(
        textwrap.dedent(
            """
        import json, os, sys
        import torch
        sys.path.insert(0, sys.argv[1])
        rank = int(os.environ["RANK"])
        torch.cuda.get_device_capability = lambda device = None: (8, 0) if rank == 0 else (7, 5)
        import loader_helpers as h
        h.DEVICE_TYPE = "cuda"
        kwargs = {"max_num_batched_tokens": 4096}
        local = h._vllm_needs_newer_gpu(["gemma4"]) is None
        final = h._agree_fast_inference_across_ranks(True, local, kwargs)
        import torch.distributed as dist
        json.dump({"local": local, "final": final, "kwargs": kwargs, "world": dist.get_world_size()},
                  open(os.path.join(sys.argv[1], "r%d.json" % rank), "w"))
        dist.destroy_process_group()
    """
        )
    )
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES = "",
        UNSLOTH_ALLOW_CPU = "1",
        UNSLOTH_DDP_TIMEOUT = "30",
        UNSLOTH_IS_PRESENT = "1",
    )
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes",
        "1",
        "--nproc-per-node",
        "2",
        "--master-addr",
        "127.0.0.1",
        "--master-port",
        str(_free_port()),
        str(script),
        str(tmp_path),
    ]
    r = subprocess.run(cmd, env = env, capture_output = True, text = True, timeout = 180, cwd = tmp_path)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    out = [json.loads((tmp_path / f"r{rank}.json").read_text()) for rank in range(2)]
    assert [o["local"] for o in out] == [True, False]  # the per-card gates disagree
    assert [o["final"] for o in out] == [False, False]
    assert out[0]["kwargs"] == {}  # rank 1's own gate drops them in from_pretrained
    assert all(o["world"] == 2 for o in out)
