# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""GRPO old / ref scoring under FSDP2 must project with the output head the forward gathered.

Runs the shipped head lookup and padded scoring loop of ``_get_per_token_logps_and_entropies`` on two
gloo ranks after a training step has resharded the model. A head captured before the scoring forward is
still the sharded DTensor ("aten.mm.default got mixed torch.Tensor and DTensor").
"""

from __future__ import annotations

import ast
import collections
import copy
import os
import socket
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
dist = pytest.importorskip("torch.distributed")
pytest.importorskip("torch.distributed.fsdp")

if not dist.is_available() or not dist.is_gloo_available():
    pytest.skip("gloo backend unavailable", allow_module_level = True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _grpo_dispatch_source import load_dispatch_helpers, load_padded_loop_source  # noqa: E402

_SOURCE_PATH = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "rl_replacements.py"

VOCAB, HIDDEN, BATCH, SEQ, LOGITS_TO_KEEP = 17, 8, 2, 9, 3
# A pytree node like transformers' ModelOutput: FSDP2 registers its backward hooks through it.
CausalLMOutput = collections.namedtuple("CausalLMOutput", "logits")


def _head_setup_source():
    """Statements from the output head lookup up to the truncated-support block, in source order."""
    text = _SOURCE_PATH.read_text(encoding = "utf-8")
    (function,) = [
        node
        for node in ast.walk(ast.parse(text))
        if isinstance(node, ast.FunctionDef) and node.name == "_get_per_token_logps_and_entropies"
    ]
    for node in ast.walk(function):
        for field in ("body", "orelse"):
            stmts = getattr(node, field, None)
            if not isinstance(stmts, list):
                continue
            starts = [
                i
                for i, s in enumerate(stmts)
                if isinstance(s, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "output_embeddings" for t in s.targets)
            ]
            if not starts:
                continue
            ends = [
                i
                for i, s in enumerate(stmts)
                if i > starts[0]
                and isinstance(s, ast.Try)
                and "grpo_truncated_support_kwargs" in ast.unparse(s)
            ]
            assert ends, "truncated-support block not found after the head lookup"
            return "\n".join(
                textwrap.dedent(ast.get_source_segment(text, s, padded = True))
                for s in stmts[starts[0] : ends[0]]
            )
    raise AssertionError("output head lookup not found")


def _hidden_log_softmax(
    hidden_states,
    lm_head,
    index,
    chunks = 4,
    logit_scale_multiply = 0.0,
    logit_scale_divide = 0.0,
    logit_softcapping = 0.0,
    temperature = 1.0,
    lm_head_bias = None,
):
    logits = hidden_states.to(lm_head.dtype) @ lm_head.t()
    if lm_head_bias is not None:
        logits = logits + lm_head_bias
    logits = logits.float() / temperature
    picked = torch.gather(logits, -1, index.unsqueeze(-1)).squeeze(-1)
    return picked - torch.logsumexp(logits, dim = -1)


def _raw_log_softmax(
    logits,
    index,
    temperature = 1.0,
    chunks = 4,
):
    logits = logits.float() / temperature
    picked = torch.gather(logits, -1, index.unsqueeze(-1)).squeeze(-1)
    return picked - torch.logsumexp(logits, dim = -1)


class TinyCausalLM(torch.nn.Module):
    """Returns hidden states as ``.logits`` while scoring, like Unsloth's generated forwards."""

    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(VOCAB, HIDDEN)
        self.block = torch.nn.Linear(HIDDEN, HIDDEN)
        self.lm_head = torch.nn.Linear(HIDDEN, VOCAB, bias = True)
        self.return_hidden = False

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
        self,
        input_ids = None,
        logits_to_keep = None,
        **kwargs,
    ):
        hidden = torch.tanh(self.block(self.embed(input_ids)))
        out = hidden if self.return_hidden else self.lm_head(hidden)
        if logits_to_keep is not None:
            out = out[:, -logits_to_keep:, :]
        return CausalLMOutput(logits = out)


def _reference(model, input_ids):
    with torch.no_grad():
        logits = model(input_ids = input_ids).logits.float()
    width = LOGITS_TO_KEEP
    logps = torch.log_softmax(logits, dim = -1)[:, -(width + 1) : -1, :]
    return torch.gather(logps, -1, input_ids[:, -width:].unsqueeze(-1)).squeeze(-1)


def _worker(rank, port, separate_head, setup_source, loop_source, errors):
    try:
        os.environ.update(MASTER_ADDR = "127.0.0.1", MASTER_PORT = str(port))
        dist.init_process_group("gloo", rank = rank, world_size = 2)
        from torch.distributed.fsdp import fully_shard

        torch.manual_seed(0)
        model = TinyCausalLM()
        reference_model = copy.deepcopy(model)
        input_ids = torch.randint(
            0, VOCAB, (BATCH, SEQ), generator = torch.Generator().manual_seed(1)
        )
        if separate_head:
            fully_shard(model.block)
            fully_shard(model.lm_head)
        fully_shard(model)
        if separate_head:
            # Mirrors unsloth/models/fsdp2.py: the root gathers units whose weights are read directly.
            model.register_forward_pre_hook(lambda module, args: model.lm_head.unshard())

        # A training step leaves every unit resharded, as the next scoring pass finds it.
        model(input_ids = input_ids).logits.float().sum().backward()

        model.return_hidden = True
        namespace = {
            "torch": torch,
            "os": os,
            **load_dispatch_helpers(),
            "UNSLOTH_ZOO_HAS_HEAD_BIAS": True,
            "chunked_hidden_states_selective_log_softmax": _hidden_log_softmax,
            "chunked_selective_log_softmax": _raw_log_softmax,
            "device_synchronize": lambda *a, **k: None,
            "DEVICE_TYPE_TORCH": "cpu",
            "_get_inference_mode_context_manager": lambda _model: torch.no_grad(),
            "model": model,
            "unwrapped_model": model,
            "self": SimpleNamespace(_autocast_dtype = torch.float32, _autocast_enabled = False),
            "pixel_values": None,
            "zipped_inputs": [
                (input_ids[i : i + 1], torch.ones(1, SEQ, dtype = torch.long), {})
                for i in range(BATCH)
            ],
            "logits_to_keep": LOGITS_TO_KEEP,
            "max_left_pad": 0,
            "multiplier": 1,
            "logit_scale_multiply": 0.0,
            "logit_scale_divide": 0.0,
            "logit_softcapping": 0.0,
            "temperature": 1.0,
            "all_logprobs_list": [],
            "logprobs": None,
        }
        exec(compile(setup_source, "<head setup>", "exec"), namespace)
        exec(compile(loop_source, "<padded loop>", "exec"), namespace)
        torch.testing.assert_close(
            namespace["logprobs"], _reference(reference_model, input_ids), atol = 1e-5, rtol = 1e-5
        )
    except BaseException as exc:
        errors.put(f"rank {rank}: {type(exc).__name__}: {exc}")
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.parametrize("separate_head", [False, True], ids = ["root_head", "separate_head"])
def test_fsdp2_scoring_reads_the_gathered_output_head(separate_head):
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    errors = ctx.SimpleQueue()
    spawn_error = None
    try:
        mp.start_processes(
            _worker,
            args = (
                _free_port(),
                separate_head,
                _head_setup_source(),
                load_padded_loop_source(),
                errors,
            ),
            nprocs = 2,
            start_method = "spawn",
        )
    except Exception as exc:
        spawn_error = exc
    found = []
    while not errors.empty():
        found.append(errors.get())
    assert not found, "\n".join(found)
    assert spawn_error is None, spawn_error
