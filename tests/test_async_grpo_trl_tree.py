# SPDX-License-Identifier: AGPL-3.0-or-later
# Unsloth - 2x faster, 60% less VRAM LLM training and finetuning
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

"""CPU checks of patch_trl_async_grpo against the installed TRL (1.14+), with a tiny Qwen3.

The backbone stands in for Unsloth's: it keeps sequences apart only by packed_seq_lengths and takes RoPE positions
from position_ids, ignoring the attention implementation TRL sets, as Unsloth's fast forward does.
"""

from __future__ import annotations

import ast
import functools
import importlib
import os
import sys
import types
import warnings

import pytest
import torch

trl = pytest.importorskip("trl")
transformers = pytest.importorskip("transformers")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
RL_SOURCE_PATH = os.path.join(REPO_ROOT, "unsloth", "models", "rl.py")
TINY = "trl-internal-testing/tiny-Qwen3ForCausalLM"
NAMES = (
    "_without_instance_bound_methods",
    "_packed_seq_lengths_from_position_ids",
    "_is_unsloth_fast_backbone",
    "_forward_tree_unrolled",
    "_unsloth_async_grpo_lm_head",
    "_fast_inference_engine",
    "_async_grpo_uses_fast_inference",
    "_dataset_has_images",
    "_start_async_grpo_vllm_server",
    "_check_async_grpo_fast_inference",
    "_text_prepended_position_ids",
    "_packing_honored",
    "_align_vlm_text_config",
    "_hold_peft_requires_grad",
    "_allow_moe_parameter_lora",
    "_check_async_grpo_adapter_servable",
    "_async_grpo_masked_compute_loss",
    "_import_trl_async_grpo_module",
    "patch_trl_async_grpo",
)


class _Logger:
    def info(self, *a, **k):
        pass


class FakeUnslothBackbone(torch.nn.Module):
    # Attention boundaries from packed_seq_lengths only (one sequence without it), positions from position_ids.
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.config = inner.config
        self.calls = []

    def forward(
        self,
        input_ids = None,
        position_ids = None,
        packed_seq_lengths = None,
        **kwargs,
    ):
        self.calls.append(
            {
                "length": input_ids.shape[1],
                "packed_seq_lengths": None
                if packed_seq_lengths is None
                else packed_seq_lengths.tolist(),
                "tree": sorted(k for k in kwargs if k.startswith("tree_")),
            }
        )
        lengths = (
            [input_ids.shape[1]] if packed_seq_lengths is None else packed_seq_lengths.tolist()
        )
        hidden, start = [], 0
        for length in lengths:
            out = self.inner(
                input_ids = input_ids[:, start : start + length],
                position_ids = position_ids[:, start : start + length],
                use_cache = False,
            )
            hidden.append(out.last_hidden_state)
            start += length
        return types.SimpleNamespace(last_hidden_state = torch.cat(hidden, 1))


def _load():
    src = open(RL_SOURCE_PATH, encoding = "utf-8").read()
    ns = {
        "copy": __import__("copy"),
        "inspect": __import__("inspect"),
        "textwrap": __import__("textwrap"),
        "types": types,
        "torch": torch,
        "functools": functools,
        "importlib": importlib,
        "warnings": warnings,
        "weakref": __import__("weakref"),
        "logger": _Logger(),
        "Version": __import__("packaging.version", fromlist = ["Version"]).Version,
        "importlib_version": __import__("importlib.metadata", fromlist = ["version"]).version,
    }
    for node in ast.parse(src).body:
        is_constant = isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id.startswith("_ASYNC_GRPO_") for t in node.targets
        )
        if is_constant or (isinstance(node, ast.FunctionDef) and node.name in NAMES):
            exec(compile(ast.Module([node], []), RL_SOURCE_PATH, "exec"), ns)
    ns["_is_unsloth_fast_backbone"] = lambda b: isinstance(b, FakeUnslothBackbone)
    return ns


def _async_module():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name in (
            "trl.trainer.async_grpo_trainer",
            "trl.experimental.async_grpo.async_grpo_trainer",
        ):
            try:
                module = importlib.import_module(name)
            except ImportError:
                continue
            if hasattr(module, "AsyncGRPOTrainer"):
                return module
    pytest.skip("this TRL has no AsyncGRPOTrainer")


@pytest.fixture
def patched(monkeypatch):
    module = _async_module()
    if getattr(module.add_fused_lm_head, "_unsloth_async_grpo_patched", False):
        # tests/conftest.py imports unsloth, whose PatchFastRL already patched TRL: test that patch.
        rl = sys.modules.get("unsloth.models.rl")
        if rl is None or not os.path.samefile(rl.__file__, RL_SOURCE_PATH):
            pytest.skip("TRL was patched by an unsloth outside this tree")
        monkeypatch.setattr(
            rl, "_is_unsloth_fast_backbone", lambda b: isinstance(b, FakeUnslothBackbone)
        )
        return vars(rl), module
    trainer = module.AsyncGRPOTrainer
    for name in (
        "create_model_from_path",
        "add_fused_lm_head",
        "patch_chunked_lm_head",
        "AsyncRolloutWorker",
        "validate_lora_for_vllm_sync",
    ):
        if hasattr(module, name):
            monkeypatch.setattr(module, name, getattr(module, name))
    for name in ("__init__", "train"):
        if name in vars(trainer):
            monkeypatch.setattr(trainer, name, vars(trainer)[name])
    ns = _load()
    ns["patch_trl_async_grpo"]()
    return ns, module


def _tiny_model():
    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(TINY, dtype = torch.float32)
    except Exception as error:
        pytest.skip(f"{TINY} not available: {error}")
    model.eval()
    model.config._attn_implementation = "sdpa"
    return model


def _reference(model, samples, token_order):
    logps, entropies = [], []
    with torch.no_grad():
        per_sample = []
        for sample in samples:
            logits = model(input_ids = sample["input_ids"][None]).logits[0].float()
            per_sample.append(torch.log_softmax(logits, -1))
    for i, p in token_order:
        logp = per_sample[i][p - 1]
        logps.append(logp[samples[i]["input_ids"][p]])
        entropies.append(-(logp.exp() * logp).sum())
    return torch.stack(logps), torch.stack(entropies)


def _head(model, module):
    # The Unsloth stand-in backbone under TRL's fused head, as AsyncGRPOTrainer installs it.
    fake = FakeUnslothBackbone(model.model)
    lm_head = model.lm_head
    model.model = fake
    # transformers 5 looks the head up through the (now non-PreTrainedModel) backbone.
    model.get_output_embeddings = lambda: lm_head
    module.add_fused_lm_head(model, temperature = 1.0, outputs = ("log_probs", "entropy"))
    return fake


def test_patch_installs_and_is_idempotent(patched):
    ns, module = patched
    assert module.add_fused_lm_head._unsloth_async_grpo_patched is True
    init = module.AsyncGRPOTrainer.__init__
    create = module.create_model_from_path
    ns["patch_trl_async_grpo"]()
    assert module.AsyncGRPOTrainer.__init__ is init
    assert module.create_model_from_path is create
    if module.__name__ == "trl.trainer.async_grpo_trainer":
        # The class users instantiate from `trl`, and the deprecated shim subclassing it, both run the patch.
        assert trl.AsyncGRPOTrainer is module.AsyncGRPOTrainer
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from trl.experimental.async_grpo import AsyncGRPOTrainer as Shim
        assert issubclass(Shim, module.AsyncGRPOTrainer) and "__init__" in vars(Shim)
    # Unsloth's TRL-wide backwards-compatible __init__ (TRL >= 1.18 lists the trainer in trl.trainer) may wrap it.
    chain, init = [], module.AsyncGRPOTrainer.__init__
    while init is not None:
        chain.append(init)
        init = getattr(init, "__wrapped__", None)
    assert any(
        "importance_sampling_mask"
        in getattr(f, "__code__", types.SimpleNamespace(co_varnames = ())).co_varnames
        for f in chain
    )


def test_packed_row_guard_engages(patched):
    # Plain padding-free rows (TRL 1.14-1.16): position_ids restart at 0 per sequence.
    ns, module = patched
    model = _tiny_model()
    torch.manual_seed(0)
    samples = [{"input_ids": torch.randint(10, 400, (n,))} for n in (9, 7, 11)]
    input_ids = torch.cat([s["input_ids"] for s in samples])[None]
    position_ids = torch.cat([torch.arange(len(s["input_ids"])) for s in samples])[None]
    shift_labels = torch.cat(
        [torch.cat([s["input_ids"][1:], torch.tensor([-100])]) for s in samples]
    )[None]
    order = [(i, p) for i, s in enumerate(samples) for p in range(1, len(s["input_ids"]))]
    ref_logps, _ = _reference(model, samples, order)
    fake = _head(model, module)
    with torch.no_grad():
        out = model(
            input_ids = input_ids,
            position_ids = position_ids,
            shift_labels = shift_labels,
            fused_lm_head = True,
        )
    assert fake.calls[-1]["packed_seq_lengths"] == [9, 7, 11]
    got = out.log_probs[0][shift_labels[0] != -100]
    torch.testing.assert_close(got, ref_logps, atol = 1e-5, rtol = 1e-5)


def _tree_samples():
    # Two prompt groups sharing a system prefix; in group 0 two completions share their first two tokens.
    system = [11, 12, 13]
    prompts = [system + [21, 22, 23, 24], system + [31, 32, 33]]
    completions = [
        [(41, 42, 43, 44), (41, 42, 51, 52, 53), (61, 62, 63)],
        [(71, 72, 73), (81, 82)],
    ]
    samples = []
    for group, (prompt, outs) in enumerate(zip(prompts, completions)):
        for completion in outs:
            ids = prompt + list(completion)
            n = len(ids)
            samples.append(
                {
                    "input_ids": torch.tensor(ids),
                    "completion_mask": torch.tensor([0] * len(prompt) + [1] * len(completion)),
                    "old_log_probs": torch.zeros(n),
                    "advantage": 1.0,
                    "group_id": group,
                    "sampling_mask": [[] for _ in range(n)],
                    "sampler_ids": __import__("numpy").zeros((n, 0), dtype = "int64"),
                    "sampler_log_probs": __import__("numpy").zeros((n, 0), dtype = "float32"),
                }
            )
    return samples


def test_tree_packed_row_matches_unpacked(patched):
    # TRL >= 1.17: a prefix-trie row through Unsloth's head equals each sample run alone.
    ns, module = patched
    if not hasattr(module, "TreeRow") or not hasattr(module, "unpad_tree"):
        pytest.skip("this TRL packs rows without a prefix trie")
    model = _tiny_model()
    samples = _tree_samples()
    row = module.TreeRow()
    for sample in samples:
        row.add(sample)
    packed = row.pack()
    assert int((packed["tree_segment_parents"] >= 0).sum()) > 0, "nothing shared"
    assert packed["input_ids"].numel() < sum(len(s["input_ids"]) for s in samples)
    order = [
        (i, p)
        for i, s in enumerate(samples)
        for p in range(1, len(s["input_ids"]))
        if s["completion_mask"][p]
    ]
    ref_logps, ref_entropy = _reference(model, samples, order)
    fake = _head(model, module)
    inputs = {k: v[None] for k, v in packed.items()}
    kwargs = dict(
        input_ids = inputs["input_ids"],
        position_ids = inputs["position_ids"],
        shift_labels = inputs["shift_labels"],
        support_ids = None,
        score_ids = None,
        fused_lm_head = True,
    )
    with torch.no_grad():
        out = model(**kwargs, **module.unpad_tree(inputs))
    call = fake.calls[-1]
    assert call["packed_seq_lengths"] == [len(s["input_ids"]) for s in samples]
    assert call["tree"] == []
    index = packed["token_index"]
    assert out.log_probs.shape == packed["input_ids"][None].shape
    torch.testing.assert_close(out.log_probs.flatten()[index], ref_logps, atol = 1e-5, rtol = 1e-5)
    torch.testing.assert_close(out.entropy.flatten()[index], ref_entropy, atol = 1e-5, rtol = 1e-5)
    # Control: the same row read as one sequence lets samples attend to their siblings, which this test detects.
    with torch.no_grad():
        crossed = fake(
            input_ids = inputs["input_ids"],
            position_ids = inputs["position_ids"],
        ).last_hidden_state
        logps = torch.log_softmax(model.lm_head(crossed)[0].float(), -1)
    leaked = logps.gather(-1, inputs["shift_labels"][0].clamp(min = 0)[:, None])[:, 0][index]
    assert (leaked - ref_logps).abs().max() > 1e-3


def test_tree_row_gradients_match_unpacked(patched):
    ns, module = patched
    if not hasattr(module, "TreeRow"):
        pytest.skip("this TRL packs rows without a prefix trie")
    model = _tiny_model()
    samples = _tree_samples()
    row = module.TreeRow()
    for sample in samples:
        row.add(sample)
    packed = row.pack()
    inputs = {k: v[None] for k, v in packed.items()}
    weight = model.model.layers[0].self_attn.q_proj.weight
    loss = 0
    for sample in samples:
        logits = model(input_ids = sample["input_ids"][None]).logits[0].float()
        logp = torch.log_softmax(logits[:-1], -1).gather(-1, sample["input_ids"][1:, None])[:, 0]
        loss = loss + (logp * sample["completion_mask"][1:]).sum()
    (expected,) = torch.autograd.grad(loss, weight)
    _head(model, module)
    out = model(
        input_ids = inputs["input_ids"],
        position_ids = inputs["position_ids"],
        shift_labels = inputs["shift_labels"],
        fused_lm_head = True,
        **module.unpad_tree(inputs),
    )
    (got,) = torch.autograd.grad(out.log_probs.flatten()[packed["token_index"]].sum(), weight)
    torch.testing.assert_close(got, expected, atol = 1e-5, rtol = 1e-4)
