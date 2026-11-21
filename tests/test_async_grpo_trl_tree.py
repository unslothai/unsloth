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
    "_prefix_grouper_kernel",
    "_tree_attention_eligible",
    "_async_grpo_tree_segment_parents",
    "_async_grpo_tree_seg_info",
    "_forward_tree_attention",
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


def _pg_kernel():
    name = "_pg_kernel_async_grpo_test"
    if name not in sys.modules:
        path = os.path.join(REPO_ROOT, "unsloth", "utils", "prefix_grouper_kernel.py")
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


class FakeUnslothBackbone(torch.nn.Module):
    # Attention boundaries from packed_seq_lengths only (one sequence without it), positions from position_ids.
    # A prefix_seg_info (tree) row runs as one sequence under the kernel's mask_mod, as Unsloth's flex path does.
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
        prefix_seg_info = None,
        **kwargs,
    ):
        if prefix_seg_info is not None:
            self.calls.append(
                {"length": input_ids.shape[1], "prefix_seg_info": prefix_seg_info, "tree": []}
            )
            idx = torch.arange(prefix_seg_info.T)
            mask_mod = _pg_kernel()._make_tree_mask_mod(prefix_seg_info.reach_end.cpu())
            mask = mask_mod(0, 0, idx[:, None], idx[None, :])[None, None]
            out = self.inner(
                input_ids = input_ids,
                position_ids = position_ids,
                attention_mask = mask,
                use_cache = False,
            )
            return types.SimpleNamespace(last_hidden_state = out.last_hidden_state)
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
        "os": os,
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
    ns["_prefix_grouper_kernel"] = _pg_kernel
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
        monkeypatch.setattr(rl, "_tree_attention_eligible", lambda model, device: False)
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
    ns["_tree_attention_eligible"] = lambda model, device: False
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
    # TRL <= 1.15 runs that head only through its Triton kernel, left None where Triton is absent (Windows, macOS).
    if getattr(sys.modules.get("trl.trainer.utils"), "_ChunkedLogProbFunction", True) is None:
        pytest.skip("this TRL's fused LM head needs Triton, which this platform lacks")
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
    # Plain padding-free rows (TRL 1.14-1.15): position_ids restart at 0 per sequence.
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


def _tree_kwargs(module, inputs):
    # The tree_* forward kwargs AsyncGRPOTrainer.compute_loss sends: TRL >= 1.17 `unpad_tree`, TRL 1.16 inline.
    if hasattr(module, "unpad_tree"):
        return module.unpad_tree(inputs)
    tree = {
        key: inputs[key][inputs[key] >= 0]
        for key in ("tree_kv_index", "tree_cu_seqlens_q", "tree_cu_seqlens_k")
    }
    tree["tree_max_seqlen_q"], tree["tree_max_seqlen_k"] = inputs["tree_max_seqlens"][0].tolist()
    return tree


def _parents(ns, packed):
    if "tree_segment_parents" in packed:
        return packed["tree_segment_parents"]
    return ns["_async_grpo_tree_segment_parents"](packed)


def test_tree_packed_row_matches_unpacked(patched):
    # TRL >= 1.16: a prefix-trie row through Unsloth's head equals each sample run alone.
    ns, module = patched
    if not hasattr(module, "TreeRow"):
        pytest.skip("this TRL packs rows without a prefix trie")
    model = _tiny_model()
    samples = _tree_samples()
    row = module.TreeRow()
    for sample in samples:
        row.add(sample)
    packed = row.pack()
    assert int((_parents(ns, packed) >= 0).sum()) > 0, "nothing shared"
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
        out = model(**kwargs, **_tree_kwargs(module, inputs))
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
        **_tree_kwargs(module, inputs),
    )
    (got,) = torch.autograd.grad(out.log_probs.flatten()[packed["token_index"]].sum(), weight)
    torch.testing.assert_close(got, expected, atol = 1e-5, rtol = 1e-4)


def _tree_inputs(module):
    row = module.TreeRow()
    samples = _tree_samples()
    for sample in samples:
        row.add(sample)
    packed = row.pack()
    inputs = {k: v[None] for k, v in packed.items()}
    kwargs = dict(
        input_ids = inputs["input_ids"],
        position_ids = inputs["position_ids"],
        shift_labels = inputs["shift_labels"],
        fused_lm_head = True,
    )
    return samples, packed, kwargs, _tree_kwargs(module, inputs)


@pytest.mark.parametrize("tree_attention", [False, True])
def test_tree_attention_matches_unrolled(patched, monkeypatch, tree_attention):
    # Fast backbones that apply prefix_seg_info run the trie once over its unique tokens (no unroll);
    # per-token logprobs, entropies and gradients equal each sample run alone and the unrolled path.
    ns, module = patched
    if not hasattr(module, "TreeRow"):
        pytest.skip("this TRL packs rows without a prefix trie")
    monkeypatch.setitem(ns, "_tree_attention_eligible", lambda model, device: tree_attention)
    counts = ns["_ASYNC_GRPO_TREE_COUNTS"]
    before = dict(counts)
    model = _tiny_model()
    samples, packed, kwargs, tree = _tree_inputs(module)
    order = [
        (i, p)
        for i, s in enumerate(samples)
        for p in range(1, len(s["input_ids"]))
        if s["completion_mask"][p]
    ]
    ref_logps, ref_entropy = _reference(model, samples, order)
    weight = model.model.layers[0].self_attn.q_proj.weight
    loss = 0
    for sample in samples:
        logits = model(input_ids = sample["input_ids"][None]).logits[0].float()
        logp = torch.log_softmax(logits[:-1], -1).gather(-1, sample["input_ids"][1:, None])[:, 0]
        loss = loss + (logp * sample["completion_mask"][1:]).sum()
    (expected,) = torch.autograd.grad(loss, weight)
    fake = _head(model, module)
    out = model(**kwargs, **tree)
    index = packed["token_index"]
    call = fake.calls[-1]
    unique = packed["input_ids"].numel()
    if tree_attention:
        assert call["length"] == unique
        assert counts["tree_attention"] == before["tree_attention"] + 1
        assert counts["unrolled"] == before["unrolled"]
    else:
        assert call["length"] == sum(len(s["input_ids"]) for s in samples) > unique
        assert counts["unrolled"] == before["unrolled"] + 1
        assert counts["tree_attention"] == before["tree_attention"]
    torch.testing.assert_close(
        out.log_probs.flatten()[index].detach(), ref_logps, atol = 1e-5, rtol = 1e-5
    )
    torch.testing.assert_close(
        out.entropy.flatten()[index].detach(), ref_entropy, atol = 1e-5, rtol = 1e-5
    )
    (got,) = torch.autograd.grad(out.log_probs.flatten()[index].sum(), weight)
    torch.testing.assert_close(got, expected, atol = 1e-5, rtol = 1e-4)


def test_tree_attention_skips_rows_without_forks(patched, monkeypatch):
    # No shared prefix: the row is plain packing and keeps the varlen path even when tree attention is on.
    ns, module = patched
    if not hasattr(module, "TreeRow"):
        pytest.skip("this TRL packs rows without a prefix trie")
    monkeypatch.setitem(ns, "_tree_attention_eligible", lambda model, device: True)
    model = _tiny_model()
    row = module.TreeRow()
    for i, sample in enumerate(_tree_samples()):
        sample["input_ids"] = sample["input_ids"].clone()
        sample["input_ids"][0] = 100 + i  # nothing shared, not even the system prompt
        sample["group_id"] = i
        row.add(sample)
    packed = row.pack()
    assert bool((_parents(ns, packed) < 0).all())
    inputs = {k: v[None] for k, v in packed.items()}
    fake = _head(model, module)
    with torch.no_grad():
        model(
            input_ids = inputs["input_ids"],
            position_ids = inputs["position_ids"],
            shift_labels = inputs["shift_labels"],
            fused_lm_head = True,
            **_tree_kwargs(module, inputs),
        )
    assert "prefix_seg_info" not in fake.calls[-1]
    assert fake.calls[-1]["packed_seq_lengths"] == packed["tree_cu_seqlens_q"].diff().tolist()


def test_tree_attention_falls_back_on_compile_error(patched, monkeypatch):
    ns, module = patched
    if not hasattr(module, "TreeRow"):
        pytest.skip("this TRL packs rows without a prefix trie")
    monkeypatch.setitem(ns, "_tree_attention_eligible", lambda model, device: True)
    kernel = ns["_prefix_grouper_kernel"]()
    model = _tiny_model()
    samples, packed, kwargs, tree = _tree_inputs(module)
    fake = _head(model, module)
    real_forward = FakeUnslothBackbone.forward

    def failing(
        self,
        *args,
        prefix_seg_info = None,
        **kw,
    ):
        if prefix_seg_info is not None:
            raise kernel.PrefixGrouperCompileError("no kernel")
        return real_forward(self, *args, **kw)

    monkeypatch.setattr(FakeUnslothBackbone, "forward", failing)
    with torch.no_grad():
        out = model(**kwargs, **tree)
    assert fake.calls[-1]["packed_seq_lengths"] == [len(s["input_ids"]) for s in samples]
    assert out.log_probs.shape == packed["input_ids"][None].shape


def test_segment_parents_follow_the_kv_paths(patched):
    # TRL 1.16 packs the trie without tree_segment_parents: Unsloth derives them from the K/V paths, and on
    # TRL >= 1.17 the derivation must equal the parents TRL sends.
    ns, module = patched
    if not hasattr(module, "TreeRow"):
        pytest.skip("this TRL packs rows without a prefix trie")
    row = module.TreeRow()
    for sample in _tree_samples():
        row.add(sample)
    packed = row.pack()
    derive = ns["_async_grpo_tree_segment_parents"]
    tree = {key: value for key, value in packed.items() if key != "tree_segment_parents"}
    got = derive(tree).tolist()
    cu_q, cu_k = packed["tree_cu_seqlens_q"].tolist(), packed["tree_cu_seqlens_k"].tolist()
    kv = packed["tree_kv_index"].tolist()
    paths = [kv[cu_k[i] : cu_k[i + 1]] for i in range(len(cu_q) - 1)]
    own = [list(range(cu_q[i], cu_q[i + 1])) for i in range(len(cu_q) - 1)]
    expected = [
        next(
            (j for j, path in enumerate(paths) if path == paths[i][: len(paths[i]) - len(own[i])]),
            -1,
        )
        if len(paths[i]) > len(own[i])
        else -1
        for i in range(len(paths))
    ]
    assert got == expected
    assert any(parent >= 0 for parent in got) and got.count(-1) >= 1
    if "tree_segment_parents" in packed:
        assert got == packed["tree_segment_parents"].tolist()
