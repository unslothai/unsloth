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

"""CPU checks of AsyncGRPO tree packing through gated DeltaNet layers (TRL >= 1.17, transformers >= 5.15).

A tiny random Qwen3.5 text model in fp32 stands in for an Unsloth one: its GatedDeltaNet forward is a copy of
transformers' whose globals are a separate namespace with transformers' __name__, as in Unsloth's compiled modeling
files, and it is wrapped in a PEFT LoRA model. TRL's Triton tree scan cannot run on CPU, so its chunk kernel is
replaced by a per-segment reference (each segment starts from its parent's final state); TRL's tree convolution
runs as is, and full attention layers use a per-segment SDPA over each segment's root path.
"""

from __future__ import annotations

import ast
import functools
import importlib
import os
import sys
import types

import pytest
import torch

trl = pytest.importorskip("trl")
transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
RL_SOURCE_PATH = os.path.join(REPO_ROOT, "unsloth", "models", "rl.py")
BLOCK_SOURCE_PATH = os.path.join(REPO_ROOT, "unsloth", "models", "_block_compile.py")
NAMES = ("_tree_gdn_callers", "_patch_tree_gdn_namespace", "_unsloth_patch_tree_gdn")
TREE_CPU = "unsloth_test_tree_cpu"


class _Logger:
    def info(self, *a, **k):
        pass


def _async_module():
    for name in (
        "trl.trainer.async_grpo_trainer",
        "trl.experimental.async_grpo.async_grpo_trainer",
    ):
        try:
            module = importlib.import_module(name)
        except Exception:
            continue
        if callable(getattr(module, "patch_tree_gdn", None)) and hasattr(module, "TreeRow"):
            return module
    pytest.skip("this TRL has no tree packing for gated DeltaNet layers (needs TRL >= 1.17)")


def _kernels_module(async_module):
    for name in (
        "trl.kernels.tree_gated_delta_rule",
        async_module.__name__.rsplit(".", 1)[0] + ".tree_gated_delta_rule",
    ):
        try:
            return importlib.import_module(name)
        except Exception:
            continue
    pytest.skip("TRL's tree gated DeltaNet kernels do not import here")


def _load():
    src = open(RL_SOURCE_PATH, encoding = "utf-8").read()
    ns = {"types": types, "torch": torch, "importlib": importlib, "logger": _Logger()}
    for node in ast.parse(src).body:
        is_constant = isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in ("_ASYNC_GRPO_TREE_GDN_COUNTS", "_TREE_GDN_NAMES")
            for t in node.targets
        )
        if is_constant or (isinstance(node, ast.FunctionDef) and node.name in NAMES):
            exec(compile(ast.Module([node], []), RL_SOURCE_PATH, "exec"), ns)
    return ns


def _modeling():
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5
    except Exception:
        pytest.skip("this transformers has no Qwen3.5")
    if not callable(getattr(modeling_qwen3_5, "torch_chunk_gated_delta_rule", None)):
        pytest.skip(
            "this transformers' Qwen3.5 stores its gated DeltaNet kernels per layer (< 5.15)"
        )
    return modeling_qwen3_5


def _cpu_tree_attention(
    module,
    query,
    key,
    value,
    attention_mask,
    scaling = None,
    **kwargs,
):
    # Each segment's queries attend to the K/V of its root path, causal within the segment (TRL's tree_attention).
    kv_index = kwargs["tree_kv_index"]
    cu_q, cu_k = kwargs["tree_cu_seqlens_q"].tolist(), kwargs["tree_cu_seqlens_k"].tolist()
    groups = query.shape[1] // key.shape[1]
    key, value = (x.repeat_interleave(groups, dim = 1) for x in (key, value))
    out = torch.empty_like(query)
    for s in range(len(cu_q) - 1):
        q = query[:, :, cu_q[s] : cu_q[s + 1]]
        path = kv_index[cu_k[s] : cu_k[s + 1]]
        k, v = key[:, :, path], value[:, :, path]
        n_q, n_k = q.shape[2], k.shape[2]
        mask = torch.ones(n_q, n_k, dtype = torch.bool).tril(diagonal = n_k - n_q)
        out[:, :, cu_q[s] : cu_q[s + 1]] = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask = mask, scale = scaling
        )
    return out.transpose(1, 2).contiguous(), None


def _sdpa_or_tree(module, query, key, value, attention_mask, **kwargs):
    if "tree_kv_index" in kwargs:
        return _cpu_tree_attention(module, query, key, value, attention_mask, **kwargs)
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)


@pytest.fixture(scope = "module")
def tree_cpu():
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface, sdpa_mask

    AttentionInterface.register(TREE_CPU, _sdpa_or_tree)
    AttentionMaskInterface.register(TREE_CPU, sdpa_mask)
    return TREE_CPU


def _reference_tree_chunk(chunk):
    # Per segment in DFS pre-order (a parent precedes its children): the segment starts from its parent's final state.
    def tree_chunk_gated_delta_rule(q, k, v, g, beta, tree, use_qk_l2norm_in_kernel):
        cu = tree["tree_cu_seqlens_q"].tolist()
        parents = tree["tree_segment_parents"].tolist()
        outs, finals = [], []
        for s, parent in enumerate(parents):
            a, b = cu[s], cu[s + 1]
            out, final = chunk(
                q[:, a:b],
                k[:, a:b],
                v[:, a:b],
                g = g[:, a:b],
                beta = beta[:, a:b],
                initial_state = finals[parent] if parent >= 0 else None,
                output_final_state = True,
                use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel,
            )
            outs.append(out)
            finals.append(final)
        return torch.cat(outs, 1)

    return tree_chunk_gated_delta_rule


def _compiled_copy(modeling):
    # An Unsloth compiled modeling file: its own module-level `Qwen3_5GatedDeltaNet_forward`, wrapped by a decorator
    # (torch_compiler_disable_unless_decode), in a separate namespace that imported transformers' functions and
    # __name__; the class forward only calls it by name.
    stock = modeling.Qwen3_5GatedDeltaNet.forward
    inner = stock
    while not any(
        n in inner.__code__.co_names for n in ("causal_conv1d_fn", "torch_chunk_gated_delta_rule")
    ):
        inner = inner.__wrapped__
    space = dict(vars(modeling))
    copy = types.FunctionType(
        inner.__code__, space, inner.__name__, inner.__defaults__, inner.__closure__
    )
    copy.__kwdefaults__ = inner.__kwdefaults__

    def decorator(fn):
        @functools.wraps(fn)
        def forward(*args, **kwargs):
            return fn(*args, **kwargs)

        return forward

    space["Qwen3_5GatedDeltaNet_forward"] = decorator(copy)
    exec(
        "def forward(self, hidden_states, cache_params = None, attention_mask = None, **kwargs):\n"
        "    return Qwen3_5GatedDeltaNet_forward(self, hidden_states = hidden_states, cache_params = cache_params, "
        "attention_mask = attention_mask, **kwargs)\n",
        space,
    )
    return space.pop("forward"), space


@pytest.fixture
def setup(monkeypatch, tree_cpu):
    async_module = _async_module()
    kernels = _kernels_module(async_module)
    modeling = _modeling()
    # Undo every global this test rebinds.
    for space_owner, name in (
        (modeling, "causal_conv1d_fn"),
        (modeling, "torch_chunk_gated_delta_rule"),
        (modeling.Qwen3_5GatedDeltaNet, "forward"),
    ):
        monkeypatch.setattr(space_owner, name, getattr(space_owner, name))
    stock_chunk = modeling.torch_chunk_gated_delta_rule
    monkeypatch.setattr(kernels, "tree_chunk_gated_delta_rule", _reference_tree_chunk(stock_chunk))
    # TRL's own function, also when an installed unsloth already wrapped it.
    original = getattr(
        async_module.patch_tree_gdn, "_unsloth_original", async_module.patch_tree_gdn
    )
    trl_module = sys.modules[original.__module__]
    if isinstance(getattr(trl_module, "_PATCHED", None), set):
        monkeypatch.setattr(trl_module, "_PATCHED", set())
    forward, space = _compiled_copy(modeling)
    monkeypatch.setattr(modeling.Qwen3_5GatedDeltaNet, "forward", forward)
    ns = _load()
    return types.SimpleNamespace(
        ns = ns,
        trl_patch = original,
        unsloth_patch = ns["_unsloth_patch_tree_gdn"](original),
        modeling = modeling,
        space = space,
        async_module = async_module,
    )


def _model(
    modeling,
    attn,
    lora = True,
):
    config = modeling.Qwen3_5TextConfig(
        vocab_size = 512,
        hidden_size = 64,
        intermediate_size = 128,
        num_hidden_layers = 4,
        layer_types = ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 16,
        linear_num_key_heads = 2,
        linear_num_value_heads = 4,
        linear_key_head_dim = 16,
        linear_value_head_dim = 16,
        linear_conv_kernel_dim = 4,
        max_position_embeddings = 256,
        tie_word_embeddings = False,
    )
    config._attn_implementation = attn
    torch.manual_seed(3407)
    model = modeling.Qwen3_5ForCausalLM(config).float().eval()
    for name, param in model.named_parameters():
        # Slow decay so the state carries far: a leak across branches is then large.
        if name.endswith(("A_log", "dt_bias")):
            param.data.fill_(-4.0)
    if not lora:
        return model
    lora_config = peft.LoraConfig(
        r = 4,
        lora_alpha = 8,
        target_modules = ["q_proj", "v_proj", "in_proj_qkv", "out_proj"],
        init_lora_weights = False,
    )
    return peft.get_peft_model(model, lora_config)


def _samples(perturb = None):
    # Two prompt groups sharing a system prefix; completions fork, some after a shared head of a few tokens.
    system = list(range(11, 19))
    prompts = [system + [21, 22, 23, 24, 25], system + [31, 32, 33]]
    completions = [
        [(41, 42, 43, 44, 45, 46), (41, 42, 43, 51, 52, 53, 54), (61, 62, 63, 64, 65)],
        [(71, 72, 73, 74), (81, 82, 83, 84, 85, 86)],
    ]
    if perturb is not None:
        group, index = perturb
        completions[group][index] = tuple(t + 300 for t in completions[group][index])
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


def _pack(async_module, samples):
    row = async_module.TreeRow()
    for sample in samples:
        row.add(sample)
    packed = row.pack()
    assert int((packed["tree_segment_parents"] >= 0).sum()) > 0, "nothing shared"
    assert packed["input_ids"].numel() < sum(len(s["input_ids"]) for s in samples)
    inputs = {k: v[None] for k, v in packed.items()}
    return packed, async_module.unpad_tree(inputs)


def _per_sample_logps(model, samples):
    # Each sample alone: the per-target log-probs in TRL's token_index order (sample by sample, position order).
    out = []
    for sample in samples:
        ids = sample["input_ids"]
        logits = model(input_ids = ids[None], use_cache = False).logits[0].float()
        logp = torch.log_softmax(logits[:-1], -1).gather(-1, ids[1:, None])[:, 0]
        out.append(logp[sample["completion_mask"][1:].bool()])
    return torch.cat(out)


def _tree_logps(model, packed, tree):
    logits = (
        model(
            input_ids = packed["input_ids"][None],
            position_ids = packed["position_ids"][None],
            use_cache = False,
            **tree,
        )
        .logits[0]
        .float()
    )
    logp = torch.log_softmax(logits, -1)
    labels = packed["shift_labels"].clamp(min = 0)
    per_position = logp.gather(-1, labels[:, None])[:, 0]
    return per_position[packed["token_index"]]


def test_trl_patch_misses_peft_and_compiled_layers(setup, tree_cpu):
    # TRL's sys.modules patch refuses PEFT and misses compiled forwards, which keep unpatched
    # kernels.
    model = _model(setup.modeling, tree_cpu)
    assert setup.trl_patch(model) is False
    base = model.get_base_model()
    assert setup.trl_patch(base) is True
    assert (
        setup.space["torch_chunk_gated_delta_rule"]
        is not setup.modeling.torch_chunk_gated_delta_rule
    )
    samples = _samples()
    packed, tree = _pack(setup.async_module, samples)
    with torch.no_grad():
        expected = _per_sample_logps(base, samples)
        leaked = _tree_logps(base, packed, tree)
    assert (leaked - expected).abs().max() > 1e-3


def test_patch_routes_the_layers_that_run(setup, tree_cpu):
    model = _model(setup.modeling, tree_cpu)
    assert setup.unsloth_patch(model) is True
    for name in ("causal_conv1d_fn", "torch_chunk_gated_delta_rule"):
        assert getattr(setup.space[name], "_unsloth_tree_gdn", False), name
    # Idempotent: a second trainer on the same model wraps nothing twice.
    before = dict((n, setup.space[n]) for n in ("causal_conv1d_fn", "torch_chunk_gated_delta_rule"))
    assert setup.unsloth_patch(model) is True
    assert all(setup.space[n] is fn for n, fn in before.items())


def test_model_without_gated_deltanet_keeps_trl_answer(setup, tree_cpu):
    calls = []

    def trl_patch(model):
        calls.append(model)
        return False

    wrapped = setup.ns["_unsloth_patch_tree_gdn"](trl_patch)
    dense = transformers.Qwen3Config(
        vocab_size = 128,
        hidden_size = 32,
        intermediate_size = 64,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        num_key_value_heads = 1,
        head_dim = 16,
    )
    model = transformers.Qwen3ForCausalLM(dense)
    assert wrapped(model) is False
    assert calls == [model]


def test_tree_row_matches_each_sample_alone(setup, tree_cpu):
    model = _model(setup.modeling, tree_cpu)
    assert setup.unsloth_patch(model) is True
    counts = setup.ns["_ASYNC_GRPO_TREE_GDN_COUNTS"]
    samples = _samples()
    packed, tree = _pack(setup.async_module, samples)
    with torch.no_grad():
        expected = _per_sample_logps(model, samples)
        before = dict(counts)
        got = _tree_logps(model, packed, tree)
    # Engagement: every gated DeltaNet layer ran its conv and scan along the tree (3 layers, one forward).
    assert counts["conv"] - before["conv"] == 3
    assert counts["chunk"] - before["chunk"] == 3
    torch.testing.assert_close(got, expected, atol = 1e-5, rtol = 1e-5)


def test_tree_row_gradients_match_each_sample_alone(setup, tree_cpu):
    model = _model(setup.modeling, tree_cpu)
    assert setup.unsloth_patch(model) is True
    model.train()
    samples = _samples()
    packed, tree = _pack(setup.async_module, samples)
    params = [p for n, p in model.named_parameters() if p.requires_grad and "in_proj_qkv" in n]
    assert params
    (expected,) = [torch.autograd.grad(_per_sample_logps(model, samples).sum(), params)]
    (got,) = [torch.autograd.grad(_tree_logps(model, packed, tree).sum(), params)]
    for a, b in zip(got, expected):
        torch.testing.assert_close(a, b, atol = 1e-5, rtol = 1e-4)


def test_changing_a_sibling_branch_does_not_move_the_others(setup, tree_cpu):
    model = _model(setup.modeling, tree_cpu)
    assert setup.unsloth_patch(model) is True
    clean, perturbed = _samples(), _samples(perturb = (0, 0))
    with torch.no_grad():
        a = _tree_logps(model, *_pack(setup.async_module, clean))
        b = _tree_logps(model, *_pack(setup.async_module, perturbed))
    # Targets in TRL order: sample by sample. Sample 0 changed; every other sample must be untouched.
    lengths = [int(s["completion_mask"][1:].sum()) for s in clean]
    first = lengths[0]
    assert (a[:first] - b[:first]).abs().max() > 1e-3
    torch.testing.assert_close(a[first:], b[first:], atol = 1e-6, rtol = 0)


def test_rows_without_a_tree_run_unchanged(setup, tree_cpu):
    model = _model(setup.modeling, tree_cpu)
    ids = torch.tensor([[11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
    with torch.no_grad():
        before = model(input_ids = ids, use_cache = False).logits
    assert setup.unsloth_patch(model) is True
    counts = dict(setup.ns["_ASYNC_GRPO_TREE_GDN_COUNTS"])
    with torch.no_grad():
        after = model(input_ids = ids, use_cache = False).logits
    assert counts == setup.ns["_ASYNC_GRPO_TREE_GDN_COUNTS"]
    torch.testing.assert_close(after, before, atol = 0, rtol = 0)


def _block_module():
    name = "_unsloth_block_compile_tree_gdn_test"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, BLOCK_SOURCE_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def test_block_compile_adapters_leave_tree_calls_to_the_tree_kernels(monkeypatch):
    # Inside a compiled block Unsloth's adapters swap in their own conv / fla op; a tree call must go to the wrapper.
    block = _block_module()
    calls = []

    def original(*args, **kwargs):
        calls.append(sorted(kwargs))
        return "original", None

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(block, "_GDN_OP", [True])
    monkeypatch.setattr(block, "_IN_BLOCK", ["full"])
    x = torch.zeros(1, 4, 2, 8)
    g = torch.zeros(1, 4, 2)
    chunk = block._gdn_chunk_adapter(original)
    tree = {"tree_segment_parents": torch.tensor([-1]), "tree_cu_seqlens_q": torch.tensor([0, 4])}
    assert chunk(x, x, x, g, g, use_qk_l2norm_in_kernel = True, **tree)[0] == "original"
    conv = block._causal_conv1d_adapter(
        original, lambda h, w: (_ for _ in ()).throw(AssertionError("conv"))
    )
    assert conv(torch.zeros(1, 8, 4), torch.zeros(8, 4), None, "silu", **tree)[0] == "original"
    assert len(calls) == 2


def test_a_conv_mixer_with_another_scan_is_refused(setup):
    # Mamba2 / KDA mixers call causal_conv1d_fn but scan with another kernel, which no tree wrapper reroutes.
    space = {
        "causal_conv1d_fn": lambda *a, **k: a[0],
        "torch_chunk_gated_delta_rule": lambda *a, **k: (a[0], None),
        "mamba_scan": lambda x: x,
    }
    exec(
        "def forward(self, x, **kwargs):\n    return mamba_scan(causal_conv1d_fn(x, None, **kwargs))",
        space,
    )
    mixer = type("MambaMixer", (torch.nn.Module,), {"forward": space["forward"]})
    model = torch.nn.Sequential(mixer())
    assert setup.ns["_unsloth_patch_tree_gdn"](lambda m: False)(model) is False
    assert not getattr(space["causal_conv1d_fn"], "_unsloth_tree_gdn", False)
