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

"""CPU checks of AsyncGRPO tree packing through recurrent layers on TRL's patch_tree_recurrent(model, layer_type).

TRL >= 1.19 replaced patch_tree_gdn(model) with patch_tree_recurrent(model, layer_type), covering gated DeltaNet
(Qwen3.5), Mamba2 (Nemotron-H) and LFM2 short convolutions. Tiny random fp32 models of each family stand in for
Unsloth ones: the mixer forward is a copy of transformers' whose globals are a separate namespace with transformers'
__name__, as in Unsloth's compiled modeling files, and the model is wrapped in a PEFT LoRA model. TRL's tree kernels
are Triton only, so on CPU they are replaced by per-segment references (each segment starts from its parent's final
state, the convolution window runs back along the root path) and attention uses a per-segment SDPA.
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
peft = pytest.importorskip("peft")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
RL_SOURCE_PATH = os.path.join(REPO_ROOT, "unsloth", "models", "rl.py")
NAMES = ("_tree_gdn_callers", "_patch_tree_recurrent_namespace", "_unsloth_patch_tree_recurrent")
TREE_CPU = "unsloth_test_tree_recurrent_cpu"
TREE_KEYS = ("tree_segment_parents", "tree_cu_seqlens_q")


class _Logger:
    def info(self, *a, **k):
        pass


def _async_module():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name in (
            "trl.trainer.async_grpo_trainer",
            "trl.experimental.async_grpo.async_grpo_trainer",
        ):
            try:
                module = importlib.import_module(name)
            except Exception:
                continue
            if hasattr(module, "TreeRow") and hasattr(module, "AsyncGRPOTrainer"):
                return module
    pytest.skip("this TRL has no AsyncGRPOTrainer tree packing")


def _new_api_module():
    module = _async_module()
    if not callable(getattr(module, "patch_tree_recurrent", None)):
        pytest.skip("this TRL has patch_tree_gdn, not patch_tree_recurrent (needs TRL >= 1.19)")
    return module


def _load():
    src = open(RL_SOURCE_PATH, encoding = "utf-8").read()
    ns = {"types": types, "torch": torch, "importlib": importlib, "logger": _Logger()}
    for node in ast.parse(src).body:
        is_constant = isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id.startswith(("_TREE_", "_ASYNC_GRPO_TREE_"))
            for t in node.targets
        )
        if is_constant or (isinstance(node, ast.FunctionDef) and node.name in NAMES):
            exec(compile(ast.Module([node], []), RL_SOURCE_PATH, "exec"), ns)
    missing = [n for n in NAMES if n not in ns]
    assert not missing, f"rl.py lacks {missing}"
    return ns


# ---------------------------------------------------------------------------------------------------------------
# CPU references for TRL's tree kernels (the tree op takes the stock op first, as patch_tree_recurrent calls it).


def _segments(tree):
    return tree["tree_cu_seqlens_q"].tolist(), tree["tree_segment_parents"].tolist()


def _reference_tree_conv(
    conv,
    hidden_states,
    weight,
    bias = None,
    activation = None,
    **tree,
):
    # (1, channels, tokens): each token's window runs back along its own root path, into its parent segments.
    cu, parents = _segments(tree)
    n_tokens, width = hidden_states.shape[-1], weight.shape[-1]
    segment_of = []
    for s in range(len(parents)):
        segment_of += [s] * (cu[s + 1] - cu[s])

    def previous(t):
        s = segment_of[t]
        if t > cu[s]:
            return t - 1
        return cu[parents[s] + 1] - 1 if parents[s] >= 0 else -1

    x = hidden_states.to(weight.dtype)
    out = torch.zeros_like(x)
    for t in range(n_tokens):
        source = t
        for k in range(width):
            if source < 0:
                break
            out[..., t] = out[..., t] + weight[:, width - 1 - k] * x[0, :, source]
            source = previous(source)
    if bias is not None:
        out = out + bias[:, None]
    if activation in ("silu", "swish"):
        out = torch.nn.functional.silu(out)
    return out.to(hidden_states.dtype)


def _reference_tree_chunk(stock):
    def tree_chunk(
        chunk,
        query,
        key,
        value,
        g,
        beta,
        use_qk_l2norm_in_kernel = False,
        **tree,
    ):
        cu, parents = _segments(tree)
        outs, finals = [], []
        for s, parent in enumerate(parents):
            a, b = cu[s], cu[s + 1]
            out, final = stock(
                query[:, a:b],
                key[:, a:b],
                value[:, a:b],
                g = g[:, a:b],
                beta = beta[:, a:b],
                initial_state = finals[parent] if parent >= 0 else None,
                output_final_state = True,
                use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel,
            )
            outs.append(out)
            finals.append(final)
        return torch.cat(outs, 1), None

    return tree_chunk


def _reference_tree_mamba2_scan(stock):
    def tree_scan(
        scan,
        hidden_states,
        dt,
        A,
        B,
        C,
        chunk_size,
        D = None,
        dt_bias = None,
        **kwargs,
    ):
        cu, parents = _segments(kwargs)
        rest = {k: kwargs[k] for k in ("dt_softplus", "dt_limit") if k in kwargs}
        outs, finals = [], []
        for s, parent in enumerate(parents):
            a, b = cu[s], cu[s + 1]
            out, final = stock(
                hidden_states[:, a:b],
                dt[:, a:b],
                A,
                B[:, a:b],
                C[:, a:b],
                chunk_size,
                D = D,
                dt_bias = dt_bias,
                initial_states = finals[parent] if parent >= 0 else None,
                return_final_states = True,
                **rest,
            )
            outs.append(out)
            finals.append(final)
        return torch.cat(outs, 1)

    return tree_scan


def _reference_tree_mamba2_fused(fused, *args, **kwargs):
    # No CPU fused mixer: None sends the mixer down its unfused path, as transformers' own fallback does.
    return None


def _cpu_tree_attention(
    module,
    query,
    key,
    value,
    attention_mask,
    scaling = None,
    **kwargs,
):
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


# ---------------------------------------------------------------------------------------------------------------
# Families: tiny random fp32 configs.


def _qwen3_5(modeling):
    return modeling.Qwen3_5ForCausalLM(
        modeling.Qwen3_5TextConfig(
            vocab_size = 512,
            hidden_size = 64,
            intermediate_size = 128,
            num_hidden_layers = 4,
            layer_types = [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
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
    )


def _nemotron_h(modeling):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        config = modeling.NemotronHConfig(
            vocab_size = 512,
            hidden_size = 64,
            intermediate_size = 128,
            num_hidden_layers = 4,
            layers_block_type = ["mamba", "mamba", "attention", "mamba"],
            num_attention_heads = 4,
            num_key_value_heads = 2,
            head_dim = 16,
            mamba_num_heads = 4,
            mamba_head_dim = 16,
            ssm_state_size = 16,
            n_groups = 2,
            chunk_size = 8,
            conv_kernel = 4,
            expand = 1,
            tie_word_embeddings = False,
        )
    return modeling.NemotronHForCausalLM(config)


def _lfm2(modeling):
    return modeling.Lfm2ForCausalLM(
        modeling.Lfm2Config(
            vocab_size = 512,
            hidden_size = 64,
            intermediate_size = 128,
            num_hidden_layers = 4,
            num_attention_heads = 4,
            num_key_value_heads = 2,
            layer_types = ["conv", "conv", "full_attention", "conv"],
            conv_L_cache = 3,
            tie_word_embeddings = False,
        )
    )


FAMILIES = {
    "gated_deltanet": dict(
        module = "transformers.models.qwen3_5.modeling_qwen3_5",
        mixer = "Qwen3_5GatedDeltaNet",
        layer_type = "linear_attention",
        names = ("causal_conv1d_fn", "torch_chunk_gated_delta_rule"),
        build = _qwen3_5,
        lora = ["q_proj", "v_proj", "in_proj_qkv", "out_proj"],
        layers = 3,
        min_transformers = "5.15.0",
    ),
    "mamba2": dict(
        module = "transformers.models.nemotron_h.modeling_nemotron_h",
        mixer = "NemotronHMamba2Mixer",
        layer_type = "linear_attention",
        names = ("causal_conv1d_fn", "mamba2_chunk_scan", "mamba2_split_conv1d_scan_combined"),
        build = _nemotron_h,
        # PEFT refuses out_proj / conv1d on Mamba models (the fused kernel reads their raw weights).
        lora = ["q_proj", "v_proj", "in_proj"],
        layers = 3,
        min_transformers = "5.15.0",
    ),
    "short_conv": dict(
        module = "transformers.models.lfm2.modeling_lfm2",
        mixer = "Lfm2ShortConv",
        layer_type = "conv",
        names = ("causal_conv1d_fn",),
        build = _lfm2,
        lora = ["q_proj", "v_proj", "in_proj", "out_proj"],
        layers = 3,
        min_transformers = "5.22.0",
    ),
}


def _family_modeling(spec):
    from packaging.version import Version

    if Version(transformers.__version__) < Version(spec["min_transformers"]):
        pytest.skip(f"tree packing of this family needs transformers >= {spec['min_transformers']}")
    try:
        modeling = importlib.import_module(spec["module"])
    except Exception:
        pytest.skip(f"this transformers has no {spec['module']}")
    if not all(callable(getattr(modeling, n, None)) for n in spec["names"]):
        pytest.skip(f"{spec['module']} does not call {spec['names']} at module level")
    return modeling


def _compiled_copy(modeling, spec):
    # An Unsloth compiled modeling file: its own module-level `<Mixer>_forward`, wrapped by a decorator, in a separate
    # namespace that imported transformers' functions and __name__; the class forward only calls it by name.
    inner = getattr(modeling, spec["mixer"]).forward
    while not any(n in inner.__code__.co_names for n in spec["names"]):
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

    name = f"{spec['mixer']}_forward"
    space[name] = decorator(copy)
    exec(f"def forward(self, *args, **kwargs):\n    return {name}(self, *args, **kwargs)\n", space)
    return space.pop("forward"), space


def _model(setup, lora = True):
    torch.manual_seed(3407)
    model = setup.spec["build"](setup.modeling).float().eval()
    model.config._attn_implementation = setup.attn
    for name, param in model.named_parameters():
        # Slow decay so the state carries far: a leak across branches is then large.
        if name.endswith(("A_log", "dt_bias")):
            param.data.fill_(-4.0)
    if not lora:
        return model
    config = peft.LoraConfig(
        r = 4, lora_alpha = 8, target_modules = setup.spec["lora"], init_lora_weights = False
    )
    return peft.get_peft_model(model, config)


@pytest.fixture(params = list(FAMILIES))
def setup(request, monkeypatch, tree_cpu):
    spec = FAMILIES[request.param]
    async_module = _new_api_module()
    modeling = _family_modeling(spec)
    kernels = {}
    for short in ("tree_gated_delta_rule", "tree_mamba2"):
        try:
            kernels[short] = importlib.import_module(f"trl.kernels.{short}")
        except Exception as error:
            pytest.fail(f"TRL main's trl.kernels.{short} does not import: {error}")
    mixer = getattr(modeling, spec["mixer"])
    for owner, name in [(modeling, n) for n in spec["names"]] + [(mixer, "forward")]:
        monkeypatch.setattr(owner, name, getattr(owner, name))
    stock = {n: getattr(modeling, n) for n in spec["names"]}
    for short, name, ref in (
        ("tree_gated_delta_rule", "tree_causal_conv1d", _reference_tree_conv),
        (
            "tree_gated_delta_rule",
            "tree_chunk_gated_delta_rule",
            _reference_tree_chunk(stock.get("torch_chunk_gated_delta_rule")),
        ),
        (
            "tree_mamba2",
            "tree_mamba2_chunk_scan",
            _reference_tree_mamba2_scan(stock.get("mamba2_chunk_scan")),
        ),
        ("tree_mamba2", "tree_mamba2_split_conv1d_scan_combined", _reference_tree_mamba2_fused),
    ):
        monkeypatch.setattr(kernels[short], name, ref)
    original = getattr(
        async_module.patch_tree_recurrent, "_unsloth_original", async_module.patch_tree_recurrent
    )
    trl_module = sys.modules[original.__module__]
    if isinstance(getattr(trl_module, "_PATCHED", None), set):
        monkeypatch.setattr(trl_module, "_PATCHED", set())
    forward, space = _compiled_copy(modeling, spec)
    monkeypatch.setattr(mixer, "forward", forward)
    ns = _load()
    unsloth_patch = ns["_unsloth_patch_tree_recurrent"](original)
    return types.SimpleNamespace(
        name = request.param,
        spec = spec,
        ns = ns,
        attn = tree_cpu,
        trl_patch = lambda model: original(model, spec["layer_type"]),
        patch = lambda model: unsloth_patch(model, spec["layer_type"]),
        modeling = modeling,
        space = space,
        async_module = async_module,
    )


# ---------------------------------------------------------------------------------------------------------------
# Samples, packing and log-probs.


def _samples(perturb = None):
    # Two prompt groups sharing a system prefix; completions fork, some after a shared head of a few tokens.
    import numpy

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
                    "sampler_ids": numpy.zeros((n, 0), dtype = "int64"),
                    "sampler_log_probs": numpy.zeros((n, 0), dtype = "float32"),
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
    return logp.gather(-1, labels[:, None])[:, 0][packed["token_index"]]


# ---------------------------------------------------------------------------------------------------------------
# Tests.


def test_trl_patch_refuses_peft_and_misses_compiled_layers(setup):
    # TRL's sys.modules lookup sees peft.peft_model for a LoRA model, and patches transformers' module, which the
    # compiled forward does not read.
    model = _model(setup)
    assert setup.trl_patch(model) is False
    assert setup.trl_patch(model.get_base_model()) is True
    for name in setup.spec["names"]:
        assert setup.space[name] is not getattr(setup.modeling, name), name
    samples = _samples()
    packed, tree = _pack(setup.async_module, samples)
    with torch.no_grad():
        expected = _per_sample_logps(model, samples)
        leaked = _tree_logps(model, packed, tree)
    assert (leaked - expected).abs().max() > 1e-3


def test_patch_routes_the_layers_that_run(setup):
    model = _model(setup)
    assert setup.patch(model) is True
    names = setup.spec["names"]
    for name in names:
        assert getattr(setup.space[name], "_unsloth_tree_recurrent", False), name
    before = {n: setup.space[n] for n in names}
    assert setup.patch(model) is True
    assert all(setup.space[n] is fn for n, fn in before.items())


def test_tree_row_matches_each_sample_alone(setup):
    model = _model(setup)
    assert setup.patch(model) is True
    counts = setup.ns["_ASYNC_GRPO_TREE_RECURRENT_COUNTS"]
    samples = _samples()
    packed, tree = _pack(setup.async_module, samples)
    with torch.no_grad():
        expected = _per_sample_logps(model, samples)
        before = dict(counts)
        got = _tree_logps(model, packed, tree)
    # Engagement: every recurrent layer ran its conv (and scan) along the tree (3 layers, one forward, eval mode).
    ran = {n: counts[n] - before[n] for n in counts}
    assert ran["causal_conv1d_fn"] == setup.spec["layers"], ran
    for scan in ("torch_chunk_gated_delta_rule", "mamba2_chunk_scan"):
        if scan in setup.spec["names"]:
            assert ran[scan] == setup.spec["layers"], ran
    torch.testing.assert_close(got, expected, atol = 1e-5, rtol = 1e-5)


def test_tree_row_gradients_match_each_sample_alone(setup):
    model = _model(setup)
    assert setup.patch(model) is True
    model.train()
    samples = _samples()
    packed, tree = _pack(setup.async_module, samples)
    params = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]
    assert params
    expected = torch.autograd.grad(_per_sample_logps(model, samples).sum(), params)
    got = torch.autograd.grad(_tree_logps(model, packed, tree).sum(), params)
    for a, b in zip(got, expected):
        torch.testing.assert_close(a, b, atol = 1e-5, rtol = 1e-4)


def test_changing_a_sibling_branch_does_not_move_the_others(setup):
    model = _model(setup)
    assert setup.patch(model) is True
    clean, perturbed = _samples(), _samples(perturb = (0, 0))
    with torch.no_grad():
        a = _tree_logps(model, *_pack(setup.async_module, clean))
        b = _tree_logps(model, *_pack(setup.async_module, perturbed))
    first = int(clean[0]["completion_mask"][1:].sum())
    assert (a[:first] - b[:first]).abs().max() > 1e-3
    torch.testing.assert_close(a[first:], b[first:], atol = 1e-6, rtol = 0)


def test_rows_without_a_tree_run_unchanged(setup):
    model = _model(setup)
    ids = torch.tensor([[11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
    with torch.no_grad():
        before = model(input_ids = ids, use_cache = False).logits
    assert setup.patch(model) is True
    counts = dict(setup.ns["_ASYNC_GRPO_TREE_RECURRENT_COUNTS"])
    with torch.no_grad():
        after = model(input_ids = ids, use_cache = False).logits
    assert counts == setup.ns["_ASYNC_GRPO_TREE_RECURRENT_COUNTS"]
    torch.testing.assert_close(after, before, atol = 0, rtol = 0)


def test_a_conv_mixer_with_another_scan_is_refused():
    # KDA-like: causal_conv1d_fn plus a scan no tree wrapper reroutes.
    ns = _load()
    space = {
        "causal_conv1d_fn": lambda *a, **k: a[0],
        "torch_chunk_gated_delta_rule": lambda *a, **k: (a[0], None),
        "kda_scan": lambda x: x,
    }
    exec(
        "def forward(self, x, **kwargs):\n    return kda_scan(causal_conv1d_fn(x, None, **kwargs))",
        space,
    )
    mixer = type("KdaMixer", (torch.nn.Module,), {"forward": space["forward"]})
    model = torch.nn.Sequential(mixer())
    patch = ns["_unsloth_patch_tree_recurrent"](lambda m, t: False)
    assert patch(model, "linear_attention") is False
    assert not getattr(space["causal_conv1d_fn"], "_unsloth_tree_recurrent", False)
    assert patch(model, "unknown_type") is False


def test_model_without_recurrent_layers_keeps_trl_answer():
    ns = _load()
    calls = []

    def trl_patch(model, layer_type):
        calls.append((model, layer_type))
        return False

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
    assert ns["_unsloth_patch_tree_recurrent"](trl_patch)(model, "linear_attention") is False
    assert calls == [(model, "linear_attention")]


# ---------------------------------------------------------------------------------------------------------------
# End to end through the hook Unsloth installs on TRL, and TRL's own AsyncGRPOTrainer.__init__.


class _PastTreeGate(Exception):
    pass


@pytest.mark.parametrize("family", list(FAMILIES))
def test_async_grpo_trainer_init_accepts_a_lora_hybrid(family, monkeypatch, tmp_path):
    # Runs TRL's real __init__ up to its next step after the tree check (add_fused_lm_head), on whichever TRL is
    # importable: patch_tree_gdn (TRL 1.17 - 1.18) or patch_tree_recurrent (TRL >= 1.19).
    module = _async_module()
    rl = sys.modules.get("unsloth.models.rl")
    if rl is None or not os.path.samefile(rl.__file__, RL_SOURCE_PATH):
        pytest.skip(
            "tests/conftest.py did not import this tree's unsloth, so its TRL patch is not installed"
        )
    if not getattr(module.add_fused_lm_head, "_unsloth_async_grpo_patched", False):
        pytest.fail("unsloth imported but did not patch TRL's AsyncGRPOTrainer")
    spec = FAMILIES[family]
    modeling = _family_modeling(spec)
    new_api = callable(getattr(module, "patch_tree_recurrent", None))
    if not new_api and family != "gated_deltanet":
        pytest.skip("TRL before patch_tree_recurrent tree-packs gated DeltaNet hybrids only")
    hook = module.patch_tree_recurrent if new_api else module.patch_tree_gdn
    mixer = getattr(modeling, spec["mixer"])
    for name in spec["names"]:
        monkeypatch.setattr(modeling, name, getattr(modeling, name))
    monkeypatch.setattr(mixer, "forward", mixer.forward)
    forward, space = _compiled_copy(modeling, spec)
    monkeypatch.setattr(mixer, "forward", forward)
    original = getattr(hook, "_unsloth_original", hook)
    trl_module = sys.modules[original.__module__]
    if isinstance(getattr(trl_module, "_PATCHED", None), set):
        monkeypatch.setattr(trl_module, "_PATCHED", set())

    def stop(*args, **kwargs):
        raise _PastTreeGate

    monkeypatch.setattr(module, "add_fused_lm_head", stop)
    torch.manual_seed(0)
    base = spec["build"](modeling).float()
    model = peft.get_peft_model(base, peft.LoraConfig(r = 4, target_modules = spec["lora"]))
    args = module.AsyncGRPOConfig(
        output_dir = str(tmp_path), report_to = "none", bf16 = False, use_cpu = True
    )
    with pytest.raises(_PastTreeGate):
        module.AsyncGRPOTrainer(model, reward_funcs = lambda **k: [0.0], args = args)
    # The compiled namespace the layers read is the one routed along the tree.
    marker = "_unsloth_tree_recurrent" if new_api else "_unsloth_tree_gdn"
    for name in spec["names"]:
        assert getattr(space[name], marker, False), name
