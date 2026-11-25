# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Inkling 2 LoRA GGUF export: the adapter copy handed to llama.cpp's convert_lora_to_gguf.py carries the names its
Inkling2 converter maps, and per-expert factors whose llama.cpp product (b[e] @ a[e] * alpha / rank) is PEFT's own
delta. Runs on CPU with real PEFT on a stand-in module tree, without `import unsloth` (which wants a GPU)."""

import importlib.util
import json
import os
import pathlib

import pytest

torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")
pytest.importorskip("safetensors")
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file

_SPEC = importlib.util.spec_from_file_location(
    "_inkling2_under_test",
    pathlib.Path(__file__).resolve().parents[1] / "unsloth" / "models" / "inkling2.py",
)
inkling2 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(inkling2)

E, H, I, R = 4, 16, 8, 4


class _Experts(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # transformers' Inkling 2 layout: (experts, out, in), gate / up halved.
        self.gate_up_proj = torch.nn.Parameter(torch.randn(E, 2 * I, H))
        self.down_proj = torch.nn.Parameter(torch.randn(E, H, I))

    def forward(self, x):
        return x


class _Wrapped(torch.nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.linear = torch.nn.Linear(i, o, bias = False)
        self.gate_proj = torch.nn.Linear(i, o, bias = False)


class _Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = _Wrapped(H, H)
        self.k_proj = _Wrapped(H, H // 2)
        self.v_proj = _Wrapped(H, H // 2)
        self.o_proj = _Wrapped(H, H)
        self.sink_proj = torch.nn.Linear(H, 2, bias = False)


class _Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attention()
        self.mlp = torch.nn.Module()
        self.mlp.experts = _Experts()


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.language_model = torch.nn.Module()
        self.model.language_model.layers = torch.nn.ModuleList([_Layer(), _Layer()])
        self.model.vision_tower = torch.nn.Module()
        self.model.vision_tower.proj = torch.nn.Linear(H, H, bias = False)

    def forward(self, x):
        return x


_LINEARS = [
    "self_attn.q_proj.linear",
    "self_attn.k_proj.linear",
    "self_attn.v_proj.linear",
    "self_attn.o_proj.linear",
    "self_attn.o_proj.gate_proj",
    "self_attn.sink_proj",
]


def _peft_model(
    tmp_path,
    vision = False,
    **config,
):
    torch.manual_seed(0)
    model = get_peft_model(
        _Model(),
        LoraConfig(
            r = R,
            lora_alpha = 8,
            target_modules = [f"layers.{i}.{leaf}" for i in range(2) for leaf in _LINEARS]
            + (["vision_tower.proj"] if vision else []),
            target_parameters = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
            **config,
        ),
    )
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_" in name:
                p.normal_(0, 0.5)
    adapter = tmp_path / "adapter"
    model.save_pretrained(adapter)
    return model, adapter


def _rewrite(model, adapter):
    directory = inkling2.inkling2_lora_gguf_adapter(model, str(adapter))
    try:
        tensors = load_file(os.path.join(directory, "adapter_model.safetensors"))
        with open(os.path.join(directory, "adapter_config.json"), encoding = "utf-8") as f:
            config = json.load(f)
        return tensors, config, sorted(os.listdir(directory))
    finally:
        import shutil
        shutil.rmtree(directory)


def _converter_base_name(key):
    # convert_lora_to_gguf.get_base_tensor_name + Inkling2Model.filter_tensors
    name = key.replace("base_model.model.", "").replace(".lora_A.weight", ".weight")
    return name.replace(".lora_B.weight", ".weight").replace("model.language_model.", "model.")


def _gguf_delta(tensors, prefix, alpha, rank):
    a = tensors[f"{prefix}.lora_A.weight"].double()
    b = tensors[f"{prefix}.lora_B.weight"].double()
    return (b @ a) * (alpha / rank)  # llama.cpp: scale = alpha / b->ne[0]


def _peft_deltas(model):
    from peft.tuners.lora.layer import ParamWrapper

    deltas = {}
    for name, module in model.named_modules():
        if isinstance(module, ParamWrapper):
            owner = name
            while owner.endswith(".base_layer"):
                owner = owner[: -len(".base_layer")]
            deltas[f"{owner}.{module.parameter_name}"] = module.get_delta_weight("default")
        elif (
            hasattr(module, "lora_A")
            and "default" in module.lora_A
            and not hasattr(module, "parameter_name")
        ):
            deltas[name] = module.get_delta_weight("default")
    return deltas


def test_names_are_the_ones_the_inkling2_converter_maps(tmp_path):
    model, adapter = _peft_model(tmp_path)
    tensors, config, files = _rewrite(model, adapter)
    assert files == ["README.md", "adapter_config.json", "adapter_model.safetensors"]
    assert config["lora_alpha"] == 8
    names = {_converter_base_name(k) for k in tensors}
    want = set()
    for i in range(2):
        for leaf in ("wq_du", "wk_dv", "wv_dv", "wo_ud", "gate_attn"):
            want.add(f"model.layers.{i}.attn.{leaf}.weight")
        want.add(f"model.layers.{i}.self_attn.sink_proj.weight")  # already a converter name
        want.add(f"model.layers.{i}.mlp.experts.gate_up_proj.weight")
        want.add(f"model.layers.{i}.mlp.experts.down_proj.weight")
    assert names == want
    assert not any(".base_layer" in k or ".linear." in k for k in tensors)


def test_expert_factors_are_per_expert_3d(tmp_path):
    model, adapter = _peft_model(tmp_path)
    tensors, _, _ = _rewrite(model, adapter)
    prefix = "base_model.model.model.language_model.layers.0.mlp.experts"
    # llama.cpp's layout whatever PEFT trained: a (E, r, in), b (E, out, r) of the (E, out, in) stack.
    assert tuple(tensors[f"{prefix}.gate_up_proj.lora_A.weight"].shape) == (E, R, H)
    assert tuple(tensors[f"{prefix}.gate_up_proj.lora_B.weight"].shape) == (E, 2 * I, R)
    assert tuple(tensors[f"{prefix}.down_proj.lora_A.weight"].shape) == (E, R, I)
    assert tuple(tensors[f"{prefix}.down_proj.lora_B.weight"].shape) == (E, H, R)


@pytest.mark.parametrize("config", [{}, {"use_rslora": True}], ids = ["alpha_over_r", "rslora"])
def test_llama_cpp_product_is_the_peft_delta(tmp_path, config):
    model, adapter = _peft_model(tmp_path, **config)
    tensors, saved, _ = _rewrite(model, adapter)
    alpha = float(saved["lora_alpha"])
    deltas = _peft_deltas(model)
    checked = 0
    for name, delta in deltas.items():
        path = name
        for leaf, checkpoint_leaf in inkling2._GGUF_LORA_LEAVES.items():
            if path.endswith("." + leaf):
                path = path[: -len(leaf)] + checkpoint_leaf
        got = _gguf_delta(tensors, path, alpha, R)
        if name.endswith(("gate_up_proj", "down_proj")):
            # PEFT's ParamWrapper delta for an (experts, out, in) parameter is (experts, out, in).
            assert tuple(got.shape) == tuple(delta.shape)
        torch.testing.assert_close(got, delta.double(), rtol = 1e-5, atol = 1e-5)
        checked += 1
    assert checked == 2 * (len(_LINEARS) + 2)


def test_recorded_grouped_by_expert_lora_b_is_unpacked_by_expert(tmp_path):
    model, adapter = _peft_model(tmp_path)
    config_path = adapter / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding = "utf-8"))
    config["unsloth_fused_expert_lora"] = {
        "lora_A_layout": "grouped_by_expert",
        "lora_B_layout": "rank_major",
        "parameters": {"down_proj": {"lora_B_layout": "grouped_by_expert", "num_experts": E}},
    }
    config_path.write_text(json.dumps(config), encoding = "utf-8")
    saved = load_file(adapter / "adapter_model.safetensors")
    tensors, _, _ = _rewrite(model, adapter)
    prefix = "base_model.model.model.language_model.layers.0.mlp.experts"
    flat = saved[f"{prefix}.lora_B.weight"]  # the outer wrapper is down_proj
    wrappers = {n: m for n, m in model.named_modules() if hasattr(m, "parameter_name")}
    assert wrappers[prefix].parameter_name == "down_proj"

    def saved_b(parameter, e):
        # PEFT 0.18 does not swap the features, so llama.cpp's a holds B^T.
        out = f"{prefix}.{parameter}"
        swapped = getattr(
            wrappers[prefix if parameter == "down_proj" else prefix + ".base_layer"],
            "_did_swap_in_out_features",
            False,
        )
        return (
            tensors[f"{out}.lora_B.weight"][e] if swapped else tensors[f"{out}.lora_A.weight"][e].T
        )

    for e in range(E):
        torch.testing.assert_close(saved_b("down_proj", e), flat[:, e * R : (e + 1) * R])
        torch.testing.assert_close(
            saved_b("gate_up_proj", e), saved[f"{prefix}.base_layer.lora_B.weight"][:, e::E]
        )


def test_unknown_layout_is_refused(tmp_path):
    model, adapter = _peft_model(tmp_path)
    config_path = adapter / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding = "utf-8"))
    config["unsloth_fused_expert_lora"] = {"lora_B_layout": "something_else"}
    config_path.write_text(json.dumps(config), encoding = "utf-8")
    with pytest.raises(NotImplementedError, match = "lora_B layout"):
        inkling2.inkling2_lora_gguf_adapter(model, str(adapter))


def test_vision_lora_is_named_as_left_out(tmp_path, capsys):
    model, adapter = _peft_model(tmp_path)
    _rewrite(model, adapter)
    assert "vision tower" not in capsys.readouterr().out
    model, adapter = _peft_model(tmp_path, vision = True)
    _rewrite(model, adapter)
    assert "vision tower's LoRA weights are not in the GGUF LoRA" in capsys.readouterr().out


def test_the_saved_default_adapter_keeps_its_own_scaling_when_another_is_active(tmp_path):
    model, adapter = _peft_model(tmp_path)
    expected, _, _ = _rewrite(model, adapter)
    model.add_adapter(
        "other",
        LoraConfig(
            r = R,
            lora_alpha = 32,
            # PEFT allows target_parameters on one adapter only.
            target_modules = [f"layers.{i}.{leaf}" for i in range(2) for leaf in _LINEARS],
        ),
    )
    model.set_adapter("other")
    got, _, _ = _rewrite(model, adapter)  # the root files are still the default adapter's
    assert set(got) == set(expected)
    for key in expected:
        torch.testing.assert_close(got[key], expected[key])
