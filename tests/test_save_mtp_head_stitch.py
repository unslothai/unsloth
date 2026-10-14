# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A trained MTP head attached as `_unsloth_mtp_head` must reach every export: the merge
copies the STOCK head through from the original shards, so without the stitch a fine-tuned
model ships a head that was never aligned to it."""

import ast
import json
from pathlib import Path

import pytest

SAVE_PY = Path(__file__).resolve().parents[1] / "unsloth" / "save.py"


@pytest.fixture(scope = "module")
def tree():
    return ast.parse(SAVE_PY.read_text(encoding = "utf-8"))


def _func(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name} is not defined in save.py")


def _calls(node):
    out = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            f = sub.func
            out.add(f.id if isinstance(f, ast.Name) else getattr(f, "attr", None))
    return out


@pytest.mark.parametrize(
    "function, expected",
    [
        (
            "unsloth_save_model",
            {
                "_stitch_mtp_head",
                "_save_mtp_head_beside_adapter",
                "_push_with_mtp_head",
                "_mtp_head_tensor_names",
            },
        ),
        (
            "unsloth_generic_save",
            {"_stitch_mtp_head", "_attached_mtp_head", "_push_merged_to_hub_revision"},
        ),
        ("unsloth_save_pretrained_gguf", {"_stitch_mtp_head"}),
        ("_gguf_reuses_loaded_checkpoint", {"_attached_mtp_head"}),
    ],
)
def test_save_paths_are_wired(tree, function, expected):
    assert expected <= _calls(_func(tree, function))


@pytest.fixture(scope = "module")
def save_module():
    pytest.importorskip(
        "unsloth_zoo.speculative.stitch", reason = "unsloth_zoo has no speculative.stitch"
    )
    pytest.importorskip("unsloth", reason = "unsloth is not importable on this runner")
    try:
        import unsloth.save as module
    except ImportError as error:
        pytest.skip(f"unsloth.save is not importable on this runner: {error}")
    return module


def _head(torch):
    head = torch.nn.Module()
    head.fc = torch.nn.Linear(16, 8, bias = False)
    head.norm = torch.nn.Module()
    head.norm.weight = torch.nn.Parameter(torch.randn(8))
    object.__setattr__(head, "_unsloth_mtp_info", {"prefix": "mtp.", "source": "tiny"})
    return head


def _wrapped(torch, head, registered):
    """PeftModel -> LoraModel -> base model, with the head on the innermost model."""
    base = torch.nn.Module()
    base.config = None
    if registered:
        base._unsloth_mtp_head = head
    else:
        object.__setattr__(base, "_unsloth_mtp_head", head)
    lora = torch.nn.Module()
    lora.model = base
    peft = torch.nn.Module()
    peft.base_model = lora
    return peft


@pytest.mark.parametrize("registered", [True, False])
def test_head_is_found_through_wrappers(save_module, registered):
    import torch

    head = _head(torch)
    assert save_module._attached_mtp_head(_wrapped(torch, head, registered)) is head
    assert save_module._attached_mtp_head(torch.nn.Linear(2, 2)) is None
    assert save_module._mtp_head_tensor_names(torch.nn.Linear(2, 2)) == []
    assert sorted(save_module._mtp_head_tensor_names(_wrapped(torch, head, registered))) == [
        "mtp.fc.weight",
        "mtp.norm.weight",
    ]


def test_no_head_is_a_no_op(save_module, tmp_path):
    import torch

    model = torch.nn.Linear(2, 2)
    assert save_module._stitch_mtp_head(model, tmp_path) is None
    assert save_module._save_mtp_head_beside_adapter(model, tmp_path) is None
    assert list(tmp_path.iterdir()) == []


def test_stitch_appends_and_declares(save_module, tmp_path):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    save_file(
        {"model.norm.weight": torch.ones(8, dtype = torch.bfloat16)}, tmp_path / "model.safetensors"
    )
    (tmp_path / "config.json").write_text(json.dumps({"text_config": {"num_hidden_layers": 2}}))
    head = _head(torch)
    head.layers = torch.nn.ModuleList([torch.nn.Module()])
    head.layers[0].up = torch.nn.Linear(8, 8, bias = False)
    report = save_module._stitch_mtp_head(_wrapped(torch, head, registered = False), tmp_path)
    assert report["mode"] == "appended"
    index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
    assert {"mtp.fc.weight", "mtp.norm.weight", "mtp.layers.0.up.weight"} <= set(
        index["weight_map"]
    )
    assert (
        json.loads((tmp_path / "config.json").read_text())["text_config"]["mtp_num_hidden_layers"]
        == 1
    )
    with safe_open(str(tmp_path / index["weight_map"]["mtp.fc.weight"]), "pt") as f:
        assert torch.equal(
            f.get_tensor("mtp.fc.weight"), head.fc.weight.detach().to(torch.bfloat16)
        )


def test_adapter_save_writes_head_file(save_module, tmp_path):
    import torch
    from unsloth_zoo.speculative.stitch import load_saved_mtp_head

    head = _head(torch)
    save_module._save_mtp_head_beside_adapter(_wrapped(torch, head, registered = True), tmp_path)
    loaded, info = load_saved_mtp_head(tmp_path)
    assert set(loaded) == {"fc.weight", "norm.weight"} and info["source"] == "tiny"


def test_gguf_does_not_reuse_checkpoint_with_head(save_module, tmp_path):
    import torch

    model = torch.nn.Module()
    model.config = type("C", (), {"_name_or_path": str(tmp_path)})()
    assert save_module._gguf_reuses_loaded_checkpoint(model) is True
    object.__setattr__(model, "_unsloth_mtp_head", _head(torch))
    assert save_module._gguf_reuses_loaded_checkpoint(model) is False


def test_plain_save_pretrained_keeps_the_head(save_module, tmp_path):
    import torch
    from unsloth.models.speculative import _wrap_save_pretrained
    from unsloth_zoo.speculative.stitch import load_saved_mtp_head

    calls = []

    class Adapter(torch.nn.Module):
        def save_pretrained(self, save_directory, **kwargs):
            calls.append(save_directory)
            (Path(save_directory) / "adapter_config.json").write_text("{}")

    model = Adapter()
    object.__setattr__(model, "_unsloth_mtp_head", _head(torch))
    _wrap_save_pretrained(model)
    _wrap_save_pretrained(model)
    model.save_pretrained(str(tmp_path))
    assert calls == [str(tmp_path)]
    loaded, _ = load_saved_mtp_head(tmp_path)
    assert set(loaded) == {"fc.weight", "norm.weight"}


def test_internal_savers_stitch_once(save_module, tmp_path, monkeypatch):
    """Savers that stitch after writing bypass the wrapper, so one save writes the head once."""
    import torch
    from unsloth.models.speculative import _wrap_save_pretrained

    stitched = []
    monkeypatch.setattr(save_module, "_stitch_mtp_head", lambda m, d: stitched.append(d))

    class Plain(torch.nn.Module):
        def save_pretrained(self, save_directory, **kwargs):
            (Path(save_directory) / "config.json").write_text("{}")

    model = Plain()
    object.__setattr__(model, "_unsloth_mtp_head", _head(torch))
    assert save_module._save_pretrained_unstitched(model) == model.save_pretrained
    _wrap_save_pretrained(model)
    save_module._save_pretrained_unstitched(model)(str(tmp_path))
    assert stitched == []
    model.save_pretrained(str(tmp_path))
    assert stitched == [str(tmp_path)]


def test_reattach_is_a_no_op_without_a_saved_head(tmp_path):
    loader_tree = ast.parse((SAVE_PY.parent / "models" / "loader.py").read_text(encoding = "utf-8"))
    calls = [
        _calls(n)
        for n in ast.walk(loader_tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_reattach_saved_mtp_head"
    ]
    assert calls, "loader.py lost _reattach_saved_mtp_head"
    source = (SAVE_PY.parent / "models" / "loader.py").read_text(encoding = "utf-8")
    assert source.count("_reattach_saved_mtp_head(model, old_model_name)") == 2


def _tiny_glm(tmp_path, torch):
    pytest.importorskip("transformers.models.glm4_moe")
    from transformers import Glm4MoeForCausalLM
    from transformers.models.glm4_moe.configuration_glm4_moe import Glm4MoeConfig

    torch.manual_seed(0)
    config = Glm4MoeConfig(
        vocab_size = 97,
        hidden_size = 64,
        intermediate_size = 128,
        moe_intermediate_size = 32,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 16,
        num_hidden_layers = 2,
        n_routed_experts = 4,
        num_experts_per_tok = 2,
        n_shared_experts = 1,
        first_k_dense_replace = 1,
        num_nextn_predict_layers = 1,
        partial_rotary_factor = 0.5,
    )
    return Glm4MoeForCausalLM(config).eval()


def test_nextn_head_round_trips_through_save_pretrained(save_module, tmp_path):
    """A GLM-style head (model.layers.{N}.*) is appended on a plain save and reloads identically."""
    import torch
    from safetensors.torch import load_file

    pytest.importorskip("unsloth_zoo.speculative.nextn")
    from unsloth.models.speculative import get_mtp_head

    model = _tiny_glm(tmp_path, torch)
    head = get_mtp_head(model, random_init = True)
    assert type(head).__name__ == "NextNHead"
    with torch.no_grad():
        for p in head.parameters():
            p.add_(0.01)
    expected = {k: v.detach().clone() for k, v in head.mtp_state_dict().items()}
    assert all(name.startswith("model.layers.2.") for name in expected)
    assert sorted(save_module._mtp_head_tensor_names(model)) == sorted(expected)

    out = tmp_path / "saved"
    model.save_pretrained(str(out))
    config = json.loads((out / "config.json").read_text())
    assert config["num_nextn_predict_layers"] == 1 and "mtp_num_hidden_layers" not in config
    written = {}
    for shard in out.glob("*.safetensors"):
        written.update(load_file(str(shard)))
    for name, tensor in expected.items():
        assert torch.equal(written[name], tensor.to(written[name].dtype)), name

    fresh = _tiny_glm(tmp_path, torch)
    reloaded = get_mtp_head(fresh, source = str(out))
    assert all(torch.equal(reloaded.mtp_state_dict()[k], v) for k, v in expected.items())

    # Saving again over a checkpoint that now carries the head rewrites it in place.
    report = save_module._stitch_mtp_head(fresh, str(out))
    assert report["mode"] == "replaced"
