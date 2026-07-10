"""CPU regressions for SentenceTransformer module loading and resume topology."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import sys
import types
from typing import Optional

import pytest


torch = pytest.importorskip("torch")
F = torch.nn.functional

_SOURCE_PATH = (
    Path(__file__).resolve().parents[2] / "unsloth" / "models" / "sentence_transformer.py"
)
_SAVE_PATH = Path(__file__).resolve().parents[2] / "unsloth" / "save.py"


def _source_node(name: str, node_type):
    source_tree = ast.parse(_SOURCE_PATH.read_text(encoding = "utf-8"))
    return next(
        node for node in source_tree.body if isinstance(node, node_type) and node.name == name
    )


def _guided_projection_classes():
    source_tree = ast.parse(_SOURCE_PATH.read_text(encoding = "utf-8"))
    nodes = [
        node
        for node in source_tree.body
        if isinstance(node, ast.ClassDef)
        and node.name in {"GuidedProjection", "GuidedProjectionPooling"}
    ]
    namespace = {
        "F": F,
        "Optional": Optional,
        "json": json,
        "nn": torch.nn,
        "os": os,
        "torch": torch,
    }
    exec(compile(ast.Module(body = nodes, type_ignores = []), str(_SOURCE_PATH), "exec"), namespace)
    return namespace["GuidedProjection"], namespace["GuidedProjectionPooling"]


def _guided_projection_api():
    source_tree = ast.parse(_SOURCE_PATH.read_text(encoding = "utf-8"))
    nodes = [
        node
        for node in source_tree.body
        if (
            isinstance(node, ast.ClassDef)
            and node.name in {"GuidedProjection", "GuidedProjectionPooling"}
        )
        or (isinstance(node, ast.FunctionDef) and node.name == "attach_guided_projection")
    ]
    namespace = {
        "F": F,
        "Optional": Optional,
        "json": json,
        "nn": torch.nn,
        "os": os,
        "torch": torch,
    }
    exec(compile(ast.Module(body = nodes, type_ignores = []), str(_SOURCE_PATH), "exec"), namespace)
    return (
        namespace["GuidedProjectionPooling"],
        namespace["attach_guided_projection"],
    )


def test_guided_projection_reports_output_dimension_and_attach_is_idempotent():
    pooling_cls, attach = _guided_projection_api()

    class Pooling(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.ones(1))

        def get_embedding_dimension(self):
            return 8

        def forward(self, features):
            return features

    model = torch.nn.Sequential(Pooling())
    first = attach(model, output_dim = 3, use_residual = False)
    second = attach(model, output_dim = 3, use_residual = False)

    assert first is second
    assert isinstance(model[0], pooling_cls)
    assert model[0].get_embedding_dimension() == 3
    assert model[0].get_sentence_embedding_dimension() == 3

    with pytest.raises(ValueError, match = "different dimension"):
        attach(model, output_dim = 4, use_residual = False)


def test_guided_projection_load_resolves_remote_module_with_full_hub_contract(
    tmp_path, monkeypatch
):
    projection_cls, pooling_cls = _guided_projection_classes()
    module_dir = tmp_path / "2_GuidedProjection"
    module_dir.mkdir()
    (module_dir / pooling_cls.PROJECTION_CONFIG_NAME).write_text(
        json.dumps(
            {
                "dim": 4,
                "output_dim": 4,
                "use_bias": False,
                "use_residual": False,
                "init": "identity",
            }
        ),
        encoding = "utf-8",
    )
    expected_projection = projection_cls(4, use_residual = False)
    with torch.no_grad():
        expected_projection.proj.weight.fill_(0.25)
    torch.save(
        expected_projection.state_dict(),
        module_dir / pooling_cls.PROJECTION_WEIGHTS_NAME,
    )

    calls = []
    util_module = types.ModuleType("sentence_transformers.util")

    def fake_load_dir_path(model_name_or_path, subfolder, **kwargs):
        calls.append((model_name_or_path, subfolder, kwargs))
        return str(module_dir)

    util_module.load_dir_path = fake_load_dir_path
    sentence_transformers_module = types.ModuleType("sentence_transformers")
    sentence_transformers_module.__path__ = []
    sentence_transformers_module.util = util_module
    monkeypatch.setitem(sys.modules, "sentence_transformers", sentence_transformers_module)
    monkeypatch.setitem(sys.modules, "sentence_transformers.util", util_module)

    pooling = torch.nn.Identity()
    loaded = pooling_cls.load(
        "org/guided-projection-model",
        pooling_module = pooling,
        subfolder = "2_GuidedProjection",
        token = "hub-token",
        revision = "0123456789abcdef",
        local_files_only = True,
        cache_folder = str(tmp_path / "cache"),
        trust_remote_code = True,
    )

    assert calls == [
        (
            "org/guided-projection-model",
            "2_GuidedProjection",
            {
                "token": "hub-token",
                "cache_folder": str(tmp_path / "cache"),
                "revision": "0123456789abcdef",
                "local_files_only": True,
            },
        )
    ]
    assert loaded.pooling is pooling
    torch.testing.assert_close(
        loaded.projection.proj.weight,
        expected_projection.proj.weight,
        rtol = 0,
        atol = 0,
    )


def test_full_finetune_resume_unfuses_before_upstream_strict_load(monkeypatch):
    patch_function = _source_node("_patch_st_trainer_load_from_checkpoint", ast.FunctionDef)
    events = []

    class FakeModel(list):
        fused = True

    def unpatch_fused_pooling(model):
        assert model.fused is True
        events.append("unpatch")
        model.fused = False
        return True

    def patch_fused_pooling(model):
        assert model.fused is False
        events.append("repatch")
        model.fused = True
        return True

    class FakeTrainer:
        def _load_from_checkpoint(self, _checkpoint_path):
            # ST 5.4 performs a strict state-dict load here. The model must have
            # its ordinary terminal-LayerNorm topology at this exact point.
            assert self.model.fused is False
            events.append("upstream-load")
            return "loaded"

    sentence_transformers_module = types.ModuleType("sentence_transformers")
    sentence_transformers_module.SentenceTransformerTrainer = FakeTrainer
    monkeypatch.setitem(sys.modules, "sentence_transformers", sentence_transformers_module)

    class FakePeftModel:
        pass

    peft_module = types.ModuleType("peft")
    peft_module.PeftModel = FakePeftModel
    peft_module.load_peft_weights = lambda *_args, **_kwargs: None
    peft_module.set_peft_model_state_dict = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "peft", peft_module)

    namespace = {
        "json": json,
        "os": os,
        "_patch_fused_pooling": patch_fused_pooling,
        "_unpatch_fused_pooling": unpatch_fused_pooling,
    }
    exec(
        compile(ast.Module(body = [patch_function], type_ignores = []), str(_SOURCE_PATH), "exec"),
        namespace,
    )
    namespace["_patch_st_trainer_load_from_checkpoint"]()

    transformer_module = types.SimpleNamespace(auto_model = object())
    trainer = FakeTrainer()
    trainer.model = FakeModel((transformer_module, torch.nn.Identity()))

    assert trainer._load_from_checkpoint("checkpoint-10") == "loaded"
    assert events == ["unpatch", "upstream-load", "repatch"]
    assert trainer.model.fused is True


def test_managed_peft_resume_repatches_pooling_after_module_validation_error(tmp_path, monkeypatch):
    patch_function = _source_node("_patch_st_trainer_load_from_checkpoint", ast.FunctionDef)
    events = []

    class FakeModel(list):
        fused = True

    def unpatch_fused_pooling(model):
        events.append("unpatch")
        model.fused = False
        return True

    def patch_fused_pooling(model):
        events.append("repatch")
        model.fused = True
        return True

    class FakePeftModel:
        active_adapter = "default"
        peft_config = {"default": object()}

    class FakeTrainer:
        def _load_from_checkpoint(self, _checkpoint_path):
            raise AssertionError("managed PEFT path should not use the upstream loader")

    sentence_transformers_module = types.ModuleType("sentence_transformers")
    sentence_transformers_module.SentenceTransformerTrainer = FakeTrainer
    monkeypatch.setitem(sys.modules, "sentence_transformers", sentence_transformers_module)
    peft_module = types.ModuleType("peft")
    peft_module.PeftModel = FakePeftModel
    peft_module.load_peft_weights = lambda *_args, **_kwargs: {}
    peft_module.set_peft_model_state_dict = lambda *_args, **_kwargs: types.SimpleNamespace(
        unexpected_keys = [], missing_keys = []
    )
    monkeypatch.setitem(sys.modules, "peft", peft_module)

    namespace = {
        "json": json,
        "os": os,
        "_patch_fused_pooling": patch_fused_pooling,
        "_unpatch_fused_pooling": unpatch_fused_pooling,
    }
    exec(
        compile(ast.Module(body = [patch_function], type_ignores = []), str(_SOURCE_PATH), "exec"),
        namespace,
    )
    namespace["_patch_st_trainer_load_from_checkpoint"]()

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "adapter_model.bin").write_bytes(b"adapter")
    (checkpoint / "modules.json").write_text(json.dumps([{"idx": 99}]), encoding = "utf-8")
    trainer = FakeTrainer()
    transformer = types.SimpleNamespace(auto_model = FakePeftModel(), _unsloth_st_managed = True)
    trainer.model = FakeModel((transformer, torch.nn.Identity()))

    with pytest.raises(RuntimeError, match = "Bad module index"):
        trainer._load_from_checkpoint(str(checkpoint))
    assert events == ["unpatch", "repatch"]
    assert trainer.model.fused is True


def test_sentence_transformer_gguf_module_path_stays_within_save_directory(tmp_path):
    source_tree = ast.parse(_SAVE_PATH.read_text(encoding = "utf-8"))
    helper = next(
        node
        for node in source_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_sentence_transformer_transformer_dir"
    )
    namespace = {"os": os}
    exec(compile(ast.Module(body = [helper], type_ignores = []), str(_SAVE_PATH), "exec"), namespace)
    resolve = namespace["_sentence_transformer_transformer_dir"]

    assert resolve(tmp_path, "0_Transformer") == str(tmp_path / "0_Transformer")
    assert resolve(tmp_path, "") == str(tmp_path)
    with pytest.raises(ValueError, match = "Invalid SentenceTransformer"):
        resolve(tmp_path, None)
    with pytest.raises(ValueError, match = "Invalid SentenceTransformer"):
        resolve(tmp_path, "../outside")


def test_sentence_transformer_gguf_recursion_forwards_imatrix_and_root_path(tmp_path):
    source_tree = ast.parse(_SAVE_PATH.read_text(encoding = "utf-8"))
    resolver = next(
        node
        for node in source_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_sentence_transformer_transformer_dir"
    )
    exporter = next(
        node
        for node in source_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "unsloth_save_pretrained_gguf"
    )
    exporter.decorator_list = []
    namespace = {
        "Callable": __import__("typing").Callable,
        "List": __import__("typing").List,
        "Optional": Optional,
        "Union": __import__("typing").Union,
        "os": os,
        "torch": torch,
    }
    exec(
        compile(ast.Module(body = [resolver, exporter], type_ignores = []), str(_SAVE_PATH), "exec"),
        namespace,
    )
    outer_exporter = namespace["unsloth_save_pretrained_gguf"]
    captured = []

    def capture_recursive(model, **kwargs):
        captured.append((model, kwargs))
        return {"gguf_files": []}

    namespace["unsloth_save_pretrained_gguf"] = capture_recursive

    class SentenceTransformer:
        tokenizer = object()

        def __init__(self):
            self.inner = types.SimpleNamespace(auto_model = object())

        def __getitem__(self, index):
            assert index == 0
            return self.inner

        def save_pretrained(self, directory):
            os.makedirs(directory, exist_ok = True)
            Path(directory, "modules.json").write_text(
                json.dumps([{"type": "sentence_transformers.models.Transformer", "path": ""}]),
                encoding = "utf-8",
            )

    imatrix = tmp_path / "calibration.dat"
    outer_exporter(
        SentenceTransformer(),
        save_directory = tmp_path / "saved",
        imatrix_file = imatrix,
    )

    assert captured[0][1]["save_directory"] == str(tmp_path / "saved")
    assert captured[0][1]["imatrix_file"] == imatrix
    assert captured[0][1]["_prefer_save_directory"] is True


def test_sentence_transformer_gguf_recursion_keeps_just_saved_weights():
    source_tree = ast.parse(_SAVE_PATH.read_text(encoding = "utf-8"))
    exporter = next(
        node
        for node in source_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "unsloth_save_pretrained_gguf"
    )
    argument_names = {argument.arg for argument in exporter.args.args}
    assert "_prefer_save_directory" in argument_names
    recursive_calls = [
        node
        for node in ast.walk(exporter)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "unsloth_save_pretrained_gguf"
    ]
    assert any(
        any(
            keyword.arg == "_prefer_save_directory"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in call.keywords
        )
        for call in recursive_calls
    )
    assert 'del arguments["_prefer_save_directory"]' in _SAVE_PATH.read_text(encoding = "utf-8")


def test_gguf_source_selection_prefers_fresh_sentence_transformer_weights(tmp_path):
    source_tree = ast.parse(_SAVE_PATH.read_text(encoding = "utf-8"))
    helper = next(
        node
        for node in source_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_gguf_source_directory"
    )
    namespace = {"os": os}
    exec(compile(ast.Module(body = [helper], type_ignores = []), str(_SAVE_PATH), "exec"), namespace)
    select_source = namespace["_gguf_source_directory"]

    original = tmp_path / "original"
    freshly_saved = tmp_path / "fresh"
    original.mkdir()
    freshly_saved.mkdir()
    (original / "weight.sentinel").write_text("stale", encoding = "utf-8")
    (freshly_saved / "weight.sentinel").write_text("fresh", encoding = "utf-8")

    assert select_source(freshly_saved, original, False) == str(original)
    selected = select_source(freshly_saved, original, True)
    assert selected == str(freshly_saved)
    assert Path(selected, "weight.sentinel").read_text(encoding = "utf-8") == "fresh"
