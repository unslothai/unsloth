"""Focused loader-contract tests without importing Unsloth's CUDA package."""

from __future__ import annotations

import ast
from collections import OrderedDict
import contextlib
import inspect
import json
import logging
import os
from pathlib import Path
import sys
import threading
import types

import pytest


_SOURCE_PATH = (
    Path(__file__).resolve().parents[2] / "unsloth" / "models" / "sentence_transformer.py"
)
_METHOD_NAMES = {
    "_sentence_transformer_constructor_kwargs",
    "_load_sentence_transformer_config",
    "_adapter_checkpoint_info",
    "_load_peft_adapter",
    "_load_adapter_processor",
    "_merge_sentence_transformer_configs",
    "_base_model_revision_for_peft",
    "_prepare_existing_peft_sentence_transformer",
    "_sentence_transformer_model_config_kwargs",
    "_sentence_transformer_processor_load_contract",
    "_read_pooling_mode",
    "_module_path",
    "_is_transformer_module_ref",
    "_create_transformer_module",
    "_load_modules",
}


def _load_contract_class():
    source_tree = ast.parse(_SOURCE_PATH.read_text(encoding = "utf-8"))
    source_class = next(
        node
        for node in source_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "FastSentenceTransformer"
    )
    methods = [
        node
        for node in source_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in _METHOD_NAMES
    ]
    test_class = ast.ClassDef(
        name = "FastSentenceTransformer",
        bases = [],
        keywords = [],
        body = methods,
        decorator_list = [],
    )
    ast.fix_missing_locations(test_class)
    namespace = {
        "OrderedDict": OrderedDict,
        "contextlib": contextlib,
        "hf_hub_download": None,
        "inspect": inspect,
        "json": json,
        "logging": logging,
        "os": os,
        "_CREATE_TRANSFORMER_MODULE_LOCK": threading.RLock(),
    }
    exec(
        compile(ast.Module(body = [test_class], type_ignores = []), str(_SOURCE_PATH), "exec"),
        namespace,
    )
    cls = namespace["FastSentenceTransformer"]
    # Methods in the extracted class resolve this name through their globals.
    namespace["FastSentenceTransformer"] = cls
    return cls, namespace


@pytest.fixture
def contract_class():
    return _load_contract_class()


def test_local_sentence_transformer_adapter_resolves_base_model(
    contract_class, tmp_path, monkeypatch
):
    cls, _namespace = contract_class
    (tmp_path / "modules.json").write_text(
        json.dumps(
            [
                {
                    "type": "sentence_transformers.models.Transformer",
                    "path": "",
                }
            ]
        ),
        encoding = "utf-8",
    )
    (tmp_path / "adapter_config.json").write_text("{}", encoding = "utf-8")
    peft_calls = []

    class FakePeftConfig:
        @classmethod
        def from_pretrained(cls, model_name):
            peft_calls.append(model_name)
            return types.SimpleNamespace(
                base_model_name_or_path = "sentence-transformers/all-MiniLM-L6-v2",
                revision = "base-commit",
            )

    peft_module = types.ModuleType("peft")
    peft_module.PeftConfig = FakePeftConfig
    monkeypatch.setitem(sys.modules, "peft", peft_module)

    info = cls._adapter_checkpoint_info(
        str(tmp_path),
        token = "token",
        revision = "adapter-commit",
        local_files_only = True,
        cache_folder = "custom-cache",
    )

    assert info["base_model_name_or_path"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert info["base_model_revision"] == "base-commit"
    assert info["adapter_model_id"] == str(tmp_path.resolve())
    assert info["adapter_subfolder"] is None
    assert info["checkpoint_subfolder"] is None
    assert peft_calls == [str(tmp_path.resolve())]


def test_hub_adapter_subfolder_uses_pinned_offline_cache_contract(
    contract_class, tmp_path, monkeypatch
):
    cls, namespace = contract_class
    modules_path = tmp_path / "modules.json"
    modules_path.write_text(
        json.dumps(
            [
                {
                    "type": "sentence_transformers.models.Transformer",
                    "path": "0_Transformer",
                }
            ]
        ),
        encoding = "utf-8",
    )
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    adapter_config_path = adapter_dir / "adapter_config.json"
    adapter_config_path.write_text("{}", encoding = "utf-8")
    download_calls = []
    peft_calls = []

    def fake_download(repo_id, filename, **kwargs):
        download_calls.append((repo_id, filename, kwargs))
        if filename == "modules.json":
            return str(modules_path)
        if filename == "0_Transformer/adapter_config.json":
            return str(adapter_config_path)
        raise AssertionError(f"unexpected download: {filename}")

    class FakePeftConfig:
        @classmethod
        def from_pretrained(cls, model_name):
            peft_calls.append(model_name)
            return types.SimpleNamespace(
                base_model_name_or_path = "org/base-model",
                revision = None,
            )

    peft_module = types.ModuleType("peft")
    peft_module.PeftConfig = FakePeftConfig
    monkeypatch.setitem(sys.modules, "peft", peft_module)
    namespace["hf_hub_download"] = fake_download

    info = cls._adapter_checkpoint_info(
        "org/adapter-checkpoint",
        token = "token",
        revision = "adapter-commit",
        local_files_only = True,
        cache_folder = "custom-cache",
    )

    loading_contract = {
        "token": "token",
        "revision": "adapter-commit",
        "local_files_only": True,
        "cache_dir": "custom-cache",
    }
    assert download_calls == [
        ("org/adapter-checkpoint", "modules.json", loading_contract),
        (
            "org/adapter-checkpoint",
            "0_Transformer/adapter_config.json",
            loading_contract,
        ),
    ]
    assert peft_calls == [str(adapter_dir)]
    assert info["base_model_name_or_path"] == "org/base-model"
    assert info["adapter_model_id"] == "org/adapter-checkpoint"
    assert info["adapter_subfolder"] == "0_Transformer"
    assert info["checkpoint_subfolder"] == "0_Transformer"


def test_resolved_adapter_is_loaded_trainably_with_checkpoint_contract(contract_class, monkeypatch):
    cls, _namespace = contract_class
    calls = []
    base_model = object()
    peft_config = object()
    loaded_model = object()

    class FakePeftModel:
        @classmethod
        def from_pretrained(cls, model, model_id, **kwargs):
            calls.append((model, model_id, kwargs))
            return loaded_model

    peft_module = types.ModuleType("peft")
    peft_module.PeftModel = FakePeftModel
    monkeypatch.setitem(sys.modules, "peft", peft_module)

    result = cls._load_peft_adapter(
        base_model,
        {
            "config": peft_config,
            "adapter_model_id": "org/adapter-checkpoint",
            "adapter_subfolder": "0_Transformer",
        },
        token = "token",
        revision = "adapter-commit",
        local_files_only = True,
        cache_folder = "custom-cache",
        is_trainable = True,
    )

    assert result is loaded_model
    assert calls == [
        (
            base_model,
            "org/adapter-checkpoint",
            {
                "config": peft_config,
                "is_trainable": True,
                "token": "token",
                "revision": "adapter-commit",
                "local_files_only": True,
                "cache_dir": "custom-cache",
                "subfolder": "0_Transformer",
            },
        )
    ]


def test_adapter_processor_is_restored_with_checkpoint_contract(contract_class, monkeypatch):
    cls, _namespace = contract_class
    calls = []
    restored_processor = object()
    fallback_processor = object()

    class FakeAutoProcessor:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            calls.append((cls.__name__, model_name, kwargs))
            return restored_processor

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            raise AssertionError("AutoTokenizer fallback should not run")

    transformers_module = types.ModuleType("transformers")
    transformers_module.AutoProcessor = FakeAutoProcessor
    transformers_module.AutoTokenizer = FakeAutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)

    result = cls._load_adapter_processor(
        "org/adapter-checkpoint",
        fallback_processor,
        processor_kwargs = {"padding_side": "right"},
        max_seq_length = 384,
        token = "token",
        revision = "adapter-commit",
        local_files_only = True,
        cache_folder = "custom-cache",
        trust_remote_code = True,
    )

    assert result is restored_processor
    assert calls == [
        (
            "FakeAutoProcessor",
            "org/adapter-checkpoint",
            {
                "padding_side": "right",
                "model_max_length": 384,
                "token": "token",
                "revision": "adapter-commit",
                "local_files_only": True,
                "trust_remote_code": True,
                "cache_dir": "custom-cache",
            },
        )
    ]


def test_adapter_processor_falls_back_to_legacy_transformer_subfolder(contract_class, monkeypatch):
    cls, _namespace = contract_class
    calls = []
    restored_processor = object()

    class FakeAutoProcessor:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            calls.append((cls.__name__, model_name, kwargs))
            if kwargs.get("subfolder") == "0_Transformer":
                return restored_processor
            raise OSError("no root processor")

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            calls.append((cls.__name__, model_name, kwargs))
            raise OSError("no root tokenizer")

    transformers_module = types.ModuleType("transformers")
    transformers_module.AutoProcessor = FakeAutoProcessor
    transformers_module.AutoTokenizer = FakeAutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)

    result = cls._load_adapter_processor(
        "org/adapter-checkpoint",
        object(),
        adapter_subfolder = "0_Transformer",
        revision = "adapter-commit",
        local_files_only = True,
    )

    assert result is restored_processor
    assert calls[-1][0:2] == ("FakeAutoProcessor", "org/adapter-checkpoint")
    assert calls[-1][2]["subfolder"] == "0_Transformer"


def test_bare_adapter_metadata_overlays_base_sentence_transformer_config(contract_class):
    cls, _namespace = contract_class
    merged = cls._merge_sentence_transformer_configs(
        {
            "prompts": {"query": "base query: ", "document": "base document: "},
            "default_prompt_name": "query",
            "similarity_fn_name": "cosine",
            "truncate_dim": 384,
        },
        {"prompts": {"query": "checkpoint query: "}},
    )

    assert merged == {
        "prompts": {
            "query": "checkpoint query: ",
            "document": "base document: ",
        },
        "default_prompt_name": "query",
        "similarity_fn_name": "cosine",
        "truncate_dim": 384,
    }


def test_peft_adapter_persists_resolved_base_revision(contract_class):
    cls, _namespace = contract_class
    config = types.SimpleNamespace(_commit_hash = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf")

    assert (
        cls._base_model_revision_for_peft(config, {}) == "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
    )
    assert cls._base_model_revision_for_peft(config, {"revision": "explicit"}) == "explicit"
    assert cls._base_model_revision_for_peft(config, {"revision": None}) is None
    source = _SOURCE_PATH.read_text(encoding = "utf-8")
    assert source.count('lora_config_kwargs["revision"] = base_revision') == 2


def test_reloaded_peft_model_is_reused_instead_of_double_wrapped(contract_class, monkeypatch):
    cls, namespace = contract_class
    events = []

    class FakePeftModel:
        config = object()
        active_adapter = "default"

        def set_adapter(
            self,
            adapter_name,
            inference_mode = True,
        ):
            events.append(("set_adapter", adapter_name, inference_mode))

        def train(self):
            events.append("train")

    peft_module = types.ModuleType("peft")
    peft_module.PeftModel = FakePeftModel
    monkeypatch.setitem(sys.modules, "peft", peft_module)
    inner_model = FakePeftModel()
    transformer_module = types.SimpleNamespace(auto_model = inner_model)

    class FakeSentenceTransformer:
        _compile_mode = None

        def __getitem__(self, index):
            assert index == 0
            return transformer_module

    cls._patch_transformer_module_save_config = staticmethod(
        lambda module, config: events.append((module, config))
    )
    namespace["_patch_encoder_attention_lora"] = lambda model: (
        events.append(("fused", model)) or 1
    )

    assert cls._prepare_existing_peft_sentence_transformer(FakeSentenceTransformer()) is True
    assert events == [
        ("set_adapter", "default", False),
        "train",
        (transformer_module, inner_model.config),
        ("fused", inner_model),
    ]


def test_adapter_reload_routes_inference_through_explicit_peft_restore():
    source = _SOURCE_PATH.read_text(encoding = "utf-8")
    assert "if for_inference and _adapter_info is None:" in source
    assert "is_trainable = not for_inference" in source
    assert "if for_inference:\n                compile_mode = None" in source


def test_module_path_forwards_pinned_offline_cache_contract(contract_class, tmp_path):
    cls, namespace = contract_class
    calls = []
    modules_path = tmp_path / "modules.json"
    modules_path.write_text("[]", encoding = "utf-8")

    def fake_download(repo_id, filename, **kwargs):
        calls.append((repo_id, filename, kwargs))
        return str(modules_path)

    namespace["hf_hub_download"] = fake_download
    result = cls._module_path(
        "org/model",
        token = "token",
        revision = "pinned-commit",
        local_files_only = True,
        cache_folder = "custom-cache",
    )

    assert result == str(modules_path)
    assert calls == [
        (
            "org/model",
            "modules.json",
            {
                "token": "token",
                "revision": "pinned-commit",
                "local_files_only": True,
                "cache_dir": "custom-cache",
            },
        )
    ]


def test_pooling_metadata_uses_same_pinned_offline_contract(contract_class, tmp_path):
    cls, namespace = contract_class
    modules_path = tmp_path / "modules.json"
    pooling_path = tmp_path / "pooling.json"
    modules_path.write_text(
        json.dumps(
            [
                {
                    "type": "sentence_transformers.models.Pooling",
                    "path": "1_Pooling",
                }
            ]
        ),
        encoding = "utf-8",
    )
    pooling_path.write_text(json.dumps({"pooling_mode_lasttoken": True}), encoding = "utf-8")
    calls = []

    def fake_download(repo_id, filename, **kwargs):
        calls.append((repo_id, filename, kwargs))
        return str(modules_path if filename == "modules.json" else pooling_path)

    namespace["hf_hub_download"] = fake_download
    assert (
        cls._read_pooling_mode(
            "org/model",
            token = "token",
            revision = "pinned-commit",
            local_files_only = True,
            cache_folder = "custom-cache",
        )
        == "lasttoken"
    )
    assert [call[1] for call in calls] == ["modules.json", "1_Pooling/config.json"]
    for _, _, kwargs in calls:
        assert kwargs == {
            "token": "token",
            "revision": "pinned-commit",
            "local_files_only": True,
            "cache_dir": "custom-cache",
        }


def test_load_modules_threads_contract_to_load_dir_path(contract_class, tmp_path, monkeypatch):
    cls, _namespace = contract_class
    modules_path = tmp_path / "modules.json"
    modules_path.write_text(
        json.dumps([{"type": "example.Pooling", "name": "pool", "path": "1_Pooling"}]),
        encoding = "utf-8",
    )
    cls._module_path = staticmethod(lambda *args, **kwargs: str(modules_path))
    cls._is_transformer_module_ref = staticmethod(lambda _class_ref: False)

    load_calls = []

    class FakeModule:
        @classmethod
        def load(cls, path):
            return (cls.__name__, path)

    util_module = types.ModuleType("sentence_transformers.util")
    util_module.import_from_string = lambda _class_ref: FakeModule

    def fake_load_dir_path(model_name, module_path, **kwargs):
        load_calls.append((model_name, module_path, kwargs))
        return "downloaded-module"

    util_module.load_dir_path = fake_load_dir_path
    models_module = types.ModuleType("sentence_transformers.models")
    models_module.Pooling = object
    models_module.Normalize = object
    root_module = types.ModuleType("sentence_transformers")
    monkeypatch.setitem(sys.modules, "sentence_transformers", root_module)
    monkeypatch.setitem(sys.modules, "sentence_transformers.util", util_module)
    monkeypatch.setitem(sys.modules, "sentence_transformers.models", models_module)

    modules, no_modules = cls._load_modules(
        "org/model",
        "token",
        object(),
        object(),
        512,
        "mean",
        revision = "pinned-commit",
        local_files_only = True,
        cache_folder = "custom-cache",
    )

    assert no_modules is False
    assert modules["pool"] == ("FakeModule", "downloaded-module")
    assert load_calls == [
        (
            "org/model",
            "1_Pooling",
            {
                "token": "token",
                "cache_folder": "custom-cache",
                "revision": "pinned-commit",
                "local_files_only": True,
            },
        )
    ]


def test_transformer_module_config_processor_and_model_share_loading_contract(
    contract_class, monkeypatch
):
    cls, namespace = contract_class
    calls = []

    class FakeAuto:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return (args, kwargs)

    class FakeTransformer:
        def __init__(
            self,
            model_name,
            *,
            model_kwargs = None,
            processor_kwargs = None,
            config_kwargs = None,
            max_seq_length = None,
            do_lower_case = False,
        ):
            calls.append(
                {
                    "model_name": model_name,
                    "model_kwargs": model_kwargs,
                    "processor_kwargs": processor_kwargs,
                    "config_kwargs": config_kwargs,
                    "max_seq_length": max_seq_length,
                    "do_lower_case": do_lower_case,
                }
            )
            self.model_forward_params = set()
            self.config_keys = []

        def save(self, *_args, **_kwargs):
            pass

    transformers_module = types.ModuleType("transformers")
    transformers_module.AutoModel = FakeAuto
    transformers_module.AutoProcessor = FakeAuto
    transformers_module.AutoTokenizer = FakeAuto
    models_module = types.ModuleType("sentence_transformers.models")
    models_module.Transformer = FakeTransformer
    root_module = types.ModuleType("sentence_transformers")
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
    monkeypatch.setitem(sys.modules, "sentence_transformers", root_module)
    monkeypatch.setitem(sys.modules, "sentence_transformers.models", models_module)
    namespace["AutoModel"] = FakeAuto
    cls._patch_transformer_module_save_config = staticmethod(lambda module, _config: module)

    class FakeModel:
        config = types.SimpleNamespace(max_position_embeddings = 4096)

        def forward(
            self,
            input_ids = None,
            attention_mask = None,
        ):
            return input_ids, attention_mask

    tokenizer = types.SimpleNamespace(do_lower_case = True, model_max_length = 2048)
    module = cls._create_transformer_module(
        "org/model",
        FakeModel(),
        tokenizer,
        512,
        True,
        token = "token",
        revision = "pinned-commit",
        local_files_only = True,
        cache_folder = "custom-cache",
        model_kwargs = {"model_option": 1},
        processor_kwargs = {"processor_option": 2},
        config_kwargs = {"config_option": 3},
    )

    loading_contract = {
        "trust_remote_code": True,
        "token": "token",
        "revision": "pinned-commit",
        "local_files_only": True,
        "cache_dir": "custom-cache",
    }
    assert calls == [
        {
            "model_name": "org/model",
            "model_kwargs": {"model_option": 1, **loading_contract},
            "processor_kwargs": {"processor_option": 2, **loading_contract},
            "config_kwargs": {"config_option": 3, **loading_contract},
            "max_seq_length": 512,
            "do_lower_case": True,
        }
    ]
    assert module.tokenizer is tokenizer
    assert module.max_seq_length == 512


def test_generic_processor_options_are_applied_during_pinned_load(contract_class, monkeypatch):
    cls, namespace = contract_class
    calls = []

    class FakeAutoConfig:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            calls.append((cls.__name__, model_name, kwargs))
            return object()

    class FakeAutoProcessor:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            calls.append((cls.__name__, model_name, kwargs))
            return object()

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            calls.append((cls.__name__, model_name, kwargs))
            return object()

    transformers_module = types.ModuleType("transformers")
    transformers_module.AutoConfig = FakeAutoConfig
    transformers_module.AutoProcessor = FakeAutoProcessor
    transformers_module.AutoTokenizer = FakeAutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
    namespace["_CREATE_TRANSFORMER_MODULE_LOCK"] = threading.RLock()

    original_config_descriptor = inspect.getattr_static(FakeAutoConfig, "from_pretrained")
    original_processor_descriptor = inspect.getattr_static(FakeAutoProcessor, "from_pretrained")
    original_tokenizer_descriptor = inspect.getattr_static(FakeAutoTokenizer, "from_pretrained")
    with cls._sentence_transformer_processor_load_contract(
        {"padding_side": "right", "truncation_side": "left"},
        max_seq_length = 512,
        token = "token",
        revision = "pinned-commit",
        local_files_only = True,
        cache_folder = "custom-cache",
        trust_remote_code = True,
    ):
        FakeAutoConfig.from_pretrained("org/model")
        FakeAutoProcessor.from_pretrained("org/model", padding_side = "left")
        FakeAutoTokenizer.from_pretrained("org/model")

    loading_contract = {
        "token": "token",
        "revision": "pinned-commit",
        "local_files_only": True,
        "trust_remote_code": True,
        "cache_dir": "custom-cache",
    }
    expected_processor_kwargs = {
        "padding_side": "right",
        "truncation_side": "left",
        "model_max_length": 512,
        **loading_contract,
    }
    assert calls == [
        ("FakeAutoConfig", "org/model", loading_contract),
        ("FakeAutoProcessor", "org/model", expected_processor_kwargs),
        ("FakeAutoTokenizer", "org/model", expected_processor_kwargs),
    ]
    assert inspect.getattr_static(FakeAutoConfig, "from_pretrained") is original_config_descriptor
    assert (
        inspect.getattr_static(FakeAutoProcessor, "from_pretrained")
        is original_processor_descriptor
    )
    assert (
        inspect.getattr_static(FakeAutoTokenizer, "from_pretrained")
        is original_tokenizer_descriptor
    )


def test_qwen_prompt_config_is_pinned_and_user_values_win(contract_class, tmp_path):
    cls, namespace = contract_class
    config_path = tmp_path / "config_sentence_transformers.json"
    config_path.write_text(
        json.dumps(
            {
                "prompts": {
                    "query": (
                        "Instruct: Given a web search query, retrieve relevant passages that "
                        "answer the query\nQuery:"
                    ),
                    "document": "",
                },
                "default_prompt_name": None,
                "similarity_fn_name": "cosine",
                "truncate_dim": 1024,
            }
        ),
        encoding = "utf-8",
    )
    calls = []

    def fake_download(repo_id, filename, **kwargs):
        calls.append((repo_id, filename, kwargs))
        return str(config_path)

    namespace["hf_hub_download"] = fake_download
    config = cls._load_sentence_transformer_config(
        "unsloth/Qwen3-Embedding-0.6B",
        token = "token",
        revision = "f2fddb42505bde9feaf19f0967b01dce52e764c6",
        local_files_only = True,
        cache_folder = "custom-cache",
    )

    class FakeSentenceTransformer:
        def __init__(
            self,
            *,
            prompts = None,
            default_prompt_name = None,
            similarity_fn_name = None,
            truncate_dim = None,
        ):
            pass

    merged = cls._sentence_transformer_model_config_kwargs(
        FakeSentenceTransformer,
        config,
        {
            "prompts": {"query": "custom-query: "},
            "default_prompt_name": "query",
            "similarity_fn_name": "dot",
            "truncate_dim": 768,
        },
    )

    assert calls == [
        (
            "unsloth/Qwen3-Embedding-0.6B",
            "config_sentence_transformers.json",
            {
                "token": "token",
                "revision": "f2fddb42505bde9feaf19f0967b01dce52e764c6",
                "local_files_only": True,
                "cache_dir": "custom-cache",
            },
        )
    ]
    assert merged == {
        "prompts": {"query": "custom-query: ", "document": ""},
        "default_prompt_name": "query",
        "similarity_fn_name": "dot",
        "truncate_dim": 768,
    }


def test_constructor_kwargs_follow_current_signature_and_translate_tokenizer(contract_class):
    cls, _namespace = contract_class

    class FakeSentenceTransformer:
        def __init__(
            self,
            *,
            cache_folder = None,
            processor_kwargs = None,
            backend = "torch",
            prompts = None,
        ):
            pass

    result = cls._sentence_transformer_constructor_kwargs(
        FakeSentenceTransformer,
        {
            "cache_folder": "cache",
            "tokenizer_kwargs": {"padding_side": "right", "shared": "tokenizer"},
            "processor_kwargs": {"shared": "processor"},
            "backend": "torch",
            "prompts": {"query": "query: "},
            "not_a_constructor_kwarg": 1,
        },
    )

    assert result == {
        "cache_folder": "cache",
        "processor_kwargs": {"padding_side": "right", "shared": "processor"},
        "backend": "torch",
        "prompts": {"query": "query: "},
    }


def test_guided_projection_explicitly_disables_fused_pooling():
    source = _SOURCE_PATH.read_text(encoding = "utf-8")
    warning = "UNSLOTH_ST_FUSED_POOLING is incompatible with "
    assert source.count(warning) == 2
    assert source.count("if _fused_pooling_requested and use_guided_projection:") == 2
