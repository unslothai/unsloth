# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Regression tests for serialized UEmbed inference reload and Hub error boundaries.

The package-level ``unsloth`` import requires an accelerator, so these CPU tests execute
its torch-only UEmbed modules directly. The inference test executes the actual
``for_inference`` branch extracted from ``FastSentenceTransformer.from_pretrained`` and
uses a constructor that reconstructs a module chain from a serialized checkpoint file.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import sys
import types
from collections import OrderedDict
from pathlib import Path

import huggingface_hub.errors as _native_hub_errors
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ST_PATH = _REPO_ROOT / "unsloth" / "models" / "sentence_transformer.py"
_POOLING_PATH = _REPO_ROOT / "unsloth" / "models" / "uembed_pooling.py"
_SPLADE_PATH = _REPO_ROOT / "unsloth" / "models" / "uembed_splade.py"
_WIRING_PATH = _REPO_ROOT / "unsloth" / "models" / "uembed_wiring.py"


def _load_source(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope = "module")
def pooling():
    return _load_source("unsloth_uembed_pooling_direct", _POOLING_PATH)


@pytest.fixture(scope = "module")
def splade():
    return _load_source("unsloth_uembed_splade_direct", _SPLADE_PATH)


@pytest.fixture(scope = "module")
def wiring():
    return _load_source("unsloth_uembed_wiring_direct", _WIRING_PATH)


class _Backend:
    post_processor = None


class _Tokenizer:
    def __init__(self) -> None:
        self._tokenizer = _Backend()
        self.padding_side = "left"
        self.unk_token_id = 0
        self.unk_token = "<unk>"

    def convert_tokens_to_ids(self, token):
        return 1 if token == "<|endoftext|>" else self.unk_token_id


class _Processor:
    def __init__(self) -> None:
        self.tokenizer = _Tokenizer()
        self.template_calls = []
        self.processor_calls = []
        self.events = []

    def apply_chat_template(self, conversations, **kwargs):
        self.events.append("format")
        self.template_calls.append((conversations, kwargs))
        return [f"rendered-{index}" for index in range(len(conversations))]

    def __call__(self, **kwargs):
        self.events.append("tokenize")
        self.processor_calls.append(kwargs)
        batch = len(kwargs["text"])
        ids = torch.arange(1, 7).repeat(batch, 1)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


class _Transformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.processor = _Processor()
        self.max_seq_length = 32
        self.table = nn.Parameter(torch.arange(64, dtype = torch.float32).reshape(16, 4))
        self.plain_calls = []
        self.forward_calls = 0

    def preprocess(self, inputs, **kwargs):
        self.plain_calls.append(inputs)
        texts = [inputs] if isinstance(inputs, str) else list(inputs)
        return self.processor(text = [str(value) for value in texts])

    def forward(self, features, **kwargs):
        self.forward_calls += 1
        features["token_embeddings"] = self.table[features["input_ids"]]
        return features


class _Pooling(nn.Module):
    def forward(self, features, **kwargs):
        features["sentence_embedding"] = features["token_embeddings"][:, -1]
        return features

    def get_sentence_embedding_dimension(self):
        return 4


class _Normalize(nn.Module):
    def forward(self, features, **kwargs):
        features["sentence_embedding"] = F.normalize(features["sentence_embedding"], dim = -1)
        return features


class _SentenceTransformer(nn.Sequential):
    def encode(
        self,
        sentences,
        output_value = "sentence_embedding",
        convert_to_numpy = True,
        convert_to_tensor = False,
    ):
        single = isinstance(sentences, (str, dict))
        batch = [sentences] if single else list(sentences)
        features = self[0].preprocess(batch)
        for module in self:
            features = module(features)
        if output_value is None:
            rows = [
                {key: value[index] for key, value in features.items() if torch.is_tensor(value)}
                for index in range(len(batch))
            ]
            return rows[0] if single else rows
        result = features[output_value]
        if not convert_to_tensor and convert_to_numpy:
            result = result.detach().numpy()
        return result[0] if single else result


def _write_serialized_chain(path: Path, sparse: bool) -> None:
    state = {"sparse": sparse}
    if sparse:
        generator = torch.Generator().manual_seed(17)
        state.update(
            heads = [torch.randn(5, 4, generator = generator), torch.randn(7, 4, generator = generator)],
            biases = [torch.randn(5, generator = generator), torch.randn(7, generator = generator)],
            num_eos_tokens = 2,
        )
    torch.save(state, path)


def _constructor(wiring, splade):
    def construct(model_name, **kwargs):
        state = torch.load(model_name, map_location = "cpu", weights_only = True)
        modules = [_Transformer(), _Pooling(), _Normalize()]
        if state["sparse"]:
            head = splade.SpladeHead(state["heads"], state["biases"], state["num_eos_tokens"])
            modules.append(wiring.UEmbedSparseOutput(head))
        return _SentenceTransformer(*modules)

    return construct


def _fresh_hf_constructor(model_name, **kwargs):
    # Native sentence-transformers sees no modules.json in the public repository and
    # therefore builds only its generic Transformer -> Pooling fallback.
    return _SentenceTransformer(_Transformer(), _Pooling())


def _load_inference_assembly(
    wiring,
    splade,
    pooling,
    read_metadata = None,
    attach_checkpoint = None,
):
    tree = ast.parse(_ST_PATH.read_text(encoding = "utf-8"), filename = str(_ST_PATH))
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {
            "_ensure_uembed_offset_pooling",
            "_assemble_uembed_inference_model",
        }
    }
    if "_assemble_uembed_inference_model" not in functions:
        return None

    isolated = ast.Module(body = list(functions.values()), type_ignores = [])
    ast.fix_missing_locations(isolated)

    def read_num_eos_tokens(model_name, **kwargs):
        state = torch.load(model_name, map_location = "cpu", weights_only = True)
        return state.get("num_eos_tokens", 0)

    def attach_sparse(model, model_name, num_eos_tokens, **kwargs):
        state = torch.load(model_name, map_location = "cpu", weights_only = True)
        head = splade.SpladeHead(state["heads"], state["biases"], num_eos_tokens)
        return wiring.attach_uembed_sparse_output(model, head)

    namespace = {
        "OrderedDict": OrderedDict,
        "OffsetLastTokenPooling": pooling.OffsetLastTokenPooling,
        "attach_uembed_sparse_checkpoint": attach_checkpoint or attach_sparse,
        "patch_uembed_sparse_encode": wiring.patch_uembed_sparse_encode,
        "read_num_eos_tokens": read_metadata or read_num_eos_tokens,
        "restore_uembed_inference_input_format": wiring.restore_uembed_inference_input_format,
    }
    exec(compile(isolated, str(_ST_PATH), "exec"), namespace)
    return namespace["_assemble_uembed_inference_model"]


def _inference_branch(
    wiring,
    pooling,
    sentence_transformer,
    checkpoint,
    splade = None,
):
    """Compile and execute the production ``if for_inference`` body in isolation."""
    tree = ast.parse(_ST_PATH.read_text(encoding = "utf-8"), filename = str(_ST_PATH))
    from_pretrained = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "from_pretrained"
    )
    branch = next(
        node
        for node in from_pretrained.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "for_inference"
    )
    function = ast.FunctionDef(
        name = "run",
        args = ast.arguments(posonlyargs = [], args = [], kwonlyargs = [], kw_defaults = [], defaults = []),
        body = branch.body,
        decorator_list = [],
    )
    isolated = ast.Module(body = [function], type_ignores = [])
    ast.fix_missing_locations(isolated)
    namespace = {
        "SentenceTransformer": sentence_transformer,
        "Pooling": _Pooling,
        "Normalize": _Normalize,
        "model_name": str(checkpoint),
        "device_map": "cpu",
        "dtype": torch.float32,
        "kwargs": {},
        "trust_remote_code": False,
        "token": None,
        "revision": None,
        "patch_uembed_sparse_encode": wiring.patch_uembed_sparse_encode,
        "restore_uembed_inference_input_format": wiring.restore_uembed_inference_input_format,
        "_assemble_uembed_inference_model": (
            _load_inference_assembly(wiring, splade, pooling) if splade is not None else None
        ),
    }
    exec(compile(isolated, str(_ST_PATH), "exec"), namespace)
    return namespace["run"]()


def _assert_reference_conversation(call, expected_text):
    conversations, kwargs = call
    assert kwargs == {"add_generation_prompt": True, "tokenize": False}
    assert len(conversations) == 1
    conversation = conversations[0]
    assert [message["role"] for message in conversation] == ["system", "user"]
    assert conversation[0]["content"] == [{"type": "text", "text": "Represent the user's input."}]
    assert conversation[1]["content"] == [{"type": "text", "text": expected_text}]


def test_fresh_serialized_for_inference_restores_uembed_format_for_all_modes(
    tmp_path, wiring, splade, pooling
):
    checkpoint = tmp_path / "serialized-uembed.pt"
    _write_serialized_chain(checkpoint, sparse = True)

    model = _inference_branch(
        wiring, pooling, _constructor(wiring, splade), checkpoint, splade = splade
    )
    processor = model[0].processor

    dense = model.encode({"text": "dense-input"}, output_mode = "dense")
    sparse = model.encode({"text": "sparse-input"}, output_mode = "sparse")
    both = model.encode({"text": "both-input"}, output_mode = "both")

    assert dense.shape == (4,)
    assert sparse.shape == (12,)
    assert set(both) == {"sentence_embedding", "sparse_embedding"}
    assert len(processor.template_calls) == 3
    for call, expected in zip(
        processor.template_calls,
        ["dense-input", "sparse-input", "both-input"],
        strict = True,
    ):
        _assert_reference_conversation(call, expected)
    assert model[0].plain_calls == []

    patched = model[0].preprocess
    assert wiring.restore_uembed_inference_input_format(model) is False
    assert model[0].preprocess is patched
    assert len(model) == 4


def test_fresh_public_uembed_for_inference_assembles_reference_semantics(
    tmp_path, wiring, splade, pooling
):
    checkpoint = tmp_path / "fresh-public-uembed.pt"
    _write_serialized_chain(checkpoint, sparse = True)

    model = _inference_branch(wiring, pooling, _fresh_hf_constructor, checkpoint, splade = splade)

    assert [type(module).__name__ for module in model] == [
        "_Transformer",
        "OffsetLastTokenPooling",
        "_Normalize",
        "UEmbedSparseOutput",
    ]
    assert model[1].num_eos_tokens == 2
    assert model[0].processor.tokenizer.padding_side == "right"
    assert model[0].processor.tokenizer._tokenizer.post_processor is not None
    assert model[0]._unsloth_uembed_input_format is True

    before = model[0].forward_calls
    both = model.encode({"text": "fresh-hf"}, output_mode = "both")
    assert model[0].forward_calls == before + 1
    assert both["sentence_embedding"].shape == (4,)
    assert both["sparse_embedding"].shape == (12,)
    assert model[0].processor.events == ["format", "tokenize"]
    _assert_reference_conversation(model[0].processor.template_calls[0], "fresh-hf")


def test_dense_only_for_inference_keeps_stock_format_and_signature(
    tmp_path, wiring, splade, pooling
):
    checkpoint = tmp_path / "serialized-dense.pt"
    _write_serialized_chain(checkpoint, sparse = False)

    model = _inference_branch(
        wiring, pooling, _constructor(wiring, splade), checkpoint, splade = splade
    )
    signature = inspect.signature(model.encode)

    dense = model.encode("plain-input")

    assert dense.shape == (4,)
    assert model[0].plain_calls == [["plain-input"]]
    assert model[0].processor.template_calls == []
    assert wiring.restore_uembed_inference_input_format(model) is False
    assert inspect.signature(model.encode) == signature
    with pytest.raises(TypeError):
        model.encode("plain-input", output_mode = "sparse")


_NativeEntryNotFoundError = _native_hub_errors.EntryNotFoundError
_NativeLocalEntryNotFoundError = _native_hub_errors.LocalEntryNotFoundError
_NativeRemoteEntryNotFoundError = getattr(
    _native_hub_errors, "RemoteEntryNotFoundError", _NativeEntryNotFoundError
)


def _native_local_entry_not_found():
    exception = _NativeLocalEntryNotFoundError("offline and not cached")
    assert isinstance(exception, _NativeEntryNotFoundError)
    assert not isinstance(exception, _NativeRemoteEntryNotFoundError) or (
        _NativeRemoteEntryNotFoundError is _NativeEntryNotFoundError
    )
    return exception


def _native_remote_entry_not_found():
    if _NativeRemoteEntryNotFoundError is _NativeEntryNotFoundError:
        exception = _NativeEntryNotFoundError("remote sidecar does not exist")
    else:
        import httpx
        response = httpx.Response(
            404,
            request = httpx.Request("GET", "https://huggingface.co/org/model/resolve/main/sidecar"),
        )
        exception = _NativeRemoteEntryNotFoundError(
            "remote sidecar does not exist", response = response
        )
    assert isinstance(exception, _NativeEntryNotFoundError)
    assert not isinstance(exception, _NativeLocalEntryNotFoundError)
    return exception


class _GatedRepoError(Exception):
    pass


class _RepositoryNotFoundError(Exception):
    pass


@pytest.fixture
def fake_hub(monkeypatch):
    errors = types.ModuleType("huggingface_hub.errors")
    errors.EntryNotFoundError = _NativeEntryNotFoundError
    errors.LocalEntryNotFoundError = _NativeLocalEntryNotFoundError
    if hasattr(_native_hub_errors, "RemoteEntryNotFoundError"):
        errors.RemoteEntryNotFoundError = _NativeRemoteEntryNotFoundError
    errors.GatedRepoError = _GatedRepoError
    errors.RepositoryNotFoundError = _RepositoryNotFoundError
    hub = types.ModuleType("huggingface_hub")
    hub.errors = errors
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors)

    def inject(exception):
        def download(*args, **kwargs):
            raise exception

        hub.hf_hub_download = download

    return inject


@pytest.mark.parametrize(
    "exception_type",
    [TimeoutError, _GatedRepoError, _RepositoryNotFoundError],
)
def test_metadata_hub_operational_failures_raise_unsloth(pooling, fake_hub, exception_type):
    failure = exception_type("metadata unavailable")
    fake_hub(failure)

    with pytest.raises(RuntimeError, match = r"^Unsloth:") as error:
        pooling.read_num_eos_tokens("org/model")

    assert error.value.__cause__ is failure
    assert "sparse_info.json" in str(error.value)


def test_metadata_remote_entry_not_found_is_optional_absence(pooling, fake_hub):
    fake_hub(_native_remote_entry_not_found())

    assert pooling.read_num_eos_tokens("org/dense-model") == 0


def test_fresh_inference_assembly_propagates_native_local_metadata_failure(
    pooling, wiring, splade, fake_hub
):
    failure = _native_local_entry_not_found()
    fake_hub(failure)
    assemble = _load_inference_assembly(
        wiring, splade, pooling, read_metadata = pooling.read_num_eos_tokens
    )
    model = _fresh_hf_constructor("unused")

    with pytest.raises(RuntimeError, match = r"^Unsloth:.*sparse_info\.json") as error:
        assemble(model, "org/uembed", _Pooling, _Normalize)

    assert error.value.__cause__ is failure
    assert [type(module).__name__ for module in model] == ["_Transformer", "_Pooling"]


def test_fresh_inference_assembly_propagates_native_local_sidecar_failure(
    pooling, wiring, splade, fake_hub
):
    failure = _native_local_entry_not_found()
    fake_hub(failure)
    assemble = _load_inference_assembly(
        wiring,
        splade,
        pooling,
        read_metadata = lambda *args, **kwargs: 2,
        attach_checkpoint = wiring.attach_uembed_sparse_checkpoint,
    )
    model = _fresh_hf_constructor("unused")

    with pytest.raises(RuntimeError, match = r"^Unsloth:.*sparse_weights\.pt") as error:
        assemble(model, "org/uembed", _Pooling, _Normalize)

    assert error.value.__cause__ is failure


def test_native_local_entry_not_found_is_loud_for_metadata_and_sparse(pooling, wiring, fake_hub):
    metadata_failure = _native_local_entry_not_found()
    fake_hub(metadata_failure)
    with pytest.raises(RuntimeError, match = r"^Unsloth:.*sparse_info\.json") as metadata_error:
        pooling.read_num_eos_tokens("org/model")
    assert metadata_error.value.__cause__ is metadata_failure

    sparse_failure = _native_local_entry_not_found()
    fake_hub(sparse_failure)
    with pytest.raises(RuntimeError, match = r"^Unsloth:.*sparse_weights\.pt") as sparse_error:
        wiring._resolve_sparse_weights_dir("org/model")
    assert sparse_error.value.__cause__ is sparse_failure


@pytest.mark.parametrize(
    "exception_type",
    [TimeoutError, _GatedRepoError, _RepositoryNotFoundError],
)
def test_sparse_weight_hub_operational_failures_raise_unsloth(wiring, fake_hub, exception_type):
    failure = exception_type("weights unavailable")
    fake_hub(failure)

    with pytest.raises(RuntimeError, match = r"^Unsloth:") as error:
        wiring._resolve_sparse_weights_dir("org/model")

    assert error.value.__cause__ is failure
    assert "sparse_weights.pt" in str(error.value)


def test_sparse_weight_remote_entry_not_found_is_optional_for_dense_checkpoint(wiring, fake_hub):
    fake_hub(_native_remote_entry_not_found())

    assert wiring._resolve_sparse_weights_dir("org/dense-model") is None


def test_positive_uembed_metadata_cannot_fall_back_to_dense_only(wiring, fake_hub):
    fake_hub(_native_remote_entry_not_found())

    with pytest.raises(RuntimeError, match = r"^Unsloth:.*sparse_weights\.pt"):
        wiring.attach_uembed_sparse_checkpoint(nn.Sequential(), "org/uembed", num_eos_tokens = 2)


def test_native_st_subfolder_load_contract_reconstructs_serialized_modules(
    tmp_path, pooling, wiring, splade
):
    pooling_dir = tmp_path / "1_OffsetLastTokenPooling"
    sparse_dir = tmp_path / "3_UEmbedSparseOutput"
    expected_pooling = pooling.OffsetLastTokenPooling(4, num_eos_tokens = 2)
    expected_sparse = wiring.UEmbedSparseOutput(
        splade.SpladeHead(
            [torch.ones(3, 4), torch.ones(5, 4)],
            [torch.zeros(3), torch.zeros(5)],
            num_eos_tokens = 2,
        )
    )
    expected_pooling.save(str(pooling_dir))
    expected_sparse.save(str(sparse_dir))
    # A root config must never be mistaken for either module's subfolder config.
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))

    loaded_pooling = pooling.OffsetLastTokenPooling.load(str(tmp_path), subfolder = pooling_dir.name)
    loaded_sparse = wiring.UEmbedSparseOutput.load(str(tmp_path), subfolder = sparse_dir.name)

    assert loaded_pooling.get_config_dict() == expected_pooling.get_config_dict()
    assert loaded_sparse.get_config_dict() == expected_sparse.get_config_dict()
    assert torch.equal(
        loaded_sparse.head.sparse_lm_heads[0], expected_sparse.head.sparse_lm_heads[0]
    )


def test_direct_local_checkpoint_without_optional_files_stays_absent(tmp_path, pooling, wiring):
    assert pooling.read_num_eos_tokens(str(tmp_path)) == 0
    assert wiring._resolve_sparse_weights_dir(str(tmp_path)) is None


def test_positive_local_metadata_without_sparse_weights_fails_loudly(tmp_path, pooling, wiring):
    (tmp_path / "sparse_info.json").write_text(json.dumps({"num_eos_tokens": 2}), encoding = "utf-8")
    count = pooling.read_num_eos_tokens(str(tmp_path))

    with pytest.raises(RuntimeError, match = r"^Unsloth:.*sparse_weights\.pt"):
        wiring.attach_uembed_sparse_checkpoint(nn.Sequential(), str(tmp_path), num_eos_tokens = count)
