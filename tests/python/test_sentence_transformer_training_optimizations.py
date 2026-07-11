"""CPU regressions for the SentenceTransformer training fast paths.

The kernel module is loaded under a private package with only its AMP decorator
dependencies stubbed.  This keeps the tests independent of Unsloth's CUDA-time
package initialization and avoids model or tokenizer downloads.
"""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import types
from typing import Optional

import pytest


torch = pytest.importorskip("torch")
F = torch.nn.functional

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONTRASTIVE_LOSS_PATH = _REPO_ROOT / "unsloth" / "kernels" / "contrastive_loss.py"
_SENTENCE_TRANSFORMER_PATH = _REPO_ROOT / "unsloth" / "models" / "sentence_transformer.py"


@pytest.fixture(scope = "module")
def contrastive_module():
    package_name = "_unsloth_sentence_optimization_test"
    kernels_name = f"{package_name}.kernels"
    utils_name = f"{kernels_name}.utils"
    module_name = f"{kernels_name}.contrastive_loss"
    shadowed_names = (package_name, kernels_name, utils_name, module_name)
    saved_modules = {name: sys.modules.get(name) for name in shadowed_names}

    package = types.ModuleType(package_name)
    package.__path__ = []
    kernels = types.ModuleType(kernels_name)
    kernels.__path__ = []
    utils = types.ModuleType(utils_name)

    def identity_decorator(function):
        return function

    utils.torch_amp_custom_fwd = identity_decorator
    utils.torch_amp_custom_bwd = identity_decorator
    sys.modules[package_name] = package
    sys.modules[kernels_name] = kernels
    sys.modules[utils_name] = utils

    spec = importlib.util.spec_from_file_location(module_name, _CONTRASTIVE_LOSS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    try:
        yield module
    finally:
        for name, saved in saved_modules.items():
            if saved is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved


class Transformer(torch.nn.Module):
    def __init__(
        self,
        *,
        batch_norm = False,
        compiled = False,
        model_type = "bert",
    ):
        super().__init__()
        self.embedding = torch.nn.Embedding(32, 4).double()
        with torch.no_grad():
            values = torch.arange(32 * 4, dtype = torch.float64).reshape(32, 4)
            self.embedding.weight.copy_((values - 64) / 16)
        if batch_norm:
            # Merely containing BatchNorm is enough to make row concatenation
            # unsafe while training, even when it is nested in an allowed module.
            self.batch_norm = torch.nn.BatchNorm1d(4).double()
        self.auto_model = types.SimpleNamespace(config = types.SimpleNamespace(model_type = model_type))
        if compiled:
            self.auto_model._orig_mod = object()

    def forward(self, features):
        result = dict(features)
        result["token_embeddings"] = self.embedding(features["input_ids"])
        return result


class Pooling(torch.nn.Module):
    def forward(self, features):
        result = dict(features)
        mask = features["attention_mask"].unsqueeze(-1).to(features["token_embeddings"].dtype)
        result["sentence_embedding"] = (features["token_embeddings"] * mask).sum(dim = 1) / mask.sum(
            dim = 1
        ).clamp_min(1)
        return result


class Dense(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 3).double()
        with torch.no_grad():
            self.linear.weight.copy_(
                torch.tensor(
                    (
                        (0.5, -0.25, 0.125, 1.0),
                        (-0.5, 0.75, 0.25, -0.125),
                        (1.0, 0.5, -0.5, 0.25),
                    ),
                    dtype = torch.float64,
                )
            )
            self.linear.bias.copy_(torch.tensor((0.25, -0.5, 0.125), dtype = torch.float64))

    def forward(self, features):
        result = dict(features)
        result["sentence_embedding"] = self.linear(features["sentence_embedding"])
        return result


class CustomModule(torch.nn.Module):
    def forward(self, features):
        return features


class ToySentenceTransformer(torch.nn.Module):
    def __init__(
        self,
        *,
        batch_norm = False,
        custom = False,
        compiled = False,
        model_type = "bert",
    ):
        super().__init__()
        self.tokenizer = types.SimpleNamespace(pad_token_id = 7)
        self.add_module(
            "0",
            Transformer(
                batch_norm = batch_norm,
                compiled = compiled,
                model_type = model_type,
            ),
        )
        self.add_module("1", Pooling())
        if custom:
            self.add_module("custom", CustomModule())
        self.add_module("2", Dense())
        self.forward_batches = []

    def forward(self, features):
        self.forward_batches.append(
            {
                "input_ids": features["input_ids"].detach().clone(),
                "attention_mask": features["attention_mask"].detach().clone(),
            }
        )
        result = features
        for module in self._modules.values():
            result = module(result)
        return result


def _clone_features(features):
    return {
        key: value.clone() if torch.is_tensor(value) else value for key, value in features.items()
    }


def _feature_batch(
    batch_size,
    sequence_length,
    *,
    prompt_length = None,
):
    input_ids = torch.arange(1, batch_size * sequence_length + 1, dtype = torch.long)
    input_ids = input_ids.reshape(batch_size, sequence_length).remainder(24).add(1)
    features = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
    }
    if prompt_length is not None:
        features["prompt_length"] = torch.full((batch_size,), prompt_length, dtype = torch.long)
    return features


def _assert_falls_back(contrastive_module, model, sentence_features):
    outputs = contrastive_module.encode_sentence_features(
        model, [_clone_features(features) for features in sentence_features]
    )
    assert len(outputs) == len(sentence_features)
    assert len(model.forward_batches) == len(sentence_features)
    assert [tuple(call["input_ids"].shape) for call in model.forward_batches] == [
        tuple(features["input_ids"].shape) for features in sentence_features
    ]


def test_combined_feature_batching_matches_separate_outputs_and_gradients_exactly(
    contrastive_module, monkeypatch
):
    sentence_features = [
        {
            "input_ids": torch.tensor(((1, 2, 0), (3, 4, 0)), dtype = torch.long),
            "attention_mask": torch.tensor(((1, 1, 0), (1, 1, 0)), dtype = torch.long),
        },
        {
            "input_ids": torch.tensor(
                ((5, 6, 8, 9), (10, 11, 0, 0)),
                dtype = torch.long,
            ),
            "attention_mask": torch.tensor(
                ((1, 1, 1, 1), (1, 1, 0, 0)),
                dtype = torch.long,
            ),
        },
    ]
    separate_model = ToySentenceTransformer()
    combined_model = copy.deepcopy(separate_model)

    expected = [
        separate_model(_clone_features(features))["sentence_embedding"]
        for features in sentence_features
    ]

    def unexpected_row_plan(*_args, **_kwargs):
        raise AssertionError("single-bucket concatenation must not build a row plan")

    monkeypatch.setattr(contrastive_module, "_prepare_bucket_row_plan", unexpected_row_plan)
    actual = contrastive_module.encode_sentence_features(
        combined_model, [_clone_features(features) for features in sentence_features]
    )

    assert len(separate_model.forward_batches) == 2
    assert len(combined_model.forward_batches) == 1
    combined_call = combined_model.forward_batches[0]
    assert tuple(combined_call["input_ids"].shape) == (4, 4)
    torch.testing.assert_close(
        combined_call["input_ids"][:2, 3], torch.tensor((7, 7)), rtol = 0, atol = 0
    )
    torch.testing.assert_close(
        combined_call["attention_mask"][:2, 3],
        torch.zeros(2, dtype = torch.long),
        rtol = 0,
        atol = 0,
    )
    for expected_part, actual_part in zip(expected, actual):
        torch.testing.assert_close(actual_part, expected_part, rtol = 0, atol = 0)

    torch.cat(expected, dim = 0).sum().backward()
    torch.cat(actual, dim = 0).sum().backward()
    separate_parameters = dict(separate_model.named_parameters())
    combined_parameters = dict(combined_model.named_parameters())
    assert separate_parameters.keys() == combined_parameters.keys()
    for name in separate_parameters:
        expected_grad = separate_parameters[name].grad
        actual_grad = combined_parameters[name].grad
        assert expected_grad is not None and actual_grad is not None
        torch.testing.assert_close(actual_grad, expected_grad, rtol = 0, atol = 0)


def test_feature_batching_falls_back_for_different_prompt_lengths(contrastive_module):
    features = [
        _feature_batch(2, 4, prompt_length = 1),
        _feature_batch(2, 4, prompt_length = 2),
    ]
    _assert_falls_back(contrastive_module, ToySentenceTransformer(), features)


def test_feature_batching_falls_back_for_unequal_column_cardinality(contrastive_module):
    features = (_feature_batch(4, 4), _feature_batch(2, 4))
    _assert_falls_back(contrastive_module, ToySentenceTransformer(), features)


@pytest.mark.parametrize(
    "model",
    (
        pytest.param(ToySentenceTransformer(batch_norm = True), id = "nested-batchnorm"),
        pytest.param(ToySentenceTransformer(custom = True), id = "custom-module"),
    ),
)
def test_feature_batching_falls_back_for_batch_dependent_or_custom_modules(
    contrastive_module, model
):
    _assert_falls_back(
        contrastive_module,
        model,
        (_feature_batch(2, 4), _feature_batch(2, 4)),
    )


def test_feature_batching_uses_two_optimal_buckets_for_dissimilar_lengths(contrastive_module):
    features = (_feature_batch(2, 2), _feature_batch(2, 8))
    model = ToySentenceTransformer()

    outputs = contrastive_module.encode_sentence_features(
        model, [_clone_features(features) for features in features]
    )

    assert len(outputs) == 2
    assert [tuple(call["input_ids"].shape) for call in model.forward_batches] == [(2, 2), (2, 8)]


def test_decoder_embedding_model_prefers_one_batch_within_double_padding_budget(contrastive_module):
    features = (_feature_batch(2, 2), _feature_batch(2, 8))
    model = ToySentenceTransformer(model_type = "qwen3")

    outputs = contrastive_module.encode_sentence_features(
        model, [_clone_features(feature) for feature in features]
    )

    assert len(outputs) == 2
    assert [tuple(call["input_ids"].shape) for call in model.forward_batches] == [(4, 8)]


def test_bucketed_feature_batching_restores_row_order_and_fp64_gradients_exactly(
    contrastive_module,
):
    sentence_features = [
        {
            "input_ids": torch.tensor(((1, 2, 3, 4, 5, 6, 7, 8), (9, 10, 0, 0, 0, 0, 0, 0))),
            "attention_mask": torch.tensor(((1, 1, 1, 1, 1, 1, 1, 1), (1, 1, 0, 0, 0, 0, 0, 0))),
        },
        {
            "input_ids": torch.tensor(((11, 12), (13, 14))),
            "attention_mask": torch.ones((2, 2), dtype = torch.long),
        },
    ]
    separate_model = ToySentenceTransformer()
    bucketed_model = copy.deepcopy(separate_model)

    expected = [
        separate_model(_clone_features(features))["sentence_embedding"]
        for features in sentence_features
    ]
    actual = contrastive_module.encode_sentence_features(
        bucketed_model, [_clone_features(features) for features in sentence_features]
    )

    assert [tuple(call["input_ids"].shape) for call in bucketed_model.forward_batches] == [
        (3, 2),
        (1, 8),
    ]
    torch.testing.assert_close(
        bucketed_model.forward_batches[0]["input_ids"],
        torch.tensor(((9, 10), (11, 12), (13, 14))),
        rtol = 0,
        atol = 0,
    )
    for expected_part, actual_part in zip(expected, actual):
        torch.testing.assert_close(actual_part, expected_part, rtol = 0, atol = 0)

    torch.cat(expected, dim = 0).sum().backward()
    torch.cat(actual, dim = 0).sum().backward()
    for (expected_name, expected_parameter), (actual_name, actual_parameter) in zip(
        separate_model.named_parameters(), bucketed_model.named_parameters()
    ):
        assert actual_name == expected_name
        assert expected_parameter.grad is not None and actual_parameter.grad is not None
        torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad, rtol = 0, atol = 0)


def test_bucket_row_plan_reuses_indices_across_feature_keys_and_preserves_gradients(
    contrastive_module,
):
    bucket_flat_indices = ((2, 0, 3), (1,))
    row_plan = contrastive_module._prepare_bucket_row_plan(
        bucket_flat_indices,
        batch_size = 2,
        column_count = 2,
    )
    first_padded = (
        torch.tensor(((10, 11, 12, 13), (20, 21, 0, 0))),
        torch.tensor(((30, 31), (40, 41))),
    )
    second_padded = tuple(value.ne(0).to(torch.long) for value in first_padded)

    first_buckets = contrastive_module._gather_padded_bucket_rows(
        first_padded, row_plan, bucket_lengths = (2, 4), pad_value = 7
    )
    second_buckets = contrastive_module._gather_padded_bucket_rows(
        second_padded, row_plan, bucket_lengths = (2, 4), pad_value = 0
    )

    torch.testing.assert_close(
        first_buckets[0],
        torch.tensor(((30, 31), (10, 11), (40, 41))),
        rtol = 0,
        atol = 0,
    )
    torch.testing.assert_close(first_buckets[1], torch.tensor(((20, 21, 0, 0),)), rtol = 0, atol = 0)
    torch.testing.assert_close(
        second_buckets[0], torch.ones((3, 2), dtype = torch.long), rtol = 0, atol = 0
    )
    assert len(row_plan["index_pool"]._by_device) == 1

    unpadded = (
        torch.tensor((1.0, 2.0), dtype = torch.float64, requires_grad = True),
        torch.tensor((3.0, 4.0), dtype = torch.float64, requires_grad = True),
    )
    gathered = contrastive_module._gather_unpadded_bucket_rows(unpadded, row_plan)
    torch.testing.assert_close(
        gathered[0],
        torch.tensor((3.0, 1.0, 4.0), dtype = torch.float64),
        rtol = 0,
        atol = 0,
    )
    torch.testing.assert_close(
        gathered[1], torch.tensor((2.0,), dtype = torch.float64), rtol = 0, atol = 0
    )
    (gathered[0] * torch.tensor((2.0, 3.0, 5.0))).sum().backward()
    torch.testing.assert_close(
        unpadded[0].grad,
        torch.tensor((3.0, 0.0), dtype = torch.float64),
        rtol = 0,
        atol = 0,
    )
    torch.testing.assert_close(
        unpadded[1].grad,
        torch.tensor((2.0, 5.0), dtype = torch.float64),
        rtol = 0,
        atol = 0,
    )


def test_feature_batching_materializes_generator_once(contrastive_module):
    features = (_feature_batch(2, 3), _feature_batch(2, 4))
    model = ToySentenceTransformer()
    yielded = []

    def feature_generator():
        for index, feature in enumerate(features):
            yielded.append(index)
            yield _clone_features(feature)

    outputs = contrastive_module.encode_sentence_features(model, feature_generator())

    assert yielded == [0, 1]
    assert len(outputs) == 2
    assert len(model.forward_batches) == 1


@pytest.mark.parametrize(
    "attention_mask",
    (
        pytest.param(torch.tensor(((0, 1, 1), (0, 1, 1))), id = "left-padded"),
        pytest.param(torch.tensor(((1, 0, 1), (1, 1, 0))), id = "mask-hole"),
    ),
)
def test_feature_batching_falls_back_for_non_right_padded_masks(contrastive_module, attention_mask):
    features = [_feature_batch(2, 3), _feature_batch(2, 3)]
    features[0]["attention_mask"] = attention_mask
    _assert_falls_back(contrastive_module, ToySentenceTransformer(), features)


def test_feature_batching_falls_back_for_unknown_sequence_feature(contrastive_module):
    features = [_feature_batch(2, 3), _feature_batch(2, 3)]
    for feature in features:
        feature["unknown_sequence"] = torch.ones_like(feature["input_ids"])
    _assert_falls_back(contrastive_module, ToySentenceTransformer(), features)


def test_feature_batching_falls_back_for_compiled_encoder_marker(contrastive_module):
    features = (_feature_batch(2, 4), _feature_batch(2, 4))
    _assert_falls_back(contrastive_module, ToySentenceTransformer(compiled = True), features)


def test_fused_contrastive_empty_batch_has_scalar_zero_and_zero_gradients(contrastive_module):
    anchors = torch.empty((0, 3), dtype = torch.float32, requires_grad = True)
    candidates = torch.arange(12, dtype = torch.float32).reshape(4, 3).requires_grad_()

    loss = contrastive_module.FusedContrastiveLoss.apply(anchors, candidates, 2.0)

    assert loss.shape == torch.Size([])
    assert loss.item() == 0.0
    loss.backward()
    assert anchors.grad is not None and anchors.grad.shape == anchors.shape
    assert candidates.grad is not None
    torch.testing.assert_close(candidates.grad, torch.zeros_like(candidates), rtol = 0, atol = 0)


@pytest.mark.parametrize(
    ("anchor_count", "candidate_count", "message"),
    (
        pytest.param(2, 0, "requires candidates", id = "missing-candidates"),
        pytest.param(3, 2, "requires B_a <= B_b", id = "too-few-candidates"),
    ),
)
def test_fused_contrastive_rejects_malformed_batch_cardinality(
    contrastive_module, anchor_count, candidate_count, message
):
    anchors = torch.zeros((anchor_count, 3), dtype = torch.float32, requires_grad = True)
    candidates = torch.zeros((candidate_count, 3), dtype = torch.float32, requires_grad = True)
    with pytest.raises(ValueError, match = message):
        contrastive_module.FusedContrastiveLoss.apply(anchors, candidates, 1.0)


def test_fused_contrastive_rejects_non_matrix_anchors(contrastive_module):
    anchors = torch.zeros(3, dtype = torch.float32, requires_grad = True)
    candidates = torch.zeros((3, 3), dtype = torch.float32, requires_grad = True)
    with pytest.raises(ValueError):
        contrastive_module.FusedContrastiveLoss.apply(anchors, candidates, 1.0)


@pytest.mark.parametrize("configured_chunk_size", ("0", "-3", "not-an-int"))
def test_fused_contrastive_sanitizes_invalid_chunk_size(
    contrastive_module, monkeypatch, configured_chunk_size
):
    monkeypatch.setenv("UNSLOTH_CONTRASTIVE_CHUNK_SIZE", configured_chunk_size)
    anchors = torch.eye(2, dtype = torch.float32, requires_grad = True)
    candidates = torch.eye(2, dtype = torch.float32, requires_grad = True)

    loss = contrastive_module.FusedContrastiveLoss.apply(anchors, candidates, 1.0)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(anchors.grad).all()
    assert torch.isfinite(candidates.grad).all()


def test_unpadding_flash_dispatch_requires_cuda_half_or_bfloat16():
    source = _SENTENCE_TRANSFORMER_PATH.read_text(encoding = "utf-8")
    tree = ast.parse(source, filename = str(_SENTENCE_TRANSFORMER_PATH))
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_can_use_flash_attn_varlen"
    )
    namespace = {"torch": torch, "_FLASH_ATTN_VARLEN_AVAILABLE": True}
    exec(
        compile(
            ast.Module(body = [helper], type_ignores = []), str(_SENTENCE_TRANSFORMER_PATH), "exec"
        ),
        namespace,
    )
    can_dispatch = namespace["_can_use_flash_attn_varlen"]

    assert not can_dispatch(types.SimpleNamespace(is_cuda = True, dtype = torch.float32))
    assert not can_dispatch(types.SimpleNamespace(is_cuda = False, dtype = torch.float16))
    assert can_dispatch(types.SimpleNamespace(is_cuda = True, dtype = torch.float16))
    assert '_use_varlen = _orig_attn_impl == "sdpa"' in source
    assert '"deberta-v2"' in source


def test_lora_fusion_rejects_trainable_bases_and_peft_variants():
    source = _SENTENCE_TRANSFORMER_PATH.read_text(encoding = "utf-8")
    assert 'getattr(base_weight, "requires_grad", False)' in source
    assert "adapter in lora_variant" in source


def test_fused_qkv_cache_requires_same_self_attention_input():
    tree = ast.parse(_SENTENCE_TRANSFORMER_PATH.read_text(encoding = "utf-8"))
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_same_projection_input"
    )
    namespace = {}
    exec(
        compile(
            ast.Module(body = [helper], type_ignores = []), str(_SENTENCE_TRANSFORMER_PATH), "exec"
        ),
        namespace,
    )
    same_input = namespace["_same_projection_input"]
    query_states = torch.randn(2, 3, 4)
    encoder_states = query_states.clone()

    assert same_input(query_states, query_states)
    assert not same_input(query_states, encoder_states)


def test_constructor_kwargs_are_collision_free_and_explicit_values_win():
    tree = ast.parse(_SENTENCE_TRANSFORMER_PATH.read_text(encoding = "utf-8"))
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_merge_constructor_kwargs"
    )
    namespace = {}
    exec(
        compile(
            ast.Module(body = [helper], type_ignores = []), str(_SENTENCE_TRANSFORMER_PATH), "exec"
        ),
        namespace,
    )
    merge_kwargs = namespace["_merge_constructor_kwargs"]

    captured = {}

    def constructor(**kwargs):
        captured.update(kwargs)

    constructor(
        **merge_kwargs(
            {"cache_folder": "stale", "custom_option": 7},
            {"cache_folder": "warmed-cache", "revision": "pinned"},
        )
    )
    assert captured == {"cache_folder": "warmed-cache", "custom_option": 7, "revision": "pinned"}


def test_transformers_v4_sdpa_patch_is_serialized():
    source = _SENTENCE_TRANSFORMER_PATH.read_text(encoding = "utf-8")
    assert "_SDPA_PATCH_LOCK = threading.RLock()" in source
    assert source.count("with _SDPA_PATCH_LOCK:") == 2


def test_compile_threshold_heuristic_has_stable_boundaries():
    tree = ast.parse(_SENTENCE_TRANSFORMER_PATH.read_text(encoding = "utf-8"))
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "FastSentenceTransformer"
    )
    helper = copy.deepcopy(
        next(
            node
            for node in owner.body
            if isinstance(node, ast.FunctionDef) and node.name == "_estimate_compile_threshold"
        )
    )
    helper.decorator_list = []
    namespace = {
        "FastSentenceTransformer": types.SimpleNamespace(ENCODER_MODEL_TYPES = {"bert", "mpnet"})
    }
    exec(
        compile(
            ast.Module(body = [helper], type_ignores = []), str(_SENTENCE_TRANSFORMER_PATH), "exec"
        ),
        namespace,
    )
    estimate = namespace["_estimate_compile_threshold"]

    def model(parameter_count, model_type):
        parameter = types.SimpleNamespace(numel = lambda: parameter_count)
        inner = types.SimpleNamespace(
            parameters = lambda: iter((parameter,)),
            config = types.SimpleNamespace(model_type = model_type),
        )
        return [types.SimpleNamespace(auto_model = inner)]

    small = estimate(model(49_000_000, "bert"), batch_size = 2, grad_accum = 4, max_seq_length = 512)
    medium = estimate(model(50_000_000, "bert"), batch_size = 2, grad_accum = 4, max_seq_length = 512)
    long_run = estimate(model(50_000_000, "bert"), batch_size = 8, grad_accum = 8, max_seq_length = 1024)
    assert small >= 20
    assert medium >= 20
    assert long_run <= medium


def test_fused_contrastive_fp32_matches_dense_loss_and_gradients_exactly(
    contrastive_module, monkeypatch
):
    monkeypatch.setenv("UNSLOTH_CONTRASTIVE_CHUNK_SIZE", "2")
    anchor_values = torch.tensor(((1.0, 0.0, 0.0, 0.0), (0.0, 2.0, 0.0, 0.0)), dtype = torch.float32)
    candidate_values = torch.tensor(
        (
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 2.0),
            (0.0, 0.0, 3.0, 0.0),
            (0.0, 0.0, 0.0, 4.0),
        ),
        dtype = torch.float32,
    )
    fused_anchors = anchor_values.clone().requires_grad_()
    fused_candidates = candidate_values.clone().requires_grad_()
    dense_anchors = anchor_values.clone().requires_grad_()
    dense_candidates = candidate_values.clone().requires_grad_()

    fused_loss = contrastive_module.FusedContrastiveLoss.apply(fused_anchors, fused_candidates, 2.0)
    dense_scores = (dense_anchors @ dense_candidates.t()) * 2.0
    dense_loss = F.cross_entropy(dense_scores, torch.arange(2))
    fused_loss.backward()
    dense_loss.backward()

    torch.testing.assert_close(fused_loss, dense_loss, rtol = 0, atol = 0)
    torch.testing.assert_close(fused_anchors.grad, dense_anchors.grad, rtol = 0, atol = 0)
    torch.testing.assert_close(fused_candidates.grad, dense_candidates.grad, rtol = 0, atol = 0)
    assert torch.count_nonzero(fused_anchors.grad) > 0
    assert torch.count_nonzero(fused_candidates.grad) > 0


@pytest.fixture
def patched_pooling_class():
    class IsolatedPooling(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pooling_mode = "max"
            self.include_prompt = True

        def forward(self, features):
            raise AssertionError("the efficient pooling patch was not installed")

    sentence_transformers = types.ModuleType("sentence_transformers")
    sentence_transformers.__path__ = []
    models = types.ModuleType("sentence_transformers.models")
    models.Pooling = IsolatedPooling
    sentence_transformers.models = models
    shadowed_names = ("sentence_transformers", "sentence_transformers.models")
    saved_modules = {name: sys.modules.get(name) for name in shadowed_names}
    sys.modules["sentence_transformers"] = sentence_transformers
    sys.modules["sentence_transformers.models"] = models

    source = _SENTENCE_TRANSFORMER_PATH.read_text(encoding = "utf-8")
    parsed = ast.parse(source, filename = str(_SENTENCE_TRANSFORMER_PATH))
    selected_nodes = []
    selected_assignments = {"_POOLING_MODE_FLAGS", "_POOLING_PATCHED"}
    selected_functions = {"_ensure_pooling_flags", "_patch_efficient_pooling"}
    for node in parsed.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            if any(
                isinstance(target, ast.Name) and target.id in selected_assignments
                for target in targets
            ):
                selected_nodes.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in selected_functions:
                selected_nodes.append(node)

    namespace = {"torch": torch}
    extracted = ast.Module(body = selected_nodes, type_ignores = [])
    exec(compile(extracted, str(_SENTENCE_TRANSFORMER_PATH), "exec"), namespace)
    namespace["_patch_efficient_pooling"]()
    assert namespace["_POOLING_PATCHED"] is True

    try:
        yield IsolatedPooling
    finally:
        for name, saved in saved_modules.items():
            if saved is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved


def test_efficient_fp16_max_pooling_masks_with_negative_infinity(patched_pooling_class):
    pooling = patched_pooling_class()
    token_embeddings = torch.tensor(
        (
            ((-5.0, -2.0), (4.0, -3.0), (60000.0, 60000.0)),
            ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0)),
        ),
        dtype = torch.float16,
        requires_grad = True,
    )
    original = token_embeddings.detach().clone()
    features = {
        "token_embeddings": token_embeddings,
        "attention_mask": torch.tensor(((1, 1, 0), (0, 0, 0)), dtype = torch.long),
    }

    sentence_embedding = pooling(features)["sentence_embedding"]

    torch.testing.assert_close(
        sentence_embedding[0],
        torch.tensor((4.0, -2.0), dtype = torch.float16),
        rtol = 0,
        atol = 0,
    )
    assert torch.isneginf(sentence_embedding[1]).all()
    torch.testing.assert_close(token_embeddings.detach(), original, rtol = 0, atol = 0)

    sentence_embedding[0].sum().backward()
    assert token_embeddings.grad is not None
    torch.testing.assert_close(
        token_embeddings.grad[0, 2],
        torch.zeros(2, dtype = torch.float16),
        rtol = 0,
        atol = 0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "requires CUDA Triton")
def test_fused_layernorm_mean_pool_cuda_matches_eager_forward_and_gradients():
    pytest.importorskip("triton")
    from unsloth.kernels.fused_pooling import fused_layernorm_mean_pool

    torch.manual_seed(0)
    device = torch.device("cuda")
    token_embeddings = torch.randn(2, 5, 16, device = device, dtype = torch.float16)
    attention_mask = torch.tensor(
        ((1, 1, 1, 0, 0), (1, 1, 1, 1, 0)), device = device, dtype = torch.long
    )
    fused_inputs = token_embeddings.detach().clone().requires_grad_()
    eager_inputs = token_embeddings.detach().clone().requires_grad_()
    fused_layernorm = torch.nn.LayerNorm(16, device = device, dtype = torch.float16)
    eager_layernorm = copy.deepcopy(fused_layernorm)

    fused_output = fused_layernorm_mean_pool(fused_layernorm, fused_inputs, attention_mask)
    eager_tokens = eager_layernorm(eager_inputs)
    eager_output = (eager_tokens * attention_mask.unsqueeze(-1).to(eager_tokens.dtype)).sum(
        dim = 1
    ) / attention_mask.sum(dim = 1, keepdim = True)
    fused_output.float().square().mean().backward()
    eager_output.float().square().mean().backward()

    torch.testing.assert_close(fused_output, eager_output, rtol = 2e-3, atol = 2e-3)
    torch.testing.assert_close(fused_inputs.grad, eager_inputs.grad, rtol = 3e-3, atol = 3e-3)
    torch.testing.assert_close(
        fused_layernorm.weight.grad, eager_layernorm.weight.grad, rtol = 3e-3, atol = 3e-3
    )
    torch.testing.assert_close(
        fused_layernorm.bias.grad, eager_layernorm.bias.grad, rtol = 3e-3, atol = 3e-3
    )


def test_guided_projection_repeated_save_removes_alternate_weight_format(tmp_path):
    pytest.importorskip("safetensors")

    source = _SENTENCE_TRANSFORMER_PATH.read_text(encoding = "utf-8")
    parsed = ast.parse(source, filename = str(_SENTENCE_TRANSFORMER_PATH))
    selected_nodes = [
        node
        for node in parsed.body
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
    extracted = ast.Module(body = selected_nodes, type_ignores = [])
    exec(compile(extracted, str(_SENTENCE_TRANSFORMER_PATH), "exec"), namespace)
    projection_cls = namespace["GuidedProjection"]
    pooling_cls = namespace["GuidedProjectionPooling"]

    class StubPooling(torch.nn.Module):
        def save(self, _output_path):
            pass

    projection = projection_cls(4, use_residual = False)
    module = pooling_cls(StubPooling(), projection)

    with torch.no_grad():
        projection.proj.weight.fill_(1.0)
    module.save(str(tmp_path), safe_serialization = True)
    assert (tmp_path / module.PROJECTION_SAFETENSORS_NAME).exists()
    assert not (tmp_path / module.PROJECTION_WEIGHTS_NAME).exists()

    with torch.no_grad():
        projection.proj.weight.fill_(2.0)
    module.save(str(tmp_path), safe_serialization = False)
    assert (tmp_path / module.PROJECTION_WEIGHTS_NAME).exists()
    assert not (tmp_path / module.PROJECTION_SAFETENSORS_NAME).exists()
    reloaded = pooling_cls.load(str(tmp_path), pooling_module = StubPooling())
    torch.testing.assert_close(
        reloaded.projection.proj.weight,
        torch.full_like(reloaded.projection.proj.weight, 2.0),
    )

    with torch.no_grad():
        projection.proj.weight.fill_(3.0)
    module.save(str(tmp_path), safe_serialization = True)
    assert (tmp_path / module.PROJECTION_SAFETENSORS_NAME).exists()
    assert not (tmp_path / module.PROJECTION_WEIGHTS_NAME).exists()
    reloaded = pooling_cls.load(str(tmp_path), pooling_module = StubPooling())
    torch.testing.assert_close(
        reloaded.projection.proj.weight,
        torch.full_like(reloaded.projection.proj.weight, 3.0),
    )


def test_combined_pooling_modes_and_observable_outputs_keep_parity(contrastive_module, monkeypatch):
    source = _SENTENCE_TRANSFORMER_PATH.read_text(encoding = "utf-8")
    tree = ast.parse(source, filename = str(_SENTENCE_TRANSFORMER_PATH))
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_ensure_pooling_flags"
    )
    namespace = {
        "_POOLING_MODE_FLAGS": {
            "cls": "pooling_mode_cls_token",
            "mean": "pooling_mode_mean_tokens",
        }
    }
    exec(
        compile(
            ast.Module(body = [helper], type_ignores = []), str(_SENTENCE_TRANSFORMER_PATH), "exec"
        ),
        namespace,
    )
    pooling = types.SimpleNamespace(pooling_mode = ["cls", "mean"])
    namespace["_ensure_pooling_flags"](pooling)
    assert pooling.pooling_mode_cls_token is True
    assert pooling.pooling_mode_mean_tokens is True

    seen = []

    def model(features):
        seen.append(features.get("_unsloth_sentence_embedding_only"))
        return {"sentence_embedding": torch.ones(1, 2)}

    monkeypatch.setattr(contrastive_module, "_bucketed_sentence_features", lambda *_: None)
    features = {"input_ids": torch.ones(1, 1, dtype = torch.long)}
    contrastive_module.encode_sentence_features(model, [features])
    assert seen == [True]
    assert "_unsloth_sentence_embedding_only" not in features

    assert 'features["token_embeddings"] = stored_ln(token_embeddings)' in source
    assert "if not sentence_embedding_only or bool(" in source
    assert "supports_flash_attn_2 and dtype in (torch.float16, torch.bfloat16)" in source
