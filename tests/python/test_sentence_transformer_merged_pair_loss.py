# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Opt-in merged MNRL must preserve the stock loss and optimizer update."""

import copy
import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn


pytest.importorskip("sentence_transformers")
SOURCE = Path(__file__).resolve().parents[2] / "unsloth/models/sentence_transformer_loss.py"
SPEC = importlib.util.spec_from_file_location("st_merged_loss_under_test", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(64, 8)
        self.projection = nn.Linear(8, 8)

    def forward(self, features):
        values = self.embedding(features["input_ids"])
        mask = features["attention_mask"].unsqueeze(-1)
        pooled = (values * mask).sum(1) / mask.sum(1)
        return {"sentence_embedding": self.projection(pooled)}


class MixedDtypeTinyEncoder(TinyEncoder):
    def __init__(self):
        super().__init__()
        self.projection.to(torch.bfloat16)

    def forward(self, features):
        values = self.embedding(features["input_ids"])
        mask = features["attention_mask"].unsqueeze(-1)
        pooled = (values * mask).sum(1) / mask.sum(1)
        return {"sentence_embedding": self.projection(pooled.to(torch.bfloat16))}


class BatchNormTinyEncoder(TinyEncoder):
    def __init__(self):
        super().__init__()
        self.normalizer = nn.BatchNorm1d(8)

    def forward(self, features):
        values = self.embedding(features["input_ids"])
        mask = features["attention_mask"].unsqueeze(-1)
        pooled = (values * mask).sum(1) / mask.sum(1)
        return {"sentence_embedding": self.projection(self.normalizer(pooled))}


def pair_features(task_mismatch = False):
    left = {
        "input_ids": torch.tensor([[1, 2, 3, 0], [4, 5, 6, 7], [8, 9, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 0, 0]]),
    }
    right = {
        "input_ids": torch.tensor([[3, 2, 1, 10, 0, 0], [7, 6, 5, 4, 11, 12], [9, 8, 13, 0, 0, 0]]),
        "attention_mask": torch.tensor(
            [[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]]
        ),
    }
    if task_mismatch:
        left["task"] = "query"
        right["task"] = "document"
    return left, right


@pytest.mark.parametrize("task_mismatch", [False, True])
def test_loss_gradients_and_adamw_update(task_mismatch):
    try:
        from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
    except ImportError:
        from sentence_transformers.losses import MultipleNegativesRankingLoss

    torch.manual_seed(4460)
    reference_model = TinyEncoder()
    candidate_model = copy.deepcopy(reference_model)
    features = pair_features(task_mismatch)
    reference = MultipleNegativesRankingLoss(reference_model)
    candidate = MODULE.FastMultipleNegativesRankingLoss(candidate_model)
    ref_loss = reference([dict(x) for x in features], None)
    got_loss = candidate([dict(x) for x in features], None)
    torch.testing.assert_close(got_loss, ref_loss, atol = 1e-6, rtol = 1e-6)
    ref_loss.backward()
    got_loss.backward()
    for a, b in zip(reference_model.parameters(), candidate_model.parameters()):
        torch.testing.assert_close(b.grad, a.grad, atol = 1e-6, rtol = 1e-6)
    ref_opt = torch.optim.AdamW(reference_model.parameters(), lr = 1e-3)
    got_opt = torch.optim.AdamW(candidate_model.parameters(), lr = 1e-3)
    ref_opt.step()
    got_opt.step()
    for a, b in zip(reference_model.parameters(), candidate_model.parameters()):
        torch.testing.assert_close(b, a, atol = 1e-6, rtol = 1e-6)
    assert (candidate.merged_calls, candidate.fallback_calls) == (
        (0, 1) if task_mismatch or MODULE.merge_feature_batches is None else (1, 0)
    )


def test_older_sentence_transformers_helper_falls_back(monkeypatch):
    monkeypatch.setattr(MODULE, "merge_feature_batches", None)
    candidate = MODULE.FastMultipleNegativesRankingLoss(TinyEncoder())
    loss = candidate(list(pair_features()), None)
    assert torch.isfinite(loss)
    assert (candidate.merged_calls, candidate.fallback_calls) == (0, 1)


def test_wrapper_can_disable_column_merging():
    if MODULE.merge_feature_batches is None:
        pytest.skip("current SentenceTransformers merged-feature helper is unavailable")
    from sentence_transformers.base.losses.merged_forward import column_merging_disabled

    candidate = MODULE.FastMultipleNegativesRankingLoss(TinyEncoder())
    with column_merging_disabled():
        loss = candidate(list(pair_features()), None)
    assert torch.isfinite(loss)
    assert (candidate.merged_calls, candidate.fallback_calls) == (0, 1)


@pytest.mark.parametrize("case", ["packed", "hard_negative", "prompt_mismatch"])
def test_ineligible_features_use_reference_loss(case):
    try:
        from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
    except ImportError:
        from sentence_transformers.losses import MultipleNegativesRankingLoss

    torch.manual_seed(4460)
    reference_model = TinyEncoder()
    candidate_model = copy.deepcopy(reference_model)
    features = list(pair_features())
    if case == "packed":
        for column in features:
            column["cu_seq_lens_q"] = torch.tensor([0, 2, 4, 6])
    elif case == "hard_negative":
        features.append({key: value.clone() for key, value in features[1].items()})
        features[-1]["input_ids"] = features[-1]["input_ids"].roll(1, dims = 0)
    else:
        features[0]["prompt_length"] = 1
        features[1]["prompt_length"] = 2
    reference = MultipleNegativesRankingLoss(reference_model)
    candidate = MODULE.FastMultipleNegativesRankingLoss(candidate_model)
    expected = reference([dict(column) for column in features], None)
    actual = candidate([dict(column) for column in features], None)
    torch.testing.assert_close(actual, expected, atol = 1e-6, rtol = 1e-6)
    actual.backward()
    assert all(parameter.grad is not None for parameter in candidate_model.parameters())
    assert (candidate.merged_calls, candidate.fallback_calls) == (0, 1)


def test_distributed_gather_option_keeps_stock_forward():
    candidate = MODULE.FastMultipleNegativesRankingLoss(TinyEncoder(), gather_across_devices = True)
    loss = candidate(list(pair_features()), None)
    assert torch.isfinite(loss)
    assert (candidate.merged_calls, candidate.fallback_calls) == (0, 1)


def test_long_asymmetric_columns_keep_stock_forward():
    features = list(pair_features())
    features[0]["input_ids"] = torch.nn.functional.pad(features[0]["input_ids"], (0, 40))
    features[0]["attention_mask"] = torch.nn.functional.pad(features[0]["attention_mask"], (0, 40))
    features[1]["input_ids"] = torch.nn.functional.pad(features[1]["input_ids"], (0, 186))
    features[1]["attention_mask"] = torch.nn.functional.pad(features[1]["attention_mask"], (0, 186))
    candidate = MODULE.FastMultipleNegativesRankingLoss(TinyEncoder())
    loss = candidate(features, None)
    assert torch.isfinite(loss)
    assert (candidate.merged_calls, candidate.fallback_calls) == (0, 1)


def test_autocast_keeps_stock_forward():
    candidate = MODULE.FastMultipleNegativesRankingLoss(TinyEncoder())
    with torch.autocast("cpu", dtype = torch.bfloat16):
        loss = candidate(list(pair_features()), None)
    assert torch.isfinite(loss)
    assert (candidate.merged_calls, candidate.fallback_calls) == (0, 1)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_non_fp32_weights_keep_stock_forward(dtype):
    candidate = MODULE.FastMultipleNegativesRankingLoss(TinyEncoder().to(dtype))
    loss = candidate(list(pair_features()), None)
    assert torch.isfinite(loss)
    assert (candidate.merged_calls, candidate.fallback_calls) == (0, 1)


def test_mixed_dtype_weights_keep_stock_forward():
    model = MixedDtypeTinyEncoder()
    assert next(model.parameters()).dtype == torch.float32
    candidate = MODULE.FastMultipleNegativesRankingLoss(model)
    loss = candidate(list(pair_features()), None)
    assert torch.isfinite(loss)
    assert (candidate.merged_calls, candidate.fallback_calls) == (0, 1)


def test_batch_norm_keeps_stock_running_statistics():
    try:
        from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
    except ImportError:
        from sentence_transformers.losses import MultipleNegativesRankingLoss

    torch.manual_seed(4460)
    reference_model = BatchNormTinyEncoder()
    candidate_model = copy.deepcopy(reference_model)
    reference = MultipleNegativesRankingLoss(reference_model)
    candidate = MODULE.FastMultipleNegativesRankingLoss(candidate_model)
    expected = reference(list(pair_features()), None)
    actual = candidate(list(pair_features()), None)
    torch.testing.assert_close(actual, expected, atol = 1e-6, rtol = 1e-6)
    expected.backward()
    actual.backward()
    for reference_parameter, candidate_parameter in zip(
        reference_model.parameters(), candidate_model.parameters()
    ):
        torch.testing.assert_close(
            candidate_parameter.grad, reference_parameter.grad, atol = 1e-6, rtol = 1e-6
        )
    reference_optimizer = torch.optim.AdamW(reference_model.parameters(), lr = 1e-3)
    candidate_optimizer = torch.optim.AdamW(candidate_model.parameters(), lr = 1e-3)
    reference_optimizer.step()
    candidate_optimizer.step()
    for reference_parameter, candidate_parameter in zip(
        reference_model.parameters(), candidate_model.parameters()
    ):
        torch.testing.assert_close(candidate_parameter, reference_parameter, atol = 1e-6, rtol = 1e-6)
    for name, buffer in reference_model.named_buffers():
        torch.testing.assert_close(dict(candidate_model.named_buffers())[name], buffer)
    assert candidate_model.normalizer.num_batches_tracked == 2
    assert (candidate.merged_calls, candidate.fallback_calls) == (0, 1)


def test_batch_norm_added_after_loss_construction_keeps_stock_semantics():
    torch.manual_seed(4460)
    reference_model = TinyEncoder()
    candidate_model = copy.deepcopy(reference_model)
    reference = MODULE.MultipleNegativesRankingLoss(reference_model)
    candidate = MODULE.FastMultipleNegativesRankingLoss(candidate_model)
    reference_model.projection = nn.Sequential(reference_model.projection, nn.BatchNorm1d(8))
    candidate_model.projection = nn.Sequential(candidate_model.projection, nn.BatchNorm1d(8))
    expected = reference(list(pair_features()), None)
    actual = candidate(list(pair_features()), None)
    torch.testing.assert_close(actual, expected, atol = 1e-6, rtol = 1e-6)
    expected.backward()
    actual.backward()
    for left, right in zip(reference_model.parameters(), candidate_model.parameters()):
        torch.testing.assert_close(right.grad, left.grad, atol = 1e-6, rtol = 1e-6)
    for name, buffer in reference_model.named_buffers():
        torch.testing.assert_close(dict(candidate_model.named_buffers())[name], buffer)
    assert candidate_model.projection[1].num_batches_tracked == 2
    assert (candidate.merged_calls, candidate.fallback_calls) == (0, 1)
    torch.optim.AdamW(reference_model.parameters(), lr = 1e-3).step()
    torch.optim.AdamW(candidate_model.parameters(), lr = 1e-3).step()
    for left, right in zip(reference_model.parameters(), candidate_model.parameters()):
        torch.testing.assert_close(right, left, atol = 1e-6, rtol = 1e-6)


@pytest.mark.parametrize(
    "options",
    [
        {"directions": ("query_to_doc", "doc_to_query")},
        {"directions": ("query_to_doc", "query_to_query", "doc_to_query", "doc_to_doc")},
        {"hardness_mode": "in_batch_negatives", "hardness_strength": 5.0},
    ],
)
def test_optional_global_negative_modes_preserve_loss_and_gradients(options):
    if MODULE.merge_feature_batches is None:
        pytest.skip("current SentenceTransformers merged-feature helper is unavailable")
    from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss

    torch.manual_seed(4460)
    reference_model = TinyEncoder()
    candidate_model = copy.deepcopy(reference_model)
    reference = MultipleNegativesRankingLoss(reference_model, **options)
    candidate = MODULE.FastMultipleNegativesRankingLoss(candidate_model, **options)
    expected = reference([dict(column) for column in pair_features()], None)
    actual = candidate([dict(column) for column in pair_features()], None)
    torch.testing.assert_close(actual, expected, atol = 1e-6, rtol = 1e-6)
    expected.backward()
    actual.backward()
    for reference_parameter, candidate_parameter in zip(
        reference_model.parameters(), candidate_model.parameters()
    ):
        torch.testing.assert_close(
            candidate_parameter.grad, reference_parameter.grad, atol = 1e-6, rtol = 1e-6
        )
    assert (candidate.merged_calls, candidate.fallback_calls) == (1, 0)
