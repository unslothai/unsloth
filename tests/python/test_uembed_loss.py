# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for `UEmbedUnifiedLoss`, the UEmbed (Qwen3.5) dense + sparse + FLOPS loss.

Paper arXiv:2608.02583 Eq. 5:

    L = L_InfoNCE_dense + lambda * L_InfoNCE_sparse + alpha_q * L_FLOPS_q + alpha_d * L_FLOPS_d

- dense InfoNCE  : cosine similarity matrix * `scale` (== MultipleNegativesRankingLoss),
                   labels `arange(B)`, cross entropy.
- sparse InfoNCE : inner-product similarity matrix / `tau_s`, same labels / cross entropy.
- FLOPS          : `sum_t (mean_i W[i, t])^2`, computed separately over the query batch
                   and the document batch.

Everything here is CPU-only and deterministic: every tensor is a hand-written literal, so
there is no seed, no clock and no download anywhere in this file. Value assertions run
against INDEPENDENT python-loop oracles (`math.exp` / `math.log`, plain loops) built from
the formulas above -- never against a copy of the module's own tensor expression -- so a
mistake mirrored into both sides cannot pass. The composition test recomputes the total
from those oracles, which is what makes it fail when any weighted term is dropped.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOSS_SOURCE_PATH = _REPO_ROOT / "unsloth" / "models" / "uembed_loss.py"

# float64 everywhere: the composition assertion compares torch arithmetic against python
# arithmetic, and both are IEEE doubles, so the comparison stays meaningful at 1e-12.
_DTYPE = torch.float64


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _load_uembed_loss():
    """Load `unsloth.models.uembed_loss`, falling back to a direct file load.

    `import unsloth` runs the accelerator / unsloth_zoo gate, which legitimately refuses to
    import on a CPU-only machine. The loss module depends on torch only, so the fallback
    executes the exact same source file -- it loads it, it does not stub it out.
    """
    try:
        from unsloth.models import uembed_loss  # noqa: PLC0415

        return uembed_loss
    except Exception:  # accelerator gate / missing unsloth_zoo / heavy optional deps
        pass

    name = "unsloth_uembed_loss_direct"
    if name in sys.modules:
        return sys.modules[name]
    assert _LOSS_SOURCE_PATH.exists(), f"missing module file: {_LOSS_SOURCE_PATH}"
    spec = importlib.util.spec_from_file_location(name, _LOSS_SOURCE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def uembed_loss():
    return _load_uembed_loss()


def _tensor(rows):
    return torch.tensor(rows, dtype=_DTYPE)


def _features(dense_anchor, dense_positive, sparse_anchor, sparse_positive):
    """The T8 contract: one dict per column, each carrying dense AND sparse."""
    return [
        {"sentence_embedding": _tensor(dense_anchor), "sparse_embedding": _tensor(sparse_anchor)},
        {
            "sentence_embedding": _tensor(dense_positive),
            "sparse_embedding": _tensor(sparse_positive),
        },
    ]


# --- independent oracles (plain python, no torch ops) ----------------------------------
def _flops_oracle(rows):
    """`sum_t (mean_i W[i, t])^2` -- written as loops, not as a tensor expression."""
    batch_size = len(rows)
    vocab_size = len(rows[0])
    total = 0.0
    for term in range(vocab_size):
        column_mean = sum(rows[row][term] for row in range(batch_size)) / batch_size
        total += column_mean * column_mean
    return total


def _cross_entropy_oracle(logits):
    """Mean `-log softmax(logits[i])[i]`, i.e. in-batch InfoNCE with labels `arange(B)`."""
    per_row = []
    for index, row in enumerate(logits):
        largest = max(row)
        denominator = sum(math.exp(value - largest) for value in row)
        per_row.append(-(row[index] - largest - math.log(denominator)))
    return sum(per_row) / len(per_row)


def _dot(left, right):
    return sum(a * b for a, b in zip(left, right))


def _cosine_logits_oracle(anchors, candidates, scale):
    return [
        [
            scale * _dot(anchor, candidate) / (math.sqrt(_dot(anchor, anchor)) * math.sqrt(_dot(candidate, candidate)))
            for candidate in candidates
        ]
        for anchor in anchors
    ]


def _inner_logits_oracle(anchors, candidates, tau_s):
    return [[_dot(anchor, candidate) / tau_s for candidate in candidates] for anchor in anchors]


# --- fixed tiny batch shared by several tests ------------------------------------------
_DENSE_ANCHOR = [[1.0, 0.0, 0.5], [0.0, 2.0, 1.0]]
_DENSE_POSITIVE = [[0.5, 0.5, 0.0], [0.25, 1.0, 2.0]]
_SPARSE_ANCHOR = [[1.5, 0.0, 0.0, 2.0], [0.0, 3.0, 0.5, 0.0]]
_SPARSE_POSITIVE = [[2.0, 0.25, 0.0, 1.0], [0.0, 1.5, 2.5, 0.0]]


# --------------------------------------------------------------------------------------
# 1. FLOPS formula, pinned exactly
# --------------------------------------------------------------------------------------
def test_flops_matches_hand_computed_column_means(uembed_loss):
    # column means are (2.0, 1.0, 2.0) -> 4 + 1 + 4 = 9
    rows = [[1.0, 2.0, 3.0], [3.0, 0.0, 1.0]]
    value = uembed_loss.flops_regularizer(_tensor(rows))
    assert value.item() == pytest.approx(9.0, abs=1e-12)
    assert value.item() == pytest.approx(_flops_oracle(rows), abs=1e-12)


def test_flops_matches_oracle_on_the_shared_batch(uembed_loss):
    for rows in (_SPARSE_ANCHOR, _SPARSE_POSITIVE):
        value = uembed_loss.flops_regularizer(_tensor(rows))
        assert value.item() == pytest.approx(_flops_oracle(rows), abs=1e-12)


def test_flops_is_zero_for_an_all_zero_batch(uembed_loss):
    assert uembed_loss.flops_regularizer(torch.zeros(4, 5, dtype=_DTYPE)).item() == 0.0


def test_flops_is_a_mean_not_a_sum_over_the_batch(uembed_loss):
    """Duplicating every row leaves the column means -- and so the term -- unchanged."""
    rows = [[1.0, 2.0, 3.0], [3.0, 0.0, 1.0]]
    single = uembed_loss.flops_regularizer(_tensor(rows))
    doubled = uembed_loss.flops_regularizer(_tensor(rows + rows))
    assert doubled.item() == pytest.approx(single.item(), abs=1e-12)


def test_flops_penalises_a_denser_batch_more(uembed_loss):
    sparse = uembed_loss.flops_regularizer(_tensor([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0]]))
    dense = uembed_loss.flops_regularizer(_tensor([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]))
    assert dense.item() > sparse.item()


# --------------------------------------------------------------------------------------
# 2. weighted composition, pinned exactly
# --------------------------------------------------------------------------------------
def test_total_equals_the_independently_recomputed_weighted_sum(uembed_loss):
    lambda_sparse, alpha_q, alpha_d, scale, tau_s = 0.75, 0.03, 0.11, 20.0, 32.0
    loss = uembed_loss.UEmbedUnifiedLoss(
        lambda_sparse=lambda_sparse,
        alpha_q=alpha_q,
        alpha_d=alpha_d,
        scale=scale,
        tau_s=tau_s,
    )
    features = _features(_DENSE_ANCHOR, _DENSE_POSITIVE, _SPARSE_ANCHOR, _SPARSE_POSITIVE)

    dense_expected = _cross_entropy_oracle(
        _cosine_logits_oracle(_DENSE_ANCHOR, _DENSE_POSITIVE, scale)
    )
    sparse_expected = _cross_entropy_oracle(
        _inner_logits_oracle(_SPARSE_ANCHOR, _SPARSE_POSITIVE, tau_s)
    )
    flops_q_expected = _flops_oracle(_SPARSE_ANCHOR)
    flops_d_expected = _flops_oracle(_SPARSE_POSITIVE)
    total_expected = (
        dense_expected
        + lambda_sparse * sparse_expected
        + alpha_q * flops_q_expected
        + alpha_d * flops_d_expected
    )

    components = loss.components(features)
    assert components["dense"].item() == pytest.approx(dense_expected, abs=1e-12)
    assert components["sparse"].item() == pytest.approx(sparse_expected, abs=1e-12)
    assert components["flops_query"].item() == pytest.approx(flops_q_expected, abs=1e-12)
    assert components["flops_document"].item() == pytest.approx(flops_d_expected, abs=1e-12)
    assert components["total"].item() == pytest.approx(total_expected, abs=1e-12)
    assert loss(features).item() == pytest.approx(total_expected, abs=1e-12)


def test_every_coefficient_moves_the_total(uembed_loss):
    """Each weight is load-bearing: zeroing it changes the total by exactly its term."""
    features = _features(_DENSE_ANCHOR, _DENSE_POSITIVE, _SPARSE_ANCHOR, _SPARSE_POSITIVE)
    base_kwargs = dict(lambda_sparse=0.75, alpha_q=0.03, alpha_d=0.11)
    base = uembed_loss.UEmbedUnifiedLoss(**base_kwargs)
    components = base.components(features)
    base_total = components["total"].item()

    for coefficient, term in (
        ("lambda_sparse", "sparse"),
        ("alpha_q", "flops_query"),
        ("alpha_d", "flops_document"),
    ):
        kwargs = dict(base_kwargs)
        kwargs[coefficient] = 0.0
        zeroed = uembed_loss.UEmbedUnifiedLoss(**kwargs)(features).item()
        removed = base_kwargs[coefficient] * components[term].item()
        assert removed > 0.0
        assert base_total - zeroed == pytest.approx(removed, abs=1e-12)


def test_default_coefficients_match_the_paper_defaults(uembed_loss):
    loss = uembed_loss.UEmbedUnifiedLoss()
    assert loss.lambda_sparse == 1.0
    assert loss.alpha_q == 0.01
    assert loss.alpha_d == 0.01
    assert loss.scale == 20.0
    assert loss.tau_s == 32.0
    assert loss.alpha_warmup_steps == 0


def test_sparse_temperature_is_not_the_dense_scale(uembed_loss):
    """Sparse logits are inner products / tau_s, dense are cosines * scale."""
    features = _features(_DENSE_ANCHOR, _DENSE_POSITIVE, _SPARSE_ANCHOR, _SPARSE_POSITIVE)
    default = uembed_loss.UEmbedUnifiedLoss().components(features)["sparse"].item()
    retuned = uembed_loss.UEmbedUnifiedLoss(tau_s=4.0).components(features)["sparse"].item()
    assert default != pytest.approx(retuned, abs=1e-9)
    expected = _cross_entropy_oracle(_inner_logits_oracle(_SPARSE_ANCHOR, _SPARSE_POSITIVE, 4.0))
    assert retuned == pytest.approx(expected, abs=1e-12)


def test_config_dict_reports_the_hyperparameters(uembed_loss):
    config = uembed_loss.UEmbedUnifiedLoss(lambda_sparse=0.5, tau_s=64.0).get_config_dict()
    assert config["lambda_sparse"] == 0.5
    assert config["tau_s"] == 64.0
    assert config["scale"] == 20.0


# --------------------------------------------------------------------------------------
# 3. behavioural direction (monotonicity)
# --------------------------------------------------------------------------------------
def test_sparse_infonce_drops_when_the_positive_inner_product_rises(uembed_loss):
    """Disjoint supports: only the (0, 0) logit moves, so the direction is unambiguous."""
    loss = uembed_loss.UEmbedUnifiedLoss()
    anchor_sparse = [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    weak = [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    strong = [[4.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]

    before = loss.components(
        _features(_DENSE_ANCHOR, _DENSE_POSITIVE, anchor_sparse, weak)
    )["sparse"].item()
    after = loss.components(
        _features(_DENSE_ANCHOR, _DENSE_POSITIVE, anchor_sparse, strong)
    )["sparse"].item()
    assert after < before


def test_dense_infonce_drops_when_the_positive_cosine_rises(uembed_loss):
    loss = uembed_loss.UEmbedUnifiedLoss()
    anchor_dense = [[1.0, 0.0], [0.0, 1.0]]
    misaligned = [[1.0, 1.0], [1.0, 1.0]]
    aligned = [[1.0, 0.0], [0.0, 1.0]]

    before = loss.components(
        _features(anchor_dense, misaligned, _SPARSE_ANCHOR, _SPARSE_POSITIVE)
    )["dense"].item()
    after = loss.components(
        _features(anchor_dense, aligned, _SPARSE_ANCHOR, _SPARSE_POSITIVE)
    )["dense"].item()
    assert after < before


def test_dense_ignores_the_magnitude_of_the_embeddings(uembed_loss):
    """Cosine, not inner product: rescaling a row must not change the dense term."""
    loss = uembed_loss.UEmbedUnifiedLoss()
    scaled = [[7.0 * value for value in row] for row in _DENSE_ANCHOR]
    plain = loss.components(
        _features(_DENSE_ANCHOR, _DENSE_POSITIVE, _SPARSE_ANCHOR, _SPARSE_POSITIVE)
    )["dense"].item()
    rescaled = loss.components(
        _features(scaled, _DENSE_POSITIVE, _SPARSE_ANCHOR, _SPARSE_POSITIVE)
    )["dense"].item()
    assert rescaled == pytest.approx(plain, abs=1e-12)


def test_gradients_reach_both_the_dense_and_the_sparse_embeddings(uembed_loss):
    loss = uembed_loss.UEmbedUnifiedLoss()
    features = _features(_DENSE_ANCHOR, _DENSE_POSITIVE, _SPARSE_ANCHOR, _SPARSE_POSITIVE)
    for column in features:
        for tensor in column.values():
            tensor.requires_grad_(True)

    loss(features).backward()
    for column in features:
        for key, tensor in column.items():
            assert tensor.grad is not None, key
            assert torch.isfinite(tensor.grad).all(), key
            assert tensor.grad.abs().sum().item() > 0.0, key


def test_repeated_calls_are_bit_identical(uembed_loss):
    """No seed, no clock, no state: the same input must give the same float."""
    loss = uembed_loss.UEmbedUnifiedLoss()
    features = _features(_DENSE_ANCHOR, _DENSE_POSITIVE, _SPARSE_ANCHOR, _SPARSE_POSITIVE)
    first = loss(features).item()
    assert loss(features).item() == first
    assert loss(features).item() == first


# --------------------------------------------------------------------------------------
# 4. edges + malformed input
# --------------------------------------------------------------------------------------
def test_missing_sparse_embedding_raises_an_unsloth_error(uembed_loss):
    loss = uembed_loss.UEmbedUnifiedLoss()
    for column in (0, 1):
        features = _features(_DENSE_ANCHOR, _DENSE_POSITIVE, _SPARSE_ANCHOR, _SPARSE_POSITIVE)
        features[column].pop("sparse_embedding")
        with pytest.raises(KeyError) as excinfo:
            loss(features)
        message = str(excinfo.value)
        assert "Unsloth:" in message
        assert "sparse_embedding" in message


def test_missing_sentence_embedding_raises_an_unsloth_error(uembed_loss):
    loss = uembed_loss.UEmbedUnifiedLoss()
    features = _features(_DENSE_ANCHOR, _DENSE_POSITIVE, _SPARSE_ANCHOR, _SPARSE_POSITIVE)
    features[0].pop("sentence_embedding")
    with pytest.raises(KeyError) as excinfo:
        loss(features)
    assert "Unsloth:" in str(excinfo.value)
    assert "sentence_embedding" in str(excinfo.value)


def test_a_single_column_raises_an_unsloth_error(uembed_loss):
    loss = uembed_loss.UEmbedUnifiedLoss()
    features = _features(_DENSE_ANCHOR, _DENSE_POSITIVE, _SPARSE_ANCHOR, _SPARSE_POSITIVE)
    with pytest.raises(ValueError) as excinfo:
        loss(features[:1])
    assert "Unsloth:" in str(excinfo.value)


def test_mismatched_batch_sizes_raise_an_unsloth_error(uembed_loss):
    loss = uembed_loss.UEmbedUnifiedLoss()
    features = _features(
        _DENSE_ANCHOR,
        _DENSE_POSITIVE[:1],
        _SPARSE_ANCHOR,
        _SPARSE_POSITIVE[:1],
    )
    with pytest.raises(ValueError) as excinfo:
        loss(features)
    assert "Unsloth:" in str(excinfo.value)
    assert "batch" in str(excinfo.value).lower()


def test_dense_and_sparse_batch_sizes_must_agree_within_a_column(uembed_loss):
    loss = uembed_loss.UEmbedUnifiedLoss()
    features = _features(_DENSE_ANCHOR, _DENSE_POSITIVE, _SPARSE_ANCHOR, _SPARSE_POSITIVE)
    features[0]["sparse_embedding"] = _tensor(_SPARSE_ANCHOR[:1])
    with pytest.raises(ValueError) as excinfo:
        loss(features)
    assert "Unsloth:" in str(excinfo.value)


def test_batch_of_one_degenerates_to_the_flops_terms_only(uembed_loss):
    """B=1 has no in-batch negatives: both cross entropies are exactly 0."""
    loss = uembed_loss.UEmbedUnifiedLoss(lambda_sparse=0.75, alpha_q=0.03, alpha_d=0.11)
    features = _features(
        _DENSE_ANCHOR[:1], _DENSE_POSITIVE[:1], _SPARSE_ANCHOR[:1], _SPARSE_POSITIVE[:1]
    )
    components = loss.components(features)
    assert components["dense"].item() == pytest.approx(0.0, abs=1e-12)
    assert components["sparse"].item() == pytest.approx(0.0, abs=1e-12)
    expected = 0.03 * _flops_oracle(_SPARSE_ANCHOR[:1]) + 0.11 * _flops_oracle(_SPARSE_POSITIVE[:1])
    assert components["total"].item() == pytest.approx(expected, abs=1e-12)


def test_labels_argument_is_accepted_and_ignored(uembed_loss):
    """The trainer passes `labels`; in-batch InfoNCE derives its own `arange(B)`."""
    loss = uembed_loss.UEmbedUnifiedLoss()
    features = _features(_DENSE_ANCHOR, _DENSE_POSITIVE, _SPARSE_ANCHOR, _SPARSE_POSITIVE)
    assert loss(features, labels=None).item() == loss(features, torch.zeros(2)).item()


# --------------------------------------------------------------------------------------
# 5. optional quadratic alpha warmup (default OFF)
# --------------------------------------------------------------------------------------
def test_alpha_warmup_is_off_by_default(uembed_loss):
    loss = uembed_loss.UEmbedUnifiedLoss()
    features = _features(_DENSE_ANCHOR, _DENSE_POSITIVE, _SPARSE_ANCHOR, _SPARSE_POSITIVE)
    assert loss.alpha_scale() == 1.0
    first = loss(features).item()
    assert loss.alpha_step == 0
    assert loss(features).item() == first


def test_alpha_warmup_ramps_quadratically_then_saturates(uembed_loss):
    loss = uembed_loss.UEmbedUnifiedLoss(alpha_warmup_steps=4)
    assert loss.alpha_scale() == 0.0
    for step, expected in ((1, 0.0625), (2, 0.25), (3, 0.5625), (4, 1.0), (9, 1.0)):
        loss.alpha_step = step
        assert loss.alpha_scale() == pytest.approx(expected, abs=1e-12)


def test_alpha_warmup_scales_only_the_flops_terms(uembed_loss):
    loss = uembed_loss.UEmbedUnifiedLoss(alpha_q=0.03, alpha_d=0.11, alpha_warmup_steps=4)
    features = _features(_DENSE_ANCHOR, _DENSE_POSITIVE, _SPARSE_ANCHOR, _SPARSE_POSITIVE)
    loss.alpha_step = 2  # (2 / 4)^2 = 0.25
    components = loss.components(features)
    expected = (
        components["dense"].item()
        + components["sparse"].item()
        + 0.25 * (0.03 * _flops_oracle(_SPARSE_ANCHOR) + 0.11 * _flops_oracle(_SPARSE_POSITIVE))
    )
    assert components["total"].item() == pytest.approx(expected, abs=1e-12)


def test_alpha_step_advances_only_while_training_with_warmup_enabled(uembed_loss):
    features = _features(_DENSE_ANCHOR, _DENSE_POSITIVE, _SPARSE_ANCHOR, _SPARSE_POSITIVE)
    loss = uembed_loss.UEmbedUnifiedLoss(alpha_warmup_steps=4)
    loss.train()
    loss(features)
    loss(features)
    assert loss.alpha_step == 2
    loss.eval()
    loss(features)
    assert loss.alpha_step == 2


def test_negative_hyperparameters_are_rejected(uembed_loss):
    for kwargs in (
        {"lambda_sparse": -1.0},
        {"alpha_q": -0.01},
        {"alpha_d": -0.01},
        {"scale": 0.0},
        {"tau_s": 0.0},
        {"alpha_warmup_steps": -1},
    ):
        with pytest.raises(ValueError) as excinfo:
            uembed_loss.UEmbedUnifiedLoss(**kwargs)
        assert "Unsloth:" in str(excinfo.value)


# --------------------------------------------------------------------------------------
# 6. raw trainer columns versus already-forwarded columns
# --------------------------------------------------------------------------------------
class _CountingFeatureModel(torch.nn.Module):
    """A tiny trainable ST-like pipeline returning a new dual-output features dict."""

    def __init__(self, *, omit_sparse=False):
        super().__init__()
        self.weight = torch.nn.Parameter(
            _tensor([[1.0, -0.5, 0.25], [0.5, 1.5, -1.0], [-0.25, 0.75, 1.25]])
        )
        self.omit_sparse = omit_sparse
        self.calls = 0

    def forward(self, features):
        self.calls += 1
        dense = features["values"] @ self.weight
        output = {"sentence_embedding": dense}
        if not self.omit_sparse:
            output["sparse_embedding"] = (dense + _tensor([[1.0, 0.5, 1.5]])).pow(2)
        return output


def _raw_features():
    return [
        {"values": _tensor([[1.0, 0.0, 0.5], [0.0, 2.0, 1.0]])},
        {"values": _tensor([[0.5, 0.5, 0.0], [0.25, 1.0, 2.0]])},
    ]


def test_raw_trainer_columns_are_forwarded_once_each_with_gradients_and_oracle(uembed_loss):
    model = _CountingFeatureModel()
    loss = uembed_loss.UEmbedUnifiedLoss(
        model,
        lambda_sparse=0.75,
        alpha_q=0.03,
        alpha_d=0.11,
        scale=20.0,
        tau_s=32.0,
    )
    raw = _raw_features()
    original_keys = [set(column) for column in raw]

    total = loss(raw)

    assert model.calls == 2, "each raw column must use exactly one shared dense+sparse forward"
    assert [set(column) for column in raw] == original_keys, "the loss mutated caller dictionaries"
    assert all("sentence_embedding" not in column for column in raw)

    # Independent scalar oracle over the model's analytically defined outputs. Do not call
    # the loss/components again: doing so could mirror a forwarding or composition defect.
    weight = model.weight.detach().tolist()
    dense = []
    sparse = []
    for column in raw:
        dense_column = []
        sparse_column = []
        for row in column["values"].tolist():
            dense_row = [sum(row[k] * weight[k][j] for k in range(3)) for j in range(3)]
            dense_column.append(dense_row)
            sparse_column.append([(value + offset) ** 2 for value, offset in zip(dense_row, (1.0, 0.5, 1.5))])
        dense.append(dense_column)
        sparse.append(sparse_column)
    expected = (
        _cross_entropy_oracle(_cosine_logits_oracle(dense[0], dense[1], 20.0))
        + 0.75 * _cross_entropy_oracle(_inner_logits_oracle(sparse[0], sparse[1], 32.0))
        + 0.03 * _flops_oracle(sparse[0])
        + 0.11 * _flops_oracle(sparse[1])
    )
    assert total.item() == pytest.approx(expected, abs=1e-12)

    total.backward()
    assert model.weight.grad is not None
    assert torch.isfinite(model.weight.grad).all()
    assert model.weight.grad.abs().sum().item() > 0.0


def test_already_forwarded_columns_never_call_the_model(uembed_loss):
    model = _CountingFeatureModel()
    loss = uembed_loss.UEmbedUnifiedLoss(model)
    forwarded = _features(
        _DENSE_ANCHOR, _DENSE_POSITIVE, _SPARSE_ANCHOR, _SPARSE_POSITIVE
    )
    # Trainer/raw keys can survive in an output dict. Presence of BOTH required output
    # keys, rather than absence of input keys or an arbitrary flag, defines forwarded data.
    for column in forwarded:
        column["input_ids"] = torch.ones(2, 3, dtype=torch.long)
    expected = uembed_loss.UEmbedUnifiedLoss()(forwarded)

    actual = loss(forwarded)

    assert model.calls == 0
    assert actual.item() == expected.item()


def test_raw_forward_missing_sparse_embedding_raises_loudly(uembed_loss):
    model = _CountingFeatureModel(omit_sparse=True)
    loss = uembed_loss.UEmbedUnifiedLoss(model)

    with pytest.raises(KeyError) as excinfo:
        loss(_raw_features())

    assert model.calls == 2
    assert "Unsloth:" in str(excinfo.value)
    assert "sparse_embedding" in str(excinfo.value)


# --------------------------------------------------------------------------------------
# 7. sentence-transformers contract (skips when ST is not importable here)
# --------------------------------------------------------------------------------------
def test_dense_term_matches_multiple_negatives_ranking_loss(uembed_loss):
    """The dense half must be MNRL exactly -- same scale, same cosine, same labels."""
    try:
        from sentence_transformers.losses import MultipleNegativesRankingLoss  # noqa: PLC0415
    except Exception as error:  # ST pulls in transformers, unavailable on some CPU boxes
        pytest.skip(f"sentence_transformers unavailable: {error}")

    class _Passthrough(torch.nn.Module):
        def forward(self, features):
            return features

    features = _features(_DENSE_ANCHOR, _DENSE_POSITIVE, _SPARSE_ANCHOR, _SPARSE_POSITIVE)
    reference = MultipleNegativesRankingLoss(_Passthrough(), scale=20.0)(features, None).item()
    ours = uembed_loss.UEmbedUnifiedLoss().components(features)["dense"].item()
    assert ours == pytest.approx(reference, abs=1e-9)
