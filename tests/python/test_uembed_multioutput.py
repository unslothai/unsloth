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

"""Tests for the UEmbed single-forward dense + sparse multi-output wiring.

UEmbed produces its dense vector and its SPLADE sparse vector from ONE causal forward
pass. The wiring under test puts the sparse head INTO the sentence-transformers module
chain, so it reads the hidden state (`token_embeddings`) the transformer already emitted
and writes `sparse_embedding` beside `sentence_embedding` in the same features dict --
which is exactly what `UEmbedUnifiedLoss` consumes.

The load-bearing assertion here is the transformer forward-call COUNT: a naive
"encode once for dense, encode again for sparse" implementation doubles the most
expensive part of training and fails these tests.

`unsloth` and `sentence_transformers` are not importable on a CPU-only box, so the wiring
is exercised through a stub pipeline that mimics the sentence-transformers contract
(tokenize -> module chain over one features dict -> extract) against the REAL wiring
module loaded from its source file. The one test that needs the genuine
Transformer->Pooling->Normalize chain skips itself when sentence-transformers is absent.
Everything is CPU-only, seeded and download-free.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WIRING_SOURCE_PATH = _REPO_ROOT / "unsloth" / "models" / "uembed_wiring.py"
_SPLADE_SOURCE_PATH = _REPO_ROOT / "unsloth" / "models" / "uembed_splade.py"

_VOCAB_SIZE = 29
_HIDDEN_DIM = 4
_MAX_LENGTH = 8
# Deliberately unequal per-head vocabularies: only a real `sum(V_i)` concatenation gives
# this width, a `num_eos * V` shortcut cannot.
_HEAD_DIMS = [3, 4, 5]
_NUM_EOS = 3
_SPARSE_DIM = sum(_HEAD_DIMS)

# Every sentence must outlive the 3-slot EOS block the sparse head reads back through.
_SENTENCES = ["alpha document", "beta document"]


# --------------------------------------------------------------------------------------
# module loading
# --------------------------------------------------------------------------------------
def _load_from_source(module_name: str, source_path: Path):
    """Execute the real source file directly.

    `import unsloth` runs the accelerator / unsloth_zoo gate, which legitimately refuses
    to import on a CPU-only machine. The modules under test depend on torch only, so this
    fallback runs the exact same source -- it loads it, it does not stub it out.
    """
    if module_name in sys.modules:
        return sys.modules[module_name]
    assert source_path.exists(), f"missing module file: {source_path}"
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope = "module")
def wiring():
    try:
        from unsloth.models import uembed_wiring  # noqa: PLC0415
        return uembed_wiring
    except Exception:  # accelerator gate / missing unsloth_zoo / heavy optional deps
        return _load_from_source("unsloth_uembed_wiring_direct", _WIRING_SOURCE_PATH)


@pytest.fixture(scope = "module")
def splade():
    try:
        from unsloth.models import uembed_splade  # noqa: PLC0415
        return uembed_splade
    except Exception:
        return _load_from_source("unsloth_uembed_splade_direct", _SPLADE_SOURCE_PATH)


# --------------------------------------------------------------------------------------
# stub pipeline: the sentence-transformers contract, minus sentence-transformers
# --------------------------------------------------------------------------------------
class _CountingTransformer(nn.Module):
    """Stands in for the ST `Transformer` module and counts how often it runs."""

    def __init__(self, seed: int = 3) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.table = nn.Parameter(torch.randn(_VOCAB_SIZE, _HIDDEN_DIM, generator = generator))
        self.forward_calls = 0

    def forward(self, features: dict, **kwargs) -> dict:
        self.forward_calls += 1
        features["token_embeddings"] = self.table[features["input_ids"]]
        return features


class _LastTokenPooling(nn.Module):
    """Stands in for the ST `Pooling` module (dense vector from the last real token)."""

    def forward(self, features: dict, **kwargs) -> dict:
        mask = features["attention_mask"]
        last = (mask.cumsum(dim = 1) * mask).argmax(dim = 1)
        rows = torch.arange(mask.shape[0])
        features["sentence_embedding"] = features["token_embeddings"][rows, last]
        return features


class _Normalize(nn.Module):
    """Stands in for the ST `Normalize` module."""

    def forward(self, features: dict, **kwargs) -> dict:
        features["sentence_embedding"] = F.normalize(features["sentence_embedding"], p = 2, dim = -1)
        return features


def _tokenize(sentences: list[str]) -> dict:
    input_ids = torch.zeros(len(sentences), _MAX_LENGTH, dtype = torch.long)
    attention_mask = torch.zeros(len(sentences), _MAX_LENGTH, dtype = torch.long)
    for row, sentence in enumerate(sentences):
        codes = [ord(character) % _VOCAB_SIZE for character in sentence][:_MAX_LENGTH]
        assert len(codes) > _NUM_EOS, f"test sentence too short for the EOS block: {sentence!r}"
        input_ids[row, : len(codes)] = torch.tensor(codes, dtype = torch.long)
        attention_mask[row, : len(codes)] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask}


class _StubSentenceTransformer(nn.Sequential):
    """Mimics `SentenceTransformer.encode`: one tokenization, one pass over the chain.

    `output_value = None` returns every feature per row, which is how sentence-transformers
    hands back keys it does not know about.
    """

    def tokenize(self, sentences: list[str]) -> dict:
        return _tokenize(sentences)

    def encode(
        self,
        sentences,
        output_value: str | None = "sentence_embedding",
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
    ):
        single = isinstance(sentences, str)
        batch = [sentences] if single else list(sentences)

        features = self.tokenize(batch)
        with torch.no_grad():
            for module in self:
                features = module(features)

        if output_value is None:
            rows = [
                {key: value[index] for key, value in features.items() if torch.is_tensor(value)}
                for index in range(len(batch))
            ]
            return rows[0] if single else rows

        embeddings = features[output_value]
        if not convert_to_tensor and convert_to_numpy:
            embeddings = embeddings.cpu().numpy()
        return embeddings[0] if single else embeddings


def _make_pipeline() -> _StubSentenceTransformer:
    return _StubSentenceTransformer(_CountingTransformer(), _LastTokenPooling(), _Normalize())


def _make_head(splade_module, seed: int = 11):
    generator = torch.Generator().manual_seed(seed)
    heads = [torch.randn(dim, _HIDDEN_DIM, generator = generator) for dim in _HEAD_DIMS]
    biases = [torch.randn(dim, generator = generator) for dim in _HEAD_DIMS]
    return splade_module.SpladeHead(heads, biases, num_eos_tokens = _NUM_EOS)


@pytest.fixture
def pipeline():
    return _make_pipeline()


@pytest.fixture
def sparse_pipeline(wiring, splade):
    model = _make_pipeline()
    wiring.attach_uembed_sparse_output(model, _make_head(splade))
    model[0].forward_calls = 0
    return model


# --------------------------------------------------------------------------------------
# 1. no regression: the default path is untouched when sparse is not requested
# --------------------------------------------------------------------------------------
def test_default_encode_matches_the_pre_wiring_result(wiring, splade, pipeline):
    """Attaching the sparse output must not change what a plain `encode()` returns."""
    baseline = pipeline.encode(_SENTENCES)
    baseline_single = pipeline.encode(_SENTENCES[0])

    wiring.attach_uembed_sparse_output(pipeline, _make_head(splade))

    after = pipeline.encode(_SENTENCES)
    after_single = pipeline.encode(_SENTENCES[0])

    assert type(after) is type(baseline) is np.ndarray
    assert after.shape == baseline.shape == (len(_SENTENCES), _HIDDEN_DIM)
    assert np.array_equal(after, baseline)
    assert type(after_single) is type(baseline_single)
    assert after_single.shape == baseline_single.shape == (_HIDDEN_DIM,)
    assert np.array_equal(after_single, baseline_single)


def test_default_encode_keeps_its_keyword_arguments(sparse_pipeline):
    """The wrapped `encode` still forwards stock sentence-transformers keywords."""
    as_tensor = sparse_pipeline.encode(_SENTENCES, convert_to_tensor = True)
    assert torch.is_tensor(as_tensor)
    assert as_tensor.shape == (len(_SENTENCES), _HIDDEN_DIM)

    explicit_dense = sparse_pipeline.encode(_SENTENCES, output_mode = "dense")
    assert type(explicit_dense) is np.ndarray
    assert np.array_equal(explicit_dense, sparse_pipeline.encode(_SENTENCES))


# --------------------------------------------------------------------------------------
# 2. the load-bearing assertion: ONE transformer forward for BOTH outputs
# --------------------------------------------------------------------------------------
def test_both_outputs_come_from_a_single_transformer_forward(sparse_pipeline):
    outputs = sparse_pipeline.encode(_SENTENCES, output_mode = "both")

    assert sparse_pipeline[0].forward_calls == 1, "the transformer ran more than once"
    assert set(outputs) == {"sentence_embedding", "sparse_embedding"}
    assert outputs["sentence_embedding"].shape == (len(_SENTENCES), _HIDDEN_DIM)
    assert outputs["sparse_embedding"].shape == (len(_SENTENCES), _SPARSE_DIM)
    assert (outputs["sparse_embedding"] >= 0.0).all(), "SPLADE weights are non-negative"


def test_sparse_only_request_also_uses_a_single_forward(sparse_pipeline):
    sparse = sparse_pipeline.encode(_SENTENCES, output_mode = "sparse")

    assert sparse_pipeline[0].forward_calls == 1
    assert type(sparse) is np.ndarray
    assert sparse.shape == (len(_SENTENCES), _SPARSE_DIM)


def test_module_chain_carries_both_keys_in_one_features_dict(sparse_pipeline):
    """The loss boundary: one forward, one dict, both embeddings."""
    features = sparse_pipeline.tokenize(_SENTENCES)
    for module in sparse_pipeline:
        features = module(features)

    assert sparse_pipeline[0].forward_calls == 1
    assert "sentence_embedding" in features and "sparse_embedding" in features
    assert features["sparse_embedding"].shape == (len(_SENTENCES), _SPARSE_DIM)
    # The dense vector still went through Normalize, i.e. the sparse module did not
    # displace or shadow the dense pooling it shares the forward with.
    norms = features["sentence_embedding"].norm(dim = -1)
    assert torch.allclose(norms, torch.ones_like(norms), atol = 1e-5)


def test_single_string_request_returns_row_vectors(sparse_pipeline):
    outputs = sparse_pipeline.encode(_SENTENCES[0], output_mode = "both")

    assert sparse_pipeline[0].forward_calls == 1
    assert outputs["sentence_embedding"].shape == (_HIDDEN_DIM,)
    assert outputs["sparse_embedding"].shape == (_SPARSE_DIM,)


def test_convert_to_tensor_applies_to_both_outputs(sparse_pipeline):
    outputs = sparse_pipeline.encode(_SENTENCES, output_mode = "both", convert_to_tensor = True)

    assert torch.is_tensor(outputs["sentence_embedding"])
    assert torch.is_tensor(outputs["sparse_embedding"])
    assert outputs["sparse_embedding"].shape == (len(_SENTENCES), _SPARSE_DIM)


# --------------------------------------------------------------------------------------
# 3. stale state: the sparse vector must come from THIS forward
# --------------------------------------------------------------------------------------
def test_sparse_output_tracks_the_current_input(sparse_pipeline):
    first = sparse_pipeline.encode("alpha document", output_mode = "sparse")
    second = sparse_pipeline.encode("zulu manuscript", output_mode = "sparse")
    repeat = sparse_pipeline.encode("alpha document", output_mode = "sparse")

    assert not np.allclose(first, second), "sparse output ignored the new input (cached?)"
    assert np.allclose(first, repeat), "sparse output is not deterministic for one input"


def test_dense_and_sparse_belong_to_the_same_forward(sparse_pipeline, splade):
    """Both outputs of one call describe the same input, not a mix of two calls."""
    together = sparse_pipeline.encode(_SENTENCES, output_mode = "both")
    dense_alone = sparse_pipeline.encode(_SENTENCES)
    sparse_alone = sparse_pipeline.encode(_SENTENCES, output_mode = "sparse")

    assert np.allclose(together["sentence_embedding"], dense_alone)
    assert np.allclose(together["sparse_embedding"], sparse_alone)


# --------------------------------------------------------------------------------------
# 4. malformed requests fail loudly
# --------------------------------------------------------------------------------------
def test_sparse_request_without_a_head_raises(wiring, pipeline):
    with pytest.raises(ValueError) as error:
        wiring.encode_uembed(pipeline, _SENTENCES, output_mode = "sparse")

    message = str(error.value)
    assert "Unsloth" in message
    assert "sparse" in message.lower()


def test_unknown_output_mode_raises(wiring, sparse_pipeline):
    with pytest.raises(ValueError) as error:
        sparse_pipeline.encode(_SENTENCES, output_mode = "splade.last")

    assert "Unsloth" in str(error.value)


def test_attaching_to_a_chain_without_encode_still_wires_the_features_dict(wiring, splade):
    """Training only needs the features dict, so a chain with no `encode` must still wire."""
    chain = nn.Sequential(_CountingTransformer(), _LastTokenPooling(), _Normalize())

    assert wiring.attach_uembed_sparse_output(chain, _make_head(splade)) is True

    features = _tokenize(_SENTENCES)
    for module in chain:
        features = module(features)
    assert features["sparse_embedding"].shape == (len(_SENTENCES), _SPARSE_DIM)
    assert chain[0].forward_calls == 1


def test_attaching_something_other_than_a_head_raises(wiring, pipeline):
    with pytest.raises(ValueError) as error:
        wiring.attach_uembed_sparse_output(pipeline, nn.Linear(_HIDDEN_DIM, _HIDDEN_DIM))

    assert "Unsloth" in str(error.value)


def test_a_head_from_a_second_module_object_is_accepted(wiring, pipeline):
    """The same source loaded twice yields two classes; a head is still a head."""
    other_splade = _load_from_source("uembed_splade_second_load", _SPLADE_SOURCE_PATH)

    assert wiring.attach_uembed_sparse_output(pipeline, _make_head(other_splade)) is True
    assert pipeline.encode(_SENTENCES, output_mode = "sparse").shape == (
        len(_SENTENCES),
        _SPARSE_DIM,
    )


def test_serialized_sparse_chain_restores_encode_patch_idempotently(wiring, splade, pipeline):
    """A module rebuilt from modules.json must regain the process-local encode wrapper."""
    baseline = pipeline.encode(_SENTENCES)
    serialized_module = wiring.UEmbedSparseOutput(_make_head(splade))
    pipeline.add_module("3", serialized_module)
    original_encode = pipeline.encode
    module_count = len(pipeline)

    # Attachment discovers the module reconstructed by SentenceTransformer.load instead
    # of appending another one, but must still restore process-local method wiring.
    assert wiring.attach_uembed_sparse_output(pipeline, _make_head(splade, seed = 99)) is False
    patched_encode = pipeline.encode
    assert patched_encode is not original_encode
    assert len(pipeline) == module_count

    pipeline[0].forward_calls = 0
    sparse = pipeline.encode(_SENTENCES, output_mode = "sparse")
    assert sparse.shape == (len(_SENTENCES), _SPARSE_DIM)
    assert pipeline[0].forward_calls == 1

    pipeline[0].forward_calls = 0
    both = pipeline.encode(_SENTENCES, output_mode = "both")
    assert set(both) == {"sentence_embedding", "sparse_embedding"}
    assert pipeline[0].forward_calls == 1

    # Re-running the load/attach seam must neither append nor wrap the wrapper.
    assert wiring.attach_uembed_sparse_output(pipeline, _make_head(splade, seed = 101)) is False
    assert pipeline.encode is patched_encode
    pipeline[0].forward_calls = 0
    assert np.array_equal(pipeline.encode(_SENTENCES, output_mode = "dense"), baseline)
    assert pipeline[0].forward_calls == 1


def test_non_uembed_chain_keeps_stock_encode_signature_and_behavior(wiring, pipeline):
    signature = inspect.signature(pipeline.encode)
    baseline = pipeline.encode(_SENTENCES)

    # The idempotent restoration helper is intentionally a no-op without the sparse module.
    assert wiring.patch_uembed_sparse_encode(pipeline) is False

    assert inspect.signature(pipeline.encode) == signature
    assert np.array_equal(pipeline.encode(_SENTENCES), baseline)
    with pytest.raises(TypeError):
        pipeline.encode(_SENTENCES, output_mode = "sparse")


def test_attaching_twice_is_a_no_op(wiring, splade, pipeline):
    head = _make_head(splade)
    assert wiring.attach_uembed_sparse_output(pipeline, head) is True
    module_count = len(pipeline)

    assert wiring.attach_uembed_sparse_output(pipeline, _make_head(splade, seed = 99)) is False
    assert len(pipeline) == module_count


# --------------------------------------------------------------------------------------
# 5. the sparse head trains: parameters registered, gradients arrive
# --------------------------------------------------------------------------------------
def test_sparse_head_parameters_are_visible_to_the_optimizer(wiring, splade, pipeline):
    head = _make_head(splade)
    wiring.attach_uembed_sparse_output(pipeline, head)

    pipeline_parameters = {id(parameter) for parameter in pipeline.parameters()}
    head_parameters = list(head.parameters())

    assert head_parameters, "the head has no parameters"
    for parameter in head_parameters:
        assert parameter.requires_grad
        assert id(parameter) in pipeline_parameters


def test_gradients_reach_the_sparse_head_through_one_forward(wiring, splade, pipeline):
    head = _make_head(splade)
    wiring.attach_uembed_sparse_output(pipeline, head)

    features = pipeline.tokenize(_SENTENCES)
    for module in pipeline:
        features = module(features)
    (features["sparse_embedding"].sum() + features["sentence_embedding"].sum()).backward()

    assert pipeline[0].forward_calls == 1
    assert head.sparse_lm_heads[0].grad is not None
    assert head.sparse_lm_heads[0].grad.abs().sum() > 0
    assert head.sparse_bias[0].grad is not None
    # The shared transformer got gradient from the same backward pass.
    assert pipeline[0].table.grad is not None


# --------------------------------------------------------------------------------------
# 6. the real sentence-transformers chain must not drop the extra key
# --------------------------------------------------------------------------------------
def test_real_sentence_transformers_chain_preserves_sparse_embedding(wiring, splade):
    models = pytest.importorskip("sentence_transformers.models")

    head = _make_head(splade)
    sparse_module = wiring.UEmbedSparseOutput(head)
    pooling = models.Pooling(word_embedding_dimension = _HIDDEN_DIM, pooling_mode = "lasttoken")
    normalize = models.Normalize()

    generator = torch.Generator().manual_seed(5)
    features = {
        "token_embeddings": torch.randn(2, _MAX_LENGTH, _HIDDEN_DIM, generator = generator),
        "attention_mask": torch.ones(2, _MAX_LENGTH, dtype = torch.long),
    }
    for module in (pooling, sparse_module, normalize):
        features = module(features)

    assert "sparse_embedding" in features, "the ST chain dropped the custom key"
    assert features["sparse_embedding"].shape == (2, _SPARSE_DIM)
    assert features["sentence_embedding"].shape == (2, _HIDDEN_DIM)
