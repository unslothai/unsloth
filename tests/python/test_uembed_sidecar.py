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

"""Tests for the adapter-level UEmbed sparse sidecar (save next to the LoRA adapter).

The SPLADE heads are trained, so they must be written out with the adapter and read back
when the saved directory is loaded again. They stay a SIDECAR file (`sparse_weights.pt` +
`sparse_info.json`), never folded into the merged safetensors, which is what lets them
survive a `merged_16bit` export.

Two properties carry the weight here:

  * opt-in - a model with no SPLADE head writes NO sidecar and leaves the save directory
    byte-for-byte as it was, so every existing dense embedder is unaffected;
  * attach-aware reload - loading into a model that already carries a head REPOPULATES
    that head instead of appending a second one, which would silently double the sparse
    dimension and break the pipeline.

`unsloth` and `sentence_transformers` are not importable on a CPU-only box, so the real
sidecar module is loaded from its source file and exercised against a stub pipeline that
mimics the sentence-transformers module chain. CPU-only, seeded, download-free; every save
goes to pytest's `tmp_path`.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODELS_DIR = _REPO_ROOT / "unsloth" / "models"

_HIDDEN_DIM = 4
# Deliberately neither 16 nor uniform: the head count and every head width must come from
# the checkpoint, not from a constant.
_HEAD_DIMS = [3, 4, 5]
_NUM_EOS = 3

_SPARSE_WEIGHTS = "sparse_weights.pt"
_SPARSE_INFO = "sparse_info.json"


# --------------------------------------------------------------------------------------
# module loading
# --------------------------------------------------------------------------------------
def _load_from_source(module_name: str, file_name: str):
    """Execute the real source file directly.

    `import unsloth` runs the accelerator / unsloth_zoo gate, which legitimately refuses to
    import on a CPU-only machine. The modules under test depend on torch only, so this
    fallback runs the exact same source -- it loads it, it does not stub it out. The module
    names match the ones the sources themselves use, so every loader sees ONE copy of each
    module (and `isinstance` keeps working across them).
    """
    if module_name in sys.modules:
        return sys.modules[module_name]
    source_path = _MODELS_DIR / file_name
    assert source_path.exists(), f"missing module file: {source_path}"
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope = "module")
def splade():
    try:
        from unsloth.models import uembed_splade  # noqa: PLC0415
        return uembed_splade
    except Exception:  # accelerator gate / missing unsloth_zoo / heavy optional deps
        return _load_from_source("unsloth_uembed_splade_direct", "uembed_splade.py")


@pytest.fixture(scope = "module")
def wiring():
    try:
        from unsloth.models import uembed_wiring  # noqa: PLC0415
        return uembed_wiring
    except Exception:
        return _load_from_source("unsloth_uembed_wiring_direct", "uembed_wiring.py")


@pytest.fixture(scope = "module")
def sidecar():
    try:
        from unsloth.models import uembed_sidecar  # noqa: PLC0415
        return uembed_sidecar
    except Exception:
        return _load_from_source("unsloth_uembed_sidecar_direct", "uembed_sidecar.py")


# --------------------------------------------------------------------------------------
# stub pipeline: the sentence-transformers module chain, minus sentence-transformers
# --------------------------------------------------------------------------------------
class _StubTransformer(nn.Module):
    """Stands in for the ST `Transformer` module (the backbone that owns the dtype)."""

    def __init__(self, seed: int = 5) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.table = nn.Parameter(torch.randn(7, _HIDDEN_DIM, generator = generator))

    def forward(self, features: dict) -> dict:
        features["token_embeddings"] = self.table[features["input_ids"]]
        return features


class _StubPooling(nn.Module):
    def forward(self, features: dict) -> dict:
        return features


class _StubNormalize(nn.Module):
    def forward(self, features: dict) -> dict:
        return features


class _StubSentenceTransformer(nn.Sequential):
    """Minimal serialized-chain stand-in with a stock, process-local `encode` method."""

    def __init__(self, *modules) -> None:
        super().__init__(*modules)
        self.encode_calls = 0

    def encode(self, sentences):
        self.encode_calls += 1
        return ("stock-dense", sentences)


def _make_pipeline() -> _StubSentenceTransformer:
    return _StubSentenceTransformer(_StubTransformer(), _StubPooling(), _StubNormalize())


def _make_head(splade_module, seed: int = 11):
    generator = torch.Generator().manual_seed(seed)
    heads = [torch.randn(dim, _HIDDEN_DIM, generator = generator) for dim in _HEAD_DIMS]
    biases = [torch.randn(dim, generator = generator) for dim in _HEAD_DIMS]
    return splade_module.SpladeHead(heads, biases, num_eos_tokens = _NUM_EOS)


def _sparse_heads(model, splade_module) -> list:
    return [module for module in model.modules() if isinstance(module, splade_module.SpladeHead)]


def _only_head(model, splade_module):
    heads = _sparse_heads(model, splade_module)
    assert len(heads) == 1, f"expected exactly one SpladeHead, found {len(heads)}"
    return heads[0]


def _head_values(head) -> list[torch.Tensor]:
    return [weight.detach().clone() for weight in head.sparse_lm_heads] + [
        bias.detach().clone() for bias in head.sparse_bias
    ]


def _mutate(head, offset: float = 0.25) -> None:
    """Stand in for training: move every head parameter away from its loaded value."""
    with torch.no_grad():
        for index, weight in enumerate(head.sparse_lm_heads):
            weight.add_(offset * (index + 1))
        for index, bias in enumerate(head.sparse_bias):
            bias.add_(-offset * (index + 1))


@pytest.fixture
def sparse_model(wiring, splade):
    model = _make_pipeline()
    assert wiring.attach_uembed_sparse_output(model, _make_head(splade)) is True
    return model


# --------------------------------------------------------------------------------------
# 1. opt-in: a model with no sparse head writes no sidecar and changes nothing
# --------------------------------------------------------------------------------------
def test_dense_only_save_writes_no_sidecar(sidecar, tmp_path):
    save_directory = tmp_path / "dense"
    save_directory.mkdir()
    (save_directory / "adapter_model.safetensors").write_bytes(b"adapter")
    before = {path.name: path.read_bytes() for path in save_directory.iterdir()}

    wrote = sidecar.save_uembed_sparse_sidecar(_make_pipeline(), str(save_directory))

    assert wrote is False
    after = {path.name: path.read_bytes() for path in save_directory.iterdir()}
    assert after == before, "a dense-only save directory must be left untouched"
    assert not (save_directory / _SPARSE_WEIGHTS).exists()
    assert not (save_directory / _SPARSE_INFO).exists()


def test_dense_only_load_attaches_nothing(sidecar, splade, tmp_path):
    save_directory = tmp_path / "dense"
    save_directory.mkdir()
    model = _make_pipeline()
    module_count = len(list(model.children()))

    loaded = sidecar.load_uembed_sparse_sidecar(model, str(save_directory))

    assert loaded is False
    assert len(list(model.children())) == module_count
    assert _sparse_heads(model, splade) == []


# --------------------------------------------------------------------------------------
# 2. save: the sidecar carries UEmbed's own key layout
# --------------------------------------------------------------------------------------
def test_save_writes_sparse_weights_and_info(sidecar, splade, sparse_model, tmp_path):
    save_directory = tmp_path / "sparse"
    head = _only_head(sparse_model, splade)

    wrote = sidecar.save_uembed_sparse_sidecar(sparse_model, str(save_directory))

    assert wrote is True
    weights_path = save_directory / _SPARSE_WEIGHTS
    info_path = save_directory / _SPARSE_INFO
    assert weights_path.is_file() and info_path.is_file()

    state = torch.load(weights_path, map_location = "cpu", weights_only = True)
    assert set(state) == {"sparse_lm_heads", "sparse_bias"}
    assert len(state["sparse_lm_heads"]) == len(head.sparse_lm_heads) == len(_HEAD_DIMS)
    assert len(state["sparse_bias"]) == len(head.sparse_bias)
    assert [tuple(weight.shape) for weight in state["sparse_lm_heads"]] == [
        (dim, _HIDDEN_DIM) for dim in _HEAD_DIMS
    ]
    assert [tuple(bias.shape) for bias in state["sparse_bias"]] == [(dim,) for dim in _HEAD_DIMS]

    info = json.loads(info_path.read_text(encoding = "utf-8"))
    assert info["num_eos_tokens"] == _NUM_EOS


def test_save_keeps_the_merged_weights_free_of_sparse_heads(sidecar, sparse_model, tmp_path):
    """merged_16bit round-trip: the heads land beside the export, never inside it."""
    save_directory = tmp_path / "merged"
    save_directory.mkdir()
    merged = save_directory / "model.safetensors"
    merged.write_bytes(b"merged-16bit-weights")

    assert sidecar.save_uembed_sparse_sidecar(sparse_model, str(save_directory)) is True

    assert merged.read_bytes() == b"merged-16bit-weights"
    assert (save_directory / _SPARSE_WEIGHTS).is_file()


def test_save_preserves_other_sparse_info_keys(sidecar, sparse_model, tmp_path):
    save_directory = tmp_path / "sparse"
    save_directory.mkdir()
    (save_directory / _SPARSE_INFO).write_text(
        json.dumps({"num_eos_tokens": 0, "sparse_dim": 12}), encoding = "utf-8"
    )

    assert sidecar.save_uembed_sparse_sidecar(sparse_model, str(save_directory)) is True

    info = json.loads((save_directory / _SPARSE_INFO).read_text(encoding = "utf-8"))
    assert info == {"num_eos_tokens": _NUM_EOS, "sparse_dim": 12}


# --------------------------------------------------------------------------------------
# 3. reload: the SAVED values come back, into a trainable head
# --------------------------------------------------------------------------------------
def test_reload_restores_the_saved_head_into_a_fresh_model(sidecar, splade, sparse_model, tmp_path):
    """Stale-state probe: the head is trained (mutated) BEFORE the save."""
    save_directory = tmp_path / "sparse"
    trained = _only_head(sparse_model, splade)
    _mutate(trained)
    expected = _head_values(trained)
    assert sidecar.save_uembed_sparse_sidecar(sparse_model, str(save_directory)) is True

    fresh = _make_pipeline()
    module_count = len(list(fresh.children()))

    loaded = sidecar.load_uembed_sparse_sidecar(fresh, str(save_directory))

    assert loaded is True
    assert len(list(fresh.children())) == module_count + 1
    reloaded = _only_head(fresh, splade)
    assert reloaded.num_eos_tokens == _NUM_EOS
    for saved, restored in zip(expected, _head_values(reloaded)):
        assert torch.allclose(saved, restored)
    assert all(weight.requires_grad for weight in reloaded.sparse_lm_heads)
    assert all(bias.requires_grad for bias in reloaded.sparse_bias)


def test_reload_is_attach_aware(sidecar, wiring, splade, sparse_model, tmp_path):
    """Loading into a model that already has a head must not stack a second one."""
    save_directory = tmp_path / "sparse"
    trained = _only_head(sparse_model, splade)
    _mutate(trained)
    expected = _head_values(trained)
    assert sidecar.save_uembed_sparse_sidecar(sparse_model, str(save_directory)) is True

    # A model rebuilt from the saved directory already carries the checkpoint's head.
    target = _make_pipeline()
    wiring.attach_uembed_sparse_output(target, _make_head(splade, seed = 99))
    module_count = len(list(target.children()))
    stale = _head_values(_only_head(target, splade))

    loaded = sidecar.load_uembed_sparse_sidecar(target, str(save_directory))

    assert loaded is True
    assert len(list(target.children())) == module_count, "the reload appended a second module"
    reloaded = _only_head(target, splade)
    for saved, restored in zip(expected, _head_values(reloaded)):
        assert torch.allclose(saved, restored)
    assert any(
        not torch.allclose(old, restored) for old, restored in zip(stale, _head_values(reloaded))
    ), "the reload kept the stale in-memory values"
    assert all(weight.requires_grad for weight in reloaded.sparse_lm_heads)


def test_reload_survives_a_merged_export_round_trip(sidecar, splade, sparse_model, tmp_path):
    """Save -> merged export overwrites the weights -> reload still finds the heads."""
    save_directory = tmp_path / "roundtrip"
    trained = _only_head(sparse_model, splade)
    _mutate(trained, offset = 0.5)
    expected = _head_values(trained)

    assert sidecar.save_uembed_sparse_sidecar(sparse_model, str(save_directory)) is True
    # The merged_16bit export rewrites the backbone weights after the sidecar is written.
    (save_directory / "model.safetensors").write_bytes(b"merged-16bit-weights")
    (save_directory / "config.json").write_text("{}", encoding = "utf-8")

    fresh = _make_pipeline()
    assert sidecar.load_uembed_sparse_sidecar(fresh, str(save_directory)) is True
    for saved, restored in zip(expected, _head_values(_only_head(fresh, splade))):
        assert torch.allclose(saved, restored)


def test_reload_reads_the_saved_num_eos_tokens(sidecar, splade, wiring, tmp_path):
    """The EOS-block size is checkpoint data, never a constant."""
    save_directory = tmp_path / "sparse"
    model = _make_pipeline()
    generator = torch.Generator().manual_seed(3)
    heads = [torch.randn(dim, _HIDDEN_DIM, generator = generator) for dim in _HEAD_DIMS]
    biases = [torch.randn(dim, generator = generator) for dim in _HEAD_DIMS]
    wiring.attach_uembed_sparse_output(model, splade.SpladeHead(heads, biases, num_eos_tokens = 2))

    assert sidecar.save_uembed_sparse_sidecar(model, str(save_directory)) is True
    assert json.loads((save_directory / _SPARSE_INFO).read_text(encoding = "utf-8")) == {
        "num_eos_tokens": 2
    }

    fresh = _make_pipeline()
    assert sidecar.load_uembed_sparse_sidecar(fresh, str(save_directory)) is True
    assert _only_head(fresh, splade).num_eos_tokens == 2


def test_reload_replaces_a_head_of_a_different_shape(sidecar, wiring, splade, tmp_path):
    """A sidecar from another checkpoint still lands as exactly one head."""
    save_directory = tmp_path / "sparse"
    source = _make_pipeline()
    generator = torch.Generator().manual_seed(7)
    heads = [torch.randn(dim, _HIDDEN_DIM, generator = generator) for dim in (2, 6)]
    biases = [torch.randn(dim, generator = generator) for dim in (2, 6)]
    wiring.attach_uembed_sparse_output(source, splade.SpladeHead(heads, biases, num_eos_tokens = 2))
    expected = _head_values(_only_head(source, splade))
    assert sidecar.save_uembed_sparse_sidecar(source, str(save_directory)) is True

    target = _make_pipeline()
    wiring.attach_uembed_sparse_output(target, _make_head(splade))
    module_count = len(list(target.children()))

    assert sidecar.load_uembed_sparse_sidecar(target, str(save_directory)) is True

    assert len(list(target.children())) == module_count
    reloaded = _only_head(target, splade)
    assert reloaded.num_heads == 2
    for saved, restored in zip(expected, _head_values(reloaded)):
        assert torch.allclose(saved, restored)


def test_reloaded_head_produces_the_saved_sparse_vector(sidecar, splade, sparse_model, tmp_path):
    """End-to-end: same hidden states in, same sparse vector out, after a round-trip."""
    save_directory = tmp_path / "sparse"
    trained = _only_head(sparse_model, splade)
    _mutate(trained)
    generator = torch.Generator().manual_seed(23)
    hidden = torch.randn(2, 6, _HIDDEN_DIM, generator = generator)
    mask = torch.ones(2, 6, dtype = torch.long)
    with torch.no_grad():
        expected = trained(hidden, mask, splade.SPLADE_LAST)

    assert sidecar.save_uembed_sparse_sidecar(sparse_model, str(save_directory)) is True
    fresh = _make_pipeline()
    assert sidecar.load_uembed_sparse_sidecar(fresh, str(save_directory)) is True
    with torch.no_grad():
        restored = _only_head(fresh, splade)(hidden, mask, splade.SPLADE_LAST)

    assert torch.allclose(expected, restored)


def test_reload_is_idempotent(sidecar, splade, sparse_model, tmp_path):
    save_directory = tmp_path / "sparse"
    assert sidecar.save_uembed_sparse_sidecar(sparse_model, str(save_directory)) is True

    fresh = _make_pipeline()
    module_count = len(list(fresh.children()))
    assert sidecar.load_uembed_sparse_sidecar(fresh, str(save_directory)) is True
    assert sidecar.load_uembed_sparse_sidecar(fresh, str(save_directory)) is True

    assert len(list(fresh.children())) == module_count + 1
    assert len(_sparse_heads(fresh, splade)) == 1


def test_attach_aware_reload_restores_encode_patch_once(
    sidecar, wiring, splade, sparse_model, tmp_path
):
    """A sparse module rebuilt by modules.json must regain its non-serialized wrapper."""
    save_directory = tmp_path / "sparse"
    assert sidecar.save_uembed_sparse_sidecar(sparse_model, str(save_directory)) is True

    target = _make_pipeline()
    target.add_module("3", wiring.UEmbedSparseOutput(_make_head(splade, seed = 99)))
    baseline = target.encode("document")
    assert not hasattr(target, "_unsloth_uembed_original_encode")

    assert sidecar.load_uembed_sparse_sidecar(target, str(save_directory)) is True
    patched_encode = target.encode
    assert hasattr(target, "_unsloth_uembed_original_encode")
    assert target.encode("document", output_mode = "dense") == baseline
    assert target.encode_calls == 2

    assert sidecar.load_uembed_sparse_sidecar(target, str(save_directory)) is True
    assert target.encode is patched_encode
    assert len(_sparse_heads(target, splade)) == 1
