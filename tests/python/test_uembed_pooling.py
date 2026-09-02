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

"""Tests for the opt-in offset last-token pooling used by UEmbed (Qwen3.5).

UEmbed appends `num_eos_tokens` `<|endoftext|>` tokens after the content and takes the
dense vector from the hidden state that *precedes* that block, i.e. at
`last_non_pad_index - num_eos_tokens`, not at the true last token.

Three layers, all CPU-only and deterministic (synthetic tensors, no model download):

- Baseline characterization: pins that the stock pooling path is untouched by this
  feature. These pass on the pre-change tree (before any wiring exists).
- Behavioural tests for the new module: index math is checked against an independent
  oracle (`max(i for i where mask[i] == 1)`), never against a copy of the
  implementation formula.
- Structural wiring test: proves the offset module is selected only for the offset
  mode and that stock `Pooling` remains the else-branch, without importing unsloth
  (the package import needs an accelerator + unsloth_zoo, which CPU boxes lack).
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ST_SOURCE_PATH = _REPO_ROOT / "unsloth" / "models" / "sentence_transformer.py"
_POOLING_SOURCE_PATH = _REPO_ROOT / "unsloth" / "models" / "uembed_pooling.py"

# The pooling modes sentence-transformers ships and unsloth auto-detects today. The
# offset mode must never appear here: it is opt-in only.
_STOCK_POOLING_MODES = frozenset(
    {"cls", "mean", "max", "mean_sqrt_len", "weightedmean", "lasttoken"}
)


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _load_uembed_pooling():
    """Load `unsloth.models.uembed_pooling`, falling back to a direct file load.

    `import unsloth` runs the accelerator / unsloth_zoo gate, which legitimately refuses
    to import on a CPU-only machine. The pooling module itself depends on torch only, so
    the fallback executes the exact same source file. A broken module still fails here --
    the fallback loads the file, it does not stub it out.
    """
    try:
        from unsloth.models import uembed_pooling  # noqa: PLC0415

        return uembed_pooling
    except Exception:  # accelerator gate / missing unsloth_zoo / heavy optional deps
        pass

    name = "unsloth_uembed_pooling_direct"
    if name in sys.modules:
        return sys.modules[name]
    assert _POOLING_SOURCE_PATH.exists(), f"missing module file: {_POOLING_SOURCE_PATH}"
    spec = importlib.util.spec_from_file_location(name, _POOLING_SOURCE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def uembed():
    return _load_uembed_pooling()


def _st_source_tree() -> ast.Module:
    return ast.parse(_ST_SOURCE_PATH.read_text(encoding="utf-8"))


def _function_def(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {_ST_SOURCE_PATH}")


def _calls_to(node: ast.AST, func_name: str) -> list[ast.Call]:
    return [
        child
        for child in ast.walk(node)
        if isinstance(child, ast.Call)
        and isinstance(child.func, ast.Name)
        and child.func.id == func_name
    ]


def _hidden(batch: int, length: int, dim: int) -> torch.Tensor:
    """Every (batch, position) row holds unique values, so a wrong index cannot pass."""
    return torch.arange(batch * length * dim, dtype=torch.float32).reshape(batch, length, dim)


def _last_non_pad_index(mask_row: torch.Tensor) -> int:
    """Independent oracle: the last position that is not padding.

    Deliberately NOT the implementation formula (`(cumsum * mask).argmax`) and NOT
    `mask.sum() - 1` (which is only correct for right padding).
    """
    positions = [index for index, value in enumerate(mask_row.tolist()) if value]
    assert positions, "oracle called on an all-padding row"
    return positions[-1]


def _mixed_layout_mask(length: int = 24) -> torch.Tensor:
    """Right-padded rows, a fully unpadded row, a short row, and a left-padded row."""
    mask = torch.zeros(4, length, dtype=torch.long)
    mask[0, :20] = 1  # right padding
    mask[1, :] = 1  # no padding at all
    mask[2, :17] = 1  # shortest row that still survives num_eos_tokens = 16
    mask[3, 6:] = 1  # left padding -> last index is length - 1
    return mask


# --------------------------------------------------------------------------------------
# baseline characterization -- these pass BEFORE the feature exists
# --------------------------------------------------------------------------------------
def test_baseline_module_assembly_still_builds_stock_pooling_with_the_passed_mode():
    """The fallback pipeline must keep building sentence-transformers' own Pooling."""
    load_modules = _function_def(_st_source_tree(), "_load_modules")

    pooling_calls = _calls_to(load_modules, "Pooling")
    assert len(pooling_calls) == 1, "expected exactly one stock Pooling construction"

    keywords = {kw.arg: kw.value for kw in pooling_calls[0].keywords}
    assert "word_embedding_dimension" in keywords
    mode_argument = keywords["pooling_mode"]
    assert isinstance(mode_argument, ast.Name) and mode_argument.id == "pooling_mode", (
        "stock Pooling must still receive the caller's pooling_mode unchanged"
    )
    assert _calls_to(load_modules, "Normalize"), "Normalize must stay in the pipeline"


def test_baseline_auto_detected_pooling_modes_stay_stock_only():
    """Auto-detection must never hand back an offset mode: the feature is opt-in."""
    read_pooling_mode = _function_def(_st_source_tree(), "_read_pooling_mode")

    detected: set[str] = set()
    for node in ast.walk(read_pooling_mode):
        if isinstance(node, ast.Dict) and any(
            isinstance(key, ast.Constant) and key.value == "pooling_mode_lasttoken"
            for key in node.keys
        ):
            detected = {value.value for value in node.values if isinstance(value, ast.Constant)}
    assert detected == _STOCK_POOLING_MODES, f"pooling auto-detection changed: {detected}"


def test_baseline_stock_lasttoken_pools_the_last_non_pad_position():
    """Characterizes stock sentence-transformers `lasttoken` against the oracle."""
    pytest.importorskip("sentence_transformers", reason="sentence-transformers not importable")
    from sentence_transformers.models import Pooling

    mask = _mixed_layout_mask()
    hidden = _hidden(*mask.shape, 5)
    pooling = Pooling(word_embedding_dimension=5, pooling_mode="lasttoken")

    pooled = pooling({"token_embeddings": hidden, "attention_mask": mask})["sentence_embedding"]

    for row in range(mask.shape[0]):
        assert torch.allclose(pooled[row], hidden[row, _last_non_pad_index(mask[row])])


def test_baseline_stock_mean_pools_the_masked_mean():
    """Characterizes stock `mean` pooling so a regression there cannot slip through."""
    pytest.importorskip("sentence_transformers", reason="sentence-transformers not importable")
    from sentence_transformers.models import Pooling

    mask = _mixed_layout_mask()
    hidden = _hidden(*mask.shape, 5)
    pooling = Pooling(word_embedding_dimension=5, pooling_mode="mean")

    pooled = pooling({"token_embeddings": hidden, "attention_mask": mask})["sentence_embedding"]

    for row in range(mask.shape[0]):
        keep = mask[row].bool()
        assert torch.allclose(pooled[row], hidden[row][keep].mean(dim=0), atol=1e-5)


# --------------------------------------------------------------------------------------
# offset pooling -- index math
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("num_eos_tokens", [0, 1, 16])
def test_offset_pooling_selects_last_index_minus_num_eos_tokens(uembed, num_eos_tokens):
    mask = _mixed_layout_mask()
    hidden = _hidden(*mask.shape, 3)

    pooled = uembed.OffsetLastTokenPooling(
        word_embedding_dimension=3, num_eos_tokens=num_eos_tokens
    )({"token_embeddings": hidden, "attention_mask": mask})["sentence_embedding"]

    assert pooled.shape == (mask.shape[0], 3)
    for row in range(mask.shape[0]):
        target = _last_non_pad_index(mask[row]) - num_eos_tokens
        assert target >= 0, "layout must keep every row valid for this offset"
        assert torch.equal(pooled[row], hidden[row, target])


def test_offset_zero_reproduces_plain_lasttoken_exactly(uembed):
    mask = _mixed_layout_mask()
    hidden = _hidden(*mask.shape, 3)
    features = {"token_embeddings": hidden, "attention_mask": mask}

    pooled = uembed.OffsetLastTokenPooling(word_embedding_dimension=3, num_eos_tokens=0)(
        dict(features)
    )["sentence_embedding"]
    reference = torch.stack(
        [hidden[row, _last_non_pad_index(mask[row])] for row in range(mask.shape[0])]
    )

    assert torch.allclose(pooled, reference)


def test_offset_zero_matches_stock_lasttoken_module(uembed):
    """The opt-in module must be a drop-in for stock `lasttoken` when the offset is 0."""
    pytest.importorskip("sentence_transformers", reason="sentence-transformers not importable")
    from sentence_transformers.models import Pooling

    mask = _mixed_layout_mask()
    hidden = _hidden(*mask.shape, 5)
    features = {"token_embeddings": hidden, "attention_mask": mask}

    stock = Pooling(word_embedding_dimension=5, pooling_mode="lasttoken")(dict(features))
    offset = uembed.OffsetLastTokenPooling(word_embedding_dimension=5, num_eos_tokens=0)(
        dict(features)
    )

    assert torch.allclose(offset["sentence_embedding"], stock["sentence_embedding"])


def test_neighbouring_offsets_select_different_rows(uembed):
    """Guards the suite itself: an off-by-one implementation cannot pass silently."""
    mask = _mixed_layout_mask()
    hidden = _hidden(*mask.shape, 3)
    features = {"token_embeddings": hidden, "attention_mask": mask}

    pooled = [
        uembed.OffsetLastTokenPooling(word_embedding_dimension=3, num_eos_tokens=offset)(
            dict(features)
        )["sentence_embedding"]
        for offset in (0, 1, 2)
    ]

    assert not torch.allclose(pooled[0], pooled[1])
    assert not torch.allclose(pooled[1], pooled[2])


def test_offset_pooling_handles_a_fully_unpadded_batch(uembed):
    hidden = _hidden(2, 5, 4)
    mask = torch.ones(2, 5, dtype=torch.long)

    pooled = uembed.OffsetLastTokenPooling(word_embedding_dimension=4, num_eos_tokens=2)(
        {"token_embeddings": hidden, "attention_mask": mask}
    )["sentence_embedding"]

    assert torch.equal(pooled, hidden[:, 2])


def test_offset_pooling_defaults_to_all_ones_when_no_mask_is_supplied(uembed):
    """Mirrors sentence-transformers' Pooling, which assumes no padding without a mask."""
    hidden = _hidden(2, 5, 4)

    pooled = uembed.OffsetLastTokenPooling(word_embedding_dimension=4, num_eos_tokens=1)(
        {"token_embeddings": hidden}
    )["sentence_embedding"]

    assert torch.equal(pooled, hidden[:, 3])


def test_offset_pooling_preserves_other_features_and_tensor_metadata(uembed):
    hidden = _hidden(2, 5, 4).to(torch.float64)
    mask = torch.ones(2, 5, dtype=torch.long)
    features = {"token_embeddings": hidden, "attention_mask": mask, "input_ids": torch.zeros(2, 5)}

    out = uembed.OffsetLastTokenPooling(word_embedding_dimension=4, num_eos_tokens=0)(features)

    assert out is features, "sentence-transformers modules mutate and return the features dict"
    assert "input_ids" in out and "token_embeddings" in out
    assert out["sentence_embedding"].dtype == torch.float64
    assert out["sentence_embedding"].device == hidden.device


def test_offset_pooling_backpropagates_only_into_the_selected_positions(uembed):
    hidden = _hidden(2, 6, 3).requires_grad_(True)
    mask = torch.zeros(2, 6, dtype=torch.long)
    mask[0, :5] = 1
    mask[1, :] = 1

    pooled = uembed.OffsetLastTokenPooling(word_embedding_dimension=3, num_eos_tokens=1)(
        {"token_embeddings": hidden, "attention_mask": mask}
    )["sentence_embedding"]
    pooled.sum().backward()

    touched = (hidden.grad.abs().sum(dim=-1) > 0).nonzero(as_tuple=False).tolist()
    assert touched == [[0, 3], [1, 4]]


def test_offset_pooling_reports_the_embedding_dimension(uembed):
    pooling = uembed.OffsetLastTokenPooling(word_embedding_dimension=7, num_eos_tokens=16)

    assert pooling.get_sentence_embedding_dimension() == 7
    assert pooling.get_embedding_dimension() == 7
    assert pooling.num_eos_tokens == 16


# --------------------------------------------------------------------------------------
# offset pooling -- malformed input
# --------------------------------------------------------------------------------------
def test_offset_larger_than_the_content_raises(uembed):
    """Pinned behaviour: raise. Negative indices would silently wrap to the tail."""
    hidden = _hidden(2, 8, 3)
    mask = torch.zeros(2, 8, dtype=torch.long)
    mask[0, :8] = 1
    mask[1, :3] = 1  # only 3 real tokens, offset 4 has nothing left to point at

    with pytest.raises(ValueError, match="num_eos_tokens"):
        uembed.OffsetLastTokenPooling(word_embedding_dimension=3, num_eos_tokens=4)(
            {"token_embeddings": hidden, "attention_mask": mask}
        )


def test_all_padding_row_raises(uembed):
    hidden = _hidden(2, 4, 3)
    mask = torch.zeros(2, 4, dtype=torch.long)
    mask[0, :4] = 1

    with pytest.raises(ValueError, match="attention_mask"):
        uembed.OffsetLastTokenPooling(word_embedding_dimension=3, num_eos_tokens=0)(
            {"token_embeddings": hidden, "attention_mask": mask}
        )


def test_single_position_sequence(uembed):
    hidden = _hidden(2, 1, 3)
    mask = torch.ones(2, 1, dtype=torch.long)
    features = {"token_embeddings": hidden, "attention_mask": mask}

    pooled = uembed.OffsetLastTokenPooling(word_embedding_dimension=3, num_eos_tokens=0)(
        dict(features)
    )["sentence_embedding"]
    assert torch.equal(pooled, hidden[:, 0])

    with pytest.raises(ValueError, match="num_eos_tokens"):
        uembed.OffsetLastTokenPooling(word_embedding_dimension=3, num_eos_tokens=1)(dict(features))


def test_empty_batch_returns_an_empty_embedding_matrix(uembed):
    hidden = torch.zeros(0, 5, 3)
    mask = torch.zeros(0, 5, dtype=torch.long)

    pooled = uembed.OffsetLastTokenPooling(word_embedding_dimension=3, num_eos_tokens=2)(
        {"token_embeddings": hidden, "attention_mask": mask}
    )["sentence_embedding"]

    assert pooled.shape == (0, 3)


def test_negative_num_eos_tokens_is_rejected_at_construction(uembed):
    with pytest.raises(ValueError, match="num_eos_tokens"):
        uembed.OffsetLastTokenPooling(word_embedding_dimension=3, num_eos_tokens=-1)


# --------------------------------------------------------------------------------------
# num_eos_tokens comes from sparse_info.json
# --------------------------------------------------------------------------------------
def test_num_eos_tokens_is_read_from_sparse_info_json(uembed, tmp_path):
    (tmp_path / "sparse_info.json").write_text(
        json.dumps({"num_eos_tokens": 16, "other": "ignored"}), encoding="utf-8"
    )

    assert uembed.read_num_eos_tokens(str(tmp_path)) == 16


def test_missing_sparse_info_json_defaults_to_zero(uembed, tmp_path):
    assert uembed.read_num_eos_tokens(str(tmp_path)) == 0
    assert uembed.read_num_eos_tokens(str(tmp_path / "does-not-exist")) == 0


def test_sparse_info_json_without_the_key_defaults_to_zero(uembed, tmp_path):
    (tmp_path / "sparse_info.json").write_text(json.dumps({"unrelated": 1}), encoding="utf-8")

    assert uembed.read_num_eos_tokens(str(tmp_path)) == 0


@pytest.mark.parametrize("value", ["sixteen", -1, 1.5, None])
def test_malformed_num_eos_tokens_raises(uembed, tmp_path, value):
    (tmp_path / "sparse_info.json").write_text(
        json.dumps({"num_eos_tokens": value}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="num_eos_tokens"):
        uembed.read_num_eos_tokens(str(tmp_path))


# --------------------------------------------------------------------------------------
# opt-in wiring
# --------------------------------------------------------------------------------------
def test_offset_mode_is_never_a_stock_mode(uembed):
    for mode in _STOCK_POOLING_MODES:
        assert not uembed.is_offset_pooling_mode(mode), f"{mode} must keep stock Pooling"
    for mode in uembed.OFFSET_POOLING_MODES:
        assert uembed.is_offset_pooling_mode(mode)
    assert not uembed.is_offset_pooling_mode(None)
    assert not _STOCK_POOLING_MODES & set(uembed.OFFSET_POOLING_MODES)


def test_offset_pooling_is_wired_as_the_opt_in_branch_of_module_assembly():
    """`OffsetLastTokenPooling` may only replace `Pooling` behind the mode guard."""
    load_modules = _function_def(_st_source_tree(), "_load_modules")

    guards = [
        node
        for node in ast.walk(load_modules)
        if isinstance(node, ast.If)
        and _calls_to_any_attr(node.test, "is_offset_pooling_mode")
        and _calls_to(node, "OffsetLastTokenPooling")
    ]
    assert len(guards) == 1, "offset pooling must sit behind exactly one mode guard"

    guard = guards[0]
    assert not any(_calls_to(statement, "Pooling") for statement in guard.body), (
        "stock Pooling must not run on the offset branch"
    )
    assert any(_calls_to(statement, "Pooling") for statement in guard.orelse), (
        "stock Pooling must remain the default branch"
    )
    assert _calls_to(guard, "read_num_eos_tokens"), "num_eos_tokens must come from the checkpoint"

    for call in _calls_to(guard, "OffsetLastTokenPooling"):
        assert not any(
            isinstance(kw.value, ast.Constant) and kw.arg == "num_eos_tokens"
            for kw in call.keywords
        ), "num_eos_tokens must not be hardcoded"


def _calls_to_any_attr(node: ast.AST, func_name: str) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if name == func_name:
                return True
    return False
