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

"""Tests for `SpladeHead`, the UEmbed (Qwen3.5) sparse pooling sidecar.

UEmbed ships `sparse_weights.pt` next to the backbone weights: `num_eos_tokens` linear
heads (`sparse_lm_heads` + `sparse_bias`) that project hidden states onto vocabulary
logits. Two pooling modes build the sparse vector (paper Eq. 3-4):

- `splade.last`: head `i` reads the hidden state at `last_index - ((N - 1) - i)`, i.e.
  head 0 reads the first EOS slot and head N-1 the last one; the per-head logits are
  concatenated and passed through `log1p(relu(.))`.
- `splade.max`: head 0 is applied to every position, `log1p(relu(.))` is taken, padding
  is masked out, and the maximum over the sequence wins.

Everything here is CPU-only, deterministic (seeded synthetic tensors, no download) and
torch-only. Every value assertion runs against an INDEPENDENT python-loop oracle built
from the paper formula -- never against a copy of the module's own expression -- so a
mistake mirrored into both sides cannot pass.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SPLADE_SOURCE_PATH = _REPO_ROOT / "unsloth" / "models" / "uembed_splade.py"

_HIDDEN_DIM = 4
# Deliberately unequal per-head vocabularies: a `16 * V` shortcut cannot pass the
# concatenated-dimension assertion, only a real `sum(V_i)` can.
_HEAD_DIMS = [3, 4, 5] * 5 + [3]
_NUM_EOS = 16


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _load_uembed_splade():
    """Load `unsloth.models.uembed_splade`, falling back to a direct file load.

    `import unsloth` runs the accelerator / unsloth_zoo gate, which legitimately refuses
    to import on a CPU-only machine. The splade module depends on torch only, so the
    fallback executes the exact same source file -- it loads it, it does not stub it out.
    """
    try:
        from unsloth.models import uembed_splade  # noqa: PLC0415

        return uembed_splade
    except Exception:  # accelerator gate / missing unsloth_zoo / heavy optional deps
        pass

    name = "unsloth_uembed_splade_direct"
    if name in sys.modules:
        return sys.modules[name]
    assert _SPLADE_SOURCE_PATH.exists(), f"missing module file: {_SPLADE_SOURCE_PATH}"
    spec = importlib.util.spec_from_file_location(name, _SPLADE_SOURCE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def splade():
    return _load_uembed_splade()


def _make_weights(head_dims, hidden_dim=_HIDDEN_DIM, seed=0):
    generator = torch.Generator().manual_seed(seed)
    heads = [torch.randn(dim, hidden_dim, generator=generator) for dim in head_dims]
    biases = [torch.randn(dim, generator=generator) for dim in head_dims]
    return heads, biases


def _write_checkpoint(directory: Path, head_dims=_HEAD_DIMS, num_eos_tokens=_NUM_EOS):
    """Write a synthetic `sparse_weights.pt` + `sparse_info.json` UEmbed checkpoint."""
    heads, biases = _make_weights(head_dims)
    torch.save(
        {"sparse_lm_heads": heads, "sparse_bias": biases},
        directory / "sparse_weights.pt",
    )
    (directory / "sparse_info.json").write_text(
        json.dumps({"num_eos_tokens": num_eos_tokens}), encoding="utf-8"
    )
    return heads, biases


@pytest.fixture
def checkpoint(tmp_path):
    heads, biases = _write_checkpoint(tmp_path)
    return tmp_path, heads, biases


def _hidden(batch, length, dim=_HIDDEN_DIM, seed=7):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, length, dim, generator=generator)


def _right_padded_mask(lengths, total_length):
    mask = torch.zeros(len(lengths), total_length, dtype=torch.long)
    for row, length in enumerate(lengths):
        mask[row, :length] = 1
    return mask


def _oracle_last_index(mask_row) -> int:
    """Independent oracle: the last non-padding position (plain python scan)."""
    positions = [index for index, value in enumerate(mask_row.tolist()) if value]
    assert positions, "oracle called on an all-padding row"
    return positions[-1]


def _oracle_linear(weight, bias, vector) -> list[float]:
    """`F.linear` rewritten as scalar python loops: dot product per output row."""
    return [
        sum(float(w) * float(v) for w, v in zip(weight[row].tolist(), vector.tolist()))
        + float(bias[row])
        for row in range(weight.shape[0])
    ]


def _oracle_splade_last(hidden, mask, heads, biases, num_eos) -> torch.Tensor:
    """Paper Eq. 3-4 by hand: per-EOS-slot head, concatenated, `log1p(relu(.))`."""
    rows = []
    for batch in range(hidden.shape[0]):
        last = _oracle_last_index(mask[batch])
        values: list[float] = []
        for index in range(num_eos):
            position = last - ((num_eos - 1) - index)
            logits = _oracle_linear(heads[index], biases[index], hidden[batch, position])
            values.extend(math.log1p(logit) if logit > 0.0 else 0.0 for logit in logits)
        rows.append(values)
    return torch.tensor(rows, dtype=torch.float32)


def _oracle_splade_max(hidden, mask, head, bias) -> torch.Tensor:
    """Masked max by hand: scan every unmasked position, keep the largest weight."""
    rows = []
    for batch in range(hidden.shape[0]):
        best: list[float] | None = None
        for position in range(hidden.shape[1]):
            if not mask[batch, position]:
                continue
            logits = _oracle_linear(head, bias, hidden[batch, position])
            weights = [math.log1p(logit) if logit > 0.0 else 0.0 for logit in logits]
            best = weights if best is None else [max(a, b) for a, b in zip(best, weights)]
        assert best is not None, "oracle called on an all-padding row"
        rows.append(best)
    return torch.tensor(rows, dtype=torch.float32)


# --------------------------------------------------------------------------------------
# construction / checkpoint loading
# --------------------------------------------------------------------------------------
def test_from_checkpoint_reads_heads_biases_and_num_eos(splade, checkpoint):
    directory, heads, biases = checkpoint
    head = splade.SpladeHead.from_checkpoint(str(directory))

    assert head.num_heads == len(heads)
    assert head.num_eos_tokens == _NUM_EOS
    for loaded, expected in zip(head.sparse_lm_heads, heads):
        assert torch.equal(loaded.detach(), expected)
    for loaded, expected in zip(head.sparse_bias, biases):
        assert torch.equal(loaded.detach(), expected)


def test_loaded_heads_are_trainable_parameters(splade, checkpoint):
    directory, heads, _ = checkpoint
    head = splade.SpladeHead.from_checkpoint(str(directory))

    assert all(isinstance(p, torch.nn.Parameter) for p in head.sparse_lm_heads)
    assert all(p.requires_grad for p in head.sparse_lm_heads)
    assert all(p.requires_grad for p in head.sparse_bias)
    # Visible to an optimizer / `modules_to_save` collection: weights + biases.
    assert len(list(head.parameters())) == 2 * len(heads)


def test_checkpoint_loading_is_independent_of_head_count(splade, tmp_path):
    """No hardcoded 16: a 3-head checkpoint loads and pools with 3 heads."""
    head_dims = [2, 3, 4]
    _write_checkpoint(tmp_path, head_dims=head_dims, num_eos_tokens=len(head_dims))
    head = splade.SpladeHead.from_checkpoint(str(tmp_path))

    hidden = _hidden(2, 6)
    mask = _right_padded_mask([6, 5], 6)
    output = head(hidden, mask, mode="splade.last")

    assert head.num_heads == len(head_dims)
    assert output.shape == (2, sum(head_dims))


# --------------------------------------------------------------------------------------
# splade.last
# --------------------------------------------------------------------------------------
def test_splade_last_matches_independent_oracle(splade, checkpoint):
    directory, heads, biases = checkpoint
    head = splade.SpladeHead.from_checkpoint(str(directory))

    hidden = _hidden(3, 20)
    mask = _right_padded_mask([20, 18, 17], 20)
    output = head(hidden, mask, mode="splade.last")
    expected = _oracle_splade_last(hidden, mask, heads, biases, _NUM_EOS)

    assert output.shape == (3, sum(_HEAD_DIMS))
    assert torch.allclose(output, expected, atol=1e-6)


def test_splade_last_dimension_is_the_sum_of_head_vocabularies(splade, checkpoint):
    directory, heads, _ = checkpoint
    head = splade.SpladeHead.from_checkpoint(str(directory))

    output = head(_hidden(2, 20), _right_padded_mask([20, 16], 20), mode="splade.last")

    assert output.shape[-1] == sum(int(w.shape[0]) for w in heads)
    assert output.shape[-1] != len(heads) * int(heads[0].shape[0])  # not `16 * V`


def test_splade_last_is_non_negative_and_actually_clamps(splade, checkpoint):
    """`log1p(relu(x))` floors at 0; the reversed order would produce NaN for x < -1."""
    directory, _, _ = checkpoint
    head = splade.SpladeHead.from_checkpoint(str(directory))

    output = head(_hidden(3, 20), _right_padded_mask([20, 18, 17], 20), mode="splade.last")

    assert torch.isfinite(output).all()
    assert float(output.min()) >= 0.0
    assert bool((output == 0.0).any()), "relu never clamped: test inputs are too tame"
    assert bool((output > 0.0).any())


def test_splade_last_uses_a_distinct_head_per_eos_slot(splade, checkpoint):
    """Head i must read `last - ((N-1) - i)`; a shared position collapses the vector."""
    directory, heads, biases = checkpoint
    head = splade.SpladeHead.from_checkpoint(str(directory))

    hidden = _hidden(1, 20)
    mask = _right_padded_mask([20], 20)
    output = head(hidden, mask, mode="splade.last")[0]

    # First head's block equals head 0 applied at `last - 15`, not at `last`.
    block = output[: _HEAD_DIMS[0]]
    at_offset = _oracle_linear(heads[0], biases[0], hidden[0, 19 - (_NUM_EOS - 1)])
    at_last = _oracle_linear(heads[0], biases[0], hidden[0, 19])
    expected = torch.tensor(
        [math.log1p(v) if v > 0.0 else 0.0 for v in at_offset], dtype=torch.float32
    )
    wrong = torch.tensor(
        [math.log1p(v) if v > 0.0 else 0.0 for v in at_last], dtype=torch.float32
    )
    assert torch.allclose(block, expected, atol=1e-6)
    assert not torch.allclose(block, wrong, atol=1e-6)


# --------------------------------------------------------------------------------------
# splade.max
# --------------------------------------------------------------------------------------
def test_splade_max_matches_independent_masked_max_oracle(splade, checkpoint):
    directory, heads, biases = checkpoint
    head = splade.SpladeHead.from_checkpoint(str(directory))

    hidden = _hidden(3, 12)
    mask = _right_padded_mask([12, 9, 5], 12)
    output = head(hidden, mask, mode="splade.max")
    expected = _oracle_splade_max(hidden, mask, heads[0], biases[0])

    assert output.shape == (3, _HEAD_DIMS[0])
    assert torch.allclose(output, expected, atol=1e-6)
    assert float(output.min()) >= 0.0


def test_splade_max_excludes_padded_positions(splade, checkpoint):
    """Padding carries huge activations: if it leaked into the max, the values explode."""
    directory, heads, biases = checkpoint
    head = splade.SpladeHead.from_checkpoint(str(directory))

    hidden = _hidden(2, 10)
    mask = _right_padded_mask([6, 4], 10)
    # Align the padding with head 0's first row so its logit is guaranteed huge and
    # positive (`1e6 * ||row||^2`), whatever sign the random weights happen to have.
    pad_vector = 1e6 * heads[0][0]
    hidden[0, 6:] = pad_vector
    hidden[1, 4:] = pad_vector
    leaked = math.log1p(1e6 * float(heads[0][0].dot(heads[0][0])) + float(biases[0][0]))
    assert leaked > 12.0, "padding was not made loud enough to detect a leak"

    output = head(hidden, mask, mode="splade.max")
    expected = _oracle_splade_max(hidden, mask, heads[0], biases[0])

    assert torch.allclose(output, expected, atol=1e-6)
    assert float(output.max()) < 5.0  # a leaked pad position would land near `leaked`


# --------------------------------------------------------------------------------------
# gradients: the sidecar must actually train
# --------------------------------------------------------------------------------------
def test_gradients_reach_every_head_and_bias(splade, checkpoint):
    directory, _, _ = checkpoint
    head = splade.SpladeHead.from_checkpoint(str(directory))

    output = head(_hidden(3, 20), _right_padded_mask([20, 18, 17], 20), mode="splade.last")
    output.sum().backward()

    for index, (weight, bias) in enumerate(zip(head.sparse_lm_heads, head.sparse_bias)):
        assert weight.grad is not None, f"head {index} received no gradient"
        assert bias.grad is not None, f"bias {index} received no gradient"
        assert float(weight.grad.norm()) > 0.0, f"head {index} gradient is all zero"


def test_two_optimizer_steps_change_the_head_weights(splade, checkpoint):
    directory, _, _ = checkpoint
    head = splade.SpladeHead.from_checkpoint(str(directory))
    before = [w.detach().clone() for w in head.sparse_lm_heads]

    hidden = _hidden(3, 20)
    mask = _right_padded_mask([20, 18, 17], 20)
    optimizer = torch.optim.SGD(head.parameters(), lr=0.1)
    grad_norms = []
    for _ in range(2):
        optimizer.zero_grad()
        loss = (head(hidden, mask, mode="splade.last") - 1.0).pow(2).mean()
        loss.backward()
        grad_norms.append(float(head.sparse_lm_heads[0].grad.norm()))
        optimizer.step()

    assert all(norm > 0.0 for norm in grad_norms)
    assert any(
        not torch.equal(old, new.detach())
        for old, new in zip(before, head.sparse_lm_heads)
    )


def test_splade_max_gradients_reach_the_first_head(splade, checkpoint):
    directory, _, _ = checkpoint
    head = splade.SpladeHead.from_checkpoint(str(directory))

    head(_hidden(2, 10), _right_padded_mask([10, 7], 10), mode="splade.max").sum().backward()

    assert head.sparse_lm_heads[0].grad is not None
    assert float(head.sparse_lm_heads[0].grad.norm()) > 0.0


# --------------------------------------------------------------------------------------
# malformed input: fail loudly, never return a silently wrong vector
# --------------------------------------------------------------------------------------
def test_splade_last_without_eos_tokens_raises(splade, tmp_path):
    _write_checkpoint(tmp_path, num_eos_tokens=0)
    head = splade.SpladeHead.from_checkpoint(str(tmp_path))

    assert head.num_eos_tokens == 0
    with pytest.raises(ValueError, match="Unsloth"):
        head(_hidden(2, 20), _right_padded_mask([20, 18], 20), mode="splade.last")


def test_splade_modes_without_loaded_heads_raise(splade):
    head = splade.SpladeHead(sparse_lm_heads=[], sparse_bias=[], num_eos_tokens=_NUM_EOS)

    assert head.num_heads == 0
    for mode in ("splade.last", "splade.max"):
        with pytest.raises(ValueError, match="Unsloth"):
            head(_hidden(2, 20), _right_padded_mask([20, 18], 20), mode=mode)


def test_missing_sparse_weights_file_raises(splade, tmp_path):
    (tmp_path / "sparse_info.json").write_text(
        json.dumps({"num_eos_tokens": _NUM_EOS}), encoding="utf-8"
    )
    with pytest.raises(FileNotFoundError, match="Unsloth"):
        splade.SpladeHead.from_checkpoint(str(tmp_path))


def test_checkpoint_missing_keys_raises(splade, tmp_path):
    heads, _ = _make_weights(_HEAD_DIMS)
    torch.save({"sparse_lm_heads": heads}, tmp_path / "sparse_weights.pt")
    with pytest.raises(ValueError, match="Unsloth"):
        splade.SpladeHead.from_checkpoint(str(tmp_path))


def test_mismatched_head_and_bias_counts_raise(splade):
    heads, biases = _make_weights(_HEAD_DIMS)
    with pytest.raises(ValueError, match="Unsloth"):
        splade.SpladeHead(heads, biases[:-1], num_eos_tokens=_NUM_EOS)


def test_more_eos_tokens_than_heads_raises(splade, tmp_path):
    _write_checkpoint(tmp_path, head_dims=[3, 3], num_eos_tokens=4)
    head = splade.SpladeHead.from_checkpoint(str(tmp_path))
    with pytest.raises(ValueError, match="Unsloth"):
        head(_hidden(1, 8), _right_padded_mask([8], 8), mode="splade.last")


def test_sequence_shorter_than_the_eos_block_raises(splade, checkpoint):
    directory, _, _ = checkpoint
    head = splade.SpladeHead.from_checkpoint(str(directory))
    # last index 9 < num_eos - 1 == 15, so head 0 would index a negative position.
    with pytest.raises(ValueError, match="Unsloth"):
        head(_hidden(1, 20), _right_padded_mask([10], 20), mode="splade.last")


def test_all_padding_row_raises(splade, checkpoint):
    directory, _, _ = checkpoint
    head = splade.SpladeHead.from_checkpoint(str(directory))
    mask = _right_padded_mask([20, 0], 20)
    for mode in ("splade.last", "splade.max"):
        with pytest.raises(ValueError, match="Unsloth"):
            head(_hidden(2, 20), mask, mode=mode)


def test_unknown_mode_raises(splade, checkpoint):
    directory, _, _ = checkpoint
    head = splade.SpladeHead.from_checkpoint(str(directory))
    with pytest.raises(ValueError, match="Unsloth"):
        head(_hidden(1, 20), _right_padded_mask([20], 20), mode="splade.mean")


def test_empty_batch_returns_empty_vectors(splade, checkpoint):
    directory, _, _ = checkpoint
    head = splade.SpladeHead.from_checkpoint(str(directory))

    hidden = torch.zeros(0, 20, _HIDDEN_DIM)
    mask = torch.zeros(0, 20, dtype=torch.long)

    assert head(hidden, mask, mode="splade.last").shape == (0, sum(_HEAD_DIMS))
    assert head(hidden, mask, mode="splade.max").shape == (0, _HEAD_DIMS[0])


# --------------------------------------------------------------------------------------
# opt-in surface
# --------------------------------------------------------------------------------------
def test_only_splade_modes_are_recognised(splade):
    assert splade.is_splade_pooling_mode("splade.last")
    assert splade.is_splade_pooling_mode("splade.max")
    for mode in ("mean", "cls", "lasttoken", "offset_lasttoken", None, 16):
        assert not splade.is_splade_pooling_mode(mode)
