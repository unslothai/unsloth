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

"""Unified dense + sparse + FLOPS training loss for UEmbed (Qwen3.5) embedders.

Paper arXiv:2608.02583 Eq. 5:

    L = L_InfoNCE_dense + lambda * L_InfoNCE_sparse + alpha_q * L_FLOPS_q + alpha_d * L_FLOPS_d

- dense InfoNCE  : cosine similarity between the anchor and candidate dense vectors, scaled
                   by `scale` (= 1 / tau_dense, default 20.0 <=> tau_dense = 0.05), labels
                   `arange(B)`, cross entropy. This is exactly sentence-transformers'
                   `MultipleNegativesRankingLoss` (in-batch negatives).
- sparse InfoNCE : inner product between the non-negative SPLADE vectors divided by the
                   sparse temperature `tau_s` (default 32.0, paper section 5.1). Sparse
                   scores have a far wider dynamic range than cosines, so the sparse
                   temperature is deliberately NOT the dense scale.
- FLOPS          : `sum_t (mean_i W[i, t])^2` (Paria et al. / SPLADE-v2), computed
                   separately over the query batch and the document batch so queries and
                   documents can be sparsified at different strengths.

The loss accepts both sentence-transformers contracts. Trainer batches contain raw tokenized
feature dictionaries, so each column is passed through the model exactly once; the UEmbed
pipeline produces `sentence_embedding` + `sparse_embedding` together in that one forward.
Already-forwarded dictionaries carrying both keys are consumed directly without another model
call. Non-negativity of the sparse vectors is guaranteed upstream by `SpladeHead`'s
`log1p(relu(.))`, so it is assumed, not re-clamped.

Torch-only, so it imports without an accelerator and without importing `unsloth`.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch import nn


# Keys in the sentence-transformers features dict. `sentence_embedding` is the stock ST
# key; `sparse_embedding` is what the UEmbed single-forward wiring adds beside it.
SENTENCE_EMBEDDING_KEY = "sentence_embedding"
SPARSE_EMBEDDING_KEY = "sparse_embedding"

# Paper-grounded defaults: tau_dense = 0.05 (scale = 20.0), tau_s = 32.0, equal query and
# document FLOPS weights. `alpha_warmup_steps = 0` keeps the regulariser constant.
DEFAULT_SCALE = 20.0
DEFAULT_TAU_SPARSE = 32.0
DEFAULT_LAMBDA_SPARSE = 1.0
DEFAULT_ALPHA = 0.01


def flops_regularizer(sparse_embeddings: torch.Tensor) -> torch.Tensor:
    """`sum_t (mean_i W[i, t])^2` for a `(batch, vocab)` batch of sparse vectors."""
    return sparse_embeddings.mean(dim = 0).pow(2).sum()


def _positive(name: str, value: Any) -> float:
    number = float(value)
    if not number > 0.0:
        raise ValueError(f"Unsloth: `{name}` must be > 0, got {value!r}.")
    return number


def _non_negative(name: str, value: Any) -> float:
    number = float(value)
    if number < 0.0:
        raise ValueError(f"Unsloth: `{name}` must be >= 0, got {value!r}.")
    return number


class UEmbedUnifiedLoss(nn.Module):
    """Dense InfoNCE + sparse InfoNCE + FLOPS, the UEmbed unified objective.

    Usage mirrors any sentence-transformers loss::

        loss = UEmbedUnifiedLoss(model)
        SentenceTransformerTrainer(model = model, loss = loss, ...)

    Raw tokenized columns from `SentenceTransformerTrainer` are forwarded through `model`
    once each. Columns that already carry both required embedding keys are not forwarded
    again, preserving the direct/precomputed contract used outside the trainer.

    Each column of the batch arrives as one features dict; column 0 is the anchors
    (queries) and every later column is a candidate (positives first, then optional hard
    negatives, which become extra in-batch candidates exactly as MNRL treats them).

    `alpha_warmup_steps > 0` enables the optional quadratic ramp described in the SPLADE
    literature: the FLOPS weights grow as `(step / warmup)^2` and then stay at full
    strength. It is off by default, i.e. the weights are constant.
    """

    def __init__(
        self,
        model: Any = None,
        lambda_sparse: float = DEFAULT_LAMBDA_SPARSE,
        alpha_q: float = DEFAULT_ALPHA,
        alpha_d: float = DEFAULT_ALPHA,
        scale: float = DEFAULT_SCALE,
        tau_s: float = DEFAULT_TAU_SPARSE,
        alpha_warmup_steps: int = 0,
    ) -> None:
        super().__init__()
        self.model = model
        self.lambda_sparse = _non_negative("lambda_sparse", lambda_sparse)
        self.alpha_q = _non_negative("alpha_q", alpha_q)
        self.alpha_d = _non_negative("alpha_d", alpha_d)
        self.scale = _positive("scale", scale)
        self.tau_s = _positive("tau_s", tau_s)
        is_integer = isinstance(alpha_warmup_steps, int) and not isinstance(alpha_warmup_steps, bool)
        if not is_integer or alpha_warmup_steps < 0:
            raise ValueError(
                f"Unsloth: `alpha_warmup_steps` must be a non-negative integer, "
                f"got {alpha_warmup_steps!r}."
            )
        self.alpha_warmup_steps = alpha_warmup_steps
        self.alpha_step = 0

    # -- FLOPS ramp ---------------------------------------------------------------------
    def alpha_scale(self) -> float:
        """Multiplier applied to both FLOPS weights; always 1.0 when warmup is off."""
        if self.alpha_warmup_steps <= 0:
            return 1.0
        progress = min(1.0, self.alpha_step / self.alpha_warmup_steps)
        return progress * progress

    # -- feature extraction -------------------------------------------------------------
    @staticmethod
    def _embedding(column: Any, key: str, index: int) -> torch.Tensor:
        if not isinstance(column, dict) or key not in column:
            if key == SPARSE_EMBEDDING_KEY:
                raise KeyError(
                    f"Unsloth: column {index} of the batch has no `{SPARSE_EMBEDDING_KEY}`. "
                    f"UEmbedUnifiedLoss needs the dense AND sparse outputs of a single model "
                    f"forward, which the UEmbed multi-output wiring adds beside "
                    f"`{SENTENCE_EMBEDDING_KEY}`. Load the model with a SPLADE pooling mode "
                    f"so the SpladeHead is attached, or use "
                    f"MultipleNegativesRankingLoss for dense-only training."
                )
            raise KeyError(
                f"Unsloth: column {index} of the batch has no `{key}`; every sentence "
                f"feature must carry the pooled dense embedding."
            )
        embedding = column[key]
        if embedding.dim() != 2:
            raise ValueError(
                f"Unsloth: `{key}` of column {index} must be a (batch, dim) matrix, got "
                f"shape {tuple(embedding.shape)}."
            )
        return embedding

    def _columns(
        self, sentence_features: Iterable[Any]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        columns = list(sentence_features)
        if len(columns) < 2:
            raise ValueError(
                f"Unsloth: UEmbedUnifiedLoss needs at least 2 columns (anchor + positive), "
                f"got {len(columns)}."
            )

        forwarded = []
        for column in columns:
            has_outputs = isinstance(column, dict) and all(
                key in column for key in (SENTENCE_EMBEDDING_KEY, SPARSE_EMBEDDING_KEY)
            )
            if has_outputs or self.model is None or not isinstance(column, dict):
                forwarded.append(column)
            else:
                # ST modules mutate their working features dict as it moves through the
                # pipeline. A shallow copy protects the caller's dictionary while keeping
                # its tensors (and their gradient graph) intact.
                forwarded.append(self.model(dict(column)))

        dense = [
            self._embedding(c, SENTENCE_EMBEDDING_KEY, i) for i, c in enumerate(forwarded)
        ]
        sparse = [
            self._embedding(c, SPARSE_EMBEDDING_KEY, i) for i, c in enumerate(forwarded)
        ]

        batch_size = dense[0].shape[0]
        for index, (dense_column, sparse_column) in enumerate(zip(dense, sparse)):
            sizes = (dense_column.shape[0], sparse_column.shape[0])
            if sizes != (batch_size, batch_size):
                raise ValueError(
                    f"Unsloth: every column must share one batch size; column 0 has "
                    f"{batch_size} rows but column {index} has dense {sizes[0]} / sparse "
                    f"{sizes[1]}. In-batch InfoNCE pairs row i with row i."
                )
        return dense, sparse

    # -- loss terms ---------------------------------------------------------------------
    @staticmethod
    def _in_batch_cross_entropy(logits: torch.Tensor) -> torch.Tensor:
        """Cross entropy against `arange(B)`: candidate i is the positive of anchor i."""
        labels = torch.arange(logits.shape[0], device = logits.device)
        return F.cross_entropy(logits, labels)

    def _dense_infonce(self, anchors: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        similarity = F.normalize(anchors, p = 2, dim = -1) @ F.normalize(
            candidates, p = 2, dim = -1
        ).transpose(0, 1)
        return self._in_batch_cross_entropy(similarity * self.scale)

    def _sparse_infonce(self, anchors: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        similarity = anchors @ candidates.transpose(0, 1)
        return self._in_batch_cross_entropy(similarity / self.tau_s)

    def components(self, sentence_features: Iterable[Any]) -> dict[str, torch.Tensor]:
        """Every term of Eq. 5 plus the weighted `total`; leaves the warmup step alone."""
        dense, sparse = self._columns(sentence_features)
        dense_candidates = torch.cat(dense[1:], dim = 0)
        sparse_candidates = torch.cat(sparse[1:], dim = 0)

        dense_loss = self._dense_infonce(dense[0], dense_candidates)
        sparse_loss = self._sparse_infonce(sparse[0], sparse_candidates)
        flops_query = flops_regularizer(sparse[0])
        flops_document = flops_regularizer(sparse_candidates)

        regulariser = self.alpha_q * flops_query + self.alpha_d * flops_document
        total = dense_loss + self.lambda_sparse * sparse_loss + self.alpha_scale() * regulariser
        return {
            "dense": dense_loss,
            "sparse": sparse_loss,
            "flops_query": flops_query,
            "flops_document": flops_document,
            "total": total,
        }

    def forward(
        self, sentence_features: Iterable[Any], labels: torch.Tensor | None = None
    ) -> torch.Tensor:
        """`labels` is ignored: in-batch InfoNCE derives its own `arange(B)` targets."""
        total = self.components(sentence_features)["total"]
        if self.alpha_warmup_steps > 0 and self.training:
            self.alpha_step += 1
        return total

    def get_config_dict(self) -> dict[str, float | int]:
        return {
            "lambda_sparse": self.lambda_sparse,
            "alpha_q": self.alpha_q,
            "alpha_d": self.alpha_d,
            "scale": self.scale,
            "tau_s": self.tau_s,
            "alpha_warmup_steps": self.alpha_warmup_steps,
        }

    @property
    def citation(self) -> str:
        return (
            "@misc{uembed2026,\n"
            "    title={UEmbed: Unified Dense and Sparse Multimodal Embeddings},\n"
            "    eprint={2608.02583},\n"
            "    archivePrefix={arXiv},\n"
            "}"
        )
