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
#
# Adapted from Q-GaLore (https://github.com/VITA-Group/Q-GaLore)
# Original paper: "Q-GaLore: Quantized GaLore with INT4 Projection and
# Layer-Adaptive Low-Rank Gradients" (arXiv:2407.08296)

from collections import deque

import torch

__all__ = ["GaLoreProjector"]


class GaLoreProjector:
    """Low-rank gradient projector with optional INT4/INT8 quantized projection
    matrices and layer-adaptive subspace update scheduling.

    SVD of the gradient gives an orthogonal basis for the top-``rank`` subspace.
    Gradients are projected in for the optimizer step, then back to full rank for
    the weight update. Two Q-GaLore innovations:

    1. Quantized projection matrices: with ``quant=True`` the orthogonal matrix
       is stored in INT4/INT8 to cut projector-state memory.
    2. Layer-adaptive update scheduling: when the rolling-average cosine
       similarity of consecutive orthogonal vectors exceeds ``cos_threshold``,
       ``update_proj_gap`` is multiplied by ``gamma_proj`` to recompute SVD less
       often for stabilized layers.
    """

    __slots__ = (
        "rank",
        "update_proj_gap",
        "scale",
        "proj_type",
        "quant",
        "quant_group_size",
        "quant_n_bit",
        "cos_threshold",
        "gamma_proj",
        "queue_size",
        "ortho_matrix",
        "ortho_matrix_scales",
        "ortho_matrix_zeros",
        "ortho_matrix_shape",
        "past_ortho_vector",
        "queue",
        "svd_count",
        "_ortho_float_cache",
    )

    def __init__(
        self,
        rank: int,
        update_proj_gap: int = 200,
        scale: float = 1.0,
        proj_type: str = "std",
        quant: bool = False,
        group_size: int = -1,
        n_bit: int = 4,
        cos_threshold: float = 0.4,
        gamma_proj: float = 2.0,
        queue_size: int = 5,
    ):
        self.rank = rank
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.proj_type = proj_type

        self.quant = quant
        self.quant_group_size = group_size
        self.quant_n_bit = n_bit

        self.cos_threshold = cos_threshold
        self.gamma_proj = gamma_proj
        self.queue_size = queue_size
        self.past_ortho_vector = None
        self.queue = deque(maxlen = queue_size)
        self.svd_count = 0
        self._ortho_float_cache = None

        self.ortho_matrix = None
        self.ortho_matrix_scales = None
        self.ortho_matrix_zeros = None
        self.ortho_matrix_shape = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def project(self, full_rank_grad: torch.Tensor, step: int) -> torch.Tensor:
        """Project a full-rank (2-D) gradient into the low-rank subspace.

        SVD is recomputed every ``update_proj_gap`` steps (subject to adaptive
        scheduling); between recomputations the cached orthogonal matrix is reused.
        """
        assert self.proj_type == "std", "Only proj_type='std' is supported."

        if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
            # Tall matrix -> right projection (grad @ Q^T)
            if self.ortho_matrix is None or step % self.update_proj_gap == 0:
                float_ortho = self._compute_orthogonal(
                    full_rank_grad,
                    self.rank,
                    side = "right",
                )
                self._update_adaptive_schedule(float_ortho, side = "right")
                self._store_ortho(float_ortho)

            self._ortho_float_cache = self._load_ortho()
            low_rank_grad = torch.matmul(full_rank_grad, self._ortho_float_cache.t())
        else:
            # Wide matrix -> left projection (Q^T @ grad)
            if self.ortho_matrix is None or step % self.update_proj_gap == 0:
                float_ortho = self._compute_orthogonal(
                    full_rank_grad,
                    self.rank,
                    side = "left",
                )
                self._update_adaptive_schedule(float_ortho, side = "left")
                self._store_ortho(float_ortho)

            self._ortho_float_cache = self._load_ortho()
            low_rank_grad = torch.matmul(self._ortho_float_cache.t(), full_rank_grad)

        return low_rank_grad

    def project_back(self, low_rank_grad: torch.Tensor) -> torch.Tensor:
        """Project a low-rank update back to full rank, scaled by ``self.scale``."""
        float_ortho = self._ortho_float_cache
        self._ortho_float_cache = None
        if float_ortho is None:
            float_ortho = self._load_ortho()

        if low_rank_grad.shape[0] >= low_rank_grad.shape[1]:
            full_rank_grad = torch.matmul(low_rank_grad, float_ortho)
        else:
            full_rank_grad = torch.matmul(float_ortho, low_rank_grad)

        return full_rank_grad * self.scale

    # ------------------------------------------------------------------
    # SVD
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_orthogonal(weights: torch.Tensor, rank: int, side: str) -> torch.Tensor:
        """Top-``rank`` orthogonal matrix of 2-D ``weights`` via truncated SVD.
        ``side='left'`` returns U[:, :rank] shape ``(M, rank)``; ``'right'``
        returns Vh[:rank, :] shape ``(rank, N)``."""
        original_dtype = weights.dtype
        original_device = weights.device

        matrix = weights.float() if original_dtype != torch.float32 else weights

        if side not in ("right", "left"):
            raise ValueError(f"side must be 'left' or 'right', got '{side}'")

        m, n = matrix.shape
        if min(m, n) <= rank * 2:
            U, s, Vh = torch.linalg.svd(matrix, full_matrices = False)
            result = Vh[:rank, :] if side == "right" else U[:, :rank]
        else:
            # Oversampling p=10 per Halko et al. 2009 (arXiv:0909.4061)
            # recommendation of p=5..10 for large low-rank matrices.
            q = min(rank + 10, min(m, n))
            U, s, V = torch.svd_lowrank(matrix, q = q, niter = 2)
            result = V[:, :rank].t() if side == "right" else U[:, :rank]

        if original_dtype != torch.float32:
            result = result.to(device = original_device, dtype = original_dtype)
        return result

    # ------------------------------------------------------------------
    # Adaptive scheduling
    # ------------------------------------------------------------------

    def _update_adaptive_schedule(self, float_ortho: torch.Tensor, side: str) -> None:
        """Track subspace stability and increase ``update_proj_gap`` if stable."""
        self.svd_count += 1

        if side == "right":
            current_vector = float_ortho[:1, :].flatten()
        else:
            current_vector = float_ortho[:, :1].flatten()

        if self.past_ortho_vector is not None:
            cos_sim = torch.dot(self.past_ortho_vector, current_vector).item()

            self.queue.append(cos_sim)

            if (
                len(self.queue) == self.queue.maxlen
                and sum(self.queue) / len(self.queue) >= self.cos_threshold
            ):
                self.update_proj_gap = int(self.update_proj_gap * self.gamma_proj)

        self.past_ortho_vector = current_vector.clone()

    # ------------------------------------------------------------------
    # Quantized projection matrix storage
    # ------------------------------------------------------------------

    def _store_ortho(self, float_ortho: torch.Tensor) -> None:
        """Store the orthogonal matrix, optionally quantized."""
        if self.quant:
            q, scales, zeros, shape = _quantize(
                float_ortho,
                q_group_size = self.quant_group_size,
                n_bit = self.quant_n_bit,
            )
            self.ortho_matrix = q
            self.ortho_matrix_scales = scales
            self.ortho_matrix_zeros = zeros
            self.ortho_matrix_shape = shape
        else:
            self.ortho_matrix = float_ortho

    def _load_ortho(self) -> torch.Tensor:
        """Load the orthogonal matrix, dequantizing if necessary."""
        if self.quant:
            return _dequantize(
                self.ortho_matrix,
                self.ortho_matrix_scales,
                self.ortho_matrix_zeros,
                self.ortho_matrix_shape,
            )
        return self.ortho_matrix


# ======================================================================
# Quantization utilities (shared with the optimizer)
# ======================================================================


@torch.no_grad()
def _quantize(
    w: torch.Tensor,
    q_group_size: int = -1,
    n_bit: int = 8,
) -> tuple:
    """Asymmetric min-max quantization to unsigned int.

    Returns:
        ``(quantized_uint8, scales, zeros, original_shape)``
    """
    org_shape = w.shape
    if q_group_size > 0:
        assert (
            w.nelement() % q_group_size == 0
        ), f"Tensor size {w.nelement()} not divisible by group_size {q_group_size}"
        w = w.reshape(-1, q_group_size)
    assert w.dim() == 2

    max_val = w.amax(dim = 1, keepdim = True)
    min_val = w.amin(dim = 1, keepdim = True)
    max_int = 2**n_bit - 1
    min_int = 0
    scales = (max_val - min_val).clamp(min = 1e-5) / max_int
    zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)

    w = torch.clamp(torch.round(w / scales) + zeros, min_int, max_int)
    w = w.reshape(org_shape).to(torch.uint8)

    return w, scales, zeros, org_shape


@torch.no_grad()
def _dequantize(
    w: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, original_shape: tuple
) -> torch.Tensor:
    """Dequantize from uint8 back to float."""
    # Infer group size: scales has shape (n_groups, 1)
    total = w.numel()
    n_groups = scales.shape[0] if scales.dim() > 1 else scales.numel()
    group_size = total // n_groups if n_groups > 0 else total

    float_w = w.to(scales.dtype).reshape(-1, group_size)
    float_w = (float_w - zeros) * scales
    return float_w.reshape(original_shape)


@torch.no_grad()
def _quantize_stochastic(
    w: torch.Tensor,
    q_group_size: int = -1,
    n_bit: int = 8,
) -> tuple:
    """Asymmetric min-max quantization with stochastic rounding.

    Rounding direction is chosen probabilistically by the fractional part,
    giving an unbiased estimator in expectation.
    Returns ``(quantized_uint8, scales, zeros, original_shape)``.
    """
    org_shape = w.shape
    if q_group_size > 0:
        assert w.nelement() % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    assert w.dim() == 2

    max_val = w.amax(dim = 1, keepdim = True)
    min_val = w.amin(dim = 1, keepdim = True)
    max_int = 2**n_bit - 1
    min_int = 0
    scales = (max_val - min_val).clamp(min = 1e-5) / max_int
    zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)

    w_scaled = w / scales
    up = torch.ceil(w_scaled)
    down = torch.floor(w_scaled)
    prob = w_scaled - down
    rng = torch.rand_like(prob)
    w = torch.where(rng < prob, up, down)
    w = torch.clamp(w + zeros, min_int, max_int)
    w = w.reshape(org_shape).to(torch.uint8)

    return w, scales, zeros, org_shape
