# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import triton
import triton.language as tl
import torch
from .utils import (
    calculate_settings,
    MAX_FUSED_SIZE,
    triton_tanh,
    triton_cast,
    torch_gpu_device,
    is_cdna,
)
from transformers.models.llama.modeling_llama import logger
from unsloth_zoo.utils import Version

from unsloth_zoo.loss_utils import (
    patch_loss_functions as _patch_loss_functions,
    post_patch_loss_function,
)


def _cross_entropy_forward(
    logits_ptr,
    logits_row_stride,
    loss_ptr,
    logsumexp_ptr,
    labels_ptr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DO_SOFTCAPPING: tl.constexpr,
    SOFTCAP: tl.constexpr,
    DO_LOGIT_SCALING: tl.constexpr,
    LOGIT_SCALE: tl.constexpr,
):
    """CE = logsumexp(x) - x[label]; logsumexp uses c = max(x) for stability."""
    row_idx = tl.program_id(0)
    logits_ptr += row_idx * triton_cast(logits_row_stride, tl.int64)
    loss_ptr += row_idx
    logsumexp_ptr += row_idx
    labels_ptr += row_idx

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE

    label_idx = tl.load(labels_ptr).to(tl.int32)
    logits = tl.load(logits_ptr + col_offsets, mask = mask, other = -float("inf")).to(tl.float32)

    if DO_LOGIT_SCALING:
        logits = LOGIT_SCALE * logits
    if DO_SOFTCAPPING:
        logits = SOFTCAP * triton_tanh(logits / SOFTCAP)
    if DO_LOGIT_SCALING or DO_SOFTCAPPING:
        # Transforms make -inf padding finite (-SOFTCAP via tanh, +inf via negative scale).
        logits = tl.where(mask, logits, -float("inf"))

    c = tl.max(logits, 0)
    logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))

    if label_idx != -100:
        x = tl.load(logits_ptr + label_idx).to(tl.float32)
        if DO_LOGIT_SCALING:
            x = LOGIT_SCALE * x
        if DO_SOFTCAPPING:
            x = SOFTCAP * triton_tanh(x / SOFTCAP)
        loss = logsumexp - x
    else:
        loss = 0.0
    tl.store(logsumexp_ptr, logsumexp)
    tl.store(loss_ptr, loss)


_cross_entropy_forward = triton.jit(_cross_entropy_forward)
_cross_entropy_forward = triton.heuristics(
    {
        "DO_SOFTCAPPING": lambda args: bool(args["DO_SOFTCAPPING"]),
        "DO_LOGIT_SCALING": lambda args: bool(args["DO_LOGIT_SCALING"]),
    }
)(_cross_entropy_forward)


def _chunked_cross_entropy_forward(
    logits_ptr,
    logits_row_stride: tl.constexpr,
    loss_ptr,
    logsumexp_ptr,
    labels_ptr,
    VOCAB_SIZE: tl.constexpr,
    N_CHUNKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DO_SOFTCAPPING: tl.constexpr,
    SOFTCAP: tl.constexpr,
    DO_LOGIT_SCALING: tl.constexpr,
    LOGIT_SCALE: tl.constexpr,
):
    """Per-chunk logsumexp; logsumexp of the chunk results equals the full-row logsumexp."""
    row_idx = tl.program_id(0)
    chunk_idx = tl.program_id(1)
    logits_ptr += row_idx * triton_cast(logits_row_stride, tl.int64)
    loss_ptr += row_idx
    logsumexp_ptr += row_idx * N_CHUNKS + chunk_idx
    labels_ptr += row_idx

    col_offsets = chunk_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE

    label_idx = tl.load(labels_ptr).to(tl.int32)
    logits = tl.load(logits_ptr + col_offsets, mask = mask, other = -float("inf")).to(tl.float32)

    if DO_LOGIT_SCALING:
        logits = LOGIT_SCALE * logits
    if DO_SOFTCAPPING:
        logits = SOFTCAP * triton_tanh(logits / SOFTCAP)
    if DO_LOGIT_SCALING or DO_SOFTCAPPING:
        # Transforms make -inf padding finite (-SOFTCAP via tanh, +inf via negative scale).
        logits = tl.where(mask, logits, -float("inf"))

    c = tl.max(logits, 0)
    logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))

    if chunk_idx == 0:
        if label_idx != -100:
            x = tl.load(logits_ptr + label_idx).to(tl.float32)
            if DO_LOGIT_SCALING:
                x = LOGIT_SCALE * x
            if DO_SOFTCAPPING:
                x = SOFTCAP * triton_tanh(x / SOFTCAP)
            loss = -1.0 * x
        else:
            loss = 0.0
        tl.store(loss_ptr, loss)
    tl.store(logsumexp_ptr, logsumexp)


_chunked_cross_entropy_forward = triton.jit(_chunked_cross_entropy_forward)
_chunked_cross_entropy_forward = triton.heuristics(
    {
        "DO_SOFTCAPPING": lambda args: bool(args["DO_SOFTCAPPING"]),
        "DO_LOGIT_SCALING": lambda args: bool(args["DO_LOGIT_SCALING"]),
    }
)(_chunked_cross_entropy_forward)


def _cross_entropy_backward(
    logits_ptr,
    logits_row_stride: tl.constexpr,
    dloss_ptr,
    dloss_row_stride: tl.constexpr,
    logsumexp_ptr,
    labels_ptr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DO_SOFTCAPPING: tl.constexpr,
    SOFTCAP: tl.constexpr,
    DO_LOGIT_SCALING: tl.constexpr,
    LOGIT_SCALE: tl.constexpr,
):
    """dC/dx = softmax(x) - onehot(label) = exp(x - logsumexp) - [x == label]."""
    row_idx = tl.program_id(0)
    block_idx = tl.program_id(1)

    logits_ptr += row_idx * triton_cast(logits_row_stride, tl.int64)
    dloss_ptr += row_idx * dloss_row_stride
    col_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE
    label_idx = tl.load(labels_ptr + row_idx).to(tl.int32)

    if label_idx != -100:
        dloss = tl.load(dloss_ptr)
    else:
        dloss = 0.0

    x = tl.load(logits_ptr + col_offsets, mask = mask, other = -float("inf")).to(tl.float32)

    if DO_LOGIT_SCALING:
        x = x * LOGIT_SCALE

    partial = x
    if DO_SOFTCAPPING:
        partial = triton_tanh(x / SOFTCAP)
        x = SOFTCAP * partial

    logsumexp = tl.load(logsumexp_ptr + row_idx)
    y = tl.exp(x - logsumexp)
    y = tl.where(
        col_offsets == label_idx,
        y - 1.0,
        y,
    )

    if DO_LOGIT_SCALING:
        y = y * LOGIT_SCALE

    if DO_SOFTCAPPING:
        y = y * (1.0 - partial * partial)

    tl.store(logits_ptr + col_offsets, dloss * y, mask = mask)


_cross_entropy_backward = triton.jit(_cross_entropy_backward)
_cross_entropy_backward = triton.heuristics(
    {
        "DO_SOFTCAPPING": lambda args: bool(args["DO_SOFTCAPPING"]),
        "DO_LOGIT_SCALING": lambda args: bool(args["DO_LOGIT_SCALING"]),
    }
)(_cross_entropy_backward)


class Fast_CrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        logits,
        labels,
        logit_softcapping: float = 0,
        logit_scaling: float = 0,
    ):
        n_rows: int
        vocab_size: int
        n_rows, vocab_size = logits.shape
        device = logits.device
        logits = logits.contiguous()
        labels = labels.to(device).contiguous()

        div, mod = divmod(vocab_size, MAX_FUSED_SIZE)
        n_chunks: int = div + (mod != 0)
        losses = torch.empty(n_rows, dtype = torch.float32, device = device)

        DO_SOFTCAPPING: bool = bool(logit_softcapping != 0)
        DO_LOGIT_SCALING: bool = bool(logit_scaling != 0)

        BLOCK_SIZE: int
        num_warps: int
        if n_chunks == 1:
            BLOCK_SIZE, num_warps = calculate_settings(vocab_size)
            if is_cdna():
                num_warps = num_warps // 2
            logsumexp = torch.empty(n_rows, dtype = torch.float32, device = device)

            with torch_gpu_device(device):
                _cross_entropy_forward[(n_rows,)](
                    logits,
                    logits.stride(0),
                    losses,
                    logsumexp,
                    labels,
                    VOCAB_SIZE = vocab_size,
                    BLOCK_SIZE = BLOCK_SIZE,
                    DO_SOFTCAPPING = DO_SOFTCAPPING,
                    SOFTCAP = logit_softcapping,
                    DO_LOGIT_SCALING = DO_LOGIT_SCALING,
                    LOGIT_SCALE = logit_scaling,
                    num_warps = num_warps,
                )
        else:
            logsumexp = torch.empty(
                (
                    n_rows,
                    n_chunks,
                ),
                dtype = torch.float32,
                device = device,
            )

            with torch_gpu_device(device):
                _chunked_cross_entropy_forward[
                    (
                        n_rows,
                        n_chunks,
                    )
                ](
                    logits,
                    logits.stride(0),
                    losses,
                    logsumexp,
                    labels,
                    VOCAB_SIZE = vocab_size,
                    N_CHUNKS = n_chunks,
                    BLOCK_SIZE = MAX_FUSED_SIZE,
                    DO_SOFTCAPPING = DO_SOFTCAPPING,
                    SOFTCAP = logit_softcapping,
                    DO_LOGIT_SCALING = DO_LOGIT_SCALING,
                    LOGIT_SCALE = logit_scaling,
                    num_warps = 32 if not is_cdna() else 16,
                )
            logsumexp = torch.logsumexp(logsumexp, dim = 1)
            losses += logsumexp
            losses.masked_fill_(labels == -100, 0)

        ctx.save_for_backward(logits, logsumexp, labels)
        ctx.DO_SOFTCAPPING = DO_SOFTCAPPING
        ctx.logit_softcapping = logit_softcapping
        ctx.DO_LOGIT_SCALING = DO_LOGIT_SCALING
        ctx.logit_scaling = logit_scaling
        return losses

    @staticmethod
    def backward(ctx, dlosses):
        logits, logsumexp, labels = ctx.saved_tensors
        n_rows: int
        vocab_size: int
        n_rows, vocab_size = logits.shape

        BLOCK_SIZE: int = 4096
        div: int
        mod: int
        div, mod = divmod(vocab_size, BLOCK_SIZE)
        n_blocks: int = div + (mod != 0)

        with torch_gpu_device(dlosses.device):
            _cross_entropy_backward[
                (
                    n_rows,
                    n_blocks,
                )
            ](
                logits,
                logits.stride(0),
                dlosses,
                dlosses.stride(0),
                logsumexp,
                labels,
                VOCAB_SIZE = vocab_size,
                BLOCK_SIZE = BLOCK_SIZE,
                DO_SOFTCAPPING = ctx.DO_SOFTCAPPING,
                SOFTCAP = ctx.logit_softcapping,
                DO_LOGIT_SCALING = ctx.DO_LOGIT_SCALING,
                LOGIT_SCALE = ctx.logit_scaling,
                num_warps = 8,
            )
        return (
            logits,
            None,
            None,
            None,
        )


def fast_cross_entropy_loss(
    logits,
    labels,
    logit_softcapping = 0,
    logit_scaling = 0,
    n_items = None,
):
    """logits: (batch, seq_len, vocab_size), labels: (batch, seq_len). Returns mean loss."""
    batch, seq_len, d = logits.shape
    assert labels.shape == (batch, seq_len)

    device = logits.device
    loss = Fast_CrossEntropyLoss.apply(
        logits.reshape(batch * seq_len, d),
        labels.reshape(-1),
        logit_softcapping,
        logit_scaling,
    )
    if n_items is None:
        n_items = torch.count_nonzero(labels != -100)
    if torch.is_tensor(n_items):
        n_items = n_items.to(device)
    return loss.sum() / n_items


if (Version(torch.__version__) < Version("2.4.0")) and not hasattr(
    fast_cross_entropy_loss, "__wrapped__"
):
    fast_cross_entropy_loss = torch._disable_dynamo(fast_cross_entropy_loss)


def patch_loss_functions(torch_compile = True):
    _patch_loss_functions(fast_cross_entropy_loss, torch_compile = torch_compile)

    # Redirect LOSS_MAPPING aliases (Qwen3.5, Csm) still on stock ForCausalLMLoss; drop once zoo floor has unslothai/unsloth-zoo#656.
    try:
        import transformers.loss.loss_utils as _lu
        _unsloth_loss = _lu.LOSS_MAPPING.get("ForCausalLM")
        if _unsloth_loss is not None:
            for _key, _fn in list(_lu.LOSS_MAPPING.items()):
                if getattr(_fn, "__name__", "") == "ForCausalLMLoss":
                    _lu.LOSS_MAPPING[_key] = _unsloth_loss
    except (ImportError, AttributeError):
        pass
