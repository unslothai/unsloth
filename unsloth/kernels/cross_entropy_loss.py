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

import triton
import triton.language as tl
import torch
from .utils import calculate_settings, MAX_FUSED_SIZE
from transformers.models.llama.modeling_llama import logger


@triton.jit
def _small_cross_entropy_forward(
    logits_ptr, logits_row_stride,
    loss_ptr,
    lse_ptr,
    labels_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
        Cross Entropy Loss = 1/n sum [ -yi log(Pi) ]
        Pi = exp(xi) / sum(exp(xi))
        CE_i = -y log(p) = -y log[ exp(x) / sum(exp(x)) ]
             = -y [ x - log[sum(exp(x))] ]
             = y * (log[sum(exp(x))] - x)
        If y == 0: CE_i = 0
        If y == 1: CE_i = logsumexp - x
    """
    row_idx = tl.program_id(0)
    logits_ptr += row_idx * logits_row_stride
    loss_ptr   += row_idx
    lse_ptr    += row_idx
    labels_ptr += row_idx

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # TODO: Fixup int32 locations to int64
    label_idx = tl.load(labels_ptr).to(tl.int32)
    logits = tl.load(logits_ptr + col_offsets, mask = mask, other = -float("inf")).to(tl.float32)
    max_logits = tl.max(logits, 0)
    # Maximum stops overflow
    lse = tl.log(tl.sum(tl.exp(logits - max_logits), 0)) + max_logits
    tl.store(lse_ptr, lse)

    if label_idx != -100:
        logits_label = tl.load(logits_ptr + label_idx).to(tl.float32)
        loss = lse - logits_label
    else:
        loss = 0.0
    tl.store(loss_ptr, loss)
pass


@triton.jit
def _large_cross_entropy_forward(
    logits_ptr, logits_row_stride,
    loss_ptr,
    lse_ptr,
    labels_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
        Cross Entropy Loss = 1/n sum [ -yi log(Pi) ]
        Pi = exp(xi) / sum(exp(xi))
        CE_i = -y log(p) = -y log[ exp(x) / sum(exp(x)) ]
             = -y [ x - log[sum(exp(x))] ]
             = y * (log[sum(exp(x))] - x)
        If y == 0: CE_i = 0
        If y == 1: CE_i = logsumexp - x
    """
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)
    logits_ptr += row_idx * logits_row_stride
    loss_ptr   += row_idx + col_idx*n_rows
    lse_ptr    += row_idx + col_idx*n_rows
    labels_ptr += row_idx

    col_offsets = col_idx*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Get labels and logits
    label_idx = tl.load(labels_ptr).to(tl.int64)
    logits = tl.load(logits_ptr + col_offsets, mask = mask, other = -float("inf")).to(tl.float32)
    max_logits = tl.max(logits, 0)
    # Maximum stops overflow
    lse = tl.log(tl.sum(tl.exp(logits - max_logits), 0)) + max_logits
    tl.store(lse_ptr, lse)

    loss = 0.0
    # chained boolean operators (A or B or C) are not supported; use parentheses to split the chain.
    if (label_idx != -100):
        if  (label_idx >=    (col_idx+0)*BLOCK_SIZE) and \
            (label_idx < min((col_idx+1)*BLOCK_SIZE, n_cols)):

            logits_label = tl.load(logits_ptr + label_idx).to(tl.float32)
            lse  = 0.0
            loss = lse - logits_label # We add the final logsumexp after a reduction
        pass
    pass
        
    tl.store(loss_ptr, loss)
pass


@triton.jit
def _cross_entropy_backward(
    logits_ptr, logits_row_stride,
    dloss_ptr,   dloss_row_stride,
    lse_ptr,
    labels_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
        CE_i = -y log(P) = y * (log[sum(exp(x))] - x)
        dC/dx = d/dx (y * log[sum(exp(x))] - x * y)

        From https://en.wikipedia.org/wiki/LogSumExp
        d/dx logsumexp = exp(x) / sum(exp(x)) = softmax(x)

        dC/dx = y * exp(x) / sum(exp(x)) - d/dx (x * y)
        dC/dx = y * exp[ log[exp(x) / sum(exp(x))] ] using x = exp(log(x)) trick
        dC/dx = y * exp[x - logsumexp] - d/dx (x * y)

        If y == 0: dC/dx = 0
        If y == 1 and x == label: dC/dlabel = exp[x - logsumexp] - 1
        If y == 1 and x != label: dC/dx     = exp[x - logsumexp]
    """
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)
    logits_ptr += row_idx * logits_row_stride.to(tl.int64)
    dloss_ptr  += row_idx *  dloss_row_stride
    col_offsets = col_idx*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    label_idx = tl.load(labels_ptr + row_idx).to(tl.int32)

    if label_idx != -100:
        dloss = tl.load(dloss_ptr)
    else:
        dloss = 0.0
    logits = tl.load(logits_ptr + col_offsets, mask = mask, other = -float("inf")).to(tl.float32)
    lse = tl.load(lse_ptr + row_idx)
    probs = tl.exp(logits - lse)

    probs = tl.where(col_offsets == label_idx, probs - 1.0, probs)
    tl.store(logits_ptr + col_offsets, dloss * probs, mask = mask)
pass


MAX_FUSED_SIZE = 65536 # 2**16

class Fast_CrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels):
        n_rows, n_cols = logits.shape

        div, mod = divmod(n_cols, MAX_FUSED_SIZE)
        n_splits = div + (mod != 0)

        if n_splits == 1:
            # For small vocabs <= 65336 like Llama, Mistral
            BLOCK_SIZE, num_warps = calculate_settings(n_cols)
            losses    = torch.empty(n_rows, dtype = torch.float32, device = "cuda")
            logsumexp = torch.empty(n_rows, dtype = torch.float32, device = "cuda")

            _small_cross_entropy_forward[(n_rows,)](
                logits, logits.stride(0),
                losses,
                logsumexp,
                labels,
                n_cols,
                BLOCK_SIZE = BLOCK_SIZE,
                num_warps  = num_warps,
            )
        else:
            # For small vocabs > 65336 like Gemma
            losses    = torch.empty((n_splits, n_rows), dtype = torch.float32, device = "cuda")
            logsumexp = torch.empty((n_splits, n_rows), dtype = torch.float32, device = "cuda")

            _large_cross_entropy_forward[(n_rows, n_splits,)](
                logits, logits.stride(0),
                losses,
                logsumexp,
                labels,
                n_rows,
                n_cols,
                BLOCK_SIZE = MAX_FUSED_SIZE,
                num_warps  = 32,
            )
            logsumexp = torch.logsumexp(logsumexp, dim = 0) # Row sum
            losses = losses.sum(dim = 0) # Row sum
            losses += logsumexp # loss = lse - logits_label
            losses.masked_fill_(labels == -100, 0) # Padding tokens
        pass

        ctx.save_for_backward(logits, logsumexp, labels)
        return losses
    pass

    @staticmethod
    def backward(ctx, dlosses):
        logits, logsumexp, labels = ctx.saved_tensors
        n_rows, n_cols = logits.shape
        grid = lambda meta: (n_rows, triton.cdiv(n_cols, meta["BLOCK_SIZE"]))

        _cross_entropy_backward[grid](
            logits,   logits.stride(0),
            dlosses, dlosses.stride(0),
            logsumexp,
            labels,
            n_cols,
            BLOCK_SIZE = 4096,
            num_warps  = 8,
        )
        return logits, None, None,
    pass
pass


if "all_gather_into_tensor" not in dir(torch.distributed):
    torch.distributed.all_gather_into_tensor = torch.distributed._all_gather_base


@triton.heuristics(
    {
        "HAS_SMOOTHING": lambda args: args["smoothing"] > 0.0,
    }
)
@triton.jit
def cross_entropy_fwd_kernel(
    loss_ptr,  # data ptrs
    lse_ptr,
    z_loss_ptr,
    logits_ptr,
    labels_ptr,
    smoothing,
    logit_scale,
    lse_square_scale,
    ignored_index,
    total_classes,
    class_start_idx,  # Useful for tensor parallel when each rank only has a subset of classes
    n_cols,  # shapes
    n_rows,
    logits_row_stride,  # strides
    BLOCK_SIZE: tl.constexpr,
    HAS_SMOOTHING: tl.constexpr,
    # if SPLIT (e.g. tensor parallel), don't include the LSE in the loss since it's not the final LSE
    SPLIT: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)
    logits_ptr = logits_ptr + row_idx * logits_row_stride.to(tl.int64)
    col_offsets = col_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    label_idx = tl.load(labels_ptr + row_idx)
    logits = tl.load(logits_ptr + col_offsets, mask=col_offsets < n_cols, other=-float("inf")).to(
        tl.float32
    ) * logit_scale
    max_logits = tl.max(logits, 0)
    if HAS_SMOOTHING:
        sum_logits = tl.sum(tl.where(col_offsets < n_cols, logits, 0.0), 0)
    lse = tl.log(tl.sum(tl.exp(logits - max_logits), 0)) + max_logits
    tl.store(lse_ptr + col_block_idx * n_rows + row_idx, lse)
    if label_idx == ignored_index:
        loss = 0.0
        z_loss = 0.0
    else:
        label_idx -= class_start_idx
        if label_idx >= col_block_idx * BLOCK_SIZE and label_idx < min(
            n_cols, (col_block_idx + 1) * BLOCK_SIZE
        ):
            logits_label = tl.load(logits_ptr + label_idx) * logit_scale
            if HAS_SMOOTHING:
                loss = (
                    (lse if not SPLIT else 0.0)
                    - smoothing * sum_logits / total_classes
                    - (1 - smoothing) * logits_label
                )
            else:
                loss = (lse if not SPLIT else 0.0) - logits_label
        else:
            # If label is out of bounds, we set the CE loss to 0.0. But we still want the smoothing loss
            if HAS_SMOOTHING:
                loss = smoothing * ((lse if not SPLIT else 0.0) - sum_logits / total_classes)
            else:
                loss = 0.0
        if not SPLIT:
            z_loss = lse_square_scale * lse * lse
            loss += z_loss
        else:
            z_loss = 0.0
    tl.store(loss_ptr + col_block_idx * n_rows + row_idx, loss)
    if not SPLIT:
        tl.store(z_loss_ptr + col_block_idx * n_rows + row_idx, z_loss)


@triton.heuristics(
    {
        "HAS_SMOOTHING": lambda args: args["smoothing"] > 0.0,
    }
)
@triton.jit
def cross_entropy_bwd_kernel(
    dlogits_ptr,  # data ptrs
    dloss_ptr,
    logits_ptr,
    lse_ptr,
    labels_ptr,
    smoothing,
    logit_scale,
    lse_square_scale,
    ignored_index,
    total_classes,
    class_start_idx,  # Useful for tensor parallel when each rank only has a subset of classes
    n_cols,  # shapes
    logits_row_stride,  # strides
    dlogits_row_stride,
    dloss_row_stride,
    BLOCK_SIZE: tl.constexpr,
    HAS_SMOOTHING: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)
    logits_ptr = logits_ptr + row_idx * logits_row_stride.to(tl.int64)
    dlogits_ptr = dlogits_ptr + row_idx * dlogits_row_stride.to(tl.int64)
    col_offsets = col_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    label_idx = tl.load(labels_ptr + row_idx)
    if label_idx != ignored_index:
        dloss = tl.load(dloss_ptr + row_idx * dloss_row_stride)
    else:
        dloss = 0.0
    logits = tl.load(logits_ptr + col_offsets, mask=col_offsets < n_cols, other=-float("inf")).to(
        tl.float32
    ) * logit_scale
    lse = tl.load(lse_ptr + row_idx)
    probs = tl.exp(logits - lse)
    probs += 2.0 * lse_square_scale * lse * probs
    label_idx -= class_start_idx
    if HAS_SMOOTHING:
        smooth_positive = 1.0 - smoothing
        smooth_negative = smoothing / total_classes
        probs = tl.where(col_offsets == label_idx, probs - (1 - smoothing), probs) - smooth_negative
    else:
        probs = tl.where(col_offsets == label_idx, probs - 1.0, probs)
    tl.store(dlogits_ptr + col_offsets, (dloss * logit_scale) * probs, mask=col_offsets < n_cols)


class CrossEntropyLoss(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        logits,
        labels,
        smoothing=0.0,
        logit_scale=1.0,
        lse_square_scale=0.0,
        ignored_index=-100,
        inplace_backward=False,
        process_group=None,
    ):
        n_rows, n_cols = logits.shape
        assert labels.shape == (n_rows,)
        world_size = 1 if process_group is None else torch.distributed.get_world_size(process_group)
        total_classes = world_size * n_cols
        rank = 0 if process_group is None else torch.distributed.get_rank(process_group)
        class_start_idx = rank * n_cols

        if logits.stride(-1) != 1:
            logits = logits.contiguous()
        # Set these similar to https://github.com/openai/triton/blob/main/python/tutorials/02-fused-softmax.py
        MAX_BLOCK_SIZE = 64 * 1024
        BLOCK_SIZE = min(triton.next_power_of_2(n_cols), MAX_BLOCK_SIZE)
        num_warps = (
            4
            if BLOCK_SIZE < 2048
            else (8 if BLOCK_SIZE < 8192 else (16 if BLOCK_SIZE < 128 * 1024 else 32))
        )
        # We may split the lse computation across multiple blocks, then do a reduction
        # lse(local_lse) to get the final LSE. This is faster for large n_cols (e.g., > 64k)
        # where having just one thread block processing more than 64k elements is slow.
        split = world_size > 1 or n_cols > MAX_BLOCK_SIZE
        n_splits = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
        loss_shape = (n_splits, n_rows) if n_splits > 1 else (n_rows,)
        losses = torch.empty(*loss_shape, dtype=torch.float, device=logits.device)
        lse = torch.empty(*loss_shape, dtype=torch.float, device=logits.device)
        z_losses = torch.empty(*loss_shape, dtype=torch.float, device=logits.device)
        # Need this, otherwise Triton tries to launch from cuda:0 and we get
        # ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)
        with torch.cuda.device(logits.device.index):
            cross_entropy_fwd_kernel[(n_rows, n_splits)](
                losses,  # data ptrs
                lse,
                z_losses,
                logits,
                labels,
                smoothing,
                logit_scale,
                lse_square_scale,
                ignored_index,
                total_classes,
                class_start_idx,
                n_cols,  # shapes
                n_rows,
                logits.stride(0),  # strides
                BLOCK_SIZE=BLOCK_SIZE,  # constants
                num_warps=num_warps,
                SPLIT=split,
            )

        if split:
            # If there's no smoothing, if labels are in the vocab of this partition, losses contains
            # - predicted logit, and 0 otherwise.
            # If there's smoothing=0.1, for labels in the vocab of this partition, losses contains
            # -0.9 * predicted logit - 0.1 * sum logit / total_classes.
            # For labels not in the vocab of this partition, losses contains
            # -0.1 * sum logit / total_classes.
            if n_splits > 1:
                lse = torch.logsumexp(lse, dim=0)
                losses = losses.sum(dim=0)
            if world_size > 1:
                lse_allgather = torch.empty(world_size, n_rows, dtype=lse.dtype, device=lse.device)
                torch.distributed.all_gather_into_tensor(lse_allgather, lse, group=process_group)
                handle_losses = torch.distributed.all_reduce(
                    losses, op=torch.distributed.ReduceOp.SUM, group=process_group, async_op=True
                )
                lse = torch.logsumexp(lse_allgather, dim=0)
                handle_losses.wait()
            # After the allreduce, if there's no smoothing, the total losses are - predicted_logit,
            # we just have to add the (global) lse.
            # If there's smoothing=0.1, the total losses are
            # -0.9 * predicted_logit - 0.1 * sum logit / total_classes.
            # Again, we just have to add the (global) lse.
            losses += lse
            if lse_square_scale != 0.0:
                z_losses = lse_square_scale * lse.square()
                z_losses.masked_fill_(labels == ignored_index, 0.0)
                losses += z_losses
            else:
                z_losses = torch.zeros_like(losses)
            losses.masked_fill_(labels == ignored_index, 0.0)

        ctx.save_for_backward(logits, lse, labels)
        ctx.mark_non_differentiable(z_losses)
        ctx.smoothing = smoothing
        ctx.logit_scale = logit_scale
        ctx.lse_square_scale = lse_square_scale
        ctx.ignored_index = ignored_index
        ctx.total_classes = total_classes
        ctx.class_start_idx = class_start_idx
        ctx.inplace_backward = inplace_backward

        return losses, z_losses

    @staticmethod
    def backward(ctx, grad_losses, grad_z_losses):
        del grad_z_losses  # z_losses are only for logging.

        logits, lse, labels = ctx.saved_tensors
        dlogits = logits if ctx.inplace_backward else torch.empty_like(logits)
        n_rows, n_cols = logits.shape
        BLOCK_SIZE = min(triton.next_power_of_2(n_cols), 4 * 1024)
        num_warps = 4 if BLOCK_SIZE < 2048 else (8 if BLOCK_SIZE < 8192 else 16)
        grid = lambda META: (n_rows, triton.cdiv(n_cols, META["BLOCK_SIZE"]))  # noqa
        # Need this, otherwise Triton tries to launch from cuda:0 and we get
        # ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)
        with torch.cuda.device(logits.device.index):
            cross_entropy_bwd_kernel[grid](
                dlogits,  # data ptrs
                grad_losses,
                logits,
                lse,
                labels,
                ctx.smoothing,
                ctx.logit_scale,
                ctx.lse_square_scale,
                ctx.ignored_index,
                ctx.total_classes,
                ctx.class_start_idx,
                n_cols,  # shapes
                logits.stride(0),  # strides
                dlogits.stride(0),
                grad_losses.stride(0),
                BLOCK_SIZE=BLOCK_SIZE,  # constants
                num_warps=num_warps,
            )
        return dlogits, None, None, None, None, None, None, None, None

def cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_smoothing: float = 0.0,
    logit_scale: float = 1.0,
    lse_square_scale: float = 0.0,
    ignored_index=-100,
    inplace_backward: bool = False,
    process_group=None,
) :
    """
    Arguments:
        logits: (batch, vocab_size)
        labels: (batch,)
        label_smoothing: float
        logit_scale: float. Multiply logits by this scale before calculating the loss.
        lse_square_scale: float. If > 0, we add lse_square_scale * lse(logits) ^ 2 to the loss.
            This is also referred to as "z-loss".
        ignored_index: int. If labels == ignored_index, the loss is set to 0.0.
        inplace_backward: bool. If True, we do the backward pass in-place by modifying the logits.
            This saves memory.
        process_group: if not None, we're doing Tensor Parallel: each process is responsible for
            one part of the vocab. The loss will be aggregated across processes.
    Returns:
        losses: (batch,), float
        z_losses: (batch,), float
    """
    return CrossEntropyLoss.apply(
        logits,
        labels,
        label_smoothing,
        logit_scale,
        lse_square_scale,
        ignored_index,
        inplace_backward,
        process_group,
    )
pass


# slow_cross_entropy_loss = torch.nn.functional.cross_entropy
def fast_cross_entropy_loss(logits, labels):
    """
    Arguments:
        logits: (batch, seq_len, vocab_size)
        labels: (batch, seq_len,)
    Returns:
        losses: float
    """
    batch, seq_len, d = logits.shape
    assert(labels.shape == (batch, seq_len))

    # We now support any vocab size due to Gemma!

    # Prelim support Qwen, Deepseek other large vocab sizes > 2^16
    # if d > MAX_FUSED_SIZE:
    #     logger.warning_once(
    #         f"Unsloth: Vocab size of {d} exceeds the max CUDA blocksize of {MAX_FUSED_SIZE}.\n"\
    #         "For now, Unsloth will use Pytorch's CrossEntropyLoss, which will entail a\n"\
    #         "25% increase in memory usage and be slower. Make an issue on \n"\
    #         "Unsloth's Github page if you want a faster and more memory efficient kernel!"
    #     )
    #     loss = slow_cross_entropy_loss(
    #         logits.float().view(batch*seq_len, d), # Must cast to float32 for numerical stability
    #         labels.view(-1),
    #     )
    #     return loss
    # else:
    # loss = Fast_CrossEntropyLoss.apply(
    #     logits.view(batch*seq_len, d),
    #     labels.view(-1),
    # )
    loss = cross_entropy_loss(
        logits.view(batch*seq_len, d),
        labels.view(-1),
        inplace_backward = True,
    )
    n_items = torch.count_nonzero(labels != -100)
    return loss.sum() / n_items
pass
