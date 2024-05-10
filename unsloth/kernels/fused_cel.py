import logging
import types
from math import sqrt
from typing import List, Optional, Tuple, Union

logger = logging.getLogger(__name__)
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithPast


@triton.jit
def fused_cross_entropy_fwd_bwd_kernel(
    output_loss_ptr,
    output_logit_grad_ptr,
    input_logit_ptr,
    input_targ_ptr,
    input_divisor_ptr,
    output_loss_stride,
    output_logit_grad_stride,
    input_logit_stride,
    input_targ_stride,
    n_cols,
    ignore_index,
    BLOCK_SIZE: tl.constexpr,
):
    # Get pointers to current row for all inputs/outputs
    row_idx = tl.program_id(0)
    logit_grad_row_start_ptr = (
        output_logit_grad_ptr + row_idx * output_logit_grad_stride
    )
    logit_row_start_ptr = input_logit_ptr + row_idx * input_logit_stride
    targ_ptr = input_targ_ptr + row_idx * input_targ_stride
    loss_ptr = output_loss_ptr + row_idx * output_loss_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    logit_row_ptrs = logit_row_start_ptr + col_offsets
    logit_grad_row_ptrs = logit_grad_row_start_ptr + col_offsets

    # Load data into SRAM
    logit_row_unnormalized = tl.load(
        logit_row_ptrs, mask=col_offsets < n_cols, other=float("-Inf")
    )
    targ = tl.load(targ_ptr)
    divisor = tl.load(input_divisor_ptr)

    # Normalize logits and compute some useful intermediate values
    logit_row = logit_row_unnormalized - tl.max(
        logit_row_unnormalized, axis=0
    )  # Subtract max value for numerical stability
    exp_logit_row = tl.exp(logit_row)
    sum_exp_logit_row = tl.sum(exp_logit_row, axis=0)

    # Compute loss
    log_sum_exp_logit_row = tl.log(sum_exp_logit_row)
    logit_gt_logit = tl.sum(tl.where(targ == col_offsets, logit_row, 0.0))
    loss = log_sum_exp_logit_row - logit_gt_logit
    loss = loss / divisor
    loss = tl.where(targ == ignore_index, 0.0, loss)
    tl.store(loss_ptr, loss)

    # Compute gradients
    targ_one_hot = tl.where(targ == col_offsets, 1.0, 0.0)
    grad = exp_logit_row / sum_exp_logit_row - targ_one_hot
    grad = grad / divisor
    grad = tl.where(targ == ignore_index, 0.0, grad)
    tl.store(logit_grad_row_ptrs, grad, mask=col_offsets < n_cols)


class FusedCrossEntropyLossFunction(torch.autograd.Function):
    # NOTE: Changes from original implementation:
    # - Reshape inputs within forward from bs x seqlen x hidden_dim to (bs * seqlen) x hidden_dim per kernel requirement
    # - Reshape labels within forward from bs x seqlen to (bs * seqlen)
    # - Upcast `loss` from float32 (originally initialized to autocast / in-feat dtype)
    # - Reshape dX from (bs * seqlen) x hidden_dim to bs x seqlen x hidden_dim
    @staticmethod
    def forward(
        ctx,
        in_feat: torch.Tensor,
        proj_weight: torch.Tensor,
        targ: torch.Tensor,
        n_loop_iters: int,
        ignore_index: int,
        reduction: str,
    ):
        bs, seqlen, hidden_dim = in_feat.shape
        in_feat = in_feat.view(-1, in_feat.shape[-1])
        targ = targ.view(-1)

        n_tokens = in_feat.shape[0]
        n_classes = proj_weight.shape[0]

        assert in_feat.ndim == 2, in_feat.ndim
        assert proj_weight.ndim == 2, proj_weight.ndim
        assert targ.ndim == 1, targ.shape
        assert (
            in_feat.shape[0] == targ.shape[0]
        ), f"Number of tokens in in_feat and targ is not equal: {(in_feat.shape, targ.shape) = }"
        assert reduction in ("mean", "sum"), reduction
        assert n_loop_iters > 0, n_loop_iters
        assert n_tokens % n_loop_iters == 0, (n_tokens, n_loop_iters)

        NUM_WARPS = 16

        BLOCK_SIZE = triton.next_power_of_2(n_classes)

        # Change loss from in_feat.dtype to float32
        loss = torch.empty(n_tokens, dtype=torch.float32, device=in_feat.device)
        dtype = (
            torch.get_autocast_gpu_dtype()
            if torch.is_autocast_enabled()
            else in_feat.dtype
        )

        if proj_weight.requires_grad:
            grad_proj_weight = torch.zeros_like(proj_weight, dtype=dtype)
        else:
            grad_proj_weight = None

        if in_feat.requires_grad:
            grad_in_feat = torch.zeros_like(in_feat)
        else:
            grad_in_feat = None

        divisor = (
            (targ != ignore_index).sum().to(dtype)
            if reduction == "mean"
            else torch.ones(1, dtype=dtype, device=in_feat.device)
        )

        # Divide the input into chunks of size num_tokens // n_loop_iters, then compute the loss for each of these groups
        proj_weight_cast = proj_weight.to(dtype)

        loop_chunk_size = triton.cdiv(n_tokens, n_loop_iters)
        logits_chunk_cast = torch.zeros(
            (loop_chunk_size, n_classes), dtype=dtype, device=in_feat.device
        )

        for i, in_feat_chunk in enumerate(torch.split(in_feat, loop_chunk_size)):
            token_start_idx = i * loop_chunk_size
            token_end_idx = (i + 1) * loop_chunk_size

            in_feat_chunk = in_feat_chunk.to(dtype)

            # Compute logits
            torch.matmul(in_feat_chunk, proj_weight_cast.T, out=logits_chunk_cast)
            logits_chunk = logits_chunk_cast.float()

            # Compute loss
            loss_chunk = loss[token_start_idx:token_end_idx]
            targ_chunk = targ[token_start_idx:token_end_idx]

            n_tokens_chunk = logits_chunk.shape[0]
            grad_logits_chunk = (
                logits_chunk  # NOTE: we override the logits with their gradients
            )

            fused_cross_entropy_fwd_bwd_kernel[(n_tokens_chunk,)](
                loss_chunk,
                grad_logits_chunk,
                logits_chunk,
                targ_chunk,
                divisor,
                loss_chunk.stride(0),
                grad_logits_chunk.stride(0),
                logits_chunk.stride(0),
                targ_chunk.stride(0),
                n_classes,
                ignore_index,
                num_warps=NUM_WARPS,
                BLOCK_SIZE=BLOCK_SIZE,
            )

            grad_logits_chunk = grad_logits_chunk.to(dtype)

            if in_feat.requires_grad:
                grad_in_feat[token_start_idx:token_end_idx] = (
                    grad_logits_chunk @ proj_weight_cast
                )

            if proj_weight.requires_grad:
                torch.addmm(
                    grad_proj_weight,
                    grad_logits_chunk.T,
                    in_feat_chunk,
                    out=grad_proj_weight,
                )

        # NOTE: if reduction == "mean" we already divide by an appropriate normalization factor in the kernel so we can alway sum here
        loss = loss.sum()

        # Save data for backward
        ctx.in_feat_requires_grad = in_feat.requires_grad
        ctx.proj_weight_requires_grad = proj_weight.requires_grad

        if proj_weight.requires_grad and in_feat.requires_grad:
            grad_in_feat = grad_in_feat.view(bs, seqlen, hidden_dim)
            ctx.save_for_backward(grad_in_feat, grad_proj_weight)
        elif proj_weight.requires_grad and not in_feat.requires_grad:
            ctx.save_for_backward(grad_proj_weight)
        elif not proj_weight.requires_grad and in_feat.requires_grad:
            grad_in_feat = grad_in_feat.view(bs, seqlen, hidden_dim)
            ctx.save_for_backward(grad_in_feat)

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        grad_in_feat = grad_proj_weight = None
        if ctx.in_feat_requires_grad and ctx.proj_weight_requires_grad:
            grad_in_feat, grad_proj_weight = ctx.saved_tensors
        elif not ctx.in_feat_requires_grad and ctx.proj_weight_requires_grad:
            (grad_proj_weight,) = ctx.saved_tensors
        elif ctx.in_feat_requires_grad and not ctx.proj_weight_requires_grad:
            (grad_in_feat,) = ctx.saved_tensors

        #   assert grad_output.shape == tuple(), grad_output.shape

        if grad_in_feat is not None:
            grad_in_feat *= grad_output
        if grad_proj_weight is not None:
            grad_proj_weight *= grad_output

        return grad_in_feat, grad_proj_weight, None, None, None, None


def fused_cel_linear(
    x, proj_weight, labels, n_loop_iters=1, ignore_index=-100, reduction="mean"
):
    """
    x: (bs, seqlen, hidden_dim)
    proj_weight: (vocab_size, hidden_dim)
    labels: (bs, seqlen)

    """
    return FusedCrossEntropyLossFunction.apply(
        x, proj_weight, labels, n_loop_iters, ignore_index, reduction
    )


def fused_cel_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    hidden_states = outputs[0]
    if self.config.pretraining_tp > 1:
        lm_head_slices = self.lm_head.weight.split(
            self.vocab_size // self.config.pretraining_tp, dim=0
        )
        logits = [
            F.linear(hidden_states, lm_head_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        logits = torch.cat(logits, dim=-1)
    elif self.config.use_fused_cel:
        logger.warning_once(
            "Using fused cross entropy loss, output logits will be in None"
        )
        # Need to shift, since kernel assumes labels and hidden states have same bs * seqlen
        shift_hidden_states = hidden_states[
            ..., :-1, :
        ].contiguous()  # This is important -- MUST call contiguous, otherwise will cause downstream reshaping issues
        shift_labels = labels[..., 1:].contiguous()
        shift_labels = shift_labels.to(shift_hidden_states.device)

        loss = fused_cel_linear(
            shift_hidden_states,
            self.lm_head.weight,
            shift_labels,
            n_loop_iters=self.config.fused_cel_n_loop_iters,
            ignore_index=self.config.fused_cel_ignore_index,
            reduction=self.config.fused_cel_reduction,
        )
        logits = None

    else:
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def patch_model(
    model,
    use_fused_cel=True,
    fused_cel_n_loop_iters=1,
    fused_cel_ignore_index=-100,
    fused_cel_reduction="mean",
):
    model.config.update(
        {
            "use_fused_cel": use_fused_cel,
            "fused_cel_n_loop_iters": fused_cel_n_loop_iters,
            "fused_cel_ignore_index": fused_cel_ignore_index,
            "fused_cel_reduction": fused_cel_reduction,
        }
    )
    model.forward = types.MethodType(fused_cel_forward, model)
    return model
