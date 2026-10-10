# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

# Portions are derived from Cloudflare's joint_schema_model.py (https://huggingface.co/Cloudflare/clef),
# Copyright 2026 Cloudflare, Inc., licensed under the Apache License, Version 2.0; the license text
# is in unsloth/_vendor/clef/LICENSE. That file is vendored unmodified in unsloth/_vendor/clef: this
# module imports its record encoding, layers and SystemOne answers, and adds Unsloth's head forward.

import contextlib
import math
import os
import warnings

import torch
import torch.nn.functional as functional

from .._vendor.clef.joint_schema_model import (  # noqa: F401
    IMAGE_PLACEHOLDER,
    MEDIA_BATCH_KEYS,
    QUESTION_TYPES,
    SYSTEM_PROMPT,
    VIDEO_PLACEHOLDER,
    ClefModel,
    EncodedQuestion,
    EncodedRecord,
    EvidenceRoutingLayer,
    collate_records,
    encode_record,
    question_options,
    render,
    systemone,
    systemone_answer,
)
from .._vendor.clef.joint_schema_model import JointSchemaHead as ReferenceJointSchemaHead

HEAD_FILES = ("joint_head.safetensors", "joint_head_config.json")


def _fast_enabled() -> bool:
    return os.environ.get("UNSLOTH_CLEF_FAST", "1") != "0"


_COMPILED = {}


# Eager unless UNSLOTH_CLEF_COMPILE=1: compiling costs 7-20 minutes per process and is no faster (L4, G4).


def _compile_supported(device) -> bool:
    if os.environ.get("UNSLOTH_CLEF_COMPILE") != "1" or device.type != "cuda":
        return False
    try:
        from torch.utils._triton import has_triton
    except ImportError:
        return False
    return has_triton()


def _compiled_logits(device):
    # Training only: one fullgraph compile with dynamic shapes; inference stays eager.
    if not _compile_supported(device):
        return None
    if "function" not in _COMPILED:
        _COMPILED["function"] = torch.compile(batched_logits, fullgraph = True, dynamic = True)
    return _COMPILED["function"]


def _disable_compile(error) -> None:
    warnings.warn(f"Unsloth: Clef head compile failed, running it eagerly: {error}")
    os.environ["UNSLOTH_CLEF_COMPILE"] = "0"
    _COMPILED.clear()


class ClefLogits(list):
    # Per record, per question logits like Cloudflare's head returns, plus the padded
    # [questions, options] tensor they are views of (-1e4 past each question's options).
    def __init__(self, flat_padded, records):
        rows = iter(range(flat_padded.shape[0]))
        super().__init__(
            [flat_padded[next(rows), : len(q.option_spans)] for q in record.questions]
            for record in records
        )
        self.flat_padded = flat_padded


def _span_mask(starts, ends, begin, end, dtype):
    # [B, S, C]: 1 where token begin + c lies in [start, end) of span s.
    positions = torch.arange(begin, end, device = starts.device)
    inside = (positions >= starts.unsqueeze(-1)) & (positions < ends.unsqueeze(-1))
    return inside.to(dtype)


def _norm_pool_forward(hidden, weight, bias, memory_weight, starts, ends, eps, chunk, matmul_dtype):
    # hidden_norm in fp32 one chunk of tokens at a time: only the [B, L, width] memory and the span
    # means are kept, never the [B, L, hidden] fp32 copy. The memory projection runs in
    # matmul_dtype, as autocast runs Cloudflare's head.
    batch, length, _ = hidden.shape
    memory = hidden.new_empty((batch, length, memory_weight.shape[0]), dtype = matmul_dtype)
    pooled = hidden.new_zeros((batch, starts.shape[1], hidden.shape[-1]), dtype = weight.dtype)
    projection = memory_weight.to(matmul_dtype).t()
    for begin in range(0, length, chunk):
        end = min(length, begin + chunk)
        # aten directly, as the backward does: Unsloth patches F.layer_norm into a function that
        # compiles itself, which would add a graph inside this opaque op.
        normalized = torch.ops.aten.native_layer_norm(
            hidden[:, begin:end].to(weight.dtype), weight.shape, weight, bias, eps
        )[0]
        memory[:, begin:end] = normalized.to(matmul_dtype) @ projection
        pooled += _span_mask(starts, ends, begin, end, weight.dtype) @ normalized
    counts = (ends - starts).clamp(min = 1).unsqueeze(-1).to(weight.dtype)
    return memory, pooled / counts


def _norm_pool_backward(
    grad_memory,
    grad_pooled,
    hidden,
    weight,
    bias,
    memory_weight,
    starts,
    ends,
    eps,
    chunk,
    matmul_dtype,
):
    length = hidden.shape[1]
    counts = (ends - starts).clamp(min = 1).unsqueeze(-1).to(weight.dtype)
    grad_pooled = grad_pooled / counts
    grad_hidden = torch.empty_like(hidden)
    grad_weight = torch.zeros_like(weight)
    grad_bias = torch.zeros_like(bias)
    grad_memory_weight = torch.zeros_like(memory_weight)
    projection = memory_weight.to(matmul_dtype)
    for begin in range(0, length, chunk):
        end = min(length, begin + chunk)
        x = hidden[:, begin:end].to(weight.dtype)
        normalized, mean, rstd = torch.ops.aten.native_layer_norm(
            x, weight.shape, weight, bias, eps
        )
        grad_chunk = grad_memory[:, begin:end].to(matmul_dtype)
        grad_normalized = (grad_chunk @ projection).to(weight.dtype)
        grad_normalized = (
            grad_normalized
            + _span_mask(starts, ends, begin, end, weight.dtype).transpose(1, 2) @ grad_pooled
        )
        grad_memory_weight += (
            grad_chunk.flatten(0, 1).t() @ normalized.to(matmul_dtype).flatten(0, 1)
        ).to(memory_weight.dtype)
        grad_x, grad_w, grad_b = torch.ops.aten.native_layer_norm_backward(
            grad_normalized, x, weight.shape, mean, rstd, weight, bias, [True, True, True]
        )
        grad_hidden[:, begin:end] = grad_x.to(hidden.dtype)
        grad_weight += grad_w
        grad_bias += grad_b
    return grad_hidden, grad_weight, grad_bias, grad_memory_weight


@torch.library.custom_op("unsloth::clef_norm_pool", mutates_args = ())
def _norm_pool_op(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    memory_weight: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    eps: float,
    chunk: int,
    matmul_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _norm_pool_forward(
        hidden, weight, bias, memory_weight, starts, ends, eps, chunk, matmul_dtype
    )


@_norm_pool_op.register_fake
def _norm_pool_fake(hidden, weight, bias, memory_weight, starts, ends, eps, chunk, matmul_dtype):
    batch, length, size = hidden.shape
    return (
        hidden.new_empty((batch, length, memory_weight.shape[0]), dtype = matmul_dtype),
        hidden.new_empty((batch, starts.shape[1], size), dtype = weight.dtype),
    )


@torch.library.custom_op("unsloth::clef_norm_pool_backward", mutates_args = ())
def _norm_pool_backward_op(
    grad_memory: torch.Tensor,
    grad_pooled: torch.Tensor,
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    memory_weight: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    eps: float,
    chunk: int,
    matmul_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _norm_pool_backward(
        grad_memory,
        grad_pooled,
        hidden,
        weight,
        bias,
        memory_weight,
        starts,
        ends,
        eps,
        chunk,
        matmul_dtype,
    )


@_norm_pool_backward_op.register_fake
def _norm_pool_backward_fake(
    grad_memory,
    grad_pooled,
    hidden,
    weight,
    bias,
    memory_weight,
    starts,
    ends,
    eps,
    chunk,
    matmul_dtype,
):
    return (
        torch.empty_like(hidden),
        torch.empty_like(weight),
        torch.empty_like(bias),
        torch.empty_like(memory_weight),
    )


def _norm_pool_setup(ctx, inputs, output):
    hidden, weight, bias, memory_weight, starts, ends, eps, chunk, matmul_dtype = inputs
    ctx.save_for_backward(hidden, weight, bias, memory_weight, starts, ends)
    ctx.eps, ctx.chunk, ctx.matmul_dtype = eps, chunk, matmul_dtype


def _norm_pool_grad(ctx, grad_memory, grad_pooled):
    hidden, weight, bias, memory_weight, starts, ends = ctx.saved_tensors
    grads = _norm_pool_backward_op(
        grad_memory.contiguous(),
        grad_pooled.contiguous(),
        hidden,
        weight,
        bias,
        memory_weight,
        starts,
        ends,
        ctx.eps,
        ctx.chunk,
        ctx.matmul_dtype,
    )
    return (*grads, None, None, None, None, None)


_norm_pool_op.register_autograd(_norm_pool_grad, setup_context = _norm_pool_setup)


def norm_pool(hidden, norm, memory_projection, starts, ends, chunk):
    # Custom ops skip autocast, so ask it which dtype Linear would run in.
    matmul_dtype = norm.weight.dtype
    if torch.is_autocast_enabled(hidden.device.type):
        matmul_dtype = torch.get_autocast_dtype(hidden.device.type)
    return _norm_pool_op(
        hidden,
        norm.weight,
        norm.bias,
        memory_projection.weight,
        starts,
        ends,
        norm.eps,
        chunk,
        matmul_dtype,
    )


def build_layout(records, device) -> dict:
    # Host metadata from the records, sent in one pinned copy: no device sync before the head.
    batch = len(records)
    q_max = max(len(r.questions) for r in records)
    p_max = max(sum(len(q.option_spans) for q in r.questions) for r in records)
    o_max = max(len(q.option_spans) for r in records for q in r.questions)
    spans = q_max + p_max + 1
    starts, ends = [[0] * spans for _ in range(batch)], [[0] * spans for _ in range(batch)]
    lengths, types = [], [[0] * q_max for _ in range(batch)]
    owner = [[0] * p_max for _ in range(batch)]
    option_index = [[[0] * o_max for _ in range(q_max)] for _ in range(batch)]
    option_valid = [[[0] * o_max for _ in range(q_max)] for _ in range(batch)]
    question_valid = [[0] * q_max for _ in range(batch)]
    lexical_tokens, lexical_segments, output_rows = [], [], []
    for b, record in enumerate(records):
        length = len(record.input_ids)
        lengths.append(length)
        starts[b][-1], ends[b][-1] = length - 1, length
        p = 0
        for q, question in enumerate(record.questions):
            starts[b][q], ends[b][q] = question.question_span
            types[b][q] = question.question_type
            question_valid[b][q] = 1
            output_rows.append(b * q_max + q)
            for o, (start, end) in enumerate(question.option_spans):
                starts[b][q_max + p], ends[b][q_max + p] = start, end
                owner[b][p], option_index[b][q][o], option_valid[b][q][o] = q, p, 1
                lexical_tokens.extend(record.input_ids[start:end])
                lexical_segments.extend([b * p_max + p] * (end - start))
                p += 1
    parts = {
        "starts": (starts, (batch, spans)),
        "ends": (ends, (batch, spans)),
        "lengths": (lengths, (batch,)),
        "types": (types, (batch, q_max)),
        "owner": (owner, (batch, p_max)),
        "option_index": (option_index, (batch, q_max, o_max)),
        "option_valid": (option_valid, (batch, q_max, o_max)),
        "question_valid": (question_valid, (batch, q_max)),
        "lexical_tokens": (lexical_tokens, (len(lexical_tokens),)),
        "lexical_segments": (lexical_segments, (len(lexical_segments),)),
        "output_rows": (output_rows, (len(output_rows),)),
    }
    flat = []
    for value, _ in parts.values():
        while value and isinstance(value[0], list):
            value = [x for row in value for x in row]
        flat.extend(value)
    packed = torch.tensor(flat, dtype = torch.long)
    if device.type == "cuda":
        packed = packed.pin_memory()
    packed = packed.to(device, non_blocking = True)
    layout, offset = {}, 0
    for name, (_, shape) in parts.items():
        size = math.prod(shape)
        layout[name] = packed[offset : offset + size].view(shape)
        offset += size
    layout["option_valid"] = layout["option_valid"].bool()
    layout["question_valid"] = layout["question_valid"].bool()
    return layout


def _layer_norm(norm, x):
    # Not Unsloth's compiled F.layer_norm: memory and queries differ in length, so the forward
    # recompiles it (static, then dynamic graph) and the checkpoint recompute picks the newer graph,
    # handing the first graph's backward a mean / rstd with unpadded strides (#13160).
    layer_norm = getattr(functional, "_uncompiled_layer_norm", functional.layer_norm)
    return layer_norm(x, norm.normalized_shape, norm.weight, norm.bias, norm.eps)


def _route(layer, queries, memory, padding):
    memory = _layer_norm(layer.memory_norm, memory)
    routed, _ = layer.attention(
        _layer_norm(layer.query_norm, queries),
        memory,
        memory,
        key_padding_mask = padding,
        need_weights = False,
    )
    queries = queries + layer.attention_dropout(routed)
    return queries + layer.feedforward(_layer_norm(layer.feedforward_norm, queries))


def _decode(layer, fields, memory, field_padding, padding):
    return layer(
        fields, memory, tgt_key_padding_mask = field_padding, memory_key_padding_mask = padding
    )


def _pristine_checkpoint():
    module = torch.utils.checkpoint
    for candidate in (
        getattr(module, "_unsloth_pristine_checkpoint", None),
        module.checkpoint,
        getattr(module, "_old_checkpoint", None),
    ):
        if getattr(candidate, "__module__", None) == "torch.utils.checkpoint":
            return candidate
    return None


@contextlib.contextmanager
def _torch_checkpoint(enabled):
    # Unsloth swaps torch.utils.checkpoint.checkpoint for its offloaded one, which rejects
    # use_reentrant = False and which dynamo does not recognise as activation checkpointing
    # (it matches the module attribute by identity), so the head runs under torch's own.
    module, pristine = torch.utils.checkpoint, _pristine_checkpoint() if enabled else None
    current = module.checkpoint
    if pristine is None or pristine is current:
        yield
        return
    module.checkpoint = pristine
    try:
        yield
    finally:
        module.checkpoint = current


def _maybe_checkpoint(function, checkpoint, *args):
    if checkpoint:
        return torch.utils.checkpoint.checkpoint(function, *args, use_reentrant = False)
    return function(*args)


def batched_logits(head, hidden_states, output_embedding_weight, layout, checkpoint, chunk):
    dtype = head.hidden_norm.weight.dtype
    batch, length = hidden_states.shape[:2]
    q_max, p_max = layout["types"].shape[1], layout["owner"].shape[1]
    starts, ends = layout["starts"], layout["ends"]
    memory, pooled = norm_pool(
        hidden_states, head.hidden_norm, head.memory_projection, starts, ends, chunk
    )
    padding = torch.arange(length, device = hidden_states.device) >= layout["lengths"].unsqueeze(-1)
    question_vectors = pooled[:, :q_max]
    contexts = pooled[:, q_max : q_max + p_max]
    global_vector = pooled[:, -1]

    rows = output_embedding_weight[layout["lexical_tokens"]].to(dtype)
    lexical = rows.new_zeros((batch * p_max, rows.shape[-1]))
    lexical.index_add_(0, layout["lexical_segments"], rows)
    lexical_counts = (ends[:, q_max : q_max + p_max] - starts[:, q_max : q_max + p_max]).clamp(
        min = 1
    )
    lexical = lexical.view(batch, p_max, -1) / lexical_counts.unsqueeze(-1).to(dtype)

    owner = layout["owner"]
    owner_questions = torch.gather(
        question_vectors, 1, owner.unsqueeze(-1).expand(-1, -1, question_vectors.shape[-1])
    )
    routed = (
        head.option_context_projection(contexts)
        + head.option_lexical_projection(lexical)
        + head.option_question_projection(owner_questions)
    )
    for layer in head.evidence_layers:
        routed = _maybe_checkpoint(_route, checkpoint, layer, routed, memory, padding)

    width = routed.shape[-1]
    option_index, option_valid = layout["option_index"], layout["option_valid"]
    o_max = option_index.shape[-1]
    options = torch.gather(routed, 1, option_index.view(batch, -1, 1).expand(-1, -1, width)).view(
        batch, q_max, o_max, width
    )
    base_fields = head.question_projection(question_vectors)
    scores = (options @ base_fields.unsqueeze(-1)).squeeze(-1) / math.sqrt(width)
    scores = scores.masked_fill(~option_valid, torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores, dim = -1)
    summaries = (weights.unsqueeze(-1) * options).sum(2)
    fields = (
        base_fields
        + head.option_summary_norm(summaries)
        + head.global_projection(global_vector).unsqueeze(1)
        + head.type_embedding(layout["types"])
    )
    field_padding = ~layout["question_valid"]
    for layer in head.layers:
        fields = _maybe_checkpoint(
            _decode, checkpoint, layer, fields, memory, field_padding, padding
        )
    fields = head.field_norm(fields)

    anchors = functional.normalize(question_vectors + global_vector.unsqueeze(1), dim = -1)
    owner_anchors = torch.gather(anchors, 1, owner.unsqueeze(-1).expand(-1, -1, anchors.shape[-1]))
    prior_scale = head.prior_logit_scale.clamp(max = math.log(100.0)).exp()
    prior = prior_scale * (functional.normalize(lexical, dim = -1) * owner_anchors).sum(-1)
    owner_fields = torch.gather(fields, 1, owner.unsqueeze(-1).expand(-1, -1, width))
    normed = head.option_norm(routed)
    cosine = functional.cosine_similarity(owner_fields, normed, dim = -1)
    features = torch.cat(
        [owner_fields, normed, owner_fields * normed, torch.abs(owner_fields - normed)], dim = -1
    )
    residual = head.residual_scorer(features).squeeze(-1)
    joint_scale = head.joint_logit_scale.clamp(max = math.log(100.0)).exp()
    per_option = prior + torch.sigmoid(head.residual_gate) * (joint_scale * cosine + residual)
    logits = torch.gather(per_option, 1, option_index.view(batch, -1)).view(batch, q_max, o_max)
    logits = logits.masked_fill(~option_valid, -1e4)
    return logits.view(batch * q_max, o_max).index_select(0, layout["output_rows"])


class JointSchemaHead(ReferenceJointSchemaHead):
    # Same parameters and state dict as Cloudflare's head; the forward also takes an fp32 copy of
    # the head over 16-bit hidden states.
    def __init__(
        self,
        hidden_size: int,
        width: int,
        routing_layers: int,
        layers: int,
        heads: int,
        feedforward: int,
        dropout: float = 0.0,
    ):
        super().__init__(hidden_size, width, routing_layers, layers, heads, feedforward, dropout)
        self.config = dict(
            hidden_size = hidden_size,
            width = width,
            routing_layers = routing_layers,
            layers = layers,
            heads = heads,
            feedforward = feedforward,
        )

    def forward(
        self, hidden_states, input_ids, attention_mask, records, output_embedding_weight
    ) -> list:
        if not _fast_enabled():
            return self.forward_per_record(
                hidden_states, input_ids, attention_mask, records, output_embedding_weight
            )
        layout = build_layout(records, hidden_states.device)
        training = self.training and torch.is_grad_enabled()
        checkpoint = training and os.environ.get("UNSLOTH_CLEF_CHECKPOINT", "1") != "0"
        chunk = int(os.environ.get("UNSLOTH_CLEF_CHUNK", "2048"))
        function = _compiled_logits(hidden_states.device) if training else None
        args = (self, hidden_states, output_embedding_weight, layout, checkpoint, chunk)
        with _torch_checkpoint(checkpoint):
            if function is not None:
                try:
                    flat = function(*args)
                except Exception as error:
                    if isinstance(error, torch.cuda.OutOfMemoryError):
                        raise
                    _disable_compile(error)
                    flat = batched_logits(*args)
            else:
                flat = batched_logits(*args)
        return ClefLogits(flat, records)

    def forward_per_record(
        self, hidden_states, input_ids, attention_mask, records, output_embedding_weight
    ) -> list:
        dtype = self.hidden_norm.weight.dtype
        normalized_hidden = self.hidden_norm(hidden_states.to(dtype))
        lengths = attention_mask.sum(-1).tolist()
        prior_scale = self.prior_logit_scale.clamp(max = math.log(100.0)).exp()
        joint_scale = self.joint_logit_scale.clamp(max = math.log(100.0)).exp()
        gate = torch.sigmoid(self.residual_gate)
        results = []
        for batch_index, record in enumerate(records):
            sequence_hidden = normalized_hidden[batch_index, : int(lengths[batch_index])]
            memory = self.memory_projection(sequence_hidden).unsqueeze(0)
            global_vector = sequence_hidden[-1]
            question_vectors = torch.stack(
                [sequence_hidden[slice(*q.question_span)].mean(0) for q in record.questions]
            )
            type_ids = torch.tensor(
                [q.question_type for q in record.questions], device = hidden_states.device
            )
            spans = [span for q in record.questions for span in q.option_spans]
            option_counts = [len(q.option_spans) for q in record.questions]
            contexts = torch.stack([sequence_hidden[start:end].mean(0) for start, end in spans])
            lexical = torch.stack(
                [
                    output_embedding_weight[input_ids[batch_index, start:end]].to(dtype).mean(0)
                    for start, end in spans
                ]
            )
            owner = torch.repeat_interleave(
                torch.arange(len(option_counts), device = hidden_states.device),
                torch.tensor(option_counts, device = hidden_states.device),
            )
            routed = (
                self.option_context_projection(contexts)
                + self.option_lexical_projection(lexical)
                + self.option_question_projection(question_vectors)[owner]
            ).unsqueeze(0)
            for layer in self.evidence_layers:
                routed = layer(routed, memory)
            split_options = torch.split(routed[0], option_counts, dim = 0)

            base_fields = self.question_projection(question_vectors)
            summaries = []
            for field, options in zip(base_fields, split_options):
                weights = torch.softmax(
                    torch.matmul(options, field) / math.sqrt(options.shape[-1]), dim = 0
                )
                summaries.append(torch.sum(weights.unsqueeze(-1) * options, dim = 0))
            fields = (
                base_fields
                + self.option_summary_norm(torch.stack(summaries))
                + self.global_projection(global_vector).unsqueeze(0)
                + self.type_embedding(type_ids)
            ).unsqueeze(0)
            for layer in self.layers:
                fields = layer(fields, memory)
            fields = self.field_norm(fields[0])

            anchors = functional.normalize(question_vectors + global_vector, dim = -1)
            lexical_split = torch.split(lexical, option_counts, dim = 0)
            record_logits = []
            for index, (field, options, lexical_options) in enumerate(
                zip(fields, split_options, lexical_split)
            ):
                prior = prior_scale * torch.matmul(
                    functional.normalize(lexical_options, dim = -1), anchors[index]
                )
                options = self.option_norm(options)
                repeated = field.unsqueeze(0).expand_as(options)
                cosine = functional.cosine_similarity(repeated, options, dim = -1)
                features = torch.cat(
                    [repeated, options, repeated * options, torch.abs(repeated - options)], dim = -1
                )
                residual = self.residual_scorer(features).squeeze(-1)
                record_logits.append(prior + gate * (joint_scale * cosine + residual))
            results.append(record_logits)
        return results
