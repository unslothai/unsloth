# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

# Cloudflare's Clef decision models. The record encoding, layers and SystemOne answers are
# Cloudflare's own joint_schema_model.py (Apache-2.0), vendored unmodified in unsloth/_vendor/clef;
# this module only adds Unsloth's head forward on top of it.

import math

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
