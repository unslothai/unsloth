# Copyright 2026 Cloudflare, Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Record encoding and joint schema head of Cloudflare's Clef decision models, from
# joint_schema_model.py in https://huggingface.co/Cloudflare/clef. The encoding must stay
# token-identical to it; the head also accepts an fp32 copy of itself over 16-bit hidden states.

import json
import math
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any

import torch
import torch.nn.functional as functional

SYSTEM_PROMPT = (
    "Read the complete state and schema. Decide every field jointly. Each answer "
    "must be exactly one of that field's allowed options."
)
QUESTION_TYPES = {"noul": 0, "choice": 1, "score": 2}
HEAD_FILES = ("joint_head.safetensors", "joint_head_config.json")


def render(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii = False, separators = (",", ":"), sort_keys = True)


def question_options(question: dict) -> list:
    question_type = str(question["type"])
    if question_type == "noul":
        criteria = {
            "true": "The proposition is true or the answer is yes.",
            "false": "The proposition is false or the answer is no.",
        }
        criteria.update(question.get("criteria") or {})
        return [(key, criteria[key]) for key in ("true", "false")]
    if question_type == "choice":
        return sorted((str(key), value) for key, value in question["criteria"].items())
    return [(str(index), value) for index, value in enumerate(question["criteria"])]


@dataclass(frozen = True)
class EncodedQuestion:
    question_id: str
    question_type: int
    question_span: tuple
    option_spans: tuple
    option_ids: tuple


@dataclass(frozen = True)
class EncodedRecord:
    input_ids: tuple
    questions: tuple
    record_id: str
    media: Any = dataclass_field(default = None, compare = False, repr = False)


def _tokens(tokenizer, text: str) -> list:
    return tokenizer(text, add_special_tokens = False).input_ids


def encode_record(
    tokenizer,
    record: dict,
    max_length: int = 16384,
) -> EncodedRecord:
    schema_ids = _tokens(tokenizer, "\n\nSCHEMA FIELDS:\n")
    questions = []
    for question_index, (question_id, question) in enumerate(record["questions"].items()):
        schema_ids.extend(
            _tokens(
                tokenizer,
                f"\nFIELD {question_index + 1}\nID: {question_id}\nTYPE: {question['type']}\nINSTRUCTION: ",
            )
        )
        question_start = len(schema_ids)
        instructions = question.get("instructions")
        if instructions is None or instructions == "":
            instructions = str(question_id)
        schema_ids.extend(_tokens(tokenizer, render(instructions)))
        question_end = len(schema_ids)
        schema_ids.extend(_tokens(tokenizer, "\nALLOWED OPTIONS:\n"))
        option_spans, option_ids = [], []
        for option_index, (option_id, description) in enumerate(question_options(question)):
            schema_ids.extend(_tokens(tokenizer, f"OPTION {option_index + 1}: "))
            option_start = len(schema_ids)
            semantics = {"option_id": option_id}
            if description is not None:
                semantics["description"] = description
            schema_ids.extend(_tokens(tokenizer, render(semantics)))
            option_spans.append((option_start, len(schema_ids)))
            option_ids.append(option_id)
            schema_ids.extend(_tokens(tokenizer, "\n"))
        schema_ids.extend(_tokens(tokenizer, "END FIELD\n"))
        questions.append(
            (str(question_id), question, (question_start, question_end), option_spans, option_ids)
        )

    prefix_ids = _tokens(
        tokenizer, f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n"
    )
    suffix_ids = _tokens(
        tokenizer,
        "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:",
    )
    state_ids = _tokens(tokenizer, render(record["state"]))
    fixed_length = len(prefix_ids) + len(schema_ids) + len(suffix_ids)
    if fixed_length > max_length:
        raise ValueError(
            f"schema requires {fixed_length} tokens before state; maximum is {max_length}"
        )
    state_ids = state_ids[: max_length - fixed_length]
    offset = len(prefix_ids) + len(state_ids)
    shifted = tuple(
        EncodedQuestion(
            question_id = question_id,
            question_type = QUESTION_TYPES[str(question["type"])],
            question_span = (span[0] + offset, span[1] + offset),
            option_spans = tuple((start + offset, end + offset) for start, end in option_spans),
            option_ids = tuple(option_ids),
        )
        for question_id, question, span, option_spans, option_ids in questions
    )
    input_ids = tuple(prefix_ids + state_ids + schema_ids + suffix_ids)
    if not shifted:
        raise ValueError("record produced no questions")
    return EncodedRecord(
        input_ids = input_ids, questions = shifted, record_id = str(record.get("id", "unknown"))
    )


class EvidenceRoutingLayer(torch.nn.Module):
    def __init__(
        self,
        width: int,
        heads: int,
        feedforward: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.query_norm = torch.nn.LayerNorm(width)
        self.memory_norm = torch.nn.LayerNorm(width)
        self.attention = torch.nn.MultiheadAttention(
            width, heads, dropout = dropout, batch_first = True
        )
        self.attention_dropout = torch.nn.Dropout(dropout)
        self.feedforward_norm = torch.nn.LayerNorm(width)
        self.feedforward = torch.nn.Sequential(
            torch.nn.Linear(width, feedforward),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(feedforward, width),
            torch.nn.Dropout(dropout),
        )

    def forward(self, queries, memory):
        memory = self.memory_norm(memory)
        routed, _ = self.attention(self.query_norm(queries), memory, memory, need_weights = False)
        queries = queries + self.attention_dropout(routed)
        return queries + self.feedforward(self.feedforward_norm(queries))


class JointSchemaHead(torch.nn.Module):
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
        super().__init__()
        self.config = dict(
            hidden_size = hidden_size,
            width = width,
            routing_layers = routing_layers,
            layers = layers,
            heads = heads,
            feedforward = feedforward,
        )
        self.hidden_norm = torch.nn.LayerNorm(hidden_size)
        self.memory_projection = torch.nn.Linear(hidden_size, width, bias = False)
        self.question_projection = torch.nn.Linear(hidden_size, width, bias = False)
        self.option_question_projection = torch.nn.Linear(hidden_size, width, bias = False)
        self.global_projection = torch.nn.Linear(hidden_size, width, bias = False)
        self.option_context_projection = torch.nn.Linear(hidden_size, width, bias = False)
        self.option_lexical_projection = torch.nn.Linear(hidden_size, width, bias = False)
        self.type_embedding = torch.nn.Embedding(3, width)
        self.evidence_layers = torch.nn.ModuleList(
            [
                EvidenceRoutingLayer(width, heads, feedforward, dropout)
                for _ in range(routing_layers)
            ]
        )
        self.option_summary_norm = torch.nn.LayerNorm(width)
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.TransformerDecoderLayer(
                    d_model = width,
                    nhead = heads,
                    dim_feedforward = feedforward,
                    dropout = dropout,
                    activation = "gelu",
                    batch_first = True,
                    norm_first = True,
                )
                for _ in range(layers)
            ]
        )
        self.field_norm = torch.nn.LayerNorm(width)
        self.option_norm = torch.nn.LayerNorm(width)
        self.residual_scorer = torch.nn.Sequential(
            torch.nn.Linear(width * 4, width),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(width, 1),
        )
        self.prior_logit_scale = torch.nn.Parameter(torch.zeros(()))
        self.joint_logit_scale = torch.nn.Parameter(torch.zeros(()))
        self.residual_gate = torch.nn.Parameter(torch.zeros(()))

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


def systemone_answer(question: dict, probabilities: dict) -> dict:
    if question["type"] == "noul":
        return {"type": "noul", "noul": round(probabilities["true"], 4)}
    if question["type"] == "choice":
        options = [str(option) for option in question["criteria"]]
        choice = max(options, key = probabilities.__getitem__)
        return {
            "type": "choice",
            "choice": choice,
            "confidence": round(probabilities[choice], 4),
            "probabilities": {option: round(probabilities[option], 4) for option in options},
        }
    levels = [str(index) for index in range(len(question["criteria"]))]
    return {
        "type": "score",
        "score": round(sum(index * probabilities[level] for index, level in enumerate(levels)), 4),
        "confidence": round(max(probabilities[level] for level in levels), 4),
        "legend": dict(zip(levels, question["criteria"])),
        "probabilities": {level: round(probabilities[level], 4) for level in levels},
    }
