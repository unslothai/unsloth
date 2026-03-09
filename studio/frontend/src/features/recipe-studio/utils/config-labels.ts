// SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
// Copyright © 2025 Unsloth AI

import type {
  ExpressionDtype,
  LlmType,
  SamplerType,
} from "../types";

const SAMPLER_LABELS: Record<SamplerType, string> = {
  category: "Category",
  subcategory: "Subcategory",
  uniform: "Uniform",
  gaussian: "Gaussian",
  bernoulli: "Bernoulli",
  datetime: "Datetime",
  timedelta: "Timedelta",
  uuid: "UUID",
  person: "Person",
  person_from_faker: "Person (Faker)",
};

const LLM_LABELS: Record<LlmType, string> = {
  text: "LLM Text",
  structured: "LLM Structured",
  code: "LLM Code",
  judge: "LLM Judge",
};

const EXPRESSION_LABELS: Record<ExpressionDtype, string> = {
  str: "Text",
  int: "Int",
  float: "Float",
  bool: "Bool",
};

export function labelForSampler(type: SamplerType): string {
  return SAMPLER_LABELS[type] ?? "Sampler";
}

export function labelForLlm(type: LlmType): string {
  return LLM_LABELS[type] ?? "LLM";
}

export function labelForExpression(type: ExpressionDtype): string {
  return EXPRESSION_LABELS[type] ?? "Expression";
}
