// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  ExpressionDtype,
  LlmType,
  SamplerType,
} from "../types";

const SAMPLER_LABELS: Record<SamplerType, string> = {
  category: "Category",
  subcategory: "Subcategory",
  uniform: "Random number",
  gaussian: "Bell-curve number",
  bernoulli: "Yes/no value",
  datetime: "Date and time",
  timedelta: "Time offset",
  uuid: "Unique ID",
  person: "Synthetic person",
  person_from_faker: "Synthetic person",
};

const LLM_LABELS: Record<LlmType, string> = {
  text: "AI text",
  structured: "AI structured data",
  code: "AI code",
  judge: "AI scorer",
};

const EXPRESSION_LABELS: Record<ExpressionDtype, string> = {
  str: "Text",
  int: "Int",
  float: "Float",
  bool: "Bool",
};

export function labelForSampler(type: SamplerType): string {
  return SAMPLER_LABELS[type] ?? "Generated field";
}

export function labelForLlm(type: LlmType): string {
  return LLM_LABELS[type] ?? "AI";
}

export function labelForExpression(type: ExpressionDtype): string {
  return EXPRESSION_LABELS[type] ?? "Formula";
}
