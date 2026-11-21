// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Which typed questions a decision model can answer.
 *
 * llama.cpp's /v1/systemone takes typed questions (noul / choice / score) from
 * any decision model, so all three are offered by default. A model may narrow
 * that: the backend can report `decision_question_types`, and known families are
 * listed here. Unknown future models fall back to the full set rather than to a
 * guess, so a new model is never silently offered a type it cannot answer.
 */

// eslint-disable-next-line no-restricted-imports -- Leaf module: request types only, no React.
import type { DecisionType } from "@/features/settings/lib/decision-request";

export const ALL_DECISION_TYPES: readonly DecisionType[] = [
  "noul",
  "choice",
  "score",
];

const KNOWN = new Set<string>(ALL_DECISION_TYPES);

/** Families that answer a narrower set, keyed by llama.cpp's decision type. */
const BY_DECISION_TYPE: Record<string, readonly DecisionType[]> = {
  // Clef / OpenJev / Laya style single-pass deciders: typed questions as usual.
  // Listed explicitly so a future narrowing is a one-line change here.
  clef: ALL_DECISION_TYPES,
  openjev: ALL_DECISION_TYPES,
  laya: ALL_DECISION_TYPES,
  lev: ALL_DECISION_TYPES,
  kev: ALL_DECISION_TYPES,
  nimble: ALL_DECISION_TYPES,
  "lfm2-d1": ALL_DECISION_TYPES,
  "lfm3-d1": ALL_DECISION_TYPES,
};

function normalize(values: readonly string[] | null | undefined): DecisionType[] {
  if (!values || values.length === 0) return [];
  const seen = new Set<DecisionType>();
  for (const value of values) {
    const key = String(value).trim().toLowerCase();
    if (KNOWN.has(key)) seen.add(key as DecisionType);
  }
  // Stable order: the same list in every place that renders the picker.
  return ALL_DECISION_TYPES.filter((type) => seen.has(type));
}

export function isDecisionType(value: unknown): value is DecisionType {
  return typeof value === "string" && KNOWN.has(value.trim().toLowerCase());
}

/**
 * The question types to offer for the loaded decision model.
 *
 * Order of authority: what the backend reports for this model, then the known
 * family table, then every type the API supports.
 */
export function resolveDecisionQuestionTypes(input: {
  isDecision: boolean;
  decisionType?: string | null;
  declared?: readonly string[] | null;
}): DecisionType[] {
  if (!input.isDecision) return [];

  const declared = normalize(input.declared);
  if (declared.length > 0) return declared;

  const family = normalize(
    input.decisionType
      ? (BY_DECISION_TYPE[input.decisionType.trim().toLowerCase()] ?? null)
      : null,
  );
  if (family.length > 0) return family;

  return [...ALL_DECISION_TYPES];
}
