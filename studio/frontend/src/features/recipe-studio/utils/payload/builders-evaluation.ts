// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { EvaluationDocumentScoreConfig } from "../../types";
import { parseJsonObject } from "./parse";

export function buildEvaluationDocumentScoreProcessor(
  config: EvaluationDocumentScoreConfig,
  errors: string[],
): Record<string, unknown> | null {
  const name = config.name.trim() || "doc_score";
  const predictionColumn = config.prediction_column.trim();
  const referenceColumn = config.reference_column.trim();
  let schemaValue: unknown = null;
  const schemaText = config.schema.trim();
  if (schemaText) {
    schemaValue = parseJsonObject(
      config.schema,
      `Document score ${name} schema`,
      errors,
    );
    if (!schemaValue) {
      schemaValue = null;
    }
  }
  const breakdownColumn = config.breakdown_column.trim();
  const scoreColumn = config.score_column.trim() || "doc_score";
  return {
    // biome-ignore lint/style/useNamingConvention: api schema
    processor_type: "json_document_score",
    name,
    // biome-ignore lint/style/useNamingConvention: api schema
    prediction_column: predictionColumn,
    // biome-ignore lint/style/useNamingConvention: api schema
    reference_column: referenceColumn,
    schema: schemaValue,
    // biome-ignore lint/style/useNamingConvention: api schema
    default_comparator: config.default_comparator || "string",
    // biome-ignore lint/style/useNamingConvention: api schema
    score_column: scoreColumn,
    // biome-ignore lint/style/useNamingConvention: api schema
    breakdown_column: breakdownColumn || null,
  };
}
