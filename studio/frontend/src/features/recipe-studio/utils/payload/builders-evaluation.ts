// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { EvaluationDocumentScoreConfig } from "../../types";
import { parseJsonObject } from "./parse";

// Always emit an evaluations[] entry so the block survives save/load. Missing
// required fields are tolerated (emitted as empty strings) — the backend
// scorer will skip rows where the column lookup fails, and on reload the
// block remains on the canvas in its incomplete state for the user to finish.
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
    evaluation_type: "json_document_score",
    name,
    prediction_column: predictionColumn,
    reference_column: referenceColumn,
    schema: schemaValue,
    default_comparator: config.default_comparator || "string",
    score_column: scoreColumn,
    breakdown_column: breakdownColumn || null,
  };
}
