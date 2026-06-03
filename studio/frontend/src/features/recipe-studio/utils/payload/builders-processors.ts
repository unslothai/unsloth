// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  ExpressionConfig,
  JsonDocumentScoreProcessorConfig,
  RecipeProcessorConfig,
  SchemaTransformProcessorConfig,
} from "../../types";
import { parseJsonObject } from "./parse";

export function buildExpressionColumn(
  config: ExpressionConfig,
  errors: string[],
): Record<string, unknown> {
  if (!config.expr.trim()) {
    errors.push(`Expression ${config.name}: expr required.`);
  }
  return {
    // biome-ignore lint/style/useNamingConvention: api schema
    column_type: "expression",
    name: config.name,
    drop: config.drop ?? false,
    expr: config.expr,
    dtype: config.dtype,
  };
}

function buildSchemaTransform(
  processor: SchemaTransformProcessorConfig,
  errors: string[],
): Record<string, unknown> | null {
  const name = processor.name.trim();
  if (!name) {
    errors.push("Schema transform: name is required.");
    return null;
  }
  const template = parseJsonObject(
    processor.template,
    `Schema transform ${name} template`,
    errors,
  );
  if (!template) {
    return null;
  }
  return {
    // biome-ignore lint/style/useNamingConvention: api schema
    processor_type: "schema_transform",
    name,
    template,
  };
}

function buildJsonDocumentScore(
  processor: JsonDocumentScoreProcessorConfig,
  errors: string[],
): Record<string, unknown> | null {
  const name = processor.name.trim();
  if (!name) {
    errors.push("Document score: name is required.");
    return null;
  }
  const predictionColumn = processor.prediction_column.trim();
  const referenceColumn = processor.reference_column.trim();
  const scoreColumn = processor.score_column.trim() || "doc_score";
  if (!predictionColumn) {
    errors.push(`Document score ${name}: prediction column is required.`);
    return null;
  }
  if (!referenceColumn) {
    errors.push(`Document score ${name}: reference column is required.`);
    return null;
  }
  // Empty schema = no schema (null) — default_comparator applies to every leaf.
  let schemaValue: unknown = null;
  const schemaText = processor.schema.trim();
  if (schemaText) {
    schemaValue = parseJsonObject(
      processor.schema,
      `Document score ${name} schema`,
      errors,
    );
    if (!schemaValue) {
      return null;
    }
  }
  const breakdownColumn = processor.breakdown_column.trim();
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
    default_comparator: processor.default_comparator || "string",
    // biome-ignore lint/style/useNamingConvention: api schema
    score_column: scoreColumn,
    // biome-ignore lint/style/useNamingConvention: api schema
    breakdown_column: breakdownColumn || null,
  };
}

export function buildProcessors(
  processors: RecipeProcessorConfig[],
  errors: string[],
): Record<string, unknown>[] {
  const output: Record<string, unknown>[] = [];
  for (const processor of processors) {
    if (processor.processor_type === "schema_transform") {
      const built = buildSchemaTransform(processor, errors);
      if (built) output.push(built);
      continue;
    }
    if (processor.processor_type === "json_document_score") {
      const built = buildJsonDocumentScore(processor, errors);
      if (built) output.push(built);
      continue;
    }
  }
  return output;
}
