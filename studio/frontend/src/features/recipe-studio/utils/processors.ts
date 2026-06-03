// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  JsonDocumentScoreProcessorConfig,
  RecipeProcessorConfig,
  SchemaTransformProcessorConfig,
} from "../types";

export function buildDefaultSchemaTransform(): SchemaTransformProcessorConfig {
  return {
    id: "schema-transform-1",
    // biome-ignore lint/style/useNamingConvention: api schema
    processor_type: "schema_transform",
    name: "schema_transform",
    template: '{\n  "text": "{{ column_name }}"\n}',
  };
}

export function buildDefaultJsonDocumentScore(): JsonDocumentScoreProcessorConfig {
  return {
    id: "json-document-score-1",
    // biome-ignore lint/style/useNamingConvention: api schema
    processor_type: "json_document_score",
    name: "doc_score",
    // biome-ignore lint/style/useNamingConvention: api schema
    prediction_column: "",
    // biome-ignore lint/style/useNamingConvention: api schema
    reference_column: "",
    schema: "",
    // biome-ignore lint/style/useNamingConvention: api schema
    default_comparator: "string",
    // biome-ignore lint/style/useNamingConvention: api schema
    score_column: "doc_score",
    // biome-ignore lint/style/useNamingConvention: api schema
    breakdown_column: "",
  };
}

export type { RecipeProcessorConfig };
