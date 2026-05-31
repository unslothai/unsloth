// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ExpressionConfig, RecipeProcessorConfig } from "../../types";
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

export function buildProcessors(
  processors: RecipeProcessorConfig[],
  errors: string[],
): Record<string, unknown>[] {
  const output: Record<string, unknown>[] = [];
  for (const processor of processors) {
    if (processor.processor_type !== "schema_transform") {
      continue;
    }
    const name = processor.name.trim();
    if (!name) {
      errors.push("Schema transform: name is required.");
      continue;
    }
    const template = parseJsonObject(
      processor.template,
      `Schema transform ${name} template`,
      errors,
    );
    if (!template) {
      continue;
    }
    output.push({
      // biome-ignore lint/style/useNamingConvention: api schema
      processor_type: "schema_transform",
      name,
      template,
    });
  }
  return output;
}
