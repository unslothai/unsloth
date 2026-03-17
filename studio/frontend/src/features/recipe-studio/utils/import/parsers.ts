// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { NodeConfig } from "../../types";
import { readString } from "./helpers";
import { parseExpression } from "./parsers/expression-parser";
import { parseLlm } from "./parsers/llm-parser";
export { parseModelConfig, parseModelProvider } from "./parsers/model-parser";
import { parseSampler } from "./parsers/sampler-parser";
import { parseValidator } from "./parsers/validator-parser";

type ColumnParser = (
  column: Record<string, unknown>,
  name: string,
  id: string,
  errors: string[],
) => NodeConfig | null;

const COLUMN_PARSERS: Record<string, ColumnParser> = {
  sampler: (column, name, id, errors) =>
    parseSampler(column, name, id, errors),
  expression: (column, name, id) => parseExpression(column, name, id),
  "llm-text": (column, name, id) => parseLlm(column, name, id),
  "llm-structured": (column, name, id) => parseLlm(column, name, id),
  "llm-code": (column, name, id) => parseLlm(column, name, id),
  "llm-judge": (column, name, id) => parseLlm(column, name, id),
  validation: (column, name, id) => parseValidator(column, name, id),
};

export function parseColumn(
  column: Record<string, unknown>,
  id: string,
  errors: string[],
): NodeConfig | null {
  const name = readString(column.name);
  if (!name) {
    errors.push("Column missing name.");
    return null;
  }
  const columnType = readString(column.column_type);
  const parser = columnType ? COLUMN_PARSERS[columnType] : null;
  if (parser) {
    return parser(column, name, id, errors);
  }
  errors.push(`Column ${name}: unsupported column_type.`);
  return null;
}
