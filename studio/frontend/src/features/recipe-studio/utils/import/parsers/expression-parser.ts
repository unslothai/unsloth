// SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
// Copyright © 2025 Unsloth AI

import type {
  ExpressionConfig,
  ExpressionDtype,
} from "../../../types";
import { readString } from "../helpers";

const EXPRESSION_DTYPES: ExpressionDtype[] = ["str", "int", "float", "bool"];

export function parseExpression(
  column: Record<string, unknown>,
  name: string,
  id: string,
): ExpressionConfig {
  const dtype = readString(column.dtype);
  const normalized = EXPRESSION_DTYPES.includes(dtype as ExpressionDtype)
    ? (dtype as ExpressionDtype)
    : "str";
  return {
    id,
    kind: "expression",
    name,
    drop: column.drop === true,
    expr: readString(column.expr) ?? "",
    dtype: normalized,
  };
}
