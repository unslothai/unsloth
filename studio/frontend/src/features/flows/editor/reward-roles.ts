// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

function mapped(
  params: Record<string, unknown>,
  role: string,
): string | undefined {
  const mapping = params.custom_format_mapping;
  if (!mapping || typeof mapping !== "object") return undefined;
  return Object.entries(mapping as Record<string, unknown>).find(
    ([, r]) => r === role,
  )?.[0];
}

/** The column training reads for ``role``: a non-empty Advanced mapping wins over the simple fields. */
export function roleColumn(
  params: Record<string, unknown>,
  role: "prompt" | "answer",
): unknown {
  const mapping = params.custom_format_mapping;
  if (mapping && typeof mapping === "object" && Object.keys(mapping).length)
    return mapped(params, role);
  return params[`${role}_column`] || mapped(params, role);
}

// Training's _AUTO_ROLE_NAMES (studio/backend/core/training/rl.py), in its priority order.
export const PROMPT_NAMES = [
  "prompt",
  "question",
  "instruction",
  "problem",
  "query",
  "input",
];
export const ANSWER_NAMES = [
  "answer",
  "solution",
  "final_answer",
  "target",
  "label",
];

/** The row's value for a role: the named column, else the first alias the row has, in the order
 * training's resolve_role_columns tries them (`answer` before `solution`). */
export function pick(
  row: Record<string, unknown>,
  named: unknown,
  names: string[],
) {
  const keys = Object.keys(row);
  const key =
    String(named ?? "").trim() ||
    names
      .map((name) => keys.find((k) => k.toLowerCase() === name))
      .find(Boolean);
  return key && row[key] !== undefined ? String(row[key]) : "";
}
