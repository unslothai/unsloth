// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

type Message = {
  role?: unknown;
  from?: unknown;
  content?: unknown;
  value?: unknown;
};

function ownSystemMessage(prompt: unknown): string | null {
  if (!Array.isArray(prompt)) {
    return null;
  }
  const msg = (prompt as (Message | null)[]).find(
    (m) => m && typeof m === "object" && (m.role ?? m.from) === "system",
  );
  if (!msg) {
    return null;
  }
  const content = msg.content ?? msg.value;
  return typeof content === "string" ? content : JSON.stringify(content ?? "");
}

/** The system message GRPO trains this row with: the prompt's own, else the mapped system
 * column's, else the configured one (format_rl_dataset's precedence). */
export function effectiveSystemPrompt(
  row: Record<string, unknown> | null | undefined,
  mapping: Record<string, string>,
  configured: string,
): string {
  const columnOf = (role: string) =>
    Object.entries(mapping).find(([, r]) => r === role)?.[0];
  const promptColumn = columnOf("prompt");
  const own = ownSystemMessage(promptColumn ? row?.[promptColumn] : undefined);
  if (own !== null) {
    return own;
  }
  const systemColumn = columnOf("system");
  const fromColumn = systemColumn ? row?.[systemColumn] : undefined;
  return fromColumn ? String(fromColumn) : configured;
}
