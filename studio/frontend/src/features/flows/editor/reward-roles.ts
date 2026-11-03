// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

function mapped(params: Record<string, unknown>, role: string): string | undefined {
  const mapping = params.custom_format_mapping;
  if (!mapping || typeof mapping !== "object") return undefined;
  return Object.entries(mapping as Record<string, unknown>).find(([, r]) => r === role)?.[0];
}

/** The column training reads for ``role``: a non-empty Advanced mapping wins over the simple fields. */
export function roleColumn(params: Record<string, unknown>, role: "prompt" | "answer"): unknown {
  const mapping = params.custom_format_mapping;
  if (mapping && typeof mapping === "object" && Object.keys(mapping).length)
    return mapped(params, role);
  return params[`${role}_column`] || mapped(params, role);
}
