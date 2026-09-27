// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Local GGUF auto-compaction. On or off; how it compacts is the server's call.
 *
 *  Studio used to offer the policy as a setting, but the choice needed the reader to know what a
 *  checkpoint epoch and a rolling window were before it meant anything, and both sides of it were
 *  already the server's to configure. It now always follows the server, which is what the setting
 *  shipped as anyway (UNSLOTH_CONTEXT_POLICY, default "checkpoint"). */

export const DEFAULT_AUTO_COMPACT_ENABLED = true;

export function ggufCompactionRequestFields(options: {
  isGguf: boolean;
  autoCompactEnabled: boolean;
}): {
  context_overflow?: "error" | "truncate_oldest";
} {
  if (!options.isGguf) return {};
  if (!options.autoCompactEnabled) {
    // An omitted field falls back to UNSLOTH_CONTEXT_OVERFLOW, which may still compact. "error" is an
    // explicit refusal of that fallback.
    return { context_overflow: "error" };
  }
  // No context_policy: the server applies UNSLOTH_CONTEXT_POLICY.
  return { context_overflow: "truncate_oldest" };
}
