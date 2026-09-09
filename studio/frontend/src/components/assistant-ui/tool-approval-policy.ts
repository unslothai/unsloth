// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { MAX_SERIALISED_LENGTH } from "./tool-arg-text.ts";

const ONE_CALL_APPROVAL_TOOLS = new Set(["edit_file", "python", "terminal"]);

/** Mutating local tools require a fresh decision for each exact call. */
export function canRememberToolApproval(toolName: string): boolean {
  return !ONE_CALL_APPROVAL_TOOLS.has(toolName);
}

/** Approval must never cover bytes hidden by the card's argument limit. */
export function canApproveToolArguments(
  toolName: string,
  args: unknown,
): boolean {
  if (toolName !== "edit_file") return true;
  try {
    const text = JSON.stringify(args);
    return typeof text === "string" && text.length <= MAX_SERIALISED_LENGTH;
  } catch {
    return false;
  }
}
