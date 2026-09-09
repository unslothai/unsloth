// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const ONE_CALL_APPROVAL_TOOLS = new Set(["edit_file", "python", "terminal"]);

/** Mutating local tools require a fresh decision for each exact call. */
export function canRememberToolApproval(toolName: string): boolean {
  return !ONE_CALL_APPROVAL_TOOLS.has(toolName);
}
