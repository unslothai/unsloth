// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { MAX_SERIALISED_LENGTH } from "./tool-arg-text.ts";

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
