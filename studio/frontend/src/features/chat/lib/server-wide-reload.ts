// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type ReloadHint =
  | { reloadRequired: boolean }
  | "unsupported"
  /** The read failed: says nothing, which is not "no". */
  | "unknown";

function declines(hint: ReloadHint): boolean {
  return hint === "unknown" || (hint !== "unsupported" && hint.reloadRequired);
}

/** Whether a server-wide setting (memory policy, VRAM budget) changed since launch.
 *  Unknown declines adoption; an absent route is not unknown. */
export function serverWideReloadRequired(signals: {
  modelMemory: ReloadHint;
  vramBudget: ReloadHint;
}): boolean {
  return declines(signals.modelMemory) || declines(signals.vramBudget);
}
