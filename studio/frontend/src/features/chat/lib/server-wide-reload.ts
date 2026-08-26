// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** One settings read, in the three states the shortcut has to tell apart. */
export type ReloadHint =
  | { reloadRequired: boolean }
  /** The backend does not serve this route, so it has no such state to disagree about. */
  | "unsupported"
  /** The read could not be made. It says nothing, which is not the same as "no". */
  | "unknown";

function declines(hint: ReloadHint): boolean {
  return hint === "unknown" || (hint !== "unsupported" && hint.reloadRequired);
}

/**
 * Whether a server-wide setting changed since the resident child launched.
 *
 * `adopt_load_intent_if_matched` refuses to reuse a running server when the Model Memory
 * policy no longer describes it, or when the VRAM budget moved. Neither is part of the load
 * intent, so identity and per-model config both still match and the resident shortcut would
 * skip a `/load` the backend was going to honour.
 *
 * An unknown answer declines. The two failures are not symmetric: one extra reload costs
 * the prompt this PR removes, while adopting on a read that failed after the user changed
 * a policy leaves the child on the old one with nothing on screen to say so. An absent
 * route is not an unknown answer, and keeps the shortcut working on a backend too old to
 * report either.
 */
export function serverWideReloadRequired(signals: {
  modelMemory: ReloadHint;
  vramBudget: ReloadHint;
}): boolean {
  return declines(signals.modelMemory) || declines(signals.vramBudget);
}
