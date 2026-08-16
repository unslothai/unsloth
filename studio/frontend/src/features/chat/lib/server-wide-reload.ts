// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** A settings read that describes the running child, or null when it could not be had. */
type ReloadHint = { reloadRequired: boolean } | null;

/**
 * Whether a server-wide setting changed since the resident child launched.
 *
 * `adopt_load_intent_if_matched` refuses to reuse a running server when the Model Memory
 * policy no longer describes it, or when the VRAM budget moved. Neither is part of the load
 * intent, so identity and per-model config both still match and the resident shortcut would
 * skip a `/load` the backend was going to honour.
 *
 * Only an explicit `true` declines. An answer that could not be fetched leaves the shortcut
 * as it was, which is what keeps a backend too old to serve these endpoints working.
 */
export function serverWideReloadRequired(signals: {
  modelMemory: ReloadHint;
  vramBudget: ReloadHint;
}): boolean {
  return (
    signals.modelMemory?.reloadRequired === true ||
    signals.vramBudget?.reloadRequired === true
  );
}
