// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { CodexDesktopStatus } from "../api/codex-desktop";

/** The models a Codex status lists: connected, the ones in Codex's catalog; otherwise the saved
 *  list. Empty follows whatever Studio has loaded. */
export function codexPicks(status: CodexDesktopStatus): string[] {
  if (status.connected && status.models.length > 0) {
    return status.selected_models.length > 0 ? status.models : [];
  }
  return status.selected_models;
}

/** The model Connect will make Codex's default (null: nothing to use yet). */
export function codexTarget(status: CodexDesktopStatus): string | null {
  return status.selected_models[0] ?? status.loaded_model;
}

export function sameModels(left: string[], right: string[]): boolean {
  return (
    left.length === right.length &&
    left.every((model, index) => model === right[index])
  );
}

/** Add or remove ``model``; adding past ``max`` changes nothing. */
export function toggleModel(
  models: string[],
  model: string,
  max: number,
): string[] {
  if (models.includes(model)) {
    return models.filter((m) => m !== model);
  }
  return models.length >= max ? models : [...models, model];
}
