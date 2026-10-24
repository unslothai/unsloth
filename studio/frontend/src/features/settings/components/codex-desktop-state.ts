// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { CodexDesktopStatus } from "../api/codex-desktop";
import { FOLLOW_LOADED } from "./desktop-app-models.ts";

/** The model a Codex status points at: connected, the one Codex is on; otherwise the pick. */
export function codexPick(status: CodexDesktopStatus): string {
  if (status.connected && status.model) {
    return status.selected_model ? status.model : FOLLOW_LOADED;
  }
  return status.selected_model ?? FOLLOW_LOADED;
}

/** The model Connect will put Codex on (null: nothing to use yet). */
export function codexTarget(status: CodexDesktopStatus): string | null {
  return status.selected_model ?? status.loaded_model;
}
