// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { OpenAIAutoSwitchSettings } from "../api/openai-auto-switch";

/** The idle-seconds draft after a write from elsewhere: it follows the new saved value unless
 *  it holds an edit of its own. */
export function followSavedIdleDraft(
  draft: string,
  previous: Pick<OpenAIAutoSwitchSettings, "autoUnloadIdleSeconds"> | null,
  next: Pick<OpenAIAutoSwitchSettings, "autoUnloadIdleSeconds">,
): string {
  const untouched =
    previous == null || draft === String(previous.autoUnloadIdleSeconds);
  return untouched ? String(next.autoUnloadIdleSeconds) : draft;
}
