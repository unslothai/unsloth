// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { MicTarget } from "@/features/settings/stores/voice-settings-store";

/** The mic reaches the model only while the loaded one takes audio; otherwise it transcribes. */
export function micSendsAudioToModel(
  micTarget: MicTarget,
  modelHearsAudio: boolean,
): boolean {
  return micTarget === "model" && modelHearsAudio;
}

export function micClipFilename(now: Date = new Date()): string {
  const stamp = now.toISOString().slice(0, 19).replaceAll(":", "-");
  return `recording-${stamp}.wav`;
}
