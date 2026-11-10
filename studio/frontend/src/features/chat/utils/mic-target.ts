// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { MicTarget } from "@/features/settings/stores/voice-settings-store";
import {
  AUDIO_ATTACHMENT_ACCEPT,
  MAX_AUDIO_SIZE,
  getAudioAddError,
  isAudioAttachmentFile,
} from "../../../lib/audio-utils.ts";

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

export const MIN_MIC_CLIP_BYTES = 44 + 16_000 * 2;

export type MicClipRoom = { maxBytes: number } | { reason: string };

export function micClipRoom(
  attached: readonly { name: string; type: string; size: number }[],
  maxFiles: number,
  stagedClip: boolean,
): MicClipRoom {
  if (stagedClip) {
    return {
      reason: "Send or remove the staged audio clip before attaching more audio.",
    };
  }
  const audio = attached.filter(isAudioAttachmentFile);
  const totalSize = audio.reduce((sum, file) => sum + file.size, 0);
  const reason = getAudioAddError(
    audio.length,
    totalSize,
    MIN_MIC_CLIP_BYTES,
    maxFiles,
  );
  return reason ? { reason } : { maxBytes: MAX_AUDIO_SIZE - totalSize };
}

export function pickAudioForModel(
  add: (file: File) => Promise<unknown>,
  doc: Document = document,
): void {
  const input = doc.createElement("input");
  input.type = "file";
  input.accept = AUDIO_ATTACHMENT_ACCEPT;
  input.multiple = true;
  input.hidden = true;
  const done = () => input.remove();
  input.addEventListener(
    "change",
    () => {
      // The attachment adapter toasts its own refusals.
      for (const file of Array.from(input.files ?? [])) void add(file).catch(() => {});
      done();
    },
    { once: true },
  );
  input.addEventListener("cancel", done, { once: true });
  doc.body.append(input);
  input.click();
}
