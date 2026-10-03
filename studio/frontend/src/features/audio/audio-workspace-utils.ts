// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  AUDIO_CPP_MUSIC_AUDIO_TYPE,
  AUDIO_CPP_TTS_AUDIO_TYPE,
  audioCppDisplayName,
  isAudioCppFolderId,
} from "./audio-cpp-catalog";

/** What to call a model on screen. A Hub repo is its id; a checkpoint trained here is an output
 *  directory, and the full path in a toast reads as a bug. */
export function audioModelLabel(id: string): string {
  // A package folder of the shared GGUF repo is known by its folder name, as the Hub shows it.
  if (isAudioCppFolderId(id)) return audioCppDisplayName(id);
  if (!/^(?:[a-zA-Z]:[\\/]|[\\/]|~)/.test(id)) return id;
  const leaf = id.split(/[\\/]/).filter(Boolean).pop() ?? id;
  // Training stamps the output directory with an epoch; it means nothing to a reader.
  return leaf.replace(/_\d{10,}$/, "");
}

/** The load toast's kind: the codec or runtime name, except for the GGUF runtime's internal ones. */
export function loadedAudioKind(audioType: string | null | undefined): string {
  if (audioType === AUDIO_CPP_TTS_AUDIO_TYPE) return "speech";
  if (audioType === AUDIO_CPP_MUSIC_AUDIO_TYPE) return "music";
  return audioType ?? "audio";
}

export function deviceSizeBytes(label: string): number {
  const match = label.trim().match(/^(\d+(?:\.\d+)?)\s*(MB|GB)$/i);
  if (!match) return 0;
  const value = Number(match[1]);
  return value * (match[2].toUpperCase() === "GB" ? 1024 ** 3 : 1024 ** 2);
}

export type CreateMode = "speak" | "transcribe";
export type RemoteCodeApproval = {
  trustRemoteCode: true;
  approvedRemoteCodeFingerprint: string | null;
};

export function formatClipDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds <= 0) return "0:00";
  const whole = Math.round(seconds);
  const minutes = Math.floor(whole / 60);
  const rest = whole % 60;
  return `${minutes}:${String(rest).padStart(2, "0")}`;
}
