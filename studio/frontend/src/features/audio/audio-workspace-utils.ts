// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  AUDIO_CPP_MUSIC_AUDIO_TYPE,
  AUDIO_CPP_SEP_AUDIO_TYPE,
  AUDIO_CPP_TTS_AUDIO_TYPE,
  audioCppDisplayName,
  isAudioCppFolderId,
} from "./audio-cpp-catalog";

export function audioModelLabel(id: string): string {
  if (isAudioCppFolderId(id)) return audioCppDisplayName(id);
  if (!/^(?:[a-zA-Z]:[\\/]|[\\/]|~)/.test(id)) return id;
  const leaf = id.split(/[\\/]/).filter(Boolean).pop() ?? id;
  return leaf.replace(/_\d{10,}$/, "");
}

export function loadedAudioKind(audioType: string | null | undefined): string {
  if (audioType === AUDIO_CPP_TTS_AUDIO_TYPE) return "speech";
  if (audioType === AUDIO_CPP_MUSIC_AUDIO_TYPE) return "music";
  if (audioType === AUDIO_CPP_SEP_AUDIO_TYPE) return "separation";
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
