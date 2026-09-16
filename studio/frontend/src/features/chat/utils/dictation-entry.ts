// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type DictationEntryMode = "live" | "recording-file";

function isLoopbackHostname(hostname: string): boolean {
  const normalized = hostname
    .trim()
    .toLowerCase()
    .replace(/^\[|\]$/g, "");
  return (
    normalized === "localhost" ||
    normalized.endsWith(".localhost") ||
    normalized === "127.0.0.1" ||
    normalized === "::1"
  );
}

/** Insecure HTTP on another device cannot open the page microphone. Everything
 * else stays on the existing live path so its capability and permission errors
 * remain visible instead of silently switching workflows. */
export function dictationEntryMode({
  isSecureContext,
  protocol,
  hostname,
}: {
  isSecureContext: boolean;
  protocol: string;
  hostname: string;
}): DictationEntryMode {
  if (isSecureContext || protocol !== "http:" || isLoopbackHostname(hostname)) {
    return "live";
  }
  return "recording-file";
}

export function currentDictationEntryMode(): DictationEntryMode {
  if (typeof window === "undefined") return "live";
  return dictationEntryMode({
    isSecureContext: window.isSecureContext,
    protocol: window.location.protocol,
    hostname: window.location.hostname,
  });
}

export type RecordingPickerPlatform = "android" | "ios" | "other";

/** Presentation only: native recorder availability still belongs to the
 * browser. iPadOS can identify as Macintosh, so touch capability is included. */
export function recordingPickerPlatform({
  userAgent,
  platform = "",
  maxTouchPoints = 0,
}: {
  userAgent: string;
  platform?: string;
  maxTouchPoints?: number;
}): RecordingPickerPlatform {
  if (/android/i.test(userAgent)) return "android";
  if (
    /iphone|ipad|ipod/i.test(userAgent) ||
    (/^mac/i.test(platform) && maxTouchPoints > 1)
  ) {
    return "ios";
  }
  return "other";
}

export function currentRecordingPickerPlatform(): RecordingPickerPlatform {
  if (typeof navigator === "undefined") return "other";
  return recordingPickerPlatform({
    userAgent: navigator.userAgent,
    platform: navigator.platform,
    maxTouchPoints: navigator.maxTouchPoints,
  });
}
