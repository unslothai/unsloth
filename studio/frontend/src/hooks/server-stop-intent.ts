// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The "the user stopped the server on purpose" marker.
//
// sessionStorage outlives webview reloads but not the app process, so an explicit stop
// holds across Reload while a fresh launch still auto-starts.
export const USER_STOPPED_KEY = "unsloth_server_user_stopped";

// Every access is wrapped: private browsing, blocked cookies and opaque webview origins
// all throw on access, and the read below runs before the startup screen has any state to
// fall back on. An escaping throw would reject the mount effect's floating promise and
// leave the screen on "checking" with no way out.

/** True when this webview session carries a stop the user asked for. */
export function hasServerStopIntent(): boolean {
  try {
    return sessionStorage.getItem(USER_STOPPED_KEY) !== null;
  } catch {
    // Unreadable storage cannot be holding an intent, so treat it as a fresh session
    // and auto-start, which is what a build before this marker existed did.
    return false;
  }
}

/** Record a stop so it survives a reload of this webview. */
export function markServerStopIntent(): void {
  try {
    sessionStorage.setItem(USER_STOPPED_KEY, "1");
  } catch {
    // The stop itself still happens; only its survival across a reload is lost.
  }
}

/** Drop the marker, so the next mount is free to start the server again. */
export function clearServerStopIntent(): void {
  try {
    sessionStorage.removeItem(USER_STOPPED_KEY);
  } catch {
    // Same.
  }
}
