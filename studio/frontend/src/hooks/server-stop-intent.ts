// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// sessionStorage survives webview reloads but not the process, so a fresh launch still auto-starts.
export const USER_STOPPED_KEY = "unsloth_server_user_stopped";

// Every access is wrapped: storage can throw, and an escaping throw would strand the startup screen.

export function hasServerStopIntent(): boolean {
  try {
    return sessionStorage.getItem(USER_STOPPED_KEY) !== null;
  } catch {
    // Unreadable storage cannot hold an intent, so auto-start.
    return false;
  }
}

export function markServerStopIntent(): void {
  try {
    sessionStorage.setItem(USER_STOPPED_KEY, "1");
  } catch {
    // The stop itself still happens; only its survival across a reload is lost.
  }
}

export function clearServerStopIntent(): void {
  try {
    sessionStorage.removeItem(USER_STOPPED_KEY);
  } catch {
    // Same.
  }
}
