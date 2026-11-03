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

// A disconnect from a server this app did not start keeps its port for the stopped screen after a reload.
const DISCONNECTED_PREFIX = "disconnected:";

export function markServerStopIntent(disconnectedPort?: number): void {
  try {
    sessionStorage.setItem(
      USER_STOPPED_KEY,
      disconnectedPort === undefined ? "1" : `${DISCONNECTED_PREFIX}${disconnectedPort}`,
    );
  } catch {
    // The stop itself still happens; only its survival across a reload is lost.
  }
}

export function stopIntentDisconnectedPort(): number | null {
  try {
    const value = sessionStorage.getItem(USER_STOPPED_KEY);
    if (!value?.startsWith(DISCONNECTED_PREFIX)) return null;
    const port = Number(value.slice(DISCONNECTED_PREFIX.length));
    return Number.isInteger(port) && port > 0 && port <= 65535 ? port : null;
  } catch {
    return null;
  }
}

export function clearServerStopIntent(): void {
  try {
    sessionStorage.removeItem(USER_STOPPED_KEY);
  } catch {
    // Same.
  }
}
