// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isTauri } from "@/lib/api-base";
import {
  hasAuthToken,
  hasRefreshToken,
  mustChangePassword,
  storeAuthTokens,
} from "./session";
import { refreshSession } from "./api";

type DesktopAuthResponse = {
  access_token: string;
  refresh_token: string;
};

// Concurrency guard: multiple route guards can call tauriAutoAuth simultaneously.
// Without this, the first-launch password-change could race with itself.
let pending: Promise<boolean> | null = null;
let lastTauriAuthFailure: string | null = null;

const TAURI_AUTH_FAILURE_FALLBACK =
  "Desktop authentication failed. Update or repair the managed Studio install, then restart Studio.";
const BACKEND_NOT_READY_MESSAGE = "Backend is not ready";

function authFailureMessage(error: unknown): string {
  if (typeof error === "string" && error) return error;
  if (error instanceof Error && error.message) return error.message;
  return TAURI_AUTH_FAILURE_FALLBACK;
}

export function getTauriAuthFailure(): string | null {
  return lastTauriAuthFailure;
}

export function clearTauriAuthFailure(): void {
  lastTauriAuthFailure = null;
}

function setTauriAuthFailure(error: unknown): void {
  lastTauriAuthFailure = authFailureMessage(error);
  window.dispatchEvent(
    new CustomEvent("tauri-auth-failed", { detail: lastTauriAuthFailure }),
  );
}

function isBackendNotReady(error: unknown): boolean {
  return authFailureMessage(error).includes(BACKEND_NOT_READY_MESSAGE);
}

async function doTauriAutoAuth(): Promise<boolean> {
  // Desktop must handle password-change state internally in Rust.
  if (hasAuthToken() && !mustChangePassword()) {
    clearTauriAuthFailure();
    return true;
  }

  // Try refreshing existing session
  if (hasRefreshToken()) {
    const refreshed = await refreshSession();
    if (refreshed && hasAuthToken() && !mustChangePassword()) {
      clearTauriAuthFailure();
      return true;
    }
  }

  try {
    const { invoke } = await import("@tauri-apps/api/core");
    const tokens = await invoke<DesktopAuthResponse>("desktop_auth");
    storeAuthTokens(tokens.access_token, tokens.refresh_token, false);
    clearTauriAuthFailure();
    return true;
  } catch (error) {
    if (isBackendNotReady(error)) return false;
    setTauriAuthFailure(error);
    return false;
  }
}

/**
 * Silently authenticate in Tauri desktop mode.
 *
 * Delegates bootstrap/password handling to Rust and only stores returned tokens.
 *
 * Returns true if authentication succeeded.
 * Concurrent calls are coalesced into a single in-flight attempt.
 */
export function tauriAutoAuth(): Promise<boolean> {
  if (!isTauri) return Promise.resolve(false);
  if (!pending) {
    pending = doTauriAutoAuth().finally(() => { pending = null; });
  }
  return pending;
}
