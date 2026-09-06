// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isTauri } from "@/lib/api-base";
import {
  clearAuthTokens,
  hasAuthToken,
  hasRefreshToken,
  mustChangePassword,
  setMustChangePassword,
  storeAuthTokens,
} from "./session";
import { refreshSession } from "./api";

type DesktopAuthResponse =
  | { access_token: string; refresh_token: string }
  | { login_required: true; login_mode: "multi" };

type TauriAutoAuthOptions = {
  force?: boolean;
};

// Concurrency guard: multiple route guards can call tauriAutoAuth at once;
// without this the first-launch password-change could race with itself.
let pending: { promise: Promise<boolean>; force: boolean } | null = null;
let lastTauriAuthFailure: string | null = null;
let tauriLoginRequired = false;

export function isTauriLoginRequired(): boolean {
  return tauriLoginRequired;
}

const TAURI_AUTH_FAILURE_FALLBACK =
  "Desktop authentication failed. Update or repair the managed Unsloth install, then restart Unsloth.";
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

async function doTauriAutoAuth(options: TauriAutoAuthOptions): Promise<boolean> {
  // Desktop must handle password-change state internally in Rust.
  if (!options.force && hasAuthToken() && !mustChangePassword()) {
    clearTauriAuthFailure();
    return true;
  }

  // Try refreshing an existing session.
  if (!options.force && hasRefreshToken()) {
    const refreshed = await refreshSession();
    if (refreshed && hasAuthToken() && !mustChangePassword()) {
      clearTauriAuthFailure();
      return true;
    }
  }

  try {
    const { invoke } = await import("@tauri-apps/api/core");
    const tokens = await invoke<DesktopAuthResponse>("desktop_auth");
    if ("login_required" in tokens) {
      tauriLoginRequired = true;
      clearAuthTokens();
      clearTauriAuthFailure();
      // AppProvider's forced startup probe must release the startup screen so
      // the login form can mount. Ordinary API recovery still returns false.
      const { router } = await import("@/app/router");
      await router.navigate({ to: "/login", replace: true });
      return options.force === true;
    }
    tauriLoginRequired = false;
    storeAuthTokens(tokens.access_token, tokens.refresh_token);
    setMustChangePassword(false);
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
 * Returns true if authentication succeeded, or a forced startup probe verified
 * the shell and opened the required login form. The latter stores no session.
 * Concurrent calls are coalesced into a single in-flight attempt.
 */
export function tauriAutoAuth(
  options: TauriAutoAuthOptions = {},
): Promise<boolean> {
  if (!isTauri) return Promise.resolve(false);
  const force = options.force === true;
  if (!pending || (force && !pending.force)) {
    let promise: Promise<boolean>;
    promise = doTauriAutoAuth({ force }).finally(() => {
      if (pending?.promise === promise) pending = null;
    });
    pending = { promise, force };
  }
  if (!force && pending.force) {
    // An API retry may share AppProvider's forced probe, but shell readiness
    // alone must not authorize that retry.
    return pending.promise.then(
      (ready) => ready && (!tauriLoginRequired || hasAuthToken()),
    );
  }
  return pending.promise;
}
