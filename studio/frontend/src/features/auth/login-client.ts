// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { apiUrl } from "@/lib/api-base";
import {
  normalizeAccountUsername,
  resetFullAccessForMultiUser,
} from "@/lib/account-transition";

export type LoginMode = "single" | "multi";
export type AuthStatusResponse = {
  initialized: boolean;
  requires_password_change: boolean;
  bootstrap_deadline_seconds?: number | null;
  login_mode?: LoginMode;
  full_access?: boolean;
};
export type TokenResponse = {
  access_token: string;
  refresh_token: string;
  must_change_password: boolean;
};

// A hint lets multi-account documents hide Full access before React mounts. Legacy
// owner-only browsers keep their existing startup request count (no status probe here).
export const LOGIN_MODE_HINT_KEY = "unsloth.auth-login-mode.v1";
function initialLoginMode(): LoginMode {
  try {
    return typeof window !== "undefined" &&
      window.localStorage.getItem(LOGIN_MODE_HINT_KEY) === "multi"
      ? "multi"
      : "single";
  } catch {
    return "single";
  }
}
let loginMode: LoginMode = initialLoginMode();
let statusKnown = false;
let inflight: Promise<AuthStatusResponse> | null = null;
const listeners = new Set<() => void>();
export const getLoginMode = (): LoginMode => loginMode;
export const subscribeLoginMode = (listener: () => void): (() => void) => {
  if (listeners.size === 0 && typeof window !== "undefined") {
    window.addEventListener("storage", onLoginModeStorage);
  }
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
    if (listeners.size === 0 && typeof window !== "undefined") {
      window.removeEventListener("storage", onLoginModeStorage);
    }
  };
};
function onLoginModeStorage(event: StorageEvent): void {
  // Tighten policy in peer tabs immediately. A removed hint can also be account
  // cleanup, so relaxing policy always waits for a fresh server status instead.
  if (
    event.key === LOGIN_MODE_HINT_KEY &&
    event.newValue === "multi" &&
    (!event.storageArea || event.storageArea === window.localStorage)
  )
    setLoginMode("multi");
}
// Whether Full access may be offered. The server refuses it whenever another
// account exists, active or not, so a deactivated account keeps it hidden while
// the login form is already back in single mode. Until a status arrives the
// login-mode hint is the only knowledge: multi means no.
let fullAccessAllowed: boolean = initialLoginMode() !== "multi";
export const getFullAccessAllowed = (): boolean => fullAccessAllowed;
export function setLoginMode(mode: LoginMode, fullAccess?: boolean): void {
  statusKnown = true;
  // Without an explicit answer, multi is a no and single keeps what is known.
  const allowed = fullAccess ?? (mode === "multi" ? false : fullAccessAllowed);
  if (typeof window !== "undefined") {
    if (mode === "multi" || !allowed) {
      resetFullAccessForMultiUser(window.localStorage);
    }
    if (mode === "multi") {
      window.localStorage.setItem(LOGIN_MODE_HINT_KEY, "multi");
    } else if (window.localStorage.getItem(LOGIN_MODE_HINT_KEY) !== null) {
      window.localStorage.removeItem(LOGIN_MODE_HINT_KEY);
    }
  }
  const changed = loginMode !== mode || fullAccessAllowed !== allowed;
  loginMode = mode;
  fullAccessAllowed = allowed;
  if (changed) listeners.forEach((listener) => listener());
}
export async function fetchAuthStatus(): Promise<AuthStatusResponse> {
  if (inflight) return inflight;
  inflight = (async () => {
    const response = await fetch(apiUrl("/api/auth/status"));
    if (!response.ok) throw new Error("Failed to load auth status.");
    const result = (await response.json()) as AuthStatusResponse;
    setLoginMode(result.login_mode ?? "single", result.full_access);
    return result;
  })();
  try {
    return await inflight;
  } finally {
    inflight = null;
  }
}
export function ensureLoginMode(): void {
  if (!statusKnown && loginMode === "multi")
    void fetchAuthStatus().catch(() => undefined);
}

export class LoginError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

export async function loginWithPassword(
  username: string,
  password: string,
): Promise<TokenResponse> {
  const response = await fetch(apiUrl("/api/auth/login"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      username: normalizeAccountUsername(username),
      password,
    }),
  });
  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as {
      detail?: string;
    } | null;
    throw new LoginError(payload?.detail ?? "Login failed.", response.status);
  }
  return response.json();
}

/** Only a rejected single-mode submit probes for an account created since page load. */
export async function loginFromForm(
  mode: LoginMode,
  username: string,
  password: string,
): Promise<TokenResponse | null> {
  try {
    return await loginWithPassword(
      mode === "single" ? "unsloth" : username,
      password,
    );
  } catch (error) {
    if (
      mode === "single" &&
      error instanceof LoginError &&
      error.status === 401
    ) {
      const status = await fetchAuthStatus().catch(() => null);
      if (status?.login_mode === "multi") return null;
    }
    throw error;
  }
}
