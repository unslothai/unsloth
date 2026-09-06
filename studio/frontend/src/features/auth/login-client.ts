// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { apiUrl } from "@/lib/api-base";
import { normalizeAccountUsername, resetFullAccessForMultiUser } from "@/lib/account-transition";

export type LoginMode = "single" | "multi";
export type AuthStatusResponse = {
  initialized: boolean;
  requires_password_change: boolean;
  bootstrap_deadline_seconds?: number | null;
  login_mode?: LoginMode;
};
export type TokenResponse = {
  access_token: string;
  refresh_token: string;
  must_change_password: boolean;
};

let loginMode: LoginMode = "single";
let statusKnown = false;
let inflight: Promise<AuthStatusResponse> | null = null;
const listeners = new Set<() => void>();
export const getLoginMode = (): LoginMode => loginMode;
export const subscribeLoginMode = (listener: () => void): (() => void) => {
  listeners.add(listener);
  return () => { listeners.delete(listener); };
};
export function setLoginMode(mode: LoginMode): void {
  statusKnown = true;
  if (mode === "multi" && typeof window !== "undefined") {
    resetFullAccessForMultiUser(window.localStorage);
  }
  if (loginMode === mode) return;
  loginMode = mode;
  listeners.forEach((listener) => listener());
}
export async function fetchAuthStatus(): Promise<AuthStatusResponse> {
  if (inflight) return inflight;
  inflight = (async () => {
    const response = await fetch(apiUrl("/api/auth/status"));
    if (!response.ok) throw new Error("Failed to load auth status.");
    const result = await response.json() as AuthStatusResponse;
    setLoginMode(result.login_mode ?? "single");
    return result;
  })();
  try { return await inflight; } finally { inflight = null; }
}
export function ensureLoginMode(): void {
  if (!statusKnown) void fetchAuthStatus().catch(() => undefined);
}

export class LoginError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

export async function loginWithPassword(username: string, password: string): Promise<TokenResponse> {
  const response = await fetch(apiUrl("/api/auth/login"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username: normalizeAccountUsername(username), password }),
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => null) as { detail?: string } | null;
    throw new LoginError(payload?.detail ?? "Login failed.", response.status);
  }
  return response.json();
}

/** Only a rejected single-mode submit probes for an account created since page load. */
export async function loginFromForm(
  mode: LoginMode, username: string, password: string,
): Promise<TokenResponse | null> {
  try {
    return await loginWithPassword(mode === "single" ? "unsloth" : username, password);
  } catch (error) {
    if (mode === "single" && error instanceof LoginError && error.status === 401) {
      const status = await fetchAuthStatus().catch(() => null);
      if (status?.login_mode === "multi") return null;
    }
    throw error;
  }
}
