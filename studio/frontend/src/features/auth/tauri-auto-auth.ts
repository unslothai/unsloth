// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isTauri, apiUrl } from "@/lib/api-base";
import {
  hasAuthToken,
  hasRefreshToken,
  setMustChangePassword,
  storeAuthTokens,
} from "./session";
import { refreshSession } from "./api";

type TokenResponse = {
  access_token: string;
  refresh_token: string;
  must_change_password: boolean;
};

/** Ask Rust to generate and persist a random desktop password. */
async function generateAndStorePassword(): Promise<string> {
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<string>("set_desktop_password");
}

async function login(username: string, password: string): Promise<TokenResponse> {
  const res = await fetch(apiUrl("/api/auth/login"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  if (!res.ok) throw new Error("Login failed");
  return (await res.json()) as TokenResponse;
}

async function changePassword(
  accessToken: string,
  currentPassword: string,
  newPassword: string,
): Promise<TokenResponse> {
  const res = await fetch(apiUrl("/api/auth/change-password"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${accessToken}`,
    },
    body: JSON.stringify({ current_password: currentPassword, new_password: newPassword }),
  });
  if (!res.ok) throw new Error("Password change failed");
  return (await res.json()) as TokenResponse;
}

// Concurrency guard: multiple route guards can call tauriAutoAuth simultaneously.
// Without this, the first-launch password-change could race with itself.
let pending: Promise<boolean> | null = null;

async function doTauriAutoAuth(): Promise<boolean> {
  // Already have a valid token
  if (hasAuthToken()) return true;

  // Try refreshing existing session
  if (hasRefreshToken()) {
    const refreshed = await refreshSession();
    if (refreshed && hasAuthToken()) return true;
  }

  const { invoke } = await import("@tauri-apps/api/core");

  // Subsequent launches: try the persisted desktop password
  try {
    const desktopPwd = await invoke<string>("get_desktop_password");
    if (desktopPwd) {
      const tokens = await login("unsloth", desktopPwd);
      storeAuthTokens(tokens.access_token, tokens.refresh_token, false);
      setMustChangePassword(false);
      return true;
    }
  } catch {
    // File doesn't exist yet, fall through to bootstrap
  }

  // First launch: use bootstrap password
  try {
    const bootstrapPwd = await invoke<string>("get_bootstrap_password");
    if (!bootstrapPwd) return false;

    const tokens = await login("unsloth", bootstrapPwd);

    if (tokens.must_change_password) {
      const newPwd = await generateAndStorePassword();
      const freshTokens = await changePassword(tokens.access_token, bootstrapPwd, newPwd);
      storeAuthTokens(freshTokens.access_token, freshTokens.refresh_token, false);
      setMustChangePassword(false);
    } else {
      // Bootstrap password already accepted (e.g. changed via CLI but no desktop_password yet)
      storeAuthTokens(tokens.access_token, tokens.refresh_token, false);
      setMustChangePassword(false);
      await generateAndStorePassword();
    }

    return true;
  } catch {
    return false;
  }
}

/**
 * Silently authenticate in Tauri desktop mode.
 *
 * First launch: reads bootstrap password -> logs in -> auto-changes password
 *   to a random string -> persists it to ~/.unsloth/studio/auth/.desktop_password
 *
 * Subsequent launches: reads desktop password -> logs in -> stores tokens
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
