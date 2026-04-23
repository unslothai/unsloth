// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { apiUrl, isTauri } from "@/lib/api-base";
import {
  clearAuthTokens,
  getAuthToken,
  getRefreshToken,
  mustChangePassword,
  storeAuthTokens,
} from "./session";

type RefreshResponse = {
  access_token: string;
  refresh_token: string;
  must_change_password: boolean;
};

let isRedirecting = false;

async function isPasswordChangeRequiredResponse(response: Response): Promise<boolean> {
  if (response.status !== 403) return false;

  try {
    const payload = (await response.clone().json()) as { detail?: string };
    return payload.detail === "Password change required";
  } catch {
    return false;
  }
}

async function redirectToAuth(): Promise<void> {
  if (isRedirecting) return;
  isRedirecting = true;

  let target = "/login";
  try {
    const res = await fetch(apiUrl("/api/auth/status"));
    if (res.ok) {
      const data = (await res.json()) as { requires_password_change: boolean };
      if (data.requires_password_change || mustChangePassword()) target = "/change-password";
    }
  } catch {
    // Fall through to /login on error
  }

  window.location.href = target;
}

async function retryWithCurrentToken(
  input: RequestInfo | URL,
  init?: RequestInit,
): Promise<Response> {
  const retryHeaders = new Headers(init?.headers);
  const token = getAuthToken();
  if (token) retryHeaders.set("Authorization", `Bearer ${token}`);
  return fetch(input, { ...init, headers: retryHeaders });
}

async function retryWithTauriAutoAuth(
  input: RequestInfo | URL,
  init?: RequestInit,
): Promise<Response | null> {
  clearAuthTokens();
  const { tauriAutoAuth } = await import("./tauri-auto-auth");
  if (await tauriAutoAuth()) return retryWithCurrentToken(input, init);
  return null;
}

export async function refreshSession(): Promise<boolean> {
  const refreshToken = getRefreshToken();
  if (!refreshToken) return false;

  try {
    const response = await fetch(apiUrl("/api/auth/refresh"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ refresh_token: refreshToken }),
    });

    if (!response.ok) {
      clearAuthTokens();
      return false;
    }

    const payload = (await response.json()) as RefreshResponse;
    storeAuthTokens(
      payload.access_token,
      payload.refresh_token,
      payload.must_change_password,
    );
    return true;
  } catch {
    return false;
  }
}

export async function authFetch(
  input: RequestInfo | URL,
  init?: RequestInit,
): Promise<Response> {
  const resolvedInput = typeof input === 'string' ? apiUrl(input) : input;
  const headers = new Headers(init?.headers);
  const accessToken = getAuthToken();
  if (accessToken) {
    headers.set("Authorization", `Bearer ${accessToken}`);
  }

  let response: Response;
  try {
    response = await fetch(resolvedInput, { ...init, headers });
  } catch (err) {
    if (err instanceof TypeError) {
      throw new Error("Studio isn't running -- please relaunch it.");
    }
    throw err;
  }

  if (await isPasswordChangeRequiredResponse(response)) {
    if (isTauri) {
      return (await retryWithTauriAutoAuth(resolvedInput, init)) ?? response;
    }
    void redirectToAuth();
    return response;
  }
  if (response.status !== 401) return response;

  const refreshed = await refreshSession();
  if (!refreshed) {
    if (isTauri) {
      return (await retryWithTauriAutoAuth(resolvedInput, init)) ?? response;
    }
    clearAuthTokens();
    void redirectToAuth();
    return response;
  }

  if (mustChangePassword()) {
    if (isTauri) {
      return (await retryWithTauriAutoAuth(resolvedInput, init)) ?? response;
    }
    void redirectToAuth();
    return response;
  }

  if (!getAuthToken()) clearAuthTokens();
  return retryWithCurrentToken(resolvedInput, init);
}

export function logout(): void {
  clearAuthTokens();
}
