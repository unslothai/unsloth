// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { apiUrl, isTauri } from "@/lib/api-base";
import {
  clearAuthTokens,
  getAuthSubjectKey,
  getAuthToken,
  getRefreshToken,
  mustChangePassword,
  setMustChangePassword,
  storeAuthTokens,
} from "./session";

type RefreshResponse = {
  access_token: string;
  refresh_token: string;
  must_change_password: boolean;
};

let isRedirecting = false;
let refreshInflight: Promise<boolean> | null = null;
let refreshInflightToken: string | null = null;
let logoutGeneration = 0;

export type AuthFetchGuard = {
  /** Reject sends or retries under another auth subject; allow token rotation. */
  expectedSubjectKey: string;
};

export class AuthSubjectChangedError extends Error {
  constructor() {
    super("The signed-in account changed before the request completed.");
    this.name = "AuthSubjectChangedError";
  }
}

const TAURI_FETCH_RETRY_DELAYS_MS = [250, 750, 1500] as const;
const RETRYABLE_NETWORK_METHODS = new Set(["GET", "HEAD", "OPTIONS"]);

function wait(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function clearAuthTokensIfCurrent(refreshToken: string | null): void {
  if (!refreshToken || getRefreshToken() === refreshToken) clearAuthTokens();
}

async function fetchWithTauriNetworkRetry(
  input: RequestInfo | URL,
  init?: RequestInit,
): Promise<Response> {
  const method = (init?.method ?? "GET").toUpperCase();
  const retryable = RETRYABLE_NETWORK_METHODS.has(method);
  for (let attempt = 0; ; attempt++) {
    try {
      return await fetch(input, init);
    } catch (error) {
      if (
        !isTauri ||
        !retryable ||
        !(error instanceof TypeError) ||
        attempt >= TAURI_FETCH_RETRY_DELAYS_MS.length
      ) {
        throw error;
      }
      await wait(TAURI_FETCH_RETRY_DELAYS_MS[attempt]);
    }
  }
}

async function isPasswordChangeRequiredResponse(
  response: Response,
): Promise<boolean> {
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
      // Server truth wins; keep localStorage in sync both ways.
      if (data.requires_password_change !== mustChangePassword()) {
        setMustChangePassword(data.requires_password_change);
      }
      if (data.requires_password_change) target = "/change-password";
    }
  } catch {
    // Fall through to /login on error
  }

  if (window.location.pathname === target) {
    isRedirecting = false;
    return;
  }
  window.location.href = target;
}

async function retryWithCurrentToken(
  input: RequestInfo | URL,
  init?: RequestInit,
  guard?: AuthFetchGuard,
): Promise<Response> {
  assertExpectedSubject(guard);
  const retryHeaders = new Headers(init?.headers);
  const token = getAuthToken();
  if (token) retryHeaders.set("Authorization", `Bearer ${token}`);
  const response = await fetchWithTauriNetworkRetry(input, {
    ...init,
    headers: retryHeaders,
  });
  assertExpectedSubject(guard);
  return response;
}

function assertExpectedSubject(guard?: AuthFetchGuard): void {
  if (guard && getAuthSubjectKey() !== guard.expectedSubjectKey) {
    throw new AuthSubjectChangedError();
  }
}

async function retryWithTauriAutoAuth(
  input: RequestInfo | URL,
  init?: RequestInit,
  guard?: AuthFetchGuard,
): Promise<Response | null> {
  const subjectBeforeAutoAuth = getAuthSubjectKey();
  if (subjectBeforeAutoAuth !== "anonymous") {
    assertExpectedSubject(guard);
  }
  const allowAnonymousBootstrap = guard?.expectedSubjectKey === "anonymous";
  clearAuthTokens();
  const { tauriAutoAuth } = await import("./tauri-auto-auth");
  if (await tauriAutoAuth()) {
    // Bind tokenless Tauri replay to its new account; authenticated requests cannot switch.
    const retryGuard = allowAnonymousBootstrap
      ? { expectedSubjectKey: getAuthSubjectKey() }
      : guard;
    return retryWithCurrentToken(input, init, retryGuard);
  }
  return null;
}

export async function refreshSession(): Promise<boolean> {
  const refreshToken = getRefreshToken();
  if (!refreshToken) return false;
  if (refreshInflight && refreshInflightToken === refreshToken) {
    return refreshInflight;
  }

  const startGeneration = logoutGeneration;
  const promise = (async () => {
    try {
      const response = await fetchWithTauriNetworkRetry(
        apiUrl("/api/auth/refresh"),
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ refresh_token: refreshToken }),
        },
      );
      if (!response.ok) {
        clearAuthTokensIfCurrent(refreshToken);
        return false;
      }
      const payload = (await response.json()) as RefreshResponse;
      if (startGeneration !== logoutGeneration) return false;
      if (getRefreshToken() !== refreshToken) return false;
      storeAuthTokens(payload.access_token, payload.refresh_token);
      setMustChangePassword(payload.must_change_password ?? false);
      return true;
    } catch {
      return false;
    }
  })();
  refreshInflight = promise;
  refreshInflightToken = refreshToken;
  try {
    return await promise;
  } finally {
    if (refreshInflight === promise) {
      refreshInflight = null;
      refreshInflightToken = null;
    }
  }
}

export async function authFetch(
  input: RequestInfo | URL,
  init?: RequestInit,
  guard?: AuthFetchGuard,
): Promise<Response> {
  assertExpectedSubject(guard);
  const resolvedInput = typeof input === "string" ? apiUrl(input) : input;
  const headers = new Headers(init?.headers);
  const accessToken = getAuthToken();
  if (accessToken) {
    headers.set("Authorization", `Bearer ${accessToken}`);
  }

  let response: Response;
  try {
    response = await fetchWithTauriNetworkRetry(resolvedInput, {
      ...init,
      headers,
    });
  } catch (err) {
    if (err instanceof TypeError) {
      // fetch TypeError = offline | backend down | CORS/DNS. Tauri is always
      // backend-down; the web build distinguishes offline for the right message.
      if (
        !isTauri &&
        typeof navigator !== "undefined" &&
        navigator.onLine === false
      ) {
        throw new Error(
          "You appear to be offline. Check your network connection and try again.",
        );
      }
      throw new Error("Unsloth isn't running -- please relaunch it.");
    }
    throw err;
  }

  // Ignore responses from a captured account
  // after the active account changes.
  assertExpectedSubject(guard);

  if (await isPasswordChangeRequiredResponse(response)) {
    assertExpectedSubject(guard);
    if (isTauri) {
      return (
        (await retryWithTauriAutoAuth(resolvedInput, init, guard)) ?? response
      );
    }
    void redirectToAuth();
    return response;
  }
  if (response.status !== 401) return response;

  const refreshToken = getRefreshToken();
  assertExpectedSubject(guard);
  const refreshed = await refreshSession();
  if (!refreshed) {
    // An expired session may clear; never clear or redirect a newer account.
    if (
      guard &&
      getAuthSubjectKey() !== guard.expectedSubjectKey &&
      getAuthSubjectKey() !== "anonymous"
    ) {
      throw new AuthSubjectChangedError();
    }
    if (isTauri) {
      return (
        (await retryWithTauriAutoAuth(resolvedInput, init, guard)) ?? response
      );
    }
    clearAuthTokensIfCurrent(refreshToken);
    void redirectToAuth();
    return response;
  }

  const retryGuard =
    isTauri && guard?.expectedSubjectKey === "anonymous"
      ? { expectedSubjectKey: getAuthSubjectKey() }
      : guard;
  assertExpectedSubject(retryGuard);

  if (mustChangePassword()) {
    if (isTauri) {
      return (
        (await retryWithTauriAutoAuth(resolvedInput, init, retryGuard)) ??
        response
      );
    }
    void redirectToAuth();
    return response;
  }

  if (!getAuthToken()) clearAuthTokens();
  return retryWithCurrentToken(resolvedInput, init, retryGuard);
}

async function postLogout(
  accessToken: string | null,
): Promise<Response | null> {
  try {
    return await fetchWithTauriNetworkRetry(apiUrl("/api/auth/logout"), {
      method: "POST",
      headers: accessToken
        ? { Authorization: `Bearer ${accessToken}` }
        : undefined,
    });
  } catch {
    return null;
  }
}

export async function logout(): Promise<void> {
  // Server-side revoke. If the access token is expired, the 401 fires before
  // revoke runs; rotate via the refresh token and retry so the refresh family
  // is revoked. The finally generation bump invalidates in-flight refreshes.
  try {
    let response = await postLogout(getAuthToken());
    if (response && response.status === 401 && getRefreshToken()) {
      const refreshed = await refreshSession();
      if (refreshed) response = await postLogout(getAuthToken());
    }
  } finally {
    logoutGeneration += 1;
    clearAuthTokens();
  }
}
