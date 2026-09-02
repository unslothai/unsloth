// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { apiUrl, isTauri } from "@/lib/api-base";
import {
  clearAuthTokens,
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

type AuthFetchOptions = {
  retryNetworkErrors?: boolean;
  /** Synchronous policy check run immediately before any retry sends bytes. */
  beforeRetry?: () => void;
};

let isRedirecting = false;
let refreshInflight: Promise<boolean> | null = null;
let refreshInflightToken: string | null = null;
let logoutGeneration = 0;

const TAURI_FETCH_RETRY_DELAYS_MS = [250, 750, 1500] as const;
const BROWSER_TIMEZONE_HEADER = "X-Unsloth-Timezone";
const BROWSER_TIMEZONE_OFFSET_HEADER =
  "X-Unsloth-Timezone-Offset-Minutes";

function addBrowserTimezoneHeaders(headers: Headers): void {
  try {
    const timezone = Intl.DateTimeFormat().resolvedOptions().timeZone;
    if (timezone) headers.set(BROWSER_TIMEZONE_HEADER, timezone);
    headers.set(
      BROWSER_TIMEZONE_OFFSET_HEADER,
      String(new Date().getTimezoneOffset()),
    );
  } catch {
    // runtimes without Intl keep the backend-local fallback.
  }
}

function wait(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function clearAuthTokensIfCurrent(refreshToken: string | null): void {
  if (!refreshToken || getRefreshToken() === refreshToken) clearAuthTokens();
}

async function fetchWithTauriNetworkRetry(
  input: RequestInfo | URL,
  init?: RequestInit,
  retryNetworkErrors = true,
  beforeRetry?: () => void,
): Promise<Response> {
  for (let attempt = 0; ; attempt++) {
    try {
      return await fetch(input, init);
    } catch (error) {
      if (
        !isTauri ||
        !retryNetworkErrors ||
        !(error instanceof TypeError) ||
        attempt >= TAURI_FETCH_RETRY_DELAYS_MS.length
      ) {
        throw error;
      }
      await wait(TAURI_FETCH_RETRY_DELAYS_MS[attempt]);
      beforeRetry?.();
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

/**
 * ``ownRequirement`` means the server just told THIS account to change its
 * password, so no lookup is needed and none is made. Without it the caller has
 * no session left and /status, which describes the installation owner, is the
 * only thing there is to go on.
 */
async function redirectToAuth(ownRequirement = false): Promise<void> {
  if (isRedirecting) return;
  isRedirecting = true;

  let target = "/login";
  if (ownRequirement) {
    setMustChangePassword(true);
    target = "/change-password";
  } else {
    try {
      const res = await fetch(apiUrl("/api/auth/status"));
      if (res.ok) {
        const data = (await res.json()) as { requires_password_change: boolean };
        // /status is installation-owner bootstrap state, not the current
        // account's state, so it is only adopted once this session is gone.
        // Adopted while a managed account still held a token, the owner's
        // recovery pinned that account to /change-password even after it had
        // changed its own password, until the owner finished.
        if (
          data.requires_password_change &&
          !mustChangePassword() &&
          !getAuthToken()
        ) {
          setMustChangePassword(true);
        }
        if (data.requires_password_change || mustChangePassword()) {
          target = "/change-password";
        }
      }
    } catch {
      // Fall through to /login on error
    }
  }

  if (window.location.pathname === target) {
    isRedirecting = false;
    return;
  }
  window.location.href = target;
}

function asTransportFailure(err: unknown): unknown {
  // fetch TypeError = offline | backend down | CORS/DNS. Tagged so callers tell "never reached"
  // from "rejected"; Tauri is always backend-down, the web build distinguishes offline.
  if (!(err instanceof TypeError)) return err;
  if (
    !isTauri &&
    typeof navigator !== "undefined" &&
    navigator.onLine === false
  ) {
    return Object.assign(
      new Error(
        "You appear to be offline. Check your network connection and try again.",
      ),
      { unslothTransportFailure: true },
    );
  }
  return Object.assign(
    new Error("Unsloth isn't running -- please relaunch it."),
    { unslothTransportFailure: true },
  );
}

async function retryWithCurrentToken(
  input: RequestInfo | URL,
  init?: RequestInit,
  retryNetworkErrors = true,
  beforeRetry?: () => void,
): Promise<Response> {
  beforeRetry?.();
  const retryHeaders = new Headers(init?.headers);
  addBrowserTimezoneHeaders(retryHeaders);
  const token = getAuthToken();
  if (token) retryHeaders.set("Authorization", `Bearer ${token}`);
  // Retries are tagged like the first attempt; an untagged TypeError reads as a rejection.
  try {
    return await fetchWithTauriNetworkRetry(
      input,
      { ...init, headers: retryHeaders },
      retryNetworkErrors,
      beforeRetry,
    );
  } catch (err) {
    throw asTransportFailure(err);
  }
}

async function retryWithTauriAutoAuth(
  input: RequestInfo | URL,
  init?: RequestInit,
  retryNetworkErrors = true,
  beforeRetry?: () => void,
): Promise<Response | null> {
  clearAuthTokens();
  const { tauriAutoAuth } = await import("./tauri-auto-auth");
  if (await tauriAutoAuth()) {
    return retryWithCurrentToken(input, init, retryNetworkErrors, beforeRetry);
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
  options?: AuthFetchOptions,
): Promise<Response> {
  const resolvedInput = typeof input === "string" ? apiUrl(input) : input;
  // The session this request belongs to. A 401 can arrive after the account has
  // changed, and every recovery path below resends the original method, URL and
  // body: without this the retry carried Alice's write into Bob's workspace on
  // Bob's token. The logoutGeneration check inside refreshSession only covers a
  // refresh that was already running, not a request that outlived its session.
  // Compared on the epoch alone: a refresh rotates the stored token within the
  // same session, and two requests in flight together legitimately see each
  // other's rotation, so the token is not the identity of the session.
  const startGeneration = logoutGeneration;
  const sessionUnchanged = () => startGeneration === logoutGeneration;
  const headers = new Headers(init?.headers);
  addBrowserTimezoneHeaders(headers);
  const accessToken = getAuthToken();
  if (accessToken) {
    headers.set("Authorization", `Bearer ${accessToken}`);
  }

  let response: Response;
  try {
    response = await fetchWithTauriNetworkRetry(
      resolvedInput,
      {
        ...init,
        headers,
      },
      options?.retryNetworkErrors ?? true,
      options?.beforeRetry,
    );
  } catch (err) {
    throw asTransportFailure(err);
  }

  if (await isPasswordChangeRequiredResponse(response)) {
    if (isTauri) {
      return (
        (await retryWithTauriAutoAuth(
          resolvedInput,
          init,
          options?.retryNetworkErrors ?? true,
          options?.beforeRetry,
        )) ?? response
      );
    }
    void redirectToAuth(true);
    return response;
  }
  if (response.status !== 401) return response;
  // Answered for a session that is gone. Hand the 401 back rather than
  // authenticate it as whoever holds the browser now.
  if (!sessionUnchanged()) return response;

  const refreshToken = getRefreshToken();
  const refreshed = await refreshSession();
  if (!refreshed) {
    if (isTauri) {
      return (
        (await retryWithTauriAutoAuth(
          resolvedInput,
          init,
          options?.retryNetworkErrors ?? true,
          options?.beforeRetry,
        )) ?? response
      );
    }
    clearAuthTokensIfCurrent(refreshToken);
    void redirectToAuth();
    return response;
  }

  if (mustChangePassword()) {
    if (isTauri) {
      return (
        (await retryWithTauriAutoAuth(
          resolvedInput,
          init,
          options?.retryNetworkErrors ?? true,
          options?.beforeRetry,
        )) ?? response
      );
    }
    void redirectToAuth(true);
    return response;
  }

  if (!sessionUnchanged()) return response;
  if (!getAuthToken()) clearAuthTokens();
  return retryWithCurrentToken(
    resolvedInput,
    init,
    options?.retryNetworkErrors ?? true,
    options?.beforeRetry,
  );
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

/**
 * End the current auth session for the purposes of in-flight requests.
 *
 * Called when a different account signs in without a logout in between: a
 * request already waiting on a 401 belongs to the session that started it, and
 * without a bump here its retry would be authenticated as the new account.
 */
export function noteAuthSessionReplaced(): void {
  logoutGeneration += 1;
}

export async function logout(): Promise<void> {
  // Server-side revoke. If the access token is expired the 401 fires before revoke runs, so
  // rotate via the refresh token and retry to revoke the family. The finally generation bump
  // invalidates in-flight refreshes.
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
