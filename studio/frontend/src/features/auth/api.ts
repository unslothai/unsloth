// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { accountTransitionPending } from "@/lib/account-transition";
import { apiUrl, getApiPort, isTauri } from "@/lib/api-base";
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

// Sized against the launcher, not against a guess. src-tauri/src/commands.rs spends
// HEALTH_PROBE_TIMEOUT (10s) on a single liveness probe and three of those before its
// watchdog will call a backend dead, so a ladder that ran out after 250+750+1500ms was the
// first thing in the app to give up: it put "Unsloth isn't running" in front of a backend the
// launcher still considered perfectly alive. That is what a kernel-level loopback filter
// produces, and what a multi-GPU warm-up produces on its own (#10520). These delays sum to
// 10.5s, just past that per-probe budget, so the webview can no longer be the one to quit
// first. Guarded against drift by `the_frontend_retry_ladder_outlives_one_probe_budget` in
// src-tauri/src/commands.rs.
const TAURI_FETCH_RETRY_DELAYS_MS = [250, 750, 1500, 3000, 5000] as const;
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

async function redirectToAuth(passwordChangeRequired = false): Promise<void> {
  if (isRedirecting) return;
  isRedirecting = true;

  let target = "/login";
  try {
    const res = await fetch(apiUrl("/api/auth/status"));
    if (res.ok) {
      const data = (await res.json()) as {
        requires_password_change: boolean;
        login_mode?: "single" | "multi";
      };
      // Public status describes the owner. A managed session carries its own requirement.
      const requiresChange = data.login_mode === "multi"
        ? passwordChangeRequired || mustChangePassword()
        : data.requires_password_change;
      if (requiresChange !== mustChangePassword()) {
        setMustChangePassword(requiresChange);
      }
      if (requiresChange) target = "/change-password";
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

/** Copy shown when the backend really is unreachable and the launcher agrees. */
export const BACKEND_NOT_RUNNING_MESSAGE =
  "Unsloth isn't running -- please relaunch it.";
/** Copy shown when the webview could not reach the backend but the launcher says it is up. */
export const BACKEND_NOT_ANSWERING_MESSAGE =
  "Unsloth is running but did not answer in time. It may still be starting up. Please try again in a moment.";

/**
 * Ask the Rust side whether the backend it manages is still there.
 *
 * The `check_health` command probes /api/liveness from the native process with the
 * launcher's own budget and with proxies disabled, so it answers in cases where the
 * webview's own fetch was starved or refused: a firewall that filters loopback per process,
 * a proxy configuration the webview honours, or a backend whose event loop is held by the
 * GIL while the ML stack imports. Any failure to ask at all reads as "no second opinion",
 * which leaves the original verdict in place.
 */
let nativeHealthInflight: Promise<boolean> | null = null;

async function nativeBackendIsAlive(): Promise<boolean> {
  if (!isTauri) {
    return false;
  }
  const port = getApiPort();
  if (port === null) {
    return false;
  }
  // Single flight. The condition this runs under takes out every panel at once: a hub with
  // chat, training and settings polling loses all of them in the same tick, and each loss
  // would otherwise open its own probe. On the firewall host those probes are the ones that
  // actually wait out the launcher's budget rather than being refused immediately, so a
  // shared answer is the difference between one 10s probe and one per panel. Not cached
  // beyond the call: the answer is about right now, and the next failure deserves a fresh one.
  if (nativeHealthInflight !== null) {
    return nativeHealthInflight;
  }
  const probe = (async () => {
    try {
      const { invoke } = await import("@tauri-apps/api/core");
      return (await invoke<boolean>("check_health", { port })) === true;
    } catch {
      return false;
    }
  })();
  nativeHealthInflight = probe;
  try {
    return await probe;
  } finally {
    if (nativeHealthInflight === probe) {
      nativeHealthInflight = null;
    }
  }
}

async function asTransportFailure(err: unknown): Promise<unknown> {
  // fetch TypeError = offline | backend down | CORS/DNS. Tagged so callers tell "never reached"
  // from "rejected"; the web build distinguishes offline, and under Tauri the launcher is
  // asked before the app claims the backend is gone.
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
  // A failed fetch in the webview is not proof the backend died, and "please relaunch it" is
  // an instruction that throws away a running backend, an in-flight generation and, on the
  // reported host, the only session the user could get. Only tell them that when the native
  // side cannot see the backend either.
  if (await nativeBackendIsAlive()) {
    return Object.assign(new Error(BACKEND_NOT_ANSWERING_MESSAGE), {
      unslothTransportFailure: true,
      unslothBackendStillRunning: true,
    });
  }
  return Object.assign(new Error(BACKEND_NOT_RUNNING_MESSAGE), {
    unslothTransportFailure: true,
  });
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
    throw await asTransportFailure(err);
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
  // Another tab is mid-switch: its new tokens are published before this tab reloads, so a
  // request now would carry this tab's account content under the next account's credentials.
  if (accountTransitionPending())
    throw new Error("Another tab is switching accounts; this tab will reload.");
  const resolvedInput = typeof input === "string" ? apiUrl(input) : input;
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
    throw await asTransportFailure(err);
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
    void redirectToAuth();
    return response;
  }

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
