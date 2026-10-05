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

// #10520: sums to 10.5s, just past the launcher's 10s HEALTH_PROBE_TIMEOUT. Guarded by
// `the_frontend_retry_ladder_outlives_one_probe_budget` in src-tauri/src/commands.rs.
const TAURI_FETCH_RETRY_DELAYS_MS = [250, 750, 1500, 3000, 5000] as const;
// A network error is not an answer: retrying a committed POST creates a second API key,
// project or job, so non-idempotent methods keep the shorter ladder.
const TAURI_FETCH_RETRY_DELAYS_UNSAFE_MS = [250, 750, 1500] as const;
const IDEMPOTENT_METHODS = new Set(["GET", "HEAD", "OPTIONS", "PUT", "DELETE"]);

function retryDelaysFor(
  input: RequestInfo | URL,
  init?: RequestInit,
): readonly number[] {
  const method = (
    init?.method ??
    (typeof Request !== "undefined" && input instanceof Request
      ? input.method
      : "GET")
  ).toUpperCase();
  return IDEMPOTENT_METHODS.has(method)
    ? TAURI_FETCH_RETRY_DELAYS_MS
    : TAURI_FETCH_RETRY_DELAYS_UNSAFE_MS;
}
const BROWSER_TIMEZONE_HEADER = "X-Unsloth-Timezone";
const BROWSER_TIMEZONE_OFFSET_HEADER = "X-Unsloth-Timezone-Offset-Minutes";

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
  const delays = retryDelaysFor(input, init);
  for (let attempt = 0; ; attempt++) {
    try {
      return await fetch(input, init);
    } catch (error) {
      if (
        !isTauri ||
        !retryNetworkErrors ||
        !(error instanceof TypeError) ||
        attempt >= delays.length
      ) {
        throw error;
      }
      // Tauri only, below the guard above. `fetch` cannot tell a refused port from a silent
      // one and the native side can, so ask it once on the FIRST failure rather than sleeping
      // out #10520's 10.5s ladder to be told what the refusal already proved.
      if (attempt === 0 && (await nativeBackendIsGone())) throw error;
      await wait(delays[attempt]);
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
      const requiresChange =
        data.login_mode === "multi"
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

/** `check_backend_present` and NOT `check_health`: the latter reports a probe that ran out of budget exactly as a refused connection. */
// The port is carried WITH the promise: `setApiBase` can move it inside the 10s budget.
let nativeHealthInflight: { port: number; probe: Promise<boolean> } | null = null;

async function nativeBackendIsAlive(): Promise<boolean> {
  if (!isTauri) {
    return false;
  }
  const port = getApiPort();
  if (port === null) {
    return false;
  }
  // Single flight: one 10s probe rather than one per panel. Not cached beyond the call.
  if (nativeHealthInflight !== null && nativeHealthInflight.port === port) {
    return nativeHealthInflight.probe;
  }
  const probe = (async () => {
    try {
      const { invoke } = await import("@tauri-apps/api/core");
      return (
        (await invoke<boolean>("check_backend_present", { port })) === true
      );
    } catch {
      return false;
    }
  })();
  const inflight = { port, probe };
  nativeHealthInflight = inflight;
  try {
    return await probe;
  } finally {
    // Identity, not the port: a probe for a newer port owns the slot now.
    if (nativeHealthInflight === inflight) {
      nativeHealthInflight = null;
    }
  }
}

/** `check_backend_is_gone` and NOT `check_backend_present`: presence reports a backend of ours that has not bound its port yet as absent. */
let nativeGoneInflight: { port: number; probe: Promise<boolean> } | null = null;

/**
 * Whether the retry ladder has anything left to wait for.
 *
 * Only ever answers true on positive proof, so every failure mode below returns false and
 * leaves the ladder exactly as long as it is today: the browser build, which has no native
 * side to ask; a port the webview has not been given yet; and a shell too old to carry the
 * command, whose rejected `invoke` is caught here.
 */
async function nativeBackendIsGone(): Promise<boolean> {
  if (!isTauri) {
    return false;
  }
  const port = getApiPort();
  if (port === null) {
    return false;
  }
  // Single flight, like the presence probe: a hub losing the backend fails every panel at once.
  if (nativeGoneInflight !== null && nativeGoneInflight.port === port) {
    return nativeGoneInflight.probe;
  }
  const probe = (async () => {
    try {
      const { invoke } = await import("@tauri-apps/api/core");
      return (
        (await invoke<boolean>("check_backend_is_gone", { port })) === true
      );
    } catch {
      return false;
    }
  })();
  const inflight = { port, probe };
  nativeGoneInflight = inflight;
  try {
    return await probe;
  } finally {
    // Identity, not the port: a probe for a newer port owns the slot now.
    if (nativeGoneInflight === inflight) {
      nativeGoneInflight = null;
    }
  }
}

async function asTransportFailure(err: unknown): Promise<unknown> {
  // fetch TypeError = offline | backend down | CORS/DNS. Tagged so callers tell "never reached"
  // from "rejected"; under Tauri the launcher is asked before claiming the backend is gone.
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
  // A failed fetch in the webview is not proof the backend died, and "please relaunch it"
  // throws away a running backend and whatever it has in flight.
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
