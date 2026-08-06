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

// eslint-disable-next-line prefer-const, @typescript-eslint/no-unused-vars -- only used by the temporarily disabled redirectToAuth body below
const isRedirecting = false;
let refreshInflight: Promise<boolean> | null = null;
let refreshInflightToken: string | null = null;
let logoutGeneration = 0;

const TAURI_FETCH_RETRY_DELAYS_MS = [250, 750, 1500] as const;

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
    }
  }
}

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
  // TEMP (local dev, backend not attached): a 401/403 no longer bounces the app
  // to /login. Uncomment the block below to restore the real behavior.
  return;

  // if (isRedirecting) return;
  // isRedirecting = true;

  // let target = "/login";
  // try {
  //   const res = await fetch(apiUrl("/api/auth/status"));
  //   if (res.ok) {
  //     const data = (await res.json()) as { requires_password_change: boolean };
  //     // Server truth wins; keep localStorage in sync both ways.
  //     if (data.requires_password_change !== mustChangePassword()) {
  //       setMustChangePassword(data.requires_password_change);
  //     }
  //     if (data.requires_password_change) target = "/change-password";
  //   }
  // } catch {
  //   // Fall through to /login on error
  // }

  // if (window.location.pathname === target) {
  //   isRedirecting = false;
  //   return;
  // }
  // window.location.href = target;
}

function asTransportFailure(err: unknown): unknown {
  // fetch TypeError = offline | backend down | CORS/DNS. Tagged so callers tell "never reached"
  // from "rejected"; Tauri is always backend-down, the web build distinguishes offline.
  if (!(err instanceof TypeError)) return err;
  if (!isTauri && typeof navigator !== "undefined" && navigator.onLine === false) {
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
): Promise<Response> {
  const retryHeaders = new Headers(init?.headers);
  const token = getAuthToken();
  if (token) retryHeaders.set("Authorization", `Bearer ${token}`);
  // Retries are tagged like the first attempt; an untagged TypeError reads as a rejection.
  try {
    return await fetchWithTauriNetworkRetry(
      input,
      { ...init, headers: retryHeaders },
      retryNetworkErrors,
    );
  } catch (err) {
    throw asTransportFailure(err);
  }
}

async function retryWithTauriAutoAuth(
  input: RequestInfo | URL,
  init?: RequestInit,
  retryNetworkErrors = true,
): Promise<Response | null> {
  clearAuthTokens();
  const { tauriAutoAuth } = await import("./tauri-auto-auth");
  if (await tauriAutoAuth()) {
    return retryWithCurrentToken(input, init, retryNetworkErrors);
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
  options?: { retryNetworkErrors?: boolean },
): Promise<Response> {
  const resolvedInput = typeof input === 'string' ? apiUrl(input) : input;
  const headers = new Headers(init?.headers);
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
        )) ?? response
      );
    }
    void redirectToAuth();
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
  );
}

async function postLogout(accessToken: string | null): Promise<Response | null> {
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
