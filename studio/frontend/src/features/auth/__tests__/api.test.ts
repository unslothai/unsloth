// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  AUTH_MUST_CHANGE_PASSWORD_KEY,
  AUTH_REFRESH_TOKEN_KEY,
  AUTH_TOKEN_KEY,
} from "@/features/auth/session";

type FetchMock = ReturnType<typeof vi.fn>;

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

function deferred<T>(): {
  promise: Promise<T>;
  resolve: (v: T) => void;
  reject: (e: unknown) => void;
} {
  let resolve!: (v: T) => void;
  let reject!: (e: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

async function loadApi() {
  // Reset module-level singleflight + generation state between tests.
  vi.resetModules();
  return await import("@/features/auth/api");
}

let fetchMock: FetchMock;

beforeEach(() => {
  localStorage.clear();
  fetchMock = vi.fn();
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("refreshSession singleflight", () => {
  it("issues exactly one /api/auth/refresh request when N callers race", async () => {
    localStorage.setItem(AUTH_REFRESH_TOKEN_KEY, "rt-original");
    const refreshDeferred = deferred<Response>();
    fetchMock.mockImplementation(async (input: RequestInfo) => {
      const url = typeof input === "string" ? input : (input as Request).url;
      if (url.endsWith("/api/auth/refresh")) {
        return refreshDeferred.promise;
      }
      throw new Error(`unexpected fetch: ${url}`);
    });

    const { refreshSession } = await loadApi();
    // Kick off all callers BEFORE we resolve the in-flight fetch, so they
    // all subscribe to the same singleflight promise.
    const callersPromise = Promise.all([
      refreshSession(),
      refreshSession(),
      refreshSession(),
      refreshSession(),
      refreshSession(),
    ]);
    // Yield once so the refresh IIFE has reached its fetch call.
    await Promise.resolve();
    refreshDeferred.resolve(
      jsonResponse(200, {
        access_token: "new-access",
        refresh_token: "new-refresh",
        must_change_password: false,
      }),
    );
    const results = await callersPromise;

    expect(results).toEqual([true, true, true, true, true]);
    const refreshCalls = fetchMock.mock.calls.filter(([input]) => {
      const url = typeof input === "string" ? input : (input as Request).url;
      return url.endsWith("/api/auth/refresh");
    });
    expect(refreshCalls).toHaveLength(1);
    expect(localStorage.getItem(AUTH_TOKEN_KEY)).toBe("new-access");
    expect(localStorage.getItem(AUTH_REFRESH_TOKEN_KEY)).toBe("new-refresh");
  });

  it("clears the in-flight slot after resolution so the next call refreshes again", async () => {
    localStorage.setItem(AUTH_REFRESH_TOKEN_KEY, "rt-1");
    fetchMock.mockResolvedValue(
      jsonResponse(200, {
        access_token: "a",
        refresh_token: "b",
        must_change_password: false,
      }),
    );
    const { refreshSession } = await loadApi();
    await refreshSession();
    await refreshSession();
    const refreshCalls = fetchMock.mock.calls.filter(([input]) => {
      const url = typeof input === "string" ? input : (input as Request).url;
      return url.endsWith("/api/auth/refresh");
    });
    expect(refreshCalls).toHaveLength(2);
  });

  it("returns false and clears tokens on non-OK refresh response", async () => {
    localStorage.setItem(AUTH_TOKEN_KEY, "old-access");
    localStorage.setItem(AUTH_REFRESH_TOKEN_KEY, "rt-expired");
    fetchMock.mockResolvedValue(jsonResponse(401, { detail: "invalid" }));

    const { refreshSession } = await loadApi();
    const ok = await refreshSession();
    expect(ok).toBe(false);
    expect(localStorage.getItem(AUTH_TOKEN_KEY)).toBeNull();
    expect(localStorage.getItem(AUTH_REFRESH_TOKEN_KEY)).toBeNull();
  });

  it("returns false (no refresh fired) when there is no refresh token", async () => {
    const { refreshSession } = await loadApi();
    const ok = await refreshSession();
    expect(ok).toBe(false);
    expect(fetchMock).not.toHaveBeenCalled();
  });
});

describe("logoutGeneration invalidates a mid-flight refresh", () => {
  it("drops the new token pair when logout() runs while refresh is in flight", async () => {
    localStorage.setItem(AUTH_TOKEN_KEY, "access-old");
    localStorage.setItem(AUTH_REFRESH_TOKEN_KEY, "rt-pre-logout");
    const refreshDeferred = deferred<Response>();
    const logoutDeferred = deferred<Response>();

    fetchMock.mockImplementation(async (input: RequestInfo) => {
      const url = typeof input === "string" ? input : (input as Request).url;
      if (url.endsWith("/api/auth/refresh")) return refreshDeferred.promise;
      if (url.endsWith("/api/auth/logout")) return logoutDeferred.promise;
      throw new Error(`unexpected fetch: ${url}`);
    });

    const { refreshSession, logout } = await loadApi();
    // Start the refresh, do NOT await yet.
    const refreshPromise = refreshSession();
    // Yield once so the IIFE inside refreshSession starts executing.
    await Promise.resolve();
    // Run logout to completion; this should bump logoutGeneration before
    // the refresh has the chance to write tokens back.
    const logoutPromise = logout();
    logoutDeferred.resolve(new Response(null, { status: 204 }));
    await logoutPromise;
    // Now resolve the refresh with what would otherwise be a success.
    refreshDeferred.resolve(
      jsonResponse(200, {
        access_token: "access-NEW",
        refresh_token: "rt-NEW",
        must_change_password: false,
      }),
    );
    const refreshed = await refreshPromise;

    // The generation mismatch must have made refreshSession() drop the
    // new pair on the floor; the SPA must stay logged out.
    expect(refreshed).toBe(false);
    expect(localStorage.getItem(AUTH_TOKEN_KEY)).toBeNull();
    expect(localStorage.getItem(AUTH_REFRESH_TOKEN_KEY)).toBeNull();
    expect(localStorage.getItem(AUTH_MUST_CHANGE_PASSWORD_KEY)).toBeNull();
  });
});

describe("logout()", () => {
  it("clears local state on a 204 even if the network is fine", async () => {
    localStorage.setItem(AUTH_TOKEN_KEY, "a");
    localStorage.setItem(AUTH_REFRESH_TOKEN_KEY, "b");
    localStorage.setItem(AUTH_MUST_CHANGE_PASSWORD_KEY, "1");
    fetchMock.mockResolvedValue(new Response(null, { status: 204 }));

    const { logout } = await loadApi();
    await logout();
    expect(localStorage.getItem(AUTH_TOKEN_KEY)).toBeNull();
    expect(localStorage.getItem(AUTH_REFRESH_TOKEN_KEY)).toBeNull();
    expect(localStorage.getItem(AUTH_MUST_CHANGE_PASSWORD_KEY)).toBeNull();
  });

  it("retries with a fresh access token when the first POST hits 401", async () => {
    localStorage.setItem(AUTH_TOKEN_KEY, "stale-access");
    localStorage.setItem(AUTH_REFRESH_TOKEN_KEY, "rt-1");

    const calls: { url: string; auth: string | null }[] = [];
    fetchMock.mockImplementation(async (input: RequestInfo, init?: RequestInit) => {
      const url = typeof input === "string" ? input : (input as Request).url;
      const headers = new Headers(init?.headers);
      calls.push({ url, auth: headers.get("authorization") });
      if (url.endsWith("/api/auth/logout")) {
        // First logout sees the stale token and 401s; second sees the
        // post-refresh token and 204s.
        const seen = calls.filter((c) => c.url.endsWith("/api/auth/logout")).length;
        return seen === 1
          ? new Response(null, { status: 401 })
          : new Response(null, { status: 204 });
      }
      if (url.endsWith("/api/auth/refresh")) {
        return jsonResponse(200, {
          access_token: "rotated-access",
          refresh_token: "rt-2",
          must_change_password: false,
        });
      }
      throw new Error(`unexpected fetch: ${url}`);
    });

    const { logout } = await loadApi();
    await logout();

    const logoutCalls = calls.filter((c) => c.url.endsWith("/api/auth/logout"));
    const refreshCalls = calls.filter((c) => c.url.endsWith("/api/auth/refresh"));
    expect(logoutCalls).toHaveLength(2);
    expect(refreshCalls).toHaveLength(1);
    expect(logoutCalls[0].auth).toBe("Bearer stale-access");
    expect(logoutCalls[1].auth).toBe("Bearer rotated-access");
    expect(localStorage.getItem(AUTH_TOKEN_KEY)).toBeNull();
    expect(localStorage.getItem(AUTH_REFRESH_TOKEN_KEY)).toBeNull();
  });

  it("still clears local state when the network throws", async () => {
    localStorage.setItem(AUTH_TOKEN_KEY, "a");
    localStorage.setItem(AUTH_REFRESH_TOKEN_KEY, "b");
    fetchMock.mockRejectedValue(new TypeError("offline"));

    const { logout } = await loadApi();
    await logout();
    expect(localStorage.getItem(AUTH_TOKEN_KEY)).toBeNull();
    expect(localStorage.getItem(AUTH_REFRESH_TOKEN_KEY)).toBeNull();
  });
});
