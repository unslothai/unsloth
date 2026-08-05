// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Hub discovery fetches huggingface.co straight from the browser, so privacy
// tooling that filters browser traffic made the UI report "You're offline"
// while the backend still had connectivity. fetchWithTimeout now retries a
// failed direct HF fetch through the backend's /api/hub/hf-proxy passthrough
// and only marks the hub offline when both paths fail.

import assert from "node:assert/strict";
import test from "node:test";

import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";

const { store } = installLocalStorageFake();
Object.assign(globalThis.window as unknown as Record<string, unknown>, {
  dispatchEvent: () => true,
  addEventListener: () => undefined,
  removeEventListener: () => undefined,
  location: { href: "http://localhost:8888/" },
});

registerBundlerResolver();
const {
  __resetHfProxyPreferenceForTests,
  fetchWithTimeout,
  isHuggingFaceOffline,
  markRemoteNetworkOnline,
  setHubProxyAuthFetch,
} = await import("../src/features/hub/lib/network.ts");

const HF_URL = "https://huggingface.co/api/models?limit=20";
const PROXY_PREFIX = "/api/hub/hf-proxy?url=";

type FetchCall = { url: string; init: RequestInit | undefined };

/** Install a fetch stub; direct HF calls fail, proxy calls answer per mode. */
function installFetchStub(behavior: {
  direct: "network-error" | "hang" | "ok";
  proxy: "ok" | "network-error" | "gateway";
}): FetchCall[] {
  const calls: FetchCall[] = [];
  globalThis.fetch = ((input: string | URL | Request, init?: RequestInit) => {
    const url = typeof input === "string" ? input : input.toString();
    calls.push({ url, init });
    const isProxy = url.startsWith(PROXY_PREFIX);
    const mode = isProxy ? behavior.proxy : behavior.direct;
    if (mode === "network-error") {
      return Promise.reject(new TypeError("Failed to fetch"));
    }
    if (mode === "hang") {
      return new Promise<Response>((_resolve, reject) => {
        init?.signal?.addEventListener("abort", () =>
          reject(new DOMException("Aborted", "AbortError")),
        );
      });
    }
    if (mode === "gateway") {
      return Promise.resolve(
        new Response('{"detail":"bad gateway"}', { status: 502 }),
      );
    }
    return Promise.resolve(new Response("{}", { status: 200 }));
  }) as typeof fetch;
  return calls;
}

function reset(): void {
  __resetHfProxyPreferenceForTests();
  setHubProxyAuthFetch(null);
  markRemoteNetworkOnline();
  store.clear();
}

test("direct network error falls back to the backend proxy and stays online", async () => {
  reset();
  store.set("unsloth_auth_token", "session-jwt");
  const calls = installFetchStub({ direct: "network-error", proxy: "ok" });

  const response = await fetchWithTimeout(HF_URL, {
    headers: { Authorization: "Bearer hf_secret" },
  });

  assert.equal(response.status, 200);
  assert.equal(calls.length, 2);
  assert.equal(calls[0].url, HF_URL);
  assert.equal(calls[1].url, PROXY_PREFIX + encodeURIComponent(HF_URL));
  const proxyHeaders = new Headers(calls[1].init?.headers);
  assert.equal(proxyHeaders.get("X-Unsloth-HF-Token"), "hf_secret");
  assert.equal(proxyHeaders.get("Authorization"), "Bearer session-jwt");
  assert.equal(isHuggingFaceOffline(), false);
});

test("after a successful fallback the proxy is preferred, skipping the doomed direct attempt", async () => {
  reset();
  const calls = installFetchStub({ direct: "network-error", proxy: "ok" });

  await fetchWithTimeout(HF_URL);
  const callsBefore = calls.length;
  await fetchWithTimeout(HF_URL);

  assert.equal(callsBefore, 2);
  assert.equal(calls.length, 3);
  assert.ok(calls[2].url.startsWith(PROXY_PREFIX));
});

test("direct timeout falls back to the proxy instead of surfacing a timeout", async () => {
  reset();
  const calls = installFetchStub({ direct: "hang", proxy: "ok" });

  const response = await fetchWithTimeout(HF_URL, {}, 20);

  assert.equal(response.status, 200);
  assert.equal(calls.length, 2);
  assert.equal(isHuggingFaceOffline(), false);
});

test("only when both paths fail is the hub marked offline", async () => {
  reset();
  installFetchStub({ direct: "network-error", proxy: "network-error" });

  await assert.rejects(fetchWithTimeout(HF_URL), TypeError);
  assert.equal(isHuggingFaceOffline(), true);
});

test("non-hub origins never fall back to the proxy", async () => {
  reset();
  const calls = installFetchStub({ direct: "network-error", proxy: "ok" });

  await assert.rejects(fetchWithTimeout("https://example.com/data"), TypeError);
  assert.equal(calls.length, 1);
});

test("a proxy gateway error is not cached as a healthy fallback", async () => {
  reset();
  const calls = installFetchStub({ direct: "network-error", proxy: "gateway" });

  await assert.rejects(fetchWithTimeout(HF_URL), TypeError);
  assert.equal(isHuggingFaceOffline(), true);

  // proxy-first must not be armed: the next attempt goes direct again
  markRemoteNetworkOnline();
  const callsBefore = calls.length;
  await assert.rejects(fetchWithTimeout(HF_URL), TypeError);
  assert.equal(calls[callsBefore].url, HF_URL);
});

test("a registered session-aware fetch handles proxy auth instead of a raw token", async () => {
  reset();
  store.set("unsloth_auth_token", "stale-token");
  const authCalls: FetchCall[] = [];
  setHubProxyAuthFetch((input, init) => {
    authCalls.push({ url: String(input), init });
    return Promise.resolve(new Response("{}", { status: 200 }));
  });
  installFetchStub({ direct: "network-error", proxy: "ok" });

  const response = await fetchWithTimeout(HF_URL);

  assert.equal(response.status, 200);
  assert.equal(authCalls.length, 1);
  assert.ok(authCalls[0].url.startsWith(PROXY_PREFIX));
  // authFetch owns the session header; network.ts must not preset a stale one
  const headers = new Headers(authCalls[0].init?.headers);
  assert.equal(headers.get("Authorization"), null);
});

test("a caller abort is surfaced, not retried through the proxy", async () => {
  reset();
  const calls = installFetchStub({ direct: "hang", proxy: "ok" });
  const controller = new AbortController();
  const pending = fetchWithTimeout(HF_URL, { signal: controller.signal });
  controller.abort();

  await assert.rejects(pending, (error: unknown) => {
    return error instanceof DOMException && error.name === "AbortError";
  });
  assert.equal(calls.length, 1);
  assert.equal(isHuggingFaceOffline(), false);
});
