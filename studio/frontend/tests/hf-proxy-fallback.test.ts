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

// The backend stamps every response the hf-proxy route produces, so the stub
// has to as well; a proxy reply without it means "this backend has no such
// route", which is a different case entirely (mode "missing-route").
const PROXY_MARKER_HEADER = "X-Unsloth-HF-Proxy";
const proxiedResponse = (body: string, status: number) =>
  new Response(body, { status, headers: { [PROXY_MARKER_HEADER]: "1" } });

/** Install a fetch stub; direct HF calls fail, proxy calls answer per mode. */
function installFetchStub(behavior: {
  direct: "network-error" | "hang" | "ok";
  proxy: "ok" | "network-error" | "gateway" | "missing-route";
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
        proxiedResponse('{"detail":"bad gateway"}', 502),
      );
    }
    if (mode === "missing-route") {
      // An older Studio backend: the SPA catch-all answers unknown /api paths,
      // so there is no marker header on the 404.
      return Promise.resolve(
        new Response('{"detail":"API endpoint not found"}', { status: 404 }),
      );
    }
    return Promise.resolve(
      isProxy
        ? proxiedResponse("{}", 200)
        : new Response("{}", { status: 200 }),
    );
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
    return Promise.resolve(proxiedResponse("{}", 200));
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

test("an older backend without the route is not mistaken for a hub reply", async () => {
  reset();
  // A Studio backend that predates /api/hub/hf-proxy answers the SPA
  // catch-all's 404. Treating that as a Hugging Face response would hand
  // {"detail":"API endpoint not found"} to @huggingface/hub as a model
  // listing, report the hub as online, and pin every later request to the
  // missing route for ten minutes.
  const calls = installFetchStub({ direct: "network-error", proxy: "missing-route" });

  await assert.rejects(() => fetchWithTimeout(HF_URL, {}, 50), TypeError);
  assert.equal(isHuggingFaceOffline(), true);

  // Proxy-first must NOT be armed: the next call still tries direct first.
  const before = calls.length;
  await assert.rejects(() => fetchWithTimeout(HF_URL, {}, 50), TypeError);
  assert.equal(calls[before].url, HF_URL);
});

test("a genuine hub 404 still reaches the caller", async () => {
  reset();
  // The mirror image: the proxy DID answer, and Hugging Face said 404 for a
  // missing repo. That has to pass through untouched, which is why the marker
  // header exists instead of a status allowlist.
  globalThis.fetch = ((input: string | URL | Request) => {
    const url = typeof input === "string" ? input : input.toString();
    if (url.startsWith(PROXY_PREFIX)) {
      return Promise.resolve(
        new Response('{"error":"Repo not found"}', {
          status: 404,
          headers: { "X-Unsloth-HF-Proxy": "1" },
        }),
      );
    }
    return Promise.reject(new TypeError("Failed to fetch"));
  }) as typeof fetch;

  const response = await fetchWithTimeout(HF_URL, {}, 50);
  assert.equal(response.status, 404);
  assert.deepEqual(await response.json(), { error: "Repo not found" });
  assert.equal(isHuggingFaceOffline(), false);
});

test("an abort during the proxy fallback surfaces as an abort", async () => {
  reset();
  const controller = new AbortController();
  globalThis.fetch = ((input: string | URL | Request, init?: RequestInit) => {
    const url = typeof input === "string" ? input : input.toString();
    if (url.startsWith(PROXY_PREFIX)) {
      queueMicrotask(() => controller.abort());
      return new Promise<Response>((_resolve, reject) => {
        const fail = () => reject(new DOMException("Aborted", "AbortError"));
        if (init?.signal?.aborted) return fail();
        init?.signal?.addEventListener("abort", fail, { once: true });
      });
    }
    return Promise.reject(new TypeError("Failed to fetch"));
  }) as typeof fetch;

  let caught: unknown;
  try {
    await fetchWithTimeout(HF_URL, { signal: controller.signal }, 5_000);
  } catch (error) {
    caught = error;
  }
  assert.ok(caught instanceof DOMException && caught.name === "AbortError");
  // A caller changing its mind must never look like the hub going away.
  assert.equal(isHuggingFaceOffline(), false);
});

test("one call never spends more than a single proxy attempt", async () => {
  reset();
  // Arm proxy-first with a successful fallback...
  installFetchStub({ direct: "network-error", proxy: "ok" });
  await fetchWithTimeout(HF_URL, {}, 50);

  // ...then make everything hang. Without a cap this runs proxy, direct,
  // proxy and burns three times the caller's timeout budget.
  const legs: string[] = [];
  globalThis.fetch = ((input: string | URL | Request, init?: RequestInit) => {
    const url = typeof input === "string" ? input : input.toString();
    legs.push(url.startsWith(PROXY_PREFIX) ? "PROXY" : "DIRECT");
    return new Promise<Response>((_resolve, reject) => {
      const fail = () => reject(new DOMException("Aborted", "AbortError"));
      if (init?.signal?.aborted) return fail();
      init?.signal?.addEventListener("abort", fail, { once: true });
    });
  }) as typeof fetch;

  await assert.rejects(() => fetchWithTimeout(HF_URL, {}, 40));
  assert.deepEqual(legs, ["PROXY", "DIRECT"]);
});

test("the proxy preference is per origin", async () => {
  reset();
  const calls: FetchCall[] = [];
  globalThis.fetch = ((input: string | URL | Request, init?: RequestInit) => {
    const url = typeof input === "string" ? input : input.toString();
    calls.push({ url, init });
    return url.startsWith(PROXY_PREFIX)
      ? Promise.resolve(
          new Response("{}", { status: 200, headers: { "X-Unsloth-HF-Proxy": "1" } }),
        )
      : Promise.reject(new TypeError("Failed to fetch"));
  }) as typeof fetch;

  // A datasets-server fallback must not reroute huggingface.co traffic.
  await fetchWithTimeout("https://datasets-server.huggingface.co/size?dataset=x", {}, 50);
  const before = calls.length;
  await fetchWithTimeout(HF_URL, {}, 50);
  assert.equal(calls[before].url, HF_URL, "huggingface.co still tries direct first");
});

test("a hub 401 passed through the proxy is not treated as a dead session", async () => {
  reset();
  // huggingface.co answers 401 (not 404) for a private or missing repo, and the
  // route forwards that verbatim. Routed through authFetch it used to look like
  // a Studio session failure, which refreshes the session token per lookup and
  // signs the user out when the refresh fails.
  let sessionHandled = false;
  setHubProxyAuthFetch(async (input, init) => {
    const response = await fetch(input as string, init);
    if (
      response.status === 401 &&
      response.headers.get("X-Unsloth-HF-Proxy") === null
    ) {
      sessionHandled = true;
    }
    return response;
  });
  globalThis.fetch = ((input: string | URL | Request) => {
    const url = typeof input === "string" ? input : input.toString();
    if (url.startsWith(PROXY_PREFIX)) {
      return Promise.resolve(
        new Response('{"error":"Repo not found"}', {
          status: 401,
          headers: { "X-Unsloth-HF-Proxy": "1" },
        }),
      );
    }
    return Promise.reject(new TypeError("Failed to fetch"));
  }) as typeof fetch;

  const response = await fetchWithTimeout(HF_URL, {}, 50);
  assert.equal(response.status, 401);
  assert.equal(sessionHandled, false, "must not run the session-recovery path");
  assert.equal(isHuggingFaceOffline(), false);
});
