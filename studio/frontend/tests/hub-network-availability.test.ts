// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  classifyFetchFailure,
  clearRemoteBackoff,
  fetchWithTimeout,
  getHubPhase,
  getLastHubFailure,
  isHubFetchError,
  isRemoteNetworkOffline,
  markRemoteNetworkOffline,
  markRemoteNetworkOnline,
  sanitizeHubErrorMessage,
} from "../src/features/hub/lib/network.ts";

const HF = "https://huggingface.co";

// Installing a window exercises the same event path the app uses, not the
// no-op branch network.ts falls back to.
function installWindow() {
  const listeners = new Map<string, Set<() => void>>();
  (globalThis as Record<string, unknown>).window = {
    addEventListener(type: string, fn: () => void) {
      if (!listeners.has(type)) listeners.set(type, new Set());
      listeners.get(type)?.add(fn);
    },
    removeEventListener(type: string, fn: () => void) {
      listeners.get(type)?.delete(fn);
    },
    dispatchEvent(event: { type: string }) {
      for (const fn of listeners.get(event.type) ?? []) fn();
      return true;
    },
    location: { href: "http://127.0.0.1:8888/hub" },
  };
  return listeners;
}

function reset() {
  installWindow();
  markRemoteNetworkOnline();
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

test("a lapsed backoff window means probing, never proven-available", async () => {
  reset();
  markRemoteNetworkOffline(HF, 10, {
    kind: "network-opaque",
    message: "blocked",
    origin: HF,
    retryable: true,
  });
  assert.equal(getHubPhase(HF), "unavailable");

  await sleep(30);

  // The old code flipped straight back to "online" here, fired a "Back online"
  // toast, retried, failed, and looped. Only a successful request may promote.
  assert.equal(
    getHubPhase(HF),
    "probing",
    "TTL expiry must not be mistaken for a successful probe",
  );
});

test("only a success clears the recorded failure", async () => {
  reset();
  markRemoteNetworkOffline(HF, 10, {
    kind: "csp-blocked",
    message: "blocked by CSP",
    origin: HF,
    retryable: false,
  });
  await sleep(30);

  assert.equal(
    getLastHubFailure(HF)?.kind,
    "csp-blocked",
    "the cause must outlive the backoff window or the panel goes generic",
  );

  markRemoteNetworkOnline(HF);
  assert.equal(getLastHubFailure(HF), null);
  assert.equal(getHubPhase(HF), "available");
});

test("Retry clears the backoff but keeps the diagnosis on screen", () => {
  reset();
  markRemoteNetworkOffline(HF, 60_000, {
    kind: "network-opaque",
    message: "could not reach",
    origin: HF,
    retryable: true,
  });
  assert.equal(getHubPhase(HF), "unavailable");

  clearRemoteBackoff(HF);

  assert.equal(
    getHubPhase(HF),
    "probing",
    "Retry must be able to re-probe immediately, not wait out the timer",
  );
  assert.equal(getLastHubFailure(HF)?.kind, "network-opaque");
});

test("classification separates timeout, abort, offline and opaque failures", () => {
  reset();
  assert.equal(
    classifyFetchFailure(new Error("x"), HF, { timedOut: true }).kind,
    "timeout",
  );

  const abort = new DOMException("aborted", "AbortError");
  assert.equal(classifyFetchFailure(abort, HF).kind, "aborted");

  // No navigator.onLine === false in Node, so a bare TypeError stays opaque
  // rather than being reported as a confirmed loss of connectivity.
  const opaque = classifyFetchFailure(new TypeError("Failed to fetch"), HF);
  assert.equal(opaque.kind, "network-opaque");
  assert.match(opaque.message, /huggingface\.co/);
  assert.match(opaque.message, /extension|antivirus|filter/i);
});

test("a browser reporting itself offline is named as such", () => {
  reset();
  const original = Object.getOwnPropertyDescriptor(globalThis, "navigator");
  Object.defineProperty(globalThis, "navigator", {
    value: { onLine: false },
    configurable: true,
  });
  try {
    assert.equal(
      classifyFetchFailure(new TypeError("Failed to fetch"), HF).kind,
      "browser-offline",
    );
  } finally {
    if (original) Object.defineProperty(globalThis, "navigator", original);
    else delete (globalThis as Record<string, unknown>).navigator;
  }
});

test("the failure never leaks the request URL or its query", async () => {
  reset();
  const original = globalThis.fetch;
  globalThis.fetch = (async () => {
    throw new TypeError("Failed to fetch");
  }) as typeof fetch;
  try {
    await assert.rejects(
      fetchWithTimeout(
        `${HF}/api/models?search=my-private-project&author=acme`,
        {},
        1_000,
      ),
      (err: unknown) => {
        assert.ok(isHubFetchError(err), "should throw a classified HubFetchError");
        const { message, origin } = err.failure;
        assert.ok(
          !message.includes("my-private-project"),
          "the search query must never reach a rendered message",
        );
        assert.ok(!message.includes("/api/models"), "no request path in message");
        assert.equal(origin, HF, "only the origin is retained");
        return true;
      },
    );
  } finally {
    globalThis.fetch = original;
  }
});

test("a caller-driven abort does not blacklist the origin", async () => {
  reset();
  const original = globalThis.fetch;
  const controller = new AbortController();
  globalThis.fetch = (async () => {
    controller.abort();
    throw new DOMException("aborted", "AbortError");
  }) as unknown as typeof fetch;
  try {
    await assert.rejects(
      fetchWithTimeout(`${HF}/api/models`, { signal: controller.signal }, 1_000),
    );
    assert.equal(
      getHubPhase(HF),
      "available",
      "superseding a query must not mark the Hub unreachable",
    );
    assert.equal(getLastHubFailure(HF), null);
  } finally {
    globalThis.fetch = original;
  }
});

test("a successful response clears a prior failure", async () => {
  reset();
  markRemoteNetworkOffline(HF, 60_000, {
    kind: "network-opaque",
    message: "could not reach",
    origin: HF,
    retryable: true,
  });
  const original = globalThis.fetch;
  globalThis.fetch = (async () => new Response("{}", { status: 200 })) as typeof fetch;
  try {
    await fetchWithTimeout(`${HF}/api/models`, {}, 1_000);
    assert.equal(getHubPhase(HF), "available");
    assert.equal(getLastHubFailure(HF), null);
  } finally {
    globalThis.fetch = original;
  }
});

test("an unrelated origin failing does not take the Hub down", () => {
  reset();
  markRemoteNetworkOffline("https://datasets-server.huggingface.co", 60_000, {
    kind: "network-opaque",
    message: "dataset viewer down",
    origin: "https://datasets-server.huggingface.co",
    retryable: true,
  });
  assert.equal(
    getHubPhase(HF),
    "available",
    "a datasets-server outage must not make the model hub look offline",
  );
});

test("the SDK's URL trailer never survives into a rendered message", () => {
  // Exact shape from @huggingface/hub createApiError: message, then
  // ". URL: <url>. Request ID: <id>".
  const raw =
    "Api error with status 502. URL: https://huggingface.co/api/models?search=my-private-project&limit=100. Request ID: abc123";
  const clean = sanitizeHubErrorMessage(raw);
  assert.ok(
    !clean.includes("my-private-project"),
    "the search query must not reach the screen",
  );
  assert.ok(!clean.includes("URL:"), "no URL trailer");
  assert.equal(clean, "Api error with status 502");
});

test("sanitizing leaves a message with no trailer alone", () => {
  assert.equal(sanitizeHubErrorMessage("Search failed"), "Search failed");
  assert.equal(sanitizeHubErrorMessage(""), "");
});

test("a proxy URL trailer is stripped as well as a Hub one", () => {
  // The proxy path's Response.url is same-origin but still carries the query,
  // and the training and dataset pickers render this string raw.
  const raw =
    "Api error with status 502. URL: http://127.0.0.1:8888/api/hub/discovery/models?search=acme-internal&limit=100";
  const clean = sanitizeHubErrorMessage(raw);
  assert.ok(!clean.includes("acme-internal"));
  assert.equal(clean, "Api error with status 502");
});

test("recordFailure false leaves the origin unmarked for a caller with a fallback", async () => {
  reset();
  const original = globalThis.fetch;
  globalThis.fetch = (async () => {
    throw new TypeError("Failed to fetch");
  }) as typeof fetch;
  try {
    await assert.rejects(
      fetchWithTimeout(`${HF}/api/models`, {}, 1_000, { recordFailure: false }),
    );
    assert.equal(
      getHubPhase(HF),
      "available",
      "recording here re-renders consumers, whose cleanup aborts the fallback",
    );
    // The default still records, for callers with no fallback of their own.
    await assert.rejects(fetchWithTimeout(`${HF}/api/models`, {}, 1_000));
    assert.equal(getHubPhase(HF), "unavailable");
  } finally {
    globalThis.fetch = original;
  }
});

test("an avatar success does not erase the discovery diagnosis", async () => {
  reset();
  markRemoteNetworkOffline(
    HF,
    30_000,
    { kind: "csp-blocked", message: "blocked", origin: HF, retryable: true },
    "discovery",
  );
  const original = globalThis.fetch;
  // The owner-avatar probe 404s for a user account, and any resolved response
  // counted as reachability, so this is the everyday case.
  globalThis.fetch = (async () =>
    new Response("", { status: 404 })) as typeof fetch;
  try {
    await fetchWithTimeout(`${HF}/api/organizations/acme/overview`, {}, 1_000, {
      service: "other",
    });
  } finally {
    globalThis.fetch = original;
  }
  assert.equal(
    getLastHubFailure(HF)?.kind,
    "csp-blocked",
    "a different endpoint on the same origin must not clear the cause on screen",
  );
  assert.notEqual(
    getHubPhase(HF),
    "available",
    "reverting to available here restores the generic panel this change removes",
  );
});

test("a discovery success does clear the discovery diagnosis", async () => {
  reset();
  markRemoteNetworkOffline(
    HF,
    30_000,
    { kind: "network-opaque", message: "boom", origin: HF, retryable: true },
    "discovery",
  );
  const original = globalThis.fetch;
  globalThis.fetch = (async () => new Response("[]", { status: 200 })) as typeof fetch;
  try {
    await fetchWithTimeout(`${HF}/api/models`, {}, 1_000, {
      service: "discovery",
    });
  } finally {
    globalThis.fetch = original;
  }
  assert.equal(getLastHubFailure(HF), null);
  assert.equal(getHubPhase(HF), "available");
});

test("a blocked avatar path never pins the catalog at probing", async () => {
  reset();
  // An auxiliary endpoint fails on its own; discovery has never failed.
  markRemoteNetworkOffline(
    HF,
    30_000,
    { kind: "timeout", message: "slow", origin: HF, retryable: true },
    "other",
  );
  assert.equal(
    getHubPhase(HF),
    "available",
    "an avatar outage must not make the model list claim the Hub is down",
  );

  const original = globalThis.fetch;
  globalThis.fetch = (async () => new Response("[]", { status: 200 })) as typeof fetch;
  try {
    await fetchWithTimeout(`${HF}/api/models`, {}, 1_000, {
      service: "discovery",
    });
  } finally {
    globalThis.fetch = original;
  }
  assert.equal(
    getHubPhase(HF),
    "available",
    "a successful empty search must render the empty state, not a stale error",
  );
  assert.equal(getLastHubFailure(HF), null);
});

test("an auxiliary success does not retire the feed's backoff", async () => {
  markRemoteNetworkOnline();
  const feedFailure = {
    kind: "network-opaque" as const,
    message: "boom",
    origin: HF,
    retryable: true,
  };
  markRemoteNetworkOffline(HF, 30_000, feedFailure, "discovery");
  assert.equal(getHubPhase(HF, "discovery"), "unavailable");

  // A block can be per-path: an avatar or dataset-size request can succeed
  // while /api/models stays blocked, and that is service "other". An
  // origin-wide window let it retire the feed's backoff, so Load more resumed
  // probing a feed whose own failure was still recorded.
  markRemoteNetworkOnline(HF, "other");
  assert.equal(
    getHubPhase(HF, "discovery"),
    "unavailable",
    "the feed's own window is the one that gates the feed",
  );
  assert.equal(getLastHubFailure(HF, "discovery")?.kind, "network-opaque");
  markRemoteNetworkOnline();
});

test("Retry clears every feed's window on the origin", async () => {
  markRemoteNetworkOnline();
  const failure = {
    kind: "network-opaque" as const,
    message: "boom",
    origin: HF,
    retryable: true,
  };
  markRemoteNetworkOffline(HF, 30_000, failure, "discovery");
  markRemoteNetworkOffline(HF, 30_000, failure, "other");
  assert.equal(isRemoteNetworkOffline(HF), true);

  clearRemoteBackoff(HF);
  // An explicit click means "test the network now", for every client.
  assert.equal(isRemoteNetworkOffline(HF), false);
  assert.equal(isRemoteNetworkOffline(HF, "discovery"), false);
  assert.equal(isRemoteNetworkOffline(HF, "other"), false);
  markRemoteNetworkOnline();
});
