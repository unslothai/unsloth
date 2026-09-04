// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { readFile } from "node:fs/promises";

import {
  classifyFetchFailure,
  clearRemoteBackoff,
  fetchWithTimeout,
  getHubPhase,
  getBrowserOfflineRetryDelayMs,
  getLastHubFailure,
  isHuggingFaceOffline,
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
    kind: "timeout",
    message: "timed out",
    origin: HF,
    retryable: true,
  });
  await sleep(30);

  assert.equal(
    getLastHubFailure(HF)?.kind,
    "timeout",
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

test("a slow optional asset does not take the whole origin down", async () => {
  reset();
  const original = globalThis.fetch;
  // Never resolves: fetchWithTimeout's own AbortController fires the timeout.
  globalThis.fetch = ((_input: unknown, init: { signal?: AbortSignal }) =>
    new Promise((_resolve, reject) => {
      init?.signal?.addEventListener("abort", () =>
        reject(new DOMException("aborted", "AbortError")),
      );
    })) as unknown as typeof fetch;
  try {
    await assert.rejects(
      fetchWithTimeout(`${HF}/api/organizations/unsloth/avatar`, {}, 10),
      (err: unknown) => {
        assert.ok(isHubFetchError(err), "the caller still gets the diagnosis");
        assert.equal(err.failure.kind, "timeout");
        return true;
      },
    );
    // The avatar, README and dataset-size fetches all use a 10s timeout against
    // huggingface.co. Arming the 30s window here paused discovery and disabled
    // the metadata and download controls while the API itself was reachable.
    assert.equal(
      getHubPhase(HF),
      "available",
      "one slow endpoint is not evidence the origin is unreachable",
    );
    assert.equal(getLastHubFailure(HF), null, "and records no origin-wide cause");
  } finally {
    globalThis.fetch = original;
  }
});

test("a connectivity failure still does", async () => {
  reset();
  const original = globalThis.fetch;
  globalThis.fetch = (async () => {
    throw new TypeError("Failed to fetch");
  }) as typeof fetch;
  try {
    await assert.rejects(fetchWithTimeout(`${HF}/api/models`, {}, 1_000));
    assert.equal(getHubPhase(HF), "unavailable", "this is the case worth backing off");
    assert.equal(getLastHubFailure(HF)?.kind, "network-opaque");
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

test("the cause on screen describes the window that is in force", async () => {
  markRemoteNetworkOnline();
  const live = {
    kind: "network-opaque" as const,
    message: "boom",
    origin: HF,
    retryable: true,
  };
  markRemoteNetworkOffline(HF, 30_000, live);

  // A concurrent request records a second cause with no window of its own.
  // Taking that cause while keeping the longer window left the panel naming a
  // spent failure while a different, still-live one held it unavailable.
  markRemoteNetworkOffline(
    HF,
    0,
    { kind: "timeout", message: "timed out", origin: HF, retryable: true },
  );
  assert.equal(getHubPhase(HF), "unavailable");
  assert.equal(
    getLastHubFailure(HF)?.kind,
    "network-opaque",
    "the older answer must not explain a newer failure's backoff",
  );
  markRemoteNetworkOnline();
});


test("a generator that threw is not pulled again", async () => {
  // @huggingface/hub awaits the fetch inside the generator body, so a failed
  // page finishes the generator and every later next() resolves done. The
  // auto-fill then cleared the error with it, which undoes the whole point of
  // keeping the failure on screen.
  let requests = 0;
  async function* listing() {
    for (const page of [1, 2]) {
      requests += 1;
      if (page === 2) throw new Error("Failed to fetch");
      yield page;
    }
  }
  const iter = listing();
  assert.deepEqual(await iter.next(), { value: 1, done: false });
  await assert.rejects(iter.next());
  assert.equal((await iter.next()).done, true, "it is finished once it throws");
  assert.equal(requests, 2, "reusing it issues no request, so done is a lie");

  const src = await readFile(
    new URL("../src/features/hub/hooks/use-hub-paginated-search.ts", import.meta.url),
    "utf8",
  );
  // Set on the failure path, cleared only where a new generator is built.
  assert.match(src, /iterDeadRef\.current = true;/);
  assert.match(src, /iterRef\.current = iter;\s*\n\s*iterDeadRef\.current = false;/);
  const at = src.indexOf("const fetchMore = useCallback");
  assert.notEqual(at, -1);
  const guard = src.slice(at, src.indexOf("const iter = iterRef.current;", at));
  assert.match(guard, /if \(iterDeadRef\.current\) \{/);
  assert.ok(
    guard.slice(guard.indexOf("if (iterDeadRef.current) {")).includes("return false;"),
    "a dead feed starts nothing, so its error survives",
  );
});
