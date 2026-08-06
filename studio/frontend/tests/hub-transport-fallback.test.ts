// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

// hub-transport.ts resolves siblings extensionlessly, so teach the loader the
// bundler's rules before importing it.
registerBundlerResolver();

const { createHubTransport } = await import(
  "../src/features/hub/lib/hub-transport.ts"
);
const {
  HubFetchError,
  getHubPhase,
  isRemoteNetworkOffline,
  getLastHubFailure,
  isDirectHubOffline,
  isHubProxyServing,
  isHuggingFaceOffline,
  markRemoteNetworkOffline,
  markRemoteNetworkOnline,
} = await import("../src/features/hub/lib/network.ts");

const HF_URL = "https://huggingface.co/api/models?search=gemma&limit=100";
const HF_ORIGIN = "https://huggingface.co";
// What modelInfo() asks for: the path, not the query, carries the meaning.
const MODEL_INFO_URL =
  "https://huggingface.co/api/models/unsloth/gemma-3-4b-it/revision/HEAD?expand=tags";

function failWith(kind: string) {
  return async () => {
    throw new HubFetchError({
      kind: kind as never,
      message: "boom",
      origin: "https://huggingface.co",
      retryable: true,
    });
  };
}

function captureBackend() {
  const calls: { url: string; headers: Headers }[] = [];
  const backend = async (
    input: Parameters<typeof fetch>[0],
    init: Parameters<typeof fetch>[1],
  ) => {
    calls.push({ url: String(input), headers: new Headers(init?.headers) });
    return new Response("[]", { status: 200 });
  };
  return { calls, backend };
}

test("an opaque browser failure falls back to the backend", async () => {
  const backend = captureBackend();
  {
    const transport = createHubTransport("models", {
      direct: failWith("network-opaque"),
      backend: backend.backend,
    });
    const res = await transport(HF_URL, {});
    assert.equal(res.status, 200);
    assert.equal(backend.calls.length, 1);
    assert.ok(
      backend.calls[0].url.includes("/api/hub/discovery/models"),
      "should retry through the same-origin discovery route",
    );
    assert.ok(
      backend.calls[0].url.includes("search=gemma"),
      "the original query must survive the fallback",
    );
  }
});

test("a CSP block falls back too", async () => {
  const backend = captureBackend();
  {
    const transport = createHubTransport("models", {
      direct: failWith("csp-blocked"),
      backend: backend.backend,
    });
    await transport(HF_URL, {});
    assert.equal(backend.calls.length, 1);
  }
});

test("a cancelled request is never retried through the backend", async () => {
  const backend = captureBackend();
  {
    const transport = createHubTransport("models", {
      direct: failWith("aborted"),
      backend: backend.backend,
    });
    await assert.rejects(transport(HF_URL, {}));
    assert.equal(
      backend.calls.length,
      0,
      "a superseded query must not generate a second request",
    );
  }
});

test("an HTTP status response is returned as-is, never re-sent", async () => {
  const backend = captureBackend();
  {
    // fetchWithTimeout resolves for real HTTP statuses, so 429 reaches the SDK
    // untouched. Retrying it through the proxy would just double the traffic.
    const transport = createHubTransport("models", {
      direct: async () => new Response("rate limited", { status: 429 }),
      backend: backend.backend,
    });
    const res = await transport(HF_URL, {});
    assert.equal(res.status, 429);
    assert.equal(backend.calls.length, 0);
  }
});

test("the HF token moves out of Authorization into the internal header", async () => {
  const backend = captureBackend();
  {
    const transport = createHubTransport("models", {
      direct: failWith("network-opaque"),
      backend: backend.backend,
    });
    await transport(HF_URL, {
      headers: { Authorization: "Bearer hf_secrettoken" },
    });
    const { headers } = backend.calls[0];
    assert.equal(headers.get("X-Unsloth-HF-Token"), "hf_secrettoken");
    assert.equal(
      headers.get("Authorization"),
      null,
      "the HF token must not occupy the Studio session's Authorization slot",
    );
  }
});

test("once fallen back, later pages stay on the backend", async () => {
  const backend = captureBackend();
  let directCalls = 0;
  {
    const transport = createHubTransport("models", {
      direct: async () => {
        directCalls += 1;
        throw new HubFetchError({
          kind: "network-opaque",
          message: "boom",
          origin: "https://huggingface.co",
          retryable: true,
        });
      },
      backend: backend.backend,
    });
    await transport(HF_URL, {});
    await transport(HF_URL, {});
    assert.equal(
      directCalls,
      1,
      "the direct route must not be re-probed on every page",
    );
    assert.equal(backend.calls.length, 2);
  }
});

test("a relative next-page link goes straight to the backend", async () => {
  const backend = captureBackend();
  {
    const transport = createHubTransport("models", {
      direct: async () => {
        throw new Error("direct transport must not be used for a proxy URL");
      },
      backend: backend.backend,
    });
    await transport("/api/hub/discovery/models?search=gemma&p=1", {});
    assert.equal(backend.calls.length, 1);
    assert.ok(backend.calls[0].url.includes("search=gemma"));
  }
});

// ---------------------------------------------------------------------------
// The fallback must not leave the Hub marked unavailable
// ---------------------------------------------------------------------------

const OPAQUE_FAILURE = {
  kind: "network-opaque" as const,
  message: "boom",
  origin: HF_ORIGIN,
  retryable: true,
};

/** Stands in for fetchWithTimeout, which records the failure before throwing. */
function failAndMarkOffline() {
  return async () => {
    markRemoteNetworkOffline(HF_ORIGIN, 30_000, OPAQUE_FAILURE);
    throw new HubFetchError(OPAQUE_FAILURE);
  };
}

test("a served page does not tell the direct clients the origin is reachable", async () => {
  markRemoteNetworkOnline();
  const backend = captureBackend();
  const transport = createHubTransport("models", {
    direct: failAndMarkOffline(),
    backend: backend.backend,
  });
  await transport(HF_URL, {});

  // The catalog may use the proxied feed...
  assert.equal(getHubPhase(HF_ORIGIN), "available");
  // ...but README and owner-avatar fetches still go direct, so telling them the
  // origin is back would have them fail and re-mark it, restoring the flapping.
  assert.equal(
    isHuggingFaceOffline(),
    true,
    "a backend success must not promote browser-origin reachability",
  );
});

test("a page served by the backend clears the recorded Hub failure", async () => {
  markRemoteNetworkOnline();
  try {
    const backend = captureBackend();
    const transport = createHubTransport("models", {
      direct: failAndMarkOffline(),
      backend: backend.backend,
    });

    const res = await transport(HF_URL, {});

    assert.equal(res.status, 200);
    assert.equal(
      getHubPhase(HF_ORIGIN),
      "available",
      "results arriving through the proxy must not read as an unreachable " +
        "Hub: useDiscoverSearch refuses fetchMore and renders an empty page " +
        "as a connection error while that failure stands",
    );
  } finally {
    markRemoteNetworkOnline();
  }
});

test("a failed backend fallback keeps the recorded Hub failure", async () => {
  markRemoteNetworkOnline();
  try {
    const transport = createHubTransport("models", {
      direct: failAndMarkOffline(),
      backend: async () => new Response("no", { status: 502 }),
    });

    const res = await transport(HF_URL, {});

    assert.equal(res.status, 502);
    assert.equal(
      getHubPhase(HF_ORIGIN),
      "unavailable",
      "the diagnosis must survive when the server cannot reach the Hub either",
    );
  } finally {
    markRemoteNetworkOnline();
  }
});

// ---------------------------------------------------------------------------
// The proxy is listing-only: a path-bearing request must keep its path
// ---------------------------------------------------------------------------

test("a model-info request is never retargeted at the listing proxy", async () => {
  const backend = captureBackend();
  const seen: string[] = [];
  const transport = createHubTransport("models", {
    direct: async (input) => {
      seen.push(String(input));
      return new Response("{}", { status: 200 });
    },
    backend: backend.backend,
    // Mirror mode: every listing request goes to the proxy up front.
    proxyFirst: () => true,
  });

  await transport(MODEL_INFO_URL, {});

  // Never the listing route, which drops the pathname: modelInfo would parse
  // a listing array as one repo and cache a model with no id.
  for (const call of backend.calls) {
    assert.ok(
      !call.url.startsWith("/api/hub/discovery/"),
      `model-info must not hit the listing route (got ${call.url})`,
    );
  }
  assert.deepEqual(seen, []);
});

test("a model-info transport failure falls back, but never onto the listing", async () => {
  const backend = captureBackend();
  const transport = createHubTransport("models", {
    direct: failWith("network-opaque"),
    backend: backend.backend,
  });

  // A blocked browser must not leave deep links and pinned cards without
  // metadata while the feed itself works through the same server.
  await transport(MODEL_INFO_URL, {});
  assert.equal(backend.calls.length, 1);
  const { url } = backend.calls[0];
  assert.ok(url.includes("/api/hub/discovery-info/models"));
  // The listing route answers with an array, so a repo cached from it would
  // have no id at all.
  assert.ok(!url.includes("/api/hub/discovery/models"));
  assert.ok(url.includes("repo=unsloth%2Fgemma-3-4b-it"));
});

// ---------------------------------------------------------------------------
// Pagination: the backend's next-page link is absolute
// ---------------------------------------------------------------------------

test("an absolute next-page link back into the proxy goes to the backend", async () => {
  const backend = captureBackend();
  const transport = createHubTransport("models", {
    direct: async () => {
      throw new Error("direct transport must not be used for a proxy URL");
    },
    backend: backend.backend,
  });

  // The backend emits an absolute link because @huggingface/hub's
  // parseLinkHeader only matches <http(s)://...>.
  await transport(
    "http://studio.local:1234/api/hub/discovery/models?search=gemma&cursor=abc123",
    {},
  );

  assert.equal(backend.calls.length, 1);
  assert.ok(
    backend.calls[0].url.startsWith("/api/hub/discovery/models?"),
    `expected a same-origin proxy request, got ${backend.calls[0].url}`,
  );
  assert.ok(backend.calls[0].url.includes("cursor=abc123"));
});

test("a stuck-false navigator.onLine does not veto the fallback", async () => {
  const backend = captureBackend();
  // navigator.onLine reads false on WSL2 and some Tauri/WebKitGTK webviews, so
  // trusting it would strand exactly the users the fallback exists for.
  const transport = createHubTransport("models", {
    direct: failWith("browser-offline"),
    backend: backend.backend,
  });
  const res = await transport(HF_URL, {});
  assert.equal(res.status, 200);
  assert.equal(
    backend.calls.length,
    1,
    "a browser-reported offline state must still try the server",
  );
});

const INFO_URL =
  "https://huggingface.co/api/models/unsloth/gemma-3-4b-it/revision/HEAD?expand=gguf";

test("with a mirror configured, model-info goes to the path-preserving route", async () => {
  const backend = captureBackend();
  const transport = createHubTransport("models", {
    direct: async () => {
      throw new Error("a mirror user's model-info must not hit the public Hub");
    },
    backend: backend.backend,
    proxyFirst: () => true,
  });
  await transport(INFO_URL, {});
  assert.equal(backend.calls.length, 1);
  const url = backend.calls[0].url;
  assert.ok(url.startsWith("/api/hub/discovery-info/models"), url);
  assert.ok(
    url.includes("repo=unsloth%2Fgemma-3-4b-it"),
    "the repo must survive; the listing route would drop it",
  );
  assert.ok(url.includes("expand=gguf"));
});

test("without a mirror, model-info stays on the direct route", async () => {
  const backend = captureBackend();
  let direct = 0;
  const transport = createHubTransport("models", {
    direct: async () => {
      direct += 1;
      return new Response("{}", { status: 200 });
    },
    backend: backend.backend,
    proxyFirst: () => false,
  });
  await transport(INFO_URL, {});
  assert.equal(direct, 1);
  assert.equal(backend.calls.length, 0);
});

test("the shared model-info fetch honours a configured mirror", async () => {
  const backend = captureBackend();
  const transport = createHubTransport("models", {
    direct: async () => {
      throw new Error("a mirror user's model-info must not hit the public Hub");
    },
    backend: backend.backend,
    proxyFirst: () => true,
  });
  // Same path shape useSelectedModelMetadata produces for a deep-linked repo.
  await transport(
    "https://huggingface.co/api/models/unsloth/gemma-3-4b-it/revision/HEAD",
    {},
  );
  assert.equal(backend.calls.length, 1);
  assert.ok(
    backend.calls[0].url.startsWith("/api/hub/discovery-info/models"),
    backend.calls[0].url,
  );
});

test("proxy-served availability is distinguishable from a real reconnect", async () => {
  markRemoteNetworkOnline();
  const backend = captureBackend();
  const transport = createHubTransport("models", {
    direct: failAndMarkOffline(),
    backend: backend.backend,
  });
  await transport(HF_URL, {});

  assert.equal(getHubPhase(HF_ORIGIN), "available");
  assert.equal(
    isHubProxyServing(),
    true,
    "the reconnect toast and retry must not fire off the proxy standing in: " +
      "a fresh transport has no affinity and would re-attempt the blocked direct request",
  );
});

test("a pending fallback is not aborted by an offline re-render", async () => {
  markRemoteNetworkOnline();
  const backend = captureBackend();
  const transport = createHubTransport("models", {
    // The real fetchWithTimeout is told not to record, so nothing marks the
    // origin offline mid-flight and no consumer disables and aborts us.
    direct: failWith("network-opaque"),
    backend: backend.backend,
  });
  await transport(HF_URL, {});
  // The TTL is the thing that must stay unset: recording it mid-flight
  // re-renders consumers, and their cleanup aborts the fallback in progress.
  assert.equal(
    isRemoteNetworkOffline(HF_ORIGIN),
    false,
    "marking offline before the fallback returns is what cancelled it",
  );
  assert.equal(getHubPhase(HF_ORIGIN), "available", "the feed is being served");
  assert.equal(backend.calls.length, 1);
  // Separately, the direct clients must stand down: this browser demonstrably
  // cannot reach the origin, so a README or avatar fetch would only fail and
  // drop the selected row's metadata.
  assert.equal(
    isHuggingFaceOffline(),
    true,
    "a serving proxy is proof the direct route is blocked",
  );
  markRemoteNetworkOnline();
});

test("both routes failing does mark the origin offline", async () => {
  markRemoteNetworkOnline();
  const transport = createHubTransport("models", {
    direct: failWith("network-opaque"),
    backend: async () => {
      throw new Error("backend unreachable");
    },
  });
  await assert.rejects(transport(HF_URL, {}));
  assert.equal(isHuggingFaceOffline(), true);
});

test("a non-OK fallback records the direct failure instead of losing it", async () => {
  markRemoteNetworkOnline();
  const transport = createHubTransport("models", {
    direct: failWith("network-opaque"),
    // authFetch resolves on a non-2xx, so nothing threw and the classified
    // cause used to be dropped, leaving the panel with no diagnosis.
    backend: async () => new Response("bad gateway", { status: 502 }),
  });
  const res = await transport(HF_URL, {});
  assert.equal(res.status, 502);
  assert.equal(getHubPhase(HF_ORIGIN), "unavailable");
  assert.equal(getLastHubFailure(HF_ORIGIN)?.kind, "network-opaque");
});

test("a caller abort during the fallback does not start a backoff", async () => {
  markRemoteNetworkOnline();
  const controller = new AbortController();
  const transport = createHubTransport("models", {
    direct: failWith("network-opaque"),
    backend: async () => {
      // What a superseded query looks like: the caller walked away, the proxy
      // never failed, so this must not disable the replacement request.
      controller.abort();
      throw new DOMException("aborted", "AbortError");
    },
  });
  await assert.rejects(transport(HF_URL, { signal: controller.signal }));
  assert.equal(
    getHubPhase(HF_ORIGIN),
    "available",
    "an abort is not evidence that both routes are unavailable",
  );
});

test("a rejecting backend clears proxy availability instead of going stale", async () => {
  markRemoteNetworkOnline();
  let fail = false;
  const transport = createHubTransport("models", {
    direct: failWith("network-opaque"),
    backend: async () => {
      if (fail) throw new Error("studio server restarted");
      return new Response("[]", { status: 200 });
    },
  });
  await transport(HF_URL, {});
  assert.equal(getHubPhase(HF_ORIGIN), "available");

  fail = true;
  await assert.rejects(transport(HF_URL, {}));
  assert.notEqual(
    getHubPhase(HF_ORIGIN),
    "available",
    "a stale proxy flag would keep reporting a Hub served by a proxy that is gone",
  );
});

test("a later proxied page still carries the original direct failure", async () => {
  markRemoteNetworkOnline();
  let ok = true;
  const transport = createHubTransport("models", {
    direct: failWith("csp-blocked"),
    backend: async () =>
      ok
        ? new Response("[]", { status: 200 })
        : new Response("bad gateway", { status: 502 }),
  });
  await transport(HF_URL, {});
  ok = false;
  // Page two: no failure of its own, and the direct one was never recorded.
  await transport("/api/hub/discovery/models?search=gemma&cursor=x", {});
  assert.equal(getLastHubFailure(HF_ORIGIN)?.kind, "csp-blocked");
});

test("a direct listing success retires the proxy-serving flag", async () => {
  markRemoteNetworkOnline();
  const backend = captureBackend();

  // A first transport falls back, so the backend is serving the feed.
  const fallen = createHubTransport("models", {
    direct: failWith("network-opaque"),
    backend: backend.backend,
  });
  await fallen(HF_URL, {});
  assert.equal(isHubProxyServing(), true);

  // A later search reaches the Hub itself, so the proxy is out of it.
  const recovered = createHubTransport("models", {
    direct: async () => new Response("[]", { status: 200 }),
    backend: backend.backend,
  });
  await recovered(HF_URL, {});
  assert.equal(
    isHubProxyServing(),
    false,
    "a stale flag keeps forcing the phase to available off an idle proxy",
  );

  // Why it matters: the next direct failure is visible again.
  markRemoteNetworkOffline(
    HF_ORIGIN,
    30_000,
    {
      kind: "csp-blocked",
      message: "blocked",
      origin: HF_ORIGIN,
      retryable: true,
    },
    "discovery",
  );
  assert.equal(getHubPhase(HF_ORIGIN), "unavailable");
  assert.equal(getLastHubFailure(HF_ORIGIN)?.kind, "csp-blocked");
  markRemoteNetworkOnline();
});

test("a failed later page is filed under the Hub, not the Studio origin", async () => {
  markRemoteNetworkOnline();
  let served = 0;
  const backend = async (
    _input: Parameters<typeof fetch>[0],
    _init: Parameters<typeof fetch>[1],
  ) => {
    served += 1;
    // Page one falls back cleanly; the next page finds the proxy gone too.
    if (served === 1) return new Response("[]", { status: 200 });
    throw new TypeError("Failed to fetch");
  };
  const transport = createHubTransport("models", {
    direct: failWith("network-opaque"),
    backend,
  });

  await transport(HF_URL, {});
  // What the backend hands back is absolute and same-origin, so its origin is
  // Studio's; filing the Hub's failure there would hide it from the panel.
  await assert.rejects(
    transport("http://127.0.0.1:8888/api/hub/discovery/models?p=1", {}),
  );

  assert.equal(
    getLastHubFailure(HF_ORIGIN)?.kind,
    "network-opaque",
    "the cause must be readable for the Hub, which is what the catalog asks about",
  );
  assert.equal(getHubPhase(HF_ORIGIN), "unavailable");
  markRemoteNetworkOnline();
});

test("a repo lookup that succeeds does not retire the feed's diagnosis", async () => {
  markRemoteNetworkOnline();
  markRemoteNetworkOffline(
    HF_ORIGIN,
    30_000,
    { kind: "csp-blocked", message: "blocked", origin: HF_ORIGIN, retryable: true },
    "discovery",
  );
  // The transport decides the tag from the URL, so a caller cannot mislabel a
  // repo-path lookup as the feed. Whatever tag arrives here is the real one.
  let seen: string | undefined;
  const transport = createHubTransport("models", {
    direct: async (_input, _init, service) => {
      seen = service;
      return new Response("{}", { status: 200 });
    },
    backend: async () => new Response("[]", { status: 200 }),
  });

  await transport(MODEL_INFO_URL, {});
  assert.equal(seen, "info", "a /revision/ lookup is neither the feed nor an asset");
  assert.equal(
    getLastHubFailure(HF_ORIGIN)?.kind,
    "csp-blocked",
    "a detail-pane success must not erase why the catalog failed",
  );
  // Nor may its own failure suppress the card and avatar clients, which is what
  // "other" would have done for the 30s after a lookup nobody was waiting on.
  markRemoteNetworkOnline();
  markRemoteNetworkOffline(
    HF_ORIGIN,
    30_000,
    { kind: "timeout", message: "timed out", origin: HF_ORIGIN, retryable: true },
    "info",
  );
  assert.equal(isDirectHubOffline(), false, "cards and avatars keep fetching");
  markRemoteNetworkOnline();
});

test("a mirror-only failure is still recorded, with no saved direct failure", async () => {
  markRemoteNetworkOnline();
  // proxyFirst means the direct route is never attempted, so there is no saved
  // failure to carry: without a stand-in nothing is ever recorded on a mirror.
  const transport = createHubTransport("models", {
    direct: async () => {
      throw new Error("the direct route must not be used on a mirror");
    },
    backend: async () => new Response("bad gateway", { status: 502 }),
    proxyFirst: () => true,
  });

  await transport(HF_URL, {});
  const failure = getLastHubFailure(HF_ORIGIN);
  // 502 is the backend's own collapse of "could not reach the Hub", not a status
  // the Hub sent, so it is reported as a reachability failure and the number is
  // withheld rather than titled "Hugging Face returned 502".
  assert.equal(failure?.kind, "network-opaque");
  assert.equal(failure?.status, undefined);
  assert.equal(
    failure?.origin,
    null,
    "the operator's internal mirror hostname is not ours to put on screen",
  );
  assert.equal(getHubPhase(HF_ORIGIN), "unavailable", "a mirror needs a backoff too");
  markRemoteNetworkOnline();
});

test("an older backend's SPA 404 is not reported as a Hub failure", async () => {
  markRemoteNetworkOnline();
  let backendCalls = 0;
  const transport = createHubTransport("models", {
    direct: failWith("network-opaque"),
    // No marker header: this is the SPA catch-all, not the proxy route.
    backend: async () => {
      backendCalls += 1;
      return new Response("<!doctype html>", { status: 404 });
    },
  });

  const res = await transport(HF_URL, {});
  assert.equal(res.status, 404);
  assert.equal(
    getLastHubFailure(HF_ORIGIN),
    null,
    "a backend without the route is a deployment fact, not a Hub outage",
  );

  // And the fallback is not offered again on this transport.
  await assert.rejects(transport(HF_URL, {}));
  assert.equal(backendCalls, 1, "no point re-asking a backend that has no route");
  markRemoteNetworkOnline();
});

test("a stamped 404 is a real Hub answer and is reported", async () => {
  markRemoteNetworkOnline();
  const transport = createHubTransport("models", {
    direct: failWith("network-opaque"),
    backend: async () =>
      new Response("{}", {
        status: 404,
        headers: { "X-Unsloth-HF-Proxy": "1" },
      }),
  });

  await transport(HF_URL, {});
  // The server got an answer, so it reports the answer rather than the browser's
  // own failure to ask.
  const failure = getLastHubFailure(HF_ORIGIN);
  assert.equal(failure?.kind, "http");
  assert.equal(failure?.status, 404);
  markRemoteNetworkOnline();
});

test("a served model-info fallback suppresses the direct clients", async () => {
  markRemoteNetworkOnline();
  const transport = createHubTransport("models", {
    direct: failWith("csp-blocked"),
    backend: async () => new Response("{}", { status: 200 }),
  });

  await transport(MODEL_INFO_URL, {});
  assert.equal(
    isHuggingFaceOffline(),
    true,
    "this is what stops README and avatar clients hitting the blocked origin",
  );
  assert.equal(
    isHubProxyServing(),
    false,
    "a repo lookup says nothing about the catalog feed",
  );
  markRemoteNetworkOnline();
});

test("a missing repo does not retire a proxy the listing is using", async () => {
  markRemoteNetworkOnline();
  const backend = captureBackend();
  const listing = createHubTransport("models", {
    direct: failWith("network-opaque"),
    backend: backend.backend,
  });
  await listing(HF_URL, {});
  assert.equal(isHubProxyServing(), true);

  // 404 here means the repo is gone, not that the proxy is: the marker says the
  // route answered. Without it this exercises the SPA-catch-all branch instead.
  const info = createHubTransport("models", {
    direct: failWith("network-opaque"),
    backend: async () =>
      new Response("{}", {
        status: 404,
        headers: { "X-Unsloth-HF-Proxy": "1" },
      }),
  });
  await info(MODEL_INFO_URL, {});
  assert.equal(isHubProxyServing(), true, "the listing path owns demotion");
  markRemoteNetworkOnline();
});

test("the backend's own remap is not attributed to Hugging Face", async () => {
  markRemoteNetworkOnline();
  // 424 is what the proxy returns for an upstream 401/403, because a real 401
  // would take authFetch's session-expiry path and log the user out.
  const transport = createHubTransport("models", {
    direct: failWith("network-opaque"),
    backend: async () =>
      new Response("{}", { status: 424, headers: { "X-Unsloth-HF-Proxy": "1" } }),
  });

  await transport(HF_URL, {});
  const failure = getLastHubFailure(HF_ORIGIN);
  assert.equal(failure?.kind, "http");
  assert.match(failure?.message ?? "", /rejected the access token/);
  assert.equal(
    failure?.status,
    undefined,
    "424 is ours, so the panel must not title it as the Hub's answer",
  );
  markRemoteNetworkOnline();
});

test("a status the Hub answered directly with reaches the panel", async () => {
  markRemoteNetworkOnline();
  // The SDK raises this as an ApiError and fetchWithTimeout clears the origin on
  // any resolved response, so without an explicit record the http panel is dead
  // code on the direct path.
  const transport = createHubTransport("models", {
    // fetchWithTimeout clears the origin on any resolved response, so the stub
    // has to do it too or the ordering the fix depends on is never exercised.
    direct: async () => {
      markRemoteNetworkOnline(HF_ORIGIN);
      return new Response("rate limited", { status: 429 });
    },
    backend: async () => {
      throw new Error("a real status must not trigger the fallback");
    },
  });

  const res = await transport(HF_URL, {});
  assert.equal(res.status, 429);
  const failure = getLastHubFailure(HF_ORIGIN);
  assert.equal(failure?.kind, "http");
  assert.equal(failure?.status, 429);
  assert.equal(failure?.origin, HF_ORIGIN, "filed under the key it is read by");
  // A status proves the origin is reachable, so the cause is recorded without a
  // backoff: "unavailable" here would stall Load more and push the README,
  // avatar and quant clients into offline mode over a bad query.
  assert.equal(getHubPhase(HF_ORIGIN), "probing");
  assert.equal(isRemoteNetworkOffline(HF_ORIGIN), false);
  assert.equal(isHuggingFaceOffline(), false);
  markRemoteNetworkOnline();
});

test("a proxied detail-pane success does not erase why the catalog failed", async () => {
  markRemoteNetworkOnline();
  // The catalog's own feed is down and the panel is naming the cause.
  markRemoteNetworkOffline(HF_ORIGIN, 30_000, OPAQUE_FAILURE);
  assert.equal(getHubPhase(HF_ORIGIN), "unavailable");

  // A repo the user selected is served by the backend instead. That proves this
  // browser is blocked, not that the feed came back.
  const info = createHubTransport("models", {
    direct: failWith("timeout"),
    backend: async () => new Response("{}", { status: 200 }),
  });
  await info(MODEL_INFO_URL, {});

  assert.equal(
    getHubPhase(HF_ORIGIN),
    "unavailable",
    "a detail-pane request must not report the feed available",
  );
  assert.equal(getLastHubFailure(HF_ORIGIN)?.kind, "network-opaque");
  // ...but the direct clients still have to stay off the blocked origin.
  assert.equal(isHuggingFaceOffline(), true);
  markRemoteNetworkOnline();
});

test("a direct detail-pane success releases the blocked-browser suppression", async () => {
  markRemoteNetworkOnline();
  const blocked = createHubTransport("models", {
    direct: failWith("timeout"),
    backend: async () => new Response("{}", { status: 200 }),
  });
  await blocked(MODEL_INFO_URL, {});
  assert.equal(isHuggingFaceOffline(), true);

  // The flag has no expiry, so without this a single transient timeout would
  // keep README, avatars and quant listings on cached data for the session.
  const recovered = createHubTransport("models", {
    direct: async () => new Response("{}", { status: 200 }),
    backend: async () => {
      throw new Error("the direct route succeeded, so there is nothing to proxy");
    },
  });
  await recovered(MODEL_INFO_URL, {});
  assert.equal(isHuggingFaceOffline(), false);
  markRemoteNetworkOnline();
});

test("a Studio-side 5xx does not outrank the browser's own cause", async () => {
  markRemoteNetworkOnline();
  // Only an upstream 5xx is collapsed to 502; a 503 from a proxy in front of
  // Studio is a fact about our stack, so it must not be read as the Hub's
  // answer and must not discard the actionable csp-blocked diagnosis.
  const transport = createHubTransport("models", {
    direct: failWith("csp-blocked"),
    backend: async () =>
      new Response("gateway", { status: 503, headers: { "X-Unsloth-HF-Proxy": "1" } }),
  });

  await transport(HF_URL, {});
  assert.equal(getLastHubFailure(HF_ORIGIN)?.kind, "csp-blocked");
  markRemoteNetworkOnline();
});

test("an earlier backoff does not survive a status the origin answered", async () => {
  markRemoteNetworkOnline();
  // The production path: fetchWithTimeout clears the origin on any resolved
  // response, so a window opened before the answer is already gone by the time
  // the status is recorded. Nothing in the transport needs to clear it.
  markRemoteNetworkOffline(HF_ORIGIN, 30_000, OPAQUE_FAILURE);
  const transport = createHubTransport("models", {
    direct: async () => {
      markRemoteNetworkOnline(HF_ORIGIN);
      return new Response("not found", { status: 404 });
    },
    backend: async () => {
      throw new Error("a real status must not trigger the fallback");
    },
  });

  await transport(HF_URL, {});
  assert.equal(getLastHubFailure(HF_ORIGIN)?.status, 404);
  assert.equal(getHubPhase(HF_ORIGIN), "probing");
  markRemoteNetworkOnline();
});

test("a newer concurrent failure keeps the origin backed off", async () => {
  markRemoteNetworkOnline();
  const transport = createHubTransport("models", {
    direct: async () => {
      markRemoteNetworkOnline(HF_ORIGIN);
      // A second request fails at the network level after this one answered.
      // That is newer evidence than our status, and the transport must not
      // retire it: the direct README and avatar clients rely on that window.
      markRemoteNetworkOffline(HF_ORIGIN, 30_000, OPAQUE_FAILURE);
      return new Response("not found", { status: 404 });
    },
    backend: async () => {
      throw new Error("a real status must not trigger the fallback");
    },
  });

  await transport(HF_URL, {});
  assert.equal(
    getHubPhase(HF_ORIGIN),
    "unavailable",
    "a live network failure outranks an older status",
  );
  markRemoteNetworkOnline();
});

test("an older backend's SPA 404 is recognised on the info route too", async () => {
  markRemoteNetworkOnline();
  let backendCalls = 0;
  const transport = createHubTransport("models", {
    direct: failWith("network-opaque"),
    // No marker: the catch-all answered, so this backend has no info route.
    backend: async () => {
      backendCalls += 1;
      return new Response("<!doctype html>", { status: 404 });
    },
  });

  const res = await transport(MODEL_INFO_URL, {});
  assert.equal(res.status, 404);
  assert.equal(
    isHubProxyServing(),
    false,
    "a missing route is not the backend serving us",
  );
  // And the route is not offered again, rather than reporting the repo gone.
  await assert.rejects(transport(MODEL_INFO_URL, {}));
  assert.equal(backendCalls, 1, "no point re-asking a backend that has no route");
  markRemoteNetworkOnline();
});
