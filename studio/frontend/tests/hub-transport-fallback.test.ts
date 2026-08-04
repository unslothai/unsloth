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

  // In mirror mode it goes to the path-preserving info route. What it must
  // never reach is the listing route, which drops the pathname: modelInfo would
  // parse a listing array as one repo and cache a model with no id or name.
  for (const call of backend.calls) {
    assert.ok(
      !call.url.startsWith("/api/hub/discovery/"),
      `model-info must not hit the listing route (got ${call.url})`,
    );
  }
  assert.deepEqual(seen, []);
});

test("a model-info transport failure is not retried through the proxy", async () => {
  const backend = captureBackend();
  const transport = createHubTransport("models", {
    direct: failWith("network-opaque"),
    backend: backend.backend,
  });

  await assert.rejects(transport(MODEL_INFO_URL, {}));
  assert.equal(backend.calls.length, 0);
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
  // treating it as authoritative would strand exactly the users the fallback
  // exists for.
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
  assert.equal(
    isHuggingFaceOffline(),
    false,
    "marking offline before the fallback returns is what cancelled it",
  );
  assert.equal(backend.calls.length, 1);
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
