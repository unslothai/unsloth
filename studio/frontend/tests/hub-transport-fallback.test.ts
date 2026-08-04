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
const { HubFetchError } = await import("../src/features/hub/lib/network.ts");

const HF_URL = "https://huggingface.co/api/models?search=gemma&limit=100";

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
