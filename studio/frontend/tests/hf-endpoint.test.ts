// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Two things here are easy to get wrong: the module must stay importable under
 * bare node (network.ts imports it, and the tests import that directly), and a
 * reply omitting the fields must not reset a configured mirror.
 */

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

register("./bundler-resolver.mjs", import.meta.url);

const {
  DEFAULT_DATASETS_SERVER,
  DEFAULT_HF_ENDPOINT,
  getHfDatasetsServerBase,
  getHfEndpoint,
  resetHfEndpoints,
  setHfEndpoints,
  setHubSessionRefresh,
  useHfDatasetsServer,
  useHfEndpoint,
} = await import("../src/lib/hf-endpoint.ts");

const {
  isHuggingFaceOffline,
  markRemoteNetworkOffline,
  clearRemoteBackoff,
  fetchWithTimeout,
} = await import("../src/features/hub/lib/network.ts");

test("before /api/health answers, both endpoints are the official ones", () => {
  resetHfEndpoints();
  assert.equal(getHfEndpoint(), "https://huggingface.co");
  assert.equal(
    getHfDatasetsServerBase(),
    "https://datasets-server.huggingface.co",
  );
  assert.equal(DEFAULT_HF_ENDPOINT, "https://huggingface.co");
  assert.equal(
    DEFAULT_DATASETS_SERVER,
    "https://datasets-server.huggingface.co",
  );
});

test("the hooks React subscribes to are the same state the getters read", () => {
  // The hooks are thin useStore wrappers, so what matters is that they read the
  // one store the setter writes, not a second copy kept in sync by hand.
  resetHfEndpoints();
  assert.equal(typeof useHfEndpoint, "function");
  assert.equal(typeof useHfDatasetsServer, "function");
  setHfEndpoints("https://hf-mirror.com", "https://ds.example.com");
  assert.equal(getHfEndpoint(), "https://hf-mirror.com");
  assert.equal(getHfDatasetsServerBase(), "https://ds.example.com");
});

test("a mirror reported by the backend is applied to both getters", () => {
  resetHfEndpoints();
  setHfEndpoints("https://hf-mirror.com", "https://ds.example.com");
  assert.equal(getHfEndpoint(), "https://hf-mirror.com");
  assert.equal(getHfDatasetsServerBase(), "https://ds.example.com");
});

test("a reported endpoint is stored as sent, minus trailing slashes", () => {
  // Already sanitised and canonicalised by the backend; only the trailing slash
  // is dropped, since every consumer builds `${getHfEndpoint()}/path`.
  for (const [raw, expected] of [
    ["https://hf-mirror.com", "https://hf-mirror.com"],
    ["https://hf-mirror.com/", "https://hf-mirror.com"],
    ["https://hf-mirror.com///", "https://hf-mirror.com"],
    ["  https://hf-mirror.com  ", "https://hf-mirror.com"],
    ["http://127.0.0.1:9700", "http://127.0.0.1:9700"],
    ["http://[::1]:9700", "http://[::1]:9700"],
    ["https://hub.internal:8443/hf/", "https://hub.internal:8443/hf"],
    ["https://xn--fsqu00a.xn--0zwm56d", "https://xn--fsqu00a.xn--0zwm56d"],
  ] as const) {
    resetHfEndpoints();
    setHfEndpoints(raw, null);
    assert.equal(getHfEndpoint(), expected, `for ${raw}`);
  }
});

test("an older backend that reports neither field keeps the configured mirror", () => {
  // An older Studio carries no hf_endpoint: resetting would strand a mirror-only
  // deployment on a host it cannot reach.
  resetHfEndpoints();
  setHfEndpoints("https://hf-mirror.com", "https://ds.example.com");
  setHfEndpoints(undefined, undefined);
  assert.equal(getHfEndpoint(), "https://hf-mirror.com");
  assert.equal(getHfDatasetsServerBase(), "https://ds.example.com");
  setHfEndpoints(null, null);
  assert.equal(getHfEndpoint(), "https://hf-mirror.com");
  setHfEndpoints("", "   ");
  assert.equal(getHfEndpoint(), "https://hf-mirror.com");
  assert.equal(getHfDatasetsServerBase(), "https://ds.example.com");
});

test("a value that does not parse as an http(s) URL is ignored", () => {
  // Not a second copy of the policy: only a value that would throw inside
  // `new URL(getHfEndpoint())` is kept out.
  for (const junk of [
    "javascript:alert(1)",
    "file:///etc/passwd",
    "data:text/html,x",
    "ftp://hf-mirror.com",
    "https://",
    "hf-mirror.com",
    "not a url",
  ]) {
    resetHfEndpoints();
    setHfEndpoints(junk, junk);
    assert.equal(getHfEndpoint(), DEFAULT_HF_ENDPOINT, `for ${junk}`);
    assert.equal(getHfDatasetsServerBase(), DEFAULT_DATASETS_SERVER, `for ${junk}`);
  }
});

test("a non-string value cannot crash the getter", () => {
  resetHfEndpoints();
  for (const junk of [42, {}, [], true, () => undefined]) {
    setHfEndpoints(junk as unknown as string, junk as unknown as string);
  }
  assert.equal(getHfEndpoint(), "https://huggingface.co");
});

test("the datasets server stays independent of the hub mirror", () => {
  // Most mirrors do not proxy /splits, so HF_ENDPOINT alone must not redirect it.
  resetHfEndpoints();
  setHfEndpoints("https://hf-mirror.com", undefined);
  assert.equal(getHfEndpoint(), "https://hf-mirror.com");
  assert.equal(
    getHfDatasetsServerBase(),
    "https://datasets-server.huggingface.co",
  );
});

test("the Hub offline backoff keys on the configured mirror, not huggingface.co", () => {
  // The backoff maps key on the request origin: keying a mirror deployment on
  // huggingface.co would report the Hub healthy while every request failed.
  resetHfEndpoints();
  setHfEndpoints("https://hf-mirror.com", undefined);
  clearRemoteBackoff("https://hf-mirror.com");
  assert.equal(isHuggingFaceOffline(), false);
  markRemoteNetworkOffline("https://hf-mirror.com", 60_000, {
    kind: "network-opaque",
    message: "the mirror could not be reached",
    origin: "https://hf-mirror.com",
    retryable: true,
  });
  assert.equal(isHuggingFaceOffline(), true);
  clearRemoteBackoff("https://hf-mirror.com");
  assert.equal(isHuggingFaceOffline(), false);
  resetHfEndpoints();
});

test("the ModelScope adapter gets the Unsloth session, never the Hugging Face token", async () => {
  const adapter = "http://127.0.0.1:8888/api/hub/modelscope";
  const sent: (string | null)[] = [];
  let session: string | null = "expired";
  const realFetch = globalThis.fetch;
  globalThis.fetch = (async (_input: unknown, init?: RequestInit) => {
    const auth = new Headers(init?.headers).get("authorization");
    sent.push(auth);
    return new Response("[]", { status: auth === "Bearer expired" ? 401 : 200 });
  }) as typeof fetch;
  Object.defineProperty(globalThis, "localStorage", {
    configurable: true,
    value: { getItem: (key: string) => (key === "unsloth_auth_token" ? session : null) },
  });
  setHubSessionRefresh(async () => {
    session = "fresh";
    return true;
  });
  const hfToken = { headers: { Authorization: "Bearer hf_x" } };
  try {
    setHfEndpoints(adapter, undefined, "modelscope");
    const answer = await fetchWithTimeout(`${adapter}/api/models`, hfToken);
    await fetchWithTimeout("https://cdn.example.com/a.png", hfToken);
    setHfEndpoints("https://huggingface.co", undefined, "huggingface");
    await fetchWithTimeout("https://huggingface.co/api/models", hfToken);
    await fetchWithTimeout(`${adapter}/api/models/a/b`, hfToken);
    session = null;
    await fetchWithTimeout(`${adapter}/api/models/a/c`, hfToken);
    assert.equal(answer.status, 200);
  } finally {
    globalThis.fetch = realFetch;
    Reflect.deleteProperty(globalThis, "localStorage");
    setHubSessionRefresh(async () => false);
    resetHfEndpoints();
  }
  assert.deepEqual(sent, [
    "Bearer expired",
    "Bearer fresh",
    "Bearer hf_x",
    "Bearer hf_x",
    "Bearer fresh",
    null,
  ]);
});

test("a relay to a custom endpoint gets the session, and the Hugging Face token beside it", async () => {
  const relay = "http://127.0.0.1:8888/api/hub/proxy";
  const datasets = "http://127.0.0.1:8888/api/hub/datasets-server-proxy";
  const sent: (string | null)[][] = [];
  let refreshes = 0;
  const realFetch = globalThis.fetch;
  globalThis.fetch = (async (input: string, init?: RequestInit) => {
    const headers = new Headers(init?.headers);
    sent.push([headers.get("authorization"), headers.get("x-hf-authorization")]);
    const gated = input.includes("gated");
    const stale = input.includes("stale") && headers.get("authorization") === "Bearer session";
    return new Response("[]", { status: gated || stale ? 401 : 200, headers: gated ? { "X-Hub-Upstream": "1" } : {} });
  }) as typeof fetch;
  let session = "session";
  Object.defineProperty(globalThis, "localStorage", { configurable: true, value: { getItem: () => session } });
  setHubSessionRefresh(async () => Boolean((session = `fresh${++refreshes}`)));
  const hfToken = { headers: { Authorization: "Bearer hf_x" } };
  try {
    setHfEndpoints(relay, datasets, "huggingface", { endpoint: true, datasetsServer: true });
    await fetchWithTimeout(`${relay}/api/models`, hfToken);
    await fetchWithTimeout(`${datasets}/splits?dataset=a/b`, {});
    assert.equal((await fetchWithTimeout(`${relay}/api/models/org/gated`, hfToken)).status, 401);
    assert.equal((await fetchWithTimeout(`${relay}/api/models/stale`, hfToken)).status, 200);
    setHfEndpoints("https://huggingface.co", "https://datasets-server.huggingface.co");
    await fetchWithTimeout(`${datasets}/splits?dataset=a/c`, hfToken);
    await fetchWithTimeout("https://huggingface.co/api/models", hfToken);
  } finally {
    globalThis.fetch = realFetch;
    Reflect.deleteProperty(globalThis, "localStorage");
    setHubSessionRefresh(async () => false);
    resetHfEndpoints();
  }
  assert.equal(refreshes, 1);
  const both = ["Bearer session", "Bearer hf_x"];
  const fresh = ["Bearer fresh1", "Bearer hf_x"];
  assert.deepEqual(sent, [both, ["Bearer session", null], both, both, fresh, fresh, ["Bearer hf_x", null]]);
});
