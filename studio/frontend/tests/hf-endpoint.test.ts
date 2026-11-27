// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The module must stay importable under bare node, and a reply omitting the
 * fields must not reset a configured mirror.
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
  // Already canonicalised by the backend; only the trailing slash is dropped.
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
  // An older Studio carries no hf_endpoint: resetting would strand a mirror-only deployment.
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
  const anonymous = ["Bearer session", null];
  assert.deepEqual(sent, [both, anonymous, both, anonymous, both, fresh, fresh, ["Bearer hf_x", null]]);
});

test("hub page links open ModelScope directly, aliased ids included", async () => {
  const { hubWebUrl, setHubPageLinks } = await import("../src/lib/hf-endpoint.ts");
  const adapter = "http://127.0.0.1:8888/api/hub/modelscope";
  resetHfEndpoints();
  setHfEndpoints(adapter, undefined, "modelscope");
  assert.equal(hubWebUrl(`${adapter}/unsloth/Qwen3-8B`), `${adapter}/unsloth/Qwen3-8B`);
  setHubPageLinks(null, {
    base: "https://www.modelscope.cn",
    aliases: { model: {}, dataset: { "openai/gsm8k": "AI-ModelScope/gsm8k" } },
  });
  assert.equal(
    hubWebUrl(`${adapter}/unsloth/Qwen3-8B`),
    "https://www.modelscope.cn/models/unsloth/Qwen3-8B",
  );
  assert.equal(
    hubWebUrl(`${adapter}/datasets/OpenAI/GSM8K`),
    "https://www.modelscope.cn/datasets/AI-ModelScope/gsm8k",
  );
  assert.equal(
    hubWebUrl(`${adapter}/unsloth/Qwen3-8B/resolve/main/docs/guide.md#setup`),
    "https://www.modelscope.cn/models/unsloth/Qwen3-8B/resolve/master/docs/guide.md#setup",
  );
  assert.equal(
    hubWebUrl(`${adapter}/unsloth/Qwen3-8B?tab=files`),
    "https://www.modelscope.cn/models/unsloth/Qwen3-8B?tab=files",
  );
  setHubPageLinks(null, {
    base: "https://www.modelscope.cn",
    aliases: {
      model: { "zai-org/cogvlm-chat-hf": "zai-org/cogvlm-chat-hf", "zai-org/x": "AI-ModelScope/x" },
      dataset: {},
    },
    orgs: { "zai-org": "ZhipuAI" },
  });
  assert.equal(
    hubWebUrl(`${adapter}/ZAI-ORG/GLM-5.3/resolve/main/README.md`),
    "https://www.modelscope.cn/models/ZhipuAI/GLM-5.3/resolve/master/README.md",
  );
  assert.equal(
    hubWebUrl(`${adapter}/zai-org/cogvlm-chat-hf`),
    "https://www.modelscope.cn/models/zai-org/cogvlm-chat-hf",
  );
  assert.equal(hubWebUrl(`${adapter}/zai-org/x`), "https://www.modelscope.cn/models/AI-ModelScope/x");
  assert.equal(
    hubWebUrl(`${adapter}/constructor/chinese-roberta-wwm-ext-large`),
    "https://www.modelscope.cn/models/constructor/chinese-roberta-wwm-ext-large",
  );
  assert.equal(
    hubWebUrl(`${adapter}/datasets/zai-org/d`),
    "https://www.modelscope.cn/datasets/zai-org/d",
  );
  // A pinned revision needs the backend's commit mapping, so it keeps the mount.
  const pinned = `${adapter}/unsloth/Qwen3-8B/resolve/${"a".repeat(40)}/LICENSE`;
  assert.equal(hubWebUrl(pinned), pinned);
  assert.equal(hubWebUrl("https://example.com/x/y"), "https://example.com/x/y");
});

test("hub page links on a relayed mirror use the mirror's own origin", async () => {
  const { hubWebUrl, missingHubPageBase, setHubPageBase, setHubPageLinks } = await import(
    "../src/lib/hf-endpoint.ts"
  );
  const relay = "http://127.0.0.1:8888/api/hub/endpoint/abc";
  resetHfEndpoints();
  setHfEndpoints("https://hf-mirror.com", undefined, "huggingface");
  assert.equal(missingHubPageBase(), null);
  setHfEndpoints(relay, undefined, "huggingface", { endpoint: true });
  setHubPageLinks(null, undefined);
  assert.equal(missingHubPageBase(), relay);
  assert.equal(hubWebUrl(`${relay}/org/m`), `${relay}/org/m`);
  setHubPageBase(relay, "https://hf-mirror.com/");
  assert.equal(missingHubPageBase(), null);
  assert.equal(hubWebUrl(`${relay}/datasets/org/d`), "https://hf-mirror.com/datasets/org/d");
  // A base loaded for an endpoint that has since changed is never applied to the new one.
  const next = "http://127.0.0.1:8888/api/hub/endpoint/def";
  setHfEndpoints(next, undefined, "huggingface", { endpoint: true });
  assert.equal(hubWebUrl(`${next}/org/m`), `${next}/org/m`);
  setHubPageLinks("https://hub.internal", undefined);
  assert.equal(hubWebUrl(`${next}/org/m`), "https://hub.internal/org/m");
  setHubPageBase(relay, "https://stale.example");
  assert.equal(hubWebUrl(`${next}/org/m`), "https://hub.internal/org/m");
});

