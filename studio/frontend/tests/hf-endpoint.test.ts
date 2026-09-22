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
  useHfDatasetsServer,
  useHfEndpoint,
} = await import("../src/lib/hf-endpoint.ts");

const { isHuggingFaceOffline, markRemoteNetworkOffline, clearRemoteBackoff } =
  await import("../src/features/hub/lib/network.ts");

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
