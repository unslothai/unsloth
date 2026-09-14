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

test("a mirror reported by the backend is applied to both getters", () => {
  resetHfEndpoints();
  setHfEndpoints("https://hf-mirror.com", "https://ds.example.com");
  assert.equal(getHfEndpoint(), "https://hf-mirror.com");
  assert.equal(getHfDatasetsServerBase(), "https://ds.example.com");
});

test("endpoints are normalised the way the backend normalises them", () => {
  for (const [raw, expected] of [
    ["https://hf-mirror.com/", "https://hf-mirror.com"],
    ["https://hf-mirror.com///", "https://hf-mirror.com"],
    ["hf-mirror.com", "https://hf-mirror.com"],
    ["hf-mirror.com/", "https://hf-mirror.com"],
    ["  https://hf-mirror.com  ", "https://hf-mirror.com"],
    ["http://localhost:8080", "http://localhost:8080"],
    ["http://127.0.0.1:9700", "http://127.0.0.1:9700"],
    ["https://hub.internal:8443/hf/", "https://hub.internal:8443/hf"],
  ] as const) {
    resetHfEndpoints();
    setHfEndpoints(raw, null);
    assert.equal(getHfEndpoint(), expected, `for ${raw}`);
  }
});

test("the empty-port rule reads the authority, not the whole URL", () => {
  // The backend tests parts.netloc: a whole-string check would reject a mirror
  // it accepts, splitting the two apart again.
  for (const [raw, expected] of [
    ["https://hub.internal/hf:", "https://hub.internal/hf:"],
    ["https://host:", DEFAULT_HF_ENDPOINT],
    ["https://:8080", DEFAULT_HF_ENDPOINT],
    ["https://[::1]:", DEFAULT_HF_ENDPOINT],
  ] as const) {
    resetHfEndpoints();
    setHfEndpoints(raw, null);
    assert.equal(getHfEndpoint(), expected, `for ${raw}`);
  }
});

test("a Unicode host is refused, its punycode form is not", () => {
  // new URL() would punycode it; the backend refuses it (latin-1 CSP header).
  resetHfEndpoints();
  setHfEndpoints("https://例子.测试", null);
  assert.equal(getHfEndpoint(), DEFAULT_HF_ENDPOINT);
  setHfEndpoints("https://xn--fsqu00a.xn--0zwm56d", null);
  assert.equal(getHfEndpoint(), "https://xn--fsqu00a.xn--0zwm56d");
});

test("every IPv6 loopback spelling ends up as the compressed origin", () => {
  for (const raw of ["http://[0:0:0:0:0:0:0:1]:9700", "http://[::1]:9700"]) {
    resetHfEndpoints();
    setHfEndpoints(raw, null);
    assert.equal(getHfEndpoint(), "http://[::1]:9700", `for ${raw}`);
  }
  resetHfEndpoints();
  setHfEndpoints("http://[2001:db8::1]:9700", null);
  assert.equal(getHfEndpoint(), DEFAULT_HF_ENDPOINT);
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

test("a value that could widen the CSP is refused, not propagated", () => {
  // The backend rejects these before they reach connect-src; so must this.
  for (const hostile of [
    "https://hf-mirror.com; script-src *",
    "https://hf-mirror.com *",
    "https://hf-mirror.com\nscript-src *",
    "https://hf-mirror.com\tfoo",
    "https://hf-mirror.com,https://evil.com",
    "javascript:alert(1)",
    "file:///etc/passwd",
    "data:text/html,x",
    "ftp://hf-mirror.com",
    "https://",
    "https://user:pass@hf-mirror.com",
    "https://hf-mirror.com?x=1",
    "https://hf-mirror.com#frag",
    "https://hf-mirror.com:",
    "https://hf-mirror.com:not-a-port",
    "*",
    "https://*",
    "https://*.evil.com",
    "http://192.168.1.10:8080",
    "http://hf-mirror.com",
    "http://10.0.0.5:8080",
  ]) {
    resetHfEndpoints();
    setHfEndpoints(hostile, hostile);
    assert.equal(getHfEndpoint(), "https://huggingface.co", `for ${hostile}`);
    assert.equal(
      getHfDatasetsServerBase(),
      "https://datasets-server.huggingface.co",
      `for ${hostile}`,
    );
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
