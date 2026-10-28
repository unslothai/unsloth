// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { normalizeProviderBaseUrl } from "../src/features/chat/provider-base-url.ts";

const local = { appendOpenAiVersionPath: true };

test("a host without a scheme is read as http", () => {
  assert.equal(
    normalizeProviderBaseUrl("localhost:11434", local),
    "http://localhost:11434/v1",
  );
  assert.equal(
    normalizeProviderBaseUrl("192.168.1.5:8000/v1"),
    "http://192.168.1.5:8000/v1",
  );
  assert.equal(
    normalizeProviderBaseUrl("[::1]:8080", local),
    "http://[::1]:8080/v1",
  );
  assert.equal(normalizeProviderBaseUrl("gpu-box", local), "http://gpu-box/v1");
  assert.equal(
    normalizeProviderBaseUrl("https://api.example.com/v1/"),
    "https://api.example.com/v1",
  );
});

test("a public domain without a scheme is read as https", () => {
  assert.equal(
    normalizeProviderBaseUrl("api.openai.com/v1"),
    "https://api.openai.com/v1",
  );
  assert.equal(
    normalizeProviderBaseUrl("openrouter.ai:443/api/v1"),
    "https://openrouter.ai/api/v1",
  );
  assert.equal(
    normalizeProviderBaseUrl("gpu.example.com:8000/v1"),
    "http://gpu.example.com:8000/v1",
  );
  for (const host of ["box.local", "host.docker.internal", "nas.lan"]) {
    assert.equal(normalizeProviderBaseUrl(`${host}/v1`), `http://${host}/v1`);
  }
});

test("other schemes still fail", () => {
  assert.throws(
    () => normalizeProviderBaseUrl("ftp://host/v1"),
    /http or https/,
  );
  assert.throws(() => normalizeProviderBaseUrl("http://"), /valid URL/);
});

test("a pasted endpoint is trimmed back to its base", () => {
  for (const suffix of [
    "chat/completions",
    "completions",
    "responses",
    "messages",
    "models",
    "chat/completions/",
  ]) {
    assert.equal(
      normalizeProviderBaseUrl(`http://localhost:11434/v1/${suffix}`, local),
      "http://localhost:11434/v1",
    );
  }
  assert.equal(
    normalizeProviderBaseUrl("localhost:8080/chat/completions", local),
    "http://localhost:8080/v1",
  );
  assert.equal(
    normalizeProviderBaseUrl("https://gw.example/openai/v1"),
    "https://gw.example/openai/v1",
  );
  assert.equal(normalizeProviderBaseUrl("  "), null);
});
