// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { registerStoreStubResolver } from "./helpers/kit.ts";

registerStoreStubResolver();
const { setAuthFetchHandler } = await import("./helpers/store-stubs/auth.ts");
const { listGgufVariants } = await import(
  "../src/features/hub/inventory/api.ts"
);
const { ggufVariantsQuery } = await import(
  "../src/features/chat/api/gguf-variants-request.ts"
);

const cached = {
  repo_id: "Org/Cached",
  variants: [
    {
      quant: "Q4_K_M",
      filename: "model-Q4_K_M.gguf",
      size_bytes: 256,
      downloaded: true,
    },
  ],
  has_vision: false,
  default_variant: "Q4_K_M",
  dependencies_resolved: true,
};

test("chat local-only discovery skips remote work even when connectivity is reported online", () => {
  const params = ggufVariantsQuery(
    "Org/Cached",
    { localOnly: true, localPath: "/models/cached" },
    false,
  );
  assert.equal(params.get("offline"), "true");
  assert.equal(params.get("prefer_local_cache"), "true");
  assert.equal(params.get("local_path"), "/models/cached");
});

test("the cached single-quant client sends local-only discovery while navigator reports online", async () => {
  const original = Object.getOwnPropertyDescriptor(globalThis, "navigator");
  Object.defineProperty(globalThis, "navigator", {
    value: { onLine: true },
    configurable: true,
  });
  const calls: URL[] = [];
  setAuthFetchHandler(async (url) => {
    calls.push(new URL(String(url), "http://localhost"));
    return new Response(JSON.stringify(cached));
  });
  try {
    const result = await listGgufVariants("Org/OnlineWebview", undefined, {
      localOnly: true,
    });
    assert.equal(result.dependencies_resolved, true);
    assert.equal(calls.length, 1);
    assert.equal(calls[0].searchParams.get("offline"), "true");
    assert.equal(calls[0].searchParams.get("prefer_local_cache"), "true");
  } finally {
    if (original) Object.defineProperty(globalThis, "navigator", original);
    else delete (globalThis as Record<string, unknown>).navigator;
  }
});

test("a cache-preferred answer cannot replace a dependency-proven local-only answer", async () => {
  const calls: URL[] = [];
  setAuthFetchHandler(async (url) => {
    const request = new URL(String(url), "http://localhost");
    calls.push(request);
    return new Response(
      JSON.stringify({
        ...cached,
        dependencies_resolved: request.searchParams.get("offline") === "true",
      }),
    );
  });
  const preferred = await listGgufVariants("Org/ModeSwitch", undefined, {
    preferLocalCache: true,
  });
  assert.equal(preferred.dependencies_resolved, false);
  const local = await listGgufVariants("Org/ModeSwitch", undefined, {
    localOnly: true,
  });
  assert.equal(local.dependencies_resolved, true);
  assert.equal(calls.length, 2);
  const again = await listGgufVariants("Org/ModeSwitch", undefined, {
    localOnly: true,
  });
  assert.equal(again.dependencies_resolved, true);
  assert.equal(
    calls.length,
    2,
    "local-only answers still share their own cache",
  );
});
