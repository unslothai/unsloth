// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

register("./helpers/settings-api-resolver.mjs", import.meta.url);

const requests: string[] = [];
globalThis.fetch = (async (input: RequestInfo | URL) => {
  requests.push(String(input));
  return new Response("[]", {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}) as typeof fetch;

const { listProviderRegistry } = await import(
  "../src/features/chat/api/providers-api.ts"
);

test("provider registry declares OAuth UI support", async () => {
  requests.length = 0;

  await listProviderRegistry();

  assert.deepEqual(requests, [
    "/api/providers/registry?include_hidden=true&include_oauth=true",
  ]);
});
