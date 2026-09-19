// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { cachedModelInfo, primeCacheFromListing } = await import(
  "../src/features/hub/lib/hf-cache.ts"
);

test("a cold metadata lookup preserves GGUF parameters and caches the result", async () => {
  const name = "test/cold-gguf";
  const total = 8_190_735_360;
  let requests = 0;
  const fetchInfo: typeof fetch = async (input) => {
    requests += 1;
    const url = new URL(input instanceof Request ? input.url : String(input));
    const fields = url.searchParams.getAll("expand");
    return Response.json({
      _id: "cold-gguf",
      id: name,
      private: false,
      gated: false,
      lastModified: "2026-09-01T00:00:00Z",
      tags: ["gguf"],
      ...(fields.includes("gguf")
        ? { gguf: { total, architecture: "qwen3" } }
        : {}),
    });
  };

  const result = await cachedModelInfo({ name, fetch: fetchInfo });
  assert.equal(result.gguf?.total, total);
  const cached = await cachedModelInfo({ name, fetch: fetchInfo });
  assert.equal(cached.gguf?.total, total);
  assert.equal(requests, 1);
});

test("listing-primed GGUF metadata is reused without a request", async () => {
  const name = "test/primed-gguf";
  primeCacheFromListing(name, undefined, {
    id: "primed-gguf",
    name,
    private: false,
    gated: false,
    downloads: 0,
    likes: 0,
    updatedAt: new Date("2026-09-01T00:00:00Z"),
    gguf: { total: 8_190_735_360, architecture: "qwen3" },
  });
  const result = await cachedModelInfo({
    name,
    fetch: async () => {
      throw new Error("Listing metadata should satisfy this lookup");
    },
  });
  assert.equal(result.gguf?.total, 8_190_735_360);
});
