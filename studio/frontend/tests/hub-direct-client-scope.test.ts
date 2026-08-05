// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { readFile } from "node:fs/promises";

// isDirectHubOffline reads the "other" window, so it only works if the clients
// gated on it are the same ones whose failures arm that window. Gate a client on
// it without it being a producer and the window can lapse with nothing to
// re-arm it; gate a producer on the origin-wide flag and it never fetches under
// a discovery-only block, which lapses the window the same way. These pin both
// halves of that loop, because getting it wrong is silent.

function read(path: string): Promise<string> {
  return readFile(new URL(path, import.meta.url), "utf8");
}

const DIRECT_FETCHERS = [
  "../src/features/hub/lib/hf-owner-avatar.ts",
  "../src/features/hub/hooks/use-dataset-size.ts",
  "../src/features/hub/catalog/model-readme.tsx",
];

test("every direct asset fetcher records its failures as 'other'", async () => {
  for (const path of DIRECT_FETCHERS) {
    const src = await read(path);
    const producer =
      src.includes('service: "other"') ||
      src.includes("fetchReadme") ||
      src.includes("fetchDatasetSize");
    assert.ok(producer, `${path} must arm the window it is gated on`);
  }
});

test("and gates itself on that window, not the origin-wide flag", async () => {
  for (const path of DIRECT_FETCHERS) {
    const src = await read(path);
    assert.ok(
      src.includes("useDirectHubOnline"),
      `${path} must keep fetching when only the catalog listing is blocked`,
    );
    assert.ok(
      !/\buseOnlineStatus\b/.test(src),
      `${path} would stop fetching over a dead listing, and never re-arm`,
    );
  }
});

test("clients that only read, and never fetch the Hub, stay origin-wide", async () => {
  // These call our own backend, so they populate neither window. Pointing them
  // at "other" let it lapse while genuinely offline, and the quant picker lost
  // its local-only fast path and stalled on a 5s upstream timeout instead.
  for (const path of [
    "../src/features/chat/api/chat-api.ts",
    "../src/features/hub/inventory/api.ts",
  ]) {
    const src = await read(path);
    assert.ok(src.includes("isHuggingFaceOffline"), path);
    assert.ok(!src.includes("isDirectHubOffline"), path);
  }
});
