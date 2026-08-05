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

// gate file -> the module that issues its fetch, and the origin that fetch
// targets. Both halves have to line up or the window is unreachable.
const DIRECT_CLIENTS = [
  {
    gate: "../src/features/hub/lib/hf-owner-avatar.ts",
    fetcher: "../src/features/hub/lib/hf-owner-avatar.ts",
    origin: undefined,
  },
  {
    gate: "../src/features/hub/catalog/model-readme.tsx",
    fetcher: "../src/features/hub/lib/hf-readme.ts",
    origin: undefined,
  },
  {
    gate: "../src/features/hub/hooks/use-dataset-size.ts",
    fetcher: "../src/features/hub/lib/dataset-size.ts",
    origin: "DATASETS_SERVER_ORIGIN",
  },
];

test("the module behind each gate really tags its failures 'other'", async () => {
  for (const { gate, fetcher } of DIRECT_CLIENTS) {
    const src = await read(fetcher);
    // Asserted on the module that calls fetchWithTimeout, not on the gate: the
    // gate merely imports it, so a name check there passes on the import line
    // alone and pins nothing.
    assert.ok(
      src.includes('service: "other"'),
      `${fetcher} arms no "other" window, so ${gate} could never back off`,
    );
  }
});

test("and each gate reads the origin its own fetch targets", async () => {
  for (const { gate, origin } of DIRECT_CLIENTS) {
    const src = await read(gate);
    assert.ok(
      src.includes("useDirectHubOnline"),
      `${gate} must keep fetching when only the catalog listing is blocked`,
    );
    assert.ok(
      !/\buseOnlineStatus\b/.test(src),
      `${gate} would stop fetching over a dead listing, and never re-arm`,
    );
    if (origin) {
      // The default is the Hub's origin, which this client never fetches, so it
      // would read a window nothing it does can arm or clear.
      assert.match(src, new RegExp(`useDirectHubOnline\\(${origin}\\)`), gate);
    }
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
