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

// gate file -> the exported function that issues its fetch, and the origin that
// fetch targets. Both halves have to line up or the window is unreachable.
const DIRECT_CLIENTS = [
  {
    gate: "../src/features/hub/lib/hf-owner-avatar.ts",
    fetcher: "../src/features/hub/lib/hf-owner-avatar.ts",
    fn: "async function fetchAvatarUrl",
    origin: undefined,
  },
  {
    gate: "../src/features/hub/catalog/model-readme.tsx",
    fetcher: "../src/features/hub/lib/hf-readme.ts",
    fn: "async function fetchReadmeOnce",
    origin: undefined,
  },
  {
    gate: "../src/features/hub/hooks/use-dataset-size.ts",
    fetcher: "../src/features/hub/lib/dataset-size.ts",
    fn: "export function fetchDatasetSize",
    origin: "DATASETS_SERVER_ORIGIN",
  },
  {
    gate: "../src/features/hub/catalog/safetensors-download-card.tsx",
    fetcher: "../src/features/hub/lib/dataset-size.ts",
    fn: "export function fetchModelSize",
    origin: undefined,
  },
];

/** The body of one function, so a sibling producer cannot satisfy the check. */
function bodyOf(src: string, marker: string): string {
  const at = src.lastIndexOf(marker);
  assert.notEqual(at, -1, `could not find ${marker}`);
  const after = src.slice(at + marker.length);
  const next = after.search(/\n(?:export )?(?:async )?function /);
  return next === -1 ? after : after.slice(0, next);
}

test("the function behind each gate really tags its failures 'other'", async () => {
  for (const { gate, fetcher, fn } of DIRECT_CLIENTS) {
    const src = await read(fetcher);
    // Scoped to the function, not the file: dataset-size.ts hosts two separate
    // producers, so a file-wide search passes off the wrong one.
    assert.ok(
      bodyOf(src, fn).includes('service: "other"'),
      `${fn} arms no "other" window, so ${gate} could never back off`,
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
      assert.match(src, new RegExp(`useDirectHubOnline\\(${origin}\\)`), gate);
    }
  }
});

test("the datasets-server constant is the host that client actually calls", async () => {
  const network = await read("../src/features/hub/lib/network.ts");
  const decl = /DATASETS_SERVER_ORIGIN = "([^"]+)"/.exec(network);
  assert.ok(decl, "could not find DATASETS_SERVER_ORIGIN");
  const sizes = await read("../src/features/hub/lib/dataset-size.ts");
  // Naming the wrong host leaves both checks above green while the gate reads a
  // window nothing arms, which is the whole failure this file exists to catch.
  assert.ok(
    bodyOf(sizes, "export function fetchDatasetSize").includes(`${decl[1]}/`),
    `fetchDatasetSize does not call ${decl[1]}`,
  );
});

test("a repo lookup never speaks for the listing", async () => {
  const src = await read("../src/features/hub/hooks/use-hub-model-search.ts");
  // cachedModelInfo runs in parallel with listModels. On the feed's own key its
  // response, a 404 included, called markRemoteNetworkOnline and deleted the
  // listing's failure and window: the panel lost the classified cause and the
  // phase read "available" while the listing was still blocked.
  assert.ok(
    bodyOf(src, "function makeHfFetch").includes('service: "other"'),
    "the repo lookup must not retire the listing's diagnosis",
  );
  // And the listing itself stays on the feed's key, or nothing arms it at all.
  const listing = bodyOf(src, "function makeSortFetch");
  assert.ok(!listing.includes('service: "other"'), "the listing IS the feed");
  // makeSortFetch is what listModels is handed; makeHfFetch only ever reaches
  // cachedModelInfo, so tagging it cannot silence a real listing failure.
  for (const call of src.matchAll(/fetch: (makeHfFetch|sortFetch)\(/g)) {
    const before = src.slice(Math.max(0, call.index - 400), call.index);
    const isInfo = before.lastIndexOf("cachedModelInfo(") > before.lastIndexOf("listModels(");
    assert.equal(
      call[1] === "makeHfFetch",
      isInfo,
      "makeHfFetch is for repo lookups only",
    );
  }
});
