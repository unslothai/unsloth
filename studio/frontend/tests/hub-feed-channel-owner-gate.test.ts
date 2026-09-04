// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The feed's iconless gate (logo'd providers, or likes >= 30) is a general-feed
// cleanliness rule. An owner-scoped channel is curated content, so its owner's
// rows must bypass the gate — otherwise a channel like "Latest Unsloth Models"
// renders empty while its newest models sit under the threshold and match no
// provider stem (#9456).

import assert from "node:assert/strict";
import test from "node:test";
import { readFile } from "node:fs/promises";

function read(path: string): Promise<string> {
  return readFile(new URL(path, import.meta.url), "utf8");
}

test("the feed's iconless gate exempts the active channel's owner", async () => {
  const page = await read("../src/features/hub/hub-page.tsx");
  const source = page.slice(
    page.indexOf("const filteredDiscoverRows = useMemo"),
  );
  const filterBody = source.slice(0, source.indexOf(");", source.indexOf("discoverRows.filter")));

  assert.match(
    filterBody,
    /channelOwner !== null && row\.owner\.toLowerCase\(\) === channelOwner/,
    "the channel-owner exemption must be part of the feed gate",
  );
  // The exemption must fire before the likes threshold, so a sub-threshold
  // channel row never depends on the threshold to be visible.
  const exemptAt = filterBody.indexOf("channelOwner !== null");
  const likesAt = filterBody.indexOf("MIN_ICONLESS_MODEL_LIKES");
  assert.ok(
    exemptAt !== -1 && likesAt !== -1 && exemptAt < likesAt,
    "the channel-owner check must precede the likes threshold",
  );
  // The gate itself must stay scoped to feed mode.
  assert.match(filterBody, /!isFeedMode ||/);
});
