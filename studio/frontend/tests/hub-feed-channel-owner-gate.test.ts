// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The feed's iconless gate (logo'd providers, or likes >= 30) must not hide the
// feed list's own owner: "Latest Unsloth Models" is createdAt-sorted, so its
// newest releases sit under the threshold and many match no provider stem
// (FLUX, HiDream, LFM, ...), and the gate dropped them from the feed (#9456).

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { findChannel } from "../src/features/hub/lib/channels.ts";
import {
  MIN_ICONLESS_MODEL_LIKES,
  passesFeedIconlessGate,
} from "../src/features/hub/lib/feed-visibility.ts";

const latestOwner = findChannel("unsloth-latest")?.owner;

test("the feed list's owner bypasses the iconless gate", () => {
  assert.equal(latestOwner, "unsloth");
  for (const repo of ["FLUX.2-klein-4B", "HiDream-I1-Fast", "LFM2.5-VL-3B"]) {
    const row = { owner: "unsloth", repo, likes: 3 };
    assert.equal(passesFeedIconlessGate(row, null), false, repo);
    assert.equal(passesFeedIconlessGate(row, latestOwner), true, repo);
  }
  assert.equal(
    passesFeedIconlessGate(
      { owner: "Unsloth", repo: "x", likes: 0 },
      "unsloth",
    ),
    true,
  );
});

test("the gate is unchanged for rows the list owner did not publish", () => {
  const iconless = { owner: "someone", repo: "Ornith-7B", likes: 3 };
  assert.equal(passesFeedIconlessGate(iconless, latestOwner), false);
  assert.equal(
    passesFeedIconlessGate(
      { ...iconless, likes: MIN_ICONLESS_MODEL_LIKES },
      latestOwner,
    ),
    true,
  );
  assert.equal(
    passesFeedIconlessGate({ ...iconless, likes: null }, latestOwner),
    false,
  );
  assert.equal(
    passesFeedIconlessGate(
      { owner: "unsloth", repo: "Qwen3-8B", likes: 0 },
      null,
    ),
    true,
  );
});

test("the feed filter keys the exemption on the feed's list channel", async () => {
  const page = await readFile(
    new URL("../src/features/hub/hub-page.tsx", import.meta.url),
    "utf8",
  );
  const start = page.indexOf("const filteredDiscoverRows = useMemo");
  const body = page.slice(start, page.indexOf("]);", start));
  // activeChannel is null whenever isFeedMode is true, so it cannot be the key.
  assert.match(body, /const listOwner = liveListChannel\?\.owner/);
  assert.match(body, /!isFeedMode \|\|\s*passesFeedIconlessGate\(/);
  assert.match(body, /\bliveListChannel,/);
});
