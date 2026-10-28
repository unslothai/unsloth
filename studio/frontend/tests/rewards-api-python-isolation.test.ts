// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { registerStoreStubResolver } from "./helpers/kit.ts";

registerStoreStubResolver();
const { setAuthFetchHandler } = await import("./helpers/store-stubs/auth.ts");
const { previewRewards, pythonRewardIsolationRefusal } =
  await import("../src/features/training/api/rewards-api.ts");

test("Try it scores under the run's Python reward isolation", async () => {
  const bodies: Record<string, unknown>[] = [];
  setAuthFetchHandler((_url, init) => {
    bodies.push(JSON.parse(String(init?.body)));
    return Response.json({ scores: [], total: 0, isolation: null });
  });
  try {
    const reward = [{ name: "py", weight: 1 }];
    await previewRewards(reward, "x", null, null, null, null, "auto");
    await previewRewards(reward, "x", null);
    assert.equal(bodies[0].python_reward_isolation, "auto");
    assert.equal(bodies[1].python_reward_isolation, "required");
  } finally {
    setAuthFetchHandler(null);
  }
});

test("Start asks the same isolation check for its refusal", async () => {
  const urls: string[] = [];
  setAuthFetchHandler((url) => {
    urls.push(url);
    return Response.json({ refusal: "no sandbox" });
  });
  try {
    assert.equal(await pythonRewardIsolationRefusal("required"), "no sandbox");
    assert.deepEqual(urls, ["/api/rewards/python-isolation?mode=required"]);
  } finally {
    setAuthFetchHandler(null);
  }
});
