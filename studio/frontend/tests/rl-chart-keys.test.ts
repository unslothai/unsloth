// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { rlChartKeys } = await import(
  "../src/features/training/lib/rl-chart-keys.ts"
);

test("KL is not charted when beta 0 leaves it at 0 every step", () => {
  const keys = rlChartKeys([
    { step: 1, values: { reward: 0.5, kl: 0 } },
    { step: 2, values: { reward: 1, kl: 0 } },
  ]);
  assert.deepEqual([...keys], ["reward"]);
});

test("KL is charted once any step measures drift", () => {
  const keys = rlChartKeys([
    { step: 1, values: { reward: 0.5, kl: 0 } },
    { step: 2, values: { reward: 1, kl: 0.0004 } },
  ]);
  assert.ok(keys.has("kl"));
  assert.ok(keys.has("reward"));
});
