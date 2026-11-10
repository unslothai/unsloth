// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { pickPrimaryMetric } from "../src/features/benchmark/lib/compare-metrics.ts";

test("Compare leads with the run's headline metric, as history does", () => {
  const metrics = [
    { name: "acc,none", score: 0.25 },
    { name: "acc_norm,none", score: 0.3 },
  ];
  assert.equal(pickPrimaryMetric(metrics, "acc_norm,none")?.name, "acc_norm,none");
  assert.equal(pickPrimaryMetric(metrics)?.name, "acc,none");
  const dialog = readFileSync(
    new URL("../src/features/benchmark/components/benchmark-compare-dialog.tsx", import.meta.url),
    "utf8",
  );
  assert.match(dialog, /selectedRuns\.find\(\(r\) => r\.id === detail\.id\)\?\.default_metric/);
});
