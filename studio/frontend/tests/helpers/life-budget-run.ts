// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// node --experimental-strip-types tests/helpers/life-budget-run.ts: prints the village call budget as JSON.
import { registerBundlerResolver } from "./kit.ts";

registerBundlerResolver();
const { runBudget } = await import("./life-budget.ts");
const LIFE = {
  options: true,
  plan_cache: 64,
  reflect_at: 150,
  relationships: true,
};
const rows: Record<string, unknown> = {};
for (const [name, life] of [
  ["classic", {}],
  ["life", LIFE],
] as const)
  for (const [lim, perMinute] of [
    ["4/min", undefined],
    ["unthrottled", Number.POSITIVE_INFINITY],
  ] as const)
    for (const days of [1, 3])
      rows[`${name} ${lim} ${days}d`] = await runBudget({
        life,
        perMinute,
        days,
      });
console.log(JSON.stringify(rows, null, 1));
process.exit(0);
