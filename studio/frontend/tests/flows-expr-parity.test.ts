// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { evalExpr, ExprError, Prng, seedFrom } = await import(
  "../src/features/flows/expr.ts"
);

interface Case {
  name: string;
  expr: unknown;
  vars: Record<string, unknown>;
  seed?: number;
  expected?: unknown;
  error?: boolean;
}

const fixtures = JSON.parse(
  readFileSync(
    new URL(
      "../../backend/tests/fixtures/flows_expr_cases.json",
      import.meta.url,
    ),
    "utf8",
  ),
) as {
  cases: Case[];
  prng: { seed: number; values: number[]; state_after: number }[];
  seed_from: { parts: unknown[]; expected: number }[];
};

for (const c of fixtures.cases) {
  test(`expr parity: ${c.name}`, () => {
    const rng = c.seed === undefined ? null : new Prng(c.seed);
    if (c.error) {
      assert.throws(() => evalExpr(c.expr, c.vars, rng), ExprError);
    } else {
      assert.deepEqual(evalExpr(c.expr, c.vars, rng), c.expected ?? null);
    }
  });
}

test("mulberry32 matches the Python Prng bit for bit", () => {
  for (const p of fixtures.prng) {
    const rng = new Prng(p.seed);
    assert.deepEqual(
      p.values.map(() => rng.random()),
      p.values,
      `seed ${p.seed}`,
    );
    assert.equal(rng.state, p.state_after);
  }
});

test("seedFrom matches the Python seed_from", () => {
  for (const s of fixtures.seed_from)
    assert.equal(seedFrom(...s.parts), s.expected, JSON.stringify(s.parts));
});
