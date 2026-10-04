// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The notice quotes numbers back at the user, so a partial or hostile payload must
// produce nothing at all rather than something reading "undefined GB".

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { parseCarveoutAdvice } = await import(
  "../src/features/igpu-carveout/types.ts"
);

const GOOD = {
  current_gb: 32,
  needed_gb: 42.9,
  suggested_gb: 48,
  machine_gb: 127.8,
  host_left_gb: 79.8,
  message: "This model's weights are about 43 GB...",
};

test("a complete payload parses", () => {
  const parsed = parseCarveoutAdvice(GOOD);
  assert.ok(parsed);
  assert.equal(parsed.suggested_gb, 48);
  assert.equal(parsed.current_gb, 32);
});

test("absent advice is not an error", () => {
  // Missing on nearly every load, and the caller passes it through unconditionally.
  for (const empty of [null, undefined, "", 0, false]) {
    assert.equal(parseCarveoutAdvice(empty), null);
  }
});

test("a payload missing any number is rejected whole", () => {
  for (const key of [
    "current_gb",
    "needed_gb",
    "suggested_gb",
    "machine_gb",
    "host_left_gb",
  ]) {
    const partial: Record<string, unknown> = { ...GOOD };
    delete partial[key];
    assert.equal(parseCarveoutAdvice(partial), null, key);
  }
});

test("non-finite numbers are rejected", () => {
  // JSON cannot carry these, but a proxy or a hand-rolled backend double can.
  for (const bad of [Number.NaN, Number.POSITIVE_INFINITY]) {
    assert.equal(parseCarveoutAdvice({ ...GOOD, suggested_gb: bad }), null);
  }
});

test("a missing or blank message is rejected", () => {
  // The backend owns the wording; without it there is nothing to render.
  assert.equal(parseCarveoutAdvice({ ...GOOD, message: undefined }), null);
  assert.equal(parseCarveoutAdvice({ ...GOOD, message: "   " }), null);
  assert.equal(parseCarveoutAdvice({ ...GOOD, message: 42 }), null);
});

test("a string where a number belongs is rejected", () => {
  assert.equal(parseCarveoutAdvice({ ...GOOD, current_gb: "32" }), null);
});
