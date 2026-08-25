import assert from "node:assert/strict";
import { test } from "node:test";

import { resolveEstimateContext } from "../src/features/model-picker/model-config/estimate-context.ts";

// The regression this file exists for: the Context Length control needs a number to
// display before a new GGUF's header has been read and falls back to 32,768, but the
// Load button sends 0 for the same state and llama.cpp opens at the native context.
// Pricing the displayed fallback quoted an explicit 32k for a load that could open far
// wider, and the KV cache is the term that grows fastest with context, so the panel
// understated exactly where it mattered most.
test("Auto with no metadata yet prices the native context, not the displayed 32k", () => {
  assert.equal(resolveEstimateContext(null, null, null), 0);
});

test("an explicit length is priced as itself", () => {
  assert.equal(resolveEstimateContext(8192, null, 262144), 8192);
  // Even when it is larger than the header's native context: the user asked for it,
  // and llama.cpp is the one that refuses or fits it down.
  assert.equal(resolveEstimateContext(524288, null, 262144), 524288);
});

test("the resident load's context outranks the header's native one", () => {
  // A fitted load got less than native. That is what is resident, so that is what the
  // panel answers for while it stays resident.
  assert.equal(resolveEstimateContext(null, 40223, 262144), 40223);
});

test("the native context is used once the header has been read", () => {
  assert.equal(resolveEstimateContext(null, null, 262144), 262144);
});

test("an explicit length outranks both", () => {
  assert.equal(resolveEstimateContext(4096, 40223, 262144), 4096);
});

test("zero is not mistaken for unset", () => {
  // 0 already means "price the native context" on the wire, so an explicit 0 and an
  // unset length agree rather than one of them falling through to a display bound.
  assert.equal(resolveEstimateContext(0, 40223, 262144), 0);
});
