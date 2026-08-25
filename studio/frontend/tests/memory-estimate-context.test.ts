import assert from "node:assert/strict";
import { test } from "node:test";

import { resolveEstimateContext } from "../src/features/model-picker/model-config/estimate-context.ts";

// The regression this file exists for: the Context Length control needs a number to
// display before a new GGUF's header has been read and falls back to 32,768, but the
// Load button sends 0 for the same state and llama.cpp fits or opens at the model's
// native context. Pricing the displayed fallback quoted an explicit 32k for a load
// that could open far wider, and the KV cache is the term that grows fastest with
// context, so the panel understated exactly where it mattered most.
test("Auto with no metadata yet prices the native context, not the displayed 32k", () => {
  assert.equal(resolveEstimateContext(null, null), 0);
});

test("an explicit length is priced as itself", () => {
  assert.equal(resolveEstimateContext(8192, null), 8192);
  // Even when it is larger than the header's native context: the user asked for it,
  // and llama.cpp is the one that refuses or fits it down.
  assert.equal(resolveEstimateContext(524288, null), 524288);
});

test("the resident load's context is kept when reloading it", () => {
  // resolveLoadMaxSeqLength's isReloadingCurrentGguf branch: a fitted load got less
  // than native, and that is what it will be resident at again.
  assert.equal(resolveEstimateContext(null, 40223), 40223);
});

test("a known native context is NOT quoted as the figure", () => {
  // The one that would undo the fix if it were written the obvious way. Auto sends 0
  // and llama.cpp's --fit can land well below native, so pricing native claims an
  // outcome the load has not reached; 0 lets the estimate resolve it the way the
  // launch does. There is no native argument any more, and that is the point.
  assert.equal(resolveEstimateContext(null, null), 0);
});

test("an explicit length outranks the resident one", () => {
  assert.equal(resolveEstimateContext(4096, 40223), 4096);
});

test("zero is not mistaken for unset", () => {
  // 0 already means "price the native context" on the wire, so an explicit 0 and an
  // unset length agree rather than one of them falling through to a display bound.
  assert.equal(resolveEstimateContext(0, 40223), 0);
});
