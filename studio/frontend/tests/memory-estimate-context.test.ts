import assert from "node:assert/strict";
import { test } from "node:test";

import {
  resolveEstimateContext,
  resolveEstimateSourceIdentity,
} from "../src/features/model-picker/model-config/estimate-context.ts";

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

// Which MODEL the shown numbers belong to. The hook blanks the row when this changes
// and merely greys it when anything else does, so anything that selects a different
// FILE has to be in here. It keyed on modelPath alone, which is identical across a
// quantization switch while the weights roughly quadruple: Q4_K_M's footprint stayed
// on screen under F16's name until the new answer landed.
const sourceId = (
  path: string,
  variant: string | null = null,
  token = "",
  native: string | null = null,
) => resolveEstimateSourceIdentity(path, variant, token, native);

test("two quantizations of one repository are different sources", () => {
  assert.notEqual(
    sourceId("unsloth/Qwen3-8B-GGUF", "Q4_K_M"),
    sourceId("unsloth/Qwen3-8B-GGUF", "F16"),
  );
});

test("the same source is the same identity, so a slider step does not blank the row", () => {
  assert.equal(
    sourceId("unsloth/Qwen3-8B-GGUF", "Q4_K_M"),
    sourceId("unsloth/Qwen3-8B-GGUF", "Q4_K_M"),
  );
});

test("two repositories are different sources", () => {
  assert.notEqual(sourceId("org/a", "Q4_K_M"), sourceId("org/b", "Q4_K_M"));
});

test("two credentials are different sources: they resolve different files", () => {
  assert.notEqual(sourceId("org/gated", "Q4_K_M", "aaa"), sourceId("org/gated", "Q4_K_M", "bbb"));
});

test("two native picks of the same file name are different sources", () => {
  assert.notEqual(sourceId("model.gguf", null, "", "tok-1"), sourceId("model.gguf", null, "", "tok-2"));
});

test("absent and null are the same, so an unset variant does not thrash the row", () => {
  assert.equal(sourceId("org/a", null), sourceId("org/a", undefined as unknown as null));
});

// Manual memory mode with GPU Layers on Auto hands context sizing to llama.cpp --fit.
// `resolveFitMaxSeqLength` sends a positive pin or 0 there, never the resident length,
// so falling back to what is loaded right now priced the OLD fit after a change that
// moves it -- a KV dtype or a batch size, which is exactly when the two diverge.
test("when the fit or a builtin-default owns the context, the resident length is not sent", () => {
  assert.equal(resolveEstimateContext(null, 40960, true), 0);
});

test("a positive pin survives the fit path, because Load sends it", () => {
  assert.equal(resolveEstimateContext(8192, 40960, true), 8192);
});

test("a non-positive pin is still 0 under the fit path", () => {
  assert.equal(resolveEstimateContext(0, 40960, true), 0);
  assert.equal(resolveEstimateContext(-1, 40960, true), 0);
});

test("every other shape keeps the resident fallback", () => {
  assert.equal(resolveEstimateContext(null, 40960, false), 40960);
  // And the flag defaults off, so no caller gains the fit rule by accident.
  assert.equal(resolveEstimateContext(null, 40960), 40960);
});

// resolveLoadMaxSeqLength answers 0 for a builtin-default GGUF load too, before it
// reaches the reloading-current-GGUF branch that returns the resident context. Same
// flag, because the consequence is identical: pricing what is loaded right now quotes
// the OLD fit at exactly the moment a setting has moved the next one.
test("a builtin-default GGUF load prices the fit, not the resident context", () => {
  assert.equal(resolveEstimateContext(null, 131072, true), 0);
});
