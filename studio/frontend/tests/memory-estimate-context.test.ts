import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { register } from "node:module";
import { test } from "node:test";

import {
  resolveEstimateContext,
  resolveEstimateSourceIdentity,
  resolveMlxEstimateContext,
  resolveMlxServedWindow,
  shouldRequestMemoryEstimate,
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
  assert.notEqual(
    sourceId("org/gated", "Q4_K_M", "aaa"),
    sourceId("org/gated", "Q4_K_M", "bbb"),
  );
});

test("two native picks of the same file name are different sources", () => {
  assert.notEqual(
    sourceId("model.gguf", null, "", "tok-1"),
    sourceId("model.gguf", null, "", "tok-2"),
  );
});

test("absent and null are the same, so an unset variant does not thrash the row", () => {
  assert.equal(
    sourceId("org/a", null),
    sourceId("org/a", undefined as unknown as null),
  );
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

const gguf = (classifiedIsDiffusion: boolean | undefined) =>
  shouldRequestMemoryEstimate({
    isGguf: true,
    isAppleUnifiedMemory: true,
    classifiedIsDiffusion,
  });

const safetensors = (
  isAppleUnifiedMemory: boolean,
  classifiedIsDiffusion: boolean | undefined,
) =>
  shouldRequestMemoryEstimate({
    isGguf: false,
    isAppleUnifiedMemory,
    classifiedIsDiffusion,
  });

test("a GGUF is priced only once its probe has cleared it of being diffusion", () => {
  assert.equal(gguf(false), true);
  // Still in flight: a DiffusionGemma priced through a language-model plan is the wrong allocator.
  assert.equal(gguf(undefined), false);
});

test("an MLX load is priced without waiting for a GGUF-only probe", () => {
  assert.equal(safetensors(true, undefined), true);
  assert.equal(safetensors(true, false), true);
});

test("a model already known to be diffusion is priced by neither planner", () => {
  assert.equal(gguf(true), false);
  assert.equal(safetensors(true, true), false);
});

test("safetensors is not priced where MLX is not what would load it", () => {
  // Off Apple Silicon the backend answers not_gguf, so asking is one empty POST per slider release.
  assert.equal(safetensors(false, false), false);
  assert.equal(safetensors(false, undefined), false);
});

// The settings a NON-GGUF load sends: `max_seq_length`, not llama.cpp's context field.
register("./helpers/memory-estimate-resolver.mjs", import.meta.url);

const auth = await import("./helpers/store-stubs/auth.ts");
const native = await import("./helpers/native-path-stub.ts");
const { fetchMemoryEstimate, resetMemoryEstimateRouteMemo } = await import(
  "../src/features/model-picker/api/memory-estimate.ts"
);

let sent: Record<string, unknown> | null = null;

test.beforeEach(() => {
  resetMemoryEstimateRouteMemo();
  native.setNativePathHandler(null);
  sent = null;
  auth.setAuthFetchHandler((_url, init) => {
    sent = init?.body ? JSON.parse(String(init.body)) : null;
    return new Response("{}", {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  });
});

test.after(() => {
  resetMemoryEstimateRouteMemo();
  auth.setAuthFetchHandler(null);
  native.setNativePathHandler(null);
});

test("the non-GGUF context and MLX cache width reach the backend", async () => {
  await fetchMemoryEstimate({
    modelPath: "mlx-community/Qwen3-8B-4bit",
    maxSeqLength: 4096,
    mlxKvBits: 8,
  });
  assert.equal(sent!.max_seq_length, 4096);
  assert.equal(sent!.mlx_kv_bits, 8);
});

test("every field sent is also a field the hook re-fetches for", () => {
  // Two hand-written lists, one in each module: the request body and the hook's cache key.
  const read = (path: string, fn: string): Set<string> => {
    const source = readFileSync(new URL(path, import.meta.url), "utf8");
    const start = source.indexOf(`function ${fn}(`);
    assert.ok(start >= 0, `${fn} not found in ${path}; this test is stale`);
    const body = source.slice(start, source.indexOf("\n}\n", start));
    return new Set(
      [...body.matchAll(/\b(?:payload|request)\.([A-Za-z0-9_]+)/g)].map(
        (m) => m[1],
      ),
    );
  };
  const wire = read(
    "../src/features/model-picker/api/memory-estimate.ts",
    "estimateRequestBody",
  );
  const key = read(
    "../src/features/model-picker/hooks/use-memory-estimate.ts",
    "estimateKey",
  );
  assert.ok(wire.has("maxSeqLength") && wire.has("mlxKvBits"), "test is stale");
  const unwatched = [...wire].filter((field) => !key.has(field));
  assert.deepEqual(
    unwatched,
    [],
    `sent to the backend but absent from estimateKey: ${unwatched.join(", ")}`,
  );
});

test("what an MLX estimate names, and what the control shows", () => {
  // Sending the window the control displays leaves the backend unable to tell a pin from a
  // display fallback, so it could never fit one.
  assert.equal(resolveMlxEstimateContext(null), 0);
  assert.equal(resolveMlxEstimateContext(0), 0);
  assert.equal(resolveMlxEstimateContext(8192), 8192);
  // The control describes the next load, so a fit outranks the load running now: clearing a
  // pin on a resident 8192 must not leave it stating 8192 for a reload that fits. With no fit
  // the resident load is the best answer, and the declared window the last.
  assert.equal(resolveMlxServedWindow(8192, 24576, 262144), 24576);
  assert.equal(resolveMlxServedWindow(null, 24576, 262144), 24576);
  assert.equal(resolveMlxServedWindow(20480, null, 262144), 20480);
  assert.equal(resolveMlxServedWindow(null, null, 262144), 262144);
  assert.equal(resolveMlxServedWindow(null, null, null), null);
});

test("the fitted window survives the wire, and its absence reads as null", async () => {
  const fitted = async (body: unknown) => {
    auth.setAuthFetchHandler(() => Response.json(body));
    return (await fetchMemoryEstimate({ modelPath: "a" })).contextFitted;
  };
  assert.equal(await fitted({ available: true, n_ctx: 24576, context_fitted: 24576 }), 24576);
  // A backend predating the fit chose no window for this machine; there is none to show.
  assert.equal(await fitted({ available: true, n_ctx: 262144 }), null);
});
