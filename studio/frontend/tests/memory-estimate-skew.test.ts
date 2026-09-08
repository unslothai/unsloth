// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Version skew in both directions across /api/inference/estimate-memory.
//
// OLD BACKEND + NEW BUNDLE: the route is not there. Every failure shape must reach the
// panel as an unavailable estimate -- the row hides, and the Load button is not a party
// to any of it -- and the shape that says the ROUTE is missing, rather than that this
// request failed, is worth remembering so a slider drag does not POST into the void
// once per settings change forever.
//
// NEW BACKEND + OLD BUNDLE is not a thing (the bundle ships with the backend), but the
// field-level reverse is: a backend predating the drafter split, or one that never
// reported kv_estimable, must degrade to the documented fallbacks, not to a confident
// zero.

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

register("./helpers/memory-estimate-resolver.mjs", import.meta.url);

const auth = await import("./helpers/store-stubs/auth.ts");
const native = await import("./helpers/native-path-stub.ts");
const {
  fetchMemoryEstimate,
  resetMemoryEstimateRouteMemo,
} = await import("../src/features/model-picker/api/memory-estimate.ts");

const REQUEST = { modelPath: "unsloth/Qwen3-8B-GGUF", ggufVariant: "Q4_K_M" };

/** A complete, current response. The skew cases below delete fields from it. */
const FULL = {
  available: true,
  reason: null,
  weights_bytes: 4 * 1024 ** 3,
  kv_bytes: 2 * 1024 ** 3,
  compute_bytes: 1024 ** 3,
  drafter_runtime_bytes: 3 * 1024 ** 3,
  drafter_runtime_gpu_bytes: 1024 ** 3,
  projector_runtime_bytes: 0,
  drafter_kv_unsized: false,
  total_bytes: 7 * 1024 ** 3,
  gpu_bytes: 6 * 1024 ** 3,
  kv_estimable: true,
  kv_on_gpu: true,
  n_ctx: 32768,
  cache_type_kv: "f16",
  n_parallel: 1,
  layer_count: 36,
  gpu_layers: 37,
  moe_offload_unmodelled: false,
};

let requests: { url: string; body: unknown }[] = [];

function answer(
  make: (url: string, init?: RequestInit) => Response,
): void {
  requests = [];
  auth.setAuthFetchHandler((url, init) => {
    requests.push({
      url,
      body: init?.body ? JSON.parse(String(init.body)) : null,
    });
    return make(url, init);
  });
}

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });

const status = (code: number) =>
  new Response(code === 204 ? null : `error ${code}`, { status: code });

test.beforeEach(() => {
  resetMemoryEstimateRouteMemo();
  auth.setAuthFetchHandler(null);
  native.setNativePathHandler(null);
  requests = [];
});

test.after(() => {
  resetMemoryEstimateRouteMemo();
  auth.setAuthFetchHandler(null);
  native.setNativePathHandler(null);
});

// ---------------------------------------------------------------------------
// OLD BACKEND + NEW BUNDLE: every failure shape hides the row and nothing else

test("every non-OK status comes back unavailable rather than throwing", async () => {
  for (const code of [400, 401, 403, 404, 405, 409, 422, 429, 500, 501, 502, 503]) {
    resetMemoryEstimateRouteMemo();
    answer(() => status(code));
    const estimate = await fetchMemoryEstimate(REQUEST);
    assert.equal(estimate.available, false, `status ${code}`);
    // The row reads `!estimate?.available` and returns null, so it hides. Nothing
    // here can reject, so nothing reaches the panel's own error path.
    assert.equal(estimate.totalBytes, 0, `status ${code}`);
    assert.equal(estimate.gpuBytes, 0, `status ${code}`);
  }
});

test("an HTML 200 from a proxy is unavailable, not a crash", async () => {
  // A captive portal or a dev proxy serving its own page with a 200. `response.json()`
  // rejects on this, and an unhandled rejection here is a rejected promise inside the
  // panel's effect rather than a hidden row.
  answer(
    () =>
      new Response("<!doctype html><html><body>Sign in</body></html>", {
        status: 200,
        headers: { "Content-Type": "text/html" },
      }),
  );
  const estimate = await fetchMemoryEstimate(REQUEST);
  assert.equal(estimate.available, false);
});

test("a truncated JSON body is unavailable, not a crash", async () => {
  answer(
    () =>
      new Response('{"available": true, "total_byt', {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
  );
  const estimate = await fetchMemoryEstimate(REQUEST);
  assert.equal(estimate.available, false);
});

test("a 200 whose body is not an object at all is unavailable", async () => {
  for (const body of [null, 7, "ok", [1, 2, 3], true]) {
    resetMemoryEstimateRouteMemo();
    answer(() => json(body));
    const estimate = await fetchMemoryEstimate(REQUEST);
    assert.equal(estimate.available, false, JSON.stringify(body));
  }
});

test("a 200 with an empty object degrades to every documented default", async () => {
  answer(() => json({}));
  const estimate = await fetchMemoryEstimate(REQUEST);
  assert.equal(estimate.available, false);
  assert.equal(estimate.kvEstimable, false);
  assert.equal(estimate.kvOnGpu, true);
  assert.equal(estimate.nParallel, 1);
  assert.equal(estimate.layerCount, null);
  assert.equal(estimate.gpuLayers, null);
});

// ---------------------------------------------------------------------------
// THE MEMO

test("a 404 is remembered, so the next settings change does not POST again", async () => {
  answer(() => status(404));
  assert.equal((await fetchMemoryEstimate(REQUEST)).available, false);
  assert.equal(requests.length, 1);
  // A drag of the context slider, a KV dtype change, a new pin: all of them re-key the
  // hook and would each have fired their own request at a route that is not there.
  for (let i = 0; i < 20; i++) {
    assert.equal((await fetchMemoryEstimate(REQUEST)).available, false);
  }
  assert.equal(requests.length, 1);
});

test("405 and 501 are remembered for the same reason", async () => {
  for (const code of [405, 501]) {
    resetMemoryEstimateRouteMemo();
    answer(() => status(code));
    await fetchMemoryEstimate(REQUEST);
    await fetchMemoryEstimate(REQUEST);
    assert.equal(requests.length, 1, `status ${code}`);
  }
});

test("a transient 500 CANNOT latch the memo", async () => {
  // The requirement this test exists for. A backend that is answering at all, however
  // badly, is not one that is missing the route, and a 500 during a restart would
  // otherwise blank the row for the rest of the session.
  answer(() => status(500));
  await fetchMemoryEstimate(REQUEST);
  await fetchMemoryEstimate(REQUEST);
  await fetchMemoryEstimate(REQUEST);
  assert.equal(requests.length, 3);
});

test("no other failing status latches either", async () => {
  for (const code of [400, 401, 403, 408, 409, 422, 429, 502, 503, 504]) {
    resetMemoryEstimateRouteMemo();
    answer(() => status(code));
    await fetchMemoryEstimate(REQUEST);
    await fetchMemoryEstimate(REQUEST);
    assert.equal(requests.length, 2, `status ${code} must not latch`);
  }
});

test("a 500 arriving after a 404 clears the memo rather than leaving it standing", async () => {
  // The backend was replaced under us: the new one is up but unhealthy. That is
  // evidence the OLD answer no longer applies, so the memo must not survive it.
  let code = 404;
  answer(() => status(code));
  await fetchMemoryEstimate(REQUEST);
  assert.equal(requests.length, 1);
  // Still memoized: this call is suppressed.
  await fetchMemoryEstimate(REQUEST);
  assert.equal(requests.length, 1);
  resetMemoryEstimateRouteMemo();
  code = 500;
  await fetchMemoryEstimate(REQUEST);
  await fetchMemoryEstimate(REQUEST);
  assert.equal(requests.length, 3);
});

test("a success clears the memo, so an upgraded backend is believed at once", async () => {
  answer(() => json(FULL));
  const estimate = await fetchMemoryEstimate(REQUEST);
  assert.equal(estimate.available, true);
  await fetchMemoryEstimate(REQUEST);
  assert.equal(requests.length, 2);
});

test("resetting the memo re-probes", async () => {
  answer(() => status(404));
  await fetchMemoryEstimate(REQUEST);
  await fetchMemoryEstimate(REQUEST);
  assert.equal(requests.length, 1);
  resetMemoryEstimateRouteMemo();
  await fetchMemoryEstimate(REQUEST);
  assert.equal(requests.length, 2);
});

// ---------------------------------------------------------------------------
// FIELD-LEVEL SKEW: a backend predating parts of the response

test("a pre-split backend keeps the 'all of it is on the GPU' reading", async () => {
  // drafter_runtime_gpu_bytes did not exist before the placement split. Defaulting it
  // to 0 would silently drop a real VRAM charge off the row and paint a load that does
  // not fit as one that does.
  const { drafter_runtime_gpu_bytes, ...preSplit } = FULL;
  void drafter_runtime_gpu_bytes;
  answer(() => json(preSplit));
  const estimate = await fetchMemoryEstimate(REQUEST);
  assert.equal(estimate.drafterRuntimeGpuBytes, FULL.drafter_runtime_bytes);
  assert.equal(estimate.drafterRuntimeBytes, FULL.drafter_runtime_bytes);
});

test("an explicit zero GPU share is a real answer, not an absent field", async () => {
  // --spec-draft-ngl 0 puts the whole drafter in host RAM, and the row says so. The
  // fallback must not fire here and claim it is on the card.
  answer(() => json({ ...FULL, drafter_runtime_gpu_bytes: 0 }));
  const estimate = await fetchMemoryEstimate(REQUEST);
  assert.equal(estimate.drafterRuntimeGpuBytes, 0);
});

test("a missing kv_estimable degrades to the floor path, not a confident total", async () => {
  // The KV cache is the one term that can dwarf all the others, so an older backend
  // that cannot vouch for it must produce the amber, prefixed, "unknown" reading
  // rather than a total the row would print in plain text.
  const { kv_estimable, ...older } = FULL;
  void kv_estimable;
  answer(() => json(older));
  const estimate = await fetchMemoryEstimate(REQUEST);
  assert.equal(estimate.kvEstimable, false);
  const { resolveMemoryFit } = await import(
    "../src/features/model-picker/model-config/memory-fit.ts"
  );
  const result = resolveMemoryFit(estimate, {
    gpuCapacityGb: 24,
    totalCapacityGb: 88,
    systemRamCapacityGb: 64,
    freeGpuCapacityGb: 23,
    usableSystemRamGb: 60,
    singleMemoryPool: false,
  });
  assert.equal(result.bounded, true);
  assert.equal(result.prefix, "≥ ");
  assert.equal(result.advisory?.tone, "warn");
  assert.match(result.advisory?.text ?? "", /attention dimensions/);
});

test("a missing kv_on_gpu keeps the pre-flag assumption", async () => {
  const { kv_on_gpu, ...older } = FULL;
  void kv_on_gpu;
  answer(() => json(older));
  assert.equal((await fetchMemoryEstimate(REQUEST)).kvOnGpu, true);
});

test("a field arriving as a string or a non-finite number does not become a figure", async () => {
  // JSON.parse turns 1e999 into Infinity without complaint, and a backend that
  // stringified its numbers is a real skew shape. `?? 0` sees neither.
  answer(() =>
    new Response(
      '{"available": true, "kv_estimable": true, "total_bytes": 1e999, "gpu_bytes": "6442450944", "weights_bytes": -1, "n_ctx": null, "n_parallel": "4", "layer_count": 1e999}',
      { status: 200, headers: { "Content-Type": "application/json" } },
    ),
  );
  const estimate = await fetchMemoryEstimate(REQUEST);
  for (const value of [
    estimate.totalBytes, estimate.gpuBytes, estimate.weightsBytes,
    estimate.kvBytes, estimate.computeBytes, estimate.nCtx, estimate.nParallel,
  ]) {
    assert.ok(Number.isFinite(value) && value >= 0, `${value}`);
  }
  assert.equal(estimate.layerCount, null);
  const { classifyMemoryFit } = await import(
    "../src/features/model-picker/model-config/memory-fit.ts"
  );
  // And nothing drawn from them claims a fit.
  assert.equal(classifyMemoryFit(estimate.totalBytes, 24), "unknown");
});

test("a reason the panel has no copy for is dropped rather than rendered blank", async () => {
  answer(() => json({ ...FULL, available: false, reason: "brand_new_reason" }));
  assert.equal((await fetchMemoryEstimate(REQUEST)).reason, null);
  answer(() => json({ ...FULL, available: false, reason: "not_downloaded" }));
  assert.equal((await fetchMemoryEstimate(REQUEST)).reason, "not_downloaded");
});

test("a boolean sent as the string 'false' is not read as true", async () => {
  answer(() =>
    new Response('{"available": "false", "kv_estimable": "false", "kv_on_gpu": "false"}', {
      status: 200,
      headers: { "Content-Type": "application/json" },
    }),
  );
  const estimate = await fetchMemoryEstimate(REQUEST);
  assert.equal(estimate.available, false);
  assert.equal(estimate.kvEstimable, false);
  // Absent-shaped, so the documented default stands rather than a coerced true.
  assert.equal(estimate.kvOnGpu, true);
});

// ---------------------------------------------------------------------------
// The request itself

test("the settings that move the answer are all on the wire", async () => {
  answer(() => json(FULL));
  await fetchMemoryEstimate({
    ...REQUEST,
    nCtx: 8192,
    cacheTypeKv: "q8_0",
    nParallel: 4,
    selectedGpuIds: [0, 1],
    llamaExtraArgs: ["--foo"],
  });
  const body = requests[0]?.body as Record<string, unknown>;
  assert.equal(requests[0]?.url, "/api/inference/estimate-memory");
  assert.equal(body.model_path, REQUEST.modelPath);
  assert.equal(body.gguf_variant, "Q4_K_M");
  assert.equal(body.n_ctx, 8192);
  assert.equal(body.cache_type_kv, "q8_0");
  assert.equal(body.n_parallel, 4);
  assert.deepEqual(body.selected_gpu_ids, [0, 1]);
  assert.deepEqual(body.llama_extra_args, ["--foo"]);
  // The credential is sent, but the panel never keys on it: see estimate-context.
  assert.equal(body.hf_token, null);
});

test("an expired native path lease is its own reason and never reaches the network", async () => {
  answer(() => json(FULL));
  native.setNativePathHandler(() => {
    throw new Error("lease revoked");
  });
  const estimate = await fetchMemoryEstimate({
    ...REQUEST,
    nativePathToken: "tok-1",
  });
  assert.equal(estimate.available, false);
  assert.equal(estimate.reason, "unsupported_source");
  assert.equal(requests.length, 0);
});

test("a live native lease is exchanged and forwarded", async () => {
  answer(() => json(FULL));
  native.setNativePathHandler(() => ({ nativePathLease: "lease-9" }));
  await fetchMemoryEstimate({ ...REQUEST, nativePathToken: "tok-1" });
  const body = requests[0]?.body as Record<string, unknown>;
  assert.equal(body.native_path_lease, "lease-9");
});

test("an abort signal is passed through to the transport", async () => {
  const controller = new AbortController();
  let seen: AbortSignal | null | undefined;
  auth.setAuthFetchHandler((_url, init) => {
    seen = init?.signal;
    return json(FULL);
  });
  await fetchMemoryEstimate(REQUEST, controller.signal);
  assert.equal(seen, controller.signal);
});
