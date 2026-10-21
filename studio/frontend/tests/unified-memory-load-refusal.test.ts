// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Unified-memory load refusal (#14267).

import assert from "node:assert/strict";
import { register } from "node:module";
import test, { afterEach } from "node:test";

import { readSrc, readText } from "./helpers/kit.ts";

register("./helpers/settings-api-resolver.mjs", import.meta.url);

const refusal = await import("../src/lib/unified-memory-load-refusal.ts");
const {
  ALLOW_OVERSIZED_LOAD_LABEL,
  ALLOW_OVERSIZED_LOAD_STORAGE_KEY,
  UNIFIED_MEMORY_LOAD_REFUSAL,
  allowOversizedLoadField,
  isUnifiedMemoryLoadRefusal,
  shouldOfferLoadAnyway,
  withAllowOversizedLoad,
} = refusal;
const images = await import("../src/features/images/api.ts");
const video = await import("../src/features/video/api.ts");

// Settles the load notice's background load-progress poll once a test's own stub is gone.
const settledFetch = (async () =>
  new Response(JSON.stringify({ phase: "ready" }), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  })) as typeof fetch;
const store = new Map<string, string>();
Object.defineProperty(globalThis, "localStorage", {
  configurable: true,
  value: {
    getItem: (k: string) => store.get(k) ?? null,
    setItem: (k: string, v: string) => void store.set(k, String(v)),
    removeItem: (k: string) => void store.delete(k),
  },
});
afterEach(() => {
  globalThis.fetch = settledFetch;
  store.clear();
});

function captureFetch(reply: object): Record<string, unknown>[] {
  const sent: Record<string, unknown>[] = [];
  globalThis.fetch = (async (_url: unknown, init?: RequestInit) => {
    // Only the POST bodies: the load notice also polls load-progress, which sends none.
    if (init?.body) sent.push(JSON.parse(String(init.body)));
    return new Response(JSON.stringify(reply), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }) as typeof fetch;
  return sent;
}

test("the refusal is recognised from its code or its header, and nothing else", () => {
  assert.equal(isUnifiedMemoryLoadRefusal({ code: UNIFIED_MEMORY_LOAD_REFUSAL }), true);
  assert.equal(isUnifiedMemoryLoadRefusal({ code: " Unified-Memory-Load " }), true);
  assert.equal(isUnifiedMemoryLoadRefusal({ code: null }), false);
  assert.equal(isUnifiedMemoryLoadRefusal({ code: "memory-estimate" }), false);
  const tagged = Object.assign(new Error("needs about 56 GB"), {
    headers: { "x-unsloth-refusal": "unified-memory-load" },
  });
  assert.equal(isUnifiedMemoryLoadRefusal({ failure: tagged }), true);
  const other = Object.assign(new Error("x"), {
    headers: { "x-unsloth-refusal": "multi-model-always" },
  });
  assert.equal(isUnifiedMemoryLoadRefusal({ failure: other }), false);
  assert.equal(isUnifiedMemoryLoadRefusal({ failure: new Error("needs about 56 GB") }), false);
  // Wording alone never qualifies: only the backend's code may offer an override.
  assert.equal(
    isUnifiedMemoryLoadRefusal({ failure: "UNSLOTH_DIFFUSION_ALLOW_OVERSIZED_LOAD" }),
    false,
  );
});

test("Load anyway is offered for the refusal, and not again once the override was sent", () => {
  const code = UNIFIED_MEMORY_LOAD_REFUSAL;
  assert.equal(shouldOfferLoadAnyway({ code, overrideSent: false }), true);
  assert.equal(shouldOfferLoadAnyway({ code, overrideSent: true }), false);
  assert.equal(shouldOfferLoadAnyway({ code: null, overrideSent: false }), false);
});

test("allow_oversized_load is sent only when the setting or a one-shot retry asks for it", () => {
  assert.equal(allowOversizedLoadField(false, false), undefined);
  assert.equal(allowOversizedLoadField(true, false), true);
  assert.equal(allowOversizedLoadField(false, true), true);
  const plain: { model_path: string; allow_oversized_load?: boolean } = { model_path: "m" };
  assert.deepEqual(withAllowOversizedLoad(plain), { model_path: "m" });
  store.set(ALLOW_OVERSIZED_LOAD_STORAGE_KEY, "true");
  assert.deepEqual(withAllowOversizedLoad(plain), {
    model_path: "m",
    allow_oversized_load: true,
  });
  // An explicit value from the caller wins over the setting.
  assert.deepEqual(withAllowOversizedLoad({ model_path: "m", allow_oversized_load: false }), {
    model_path: "m",
    allow_oversized_load: false,
  });
});

const LOADS = [
  ["Images load", () => images.loadDiffusionModel({ model_path: "Qwen/Qwen-Image" })],
  ["Images plan", () => images.getDiffusionDownloadPlan({ model_path: "Qwen/Qwen-Image" })],
  ["Video load", () => video.loadVideoModel({ model_path: "Lightricks/LTX-2" })],
  ["Video plan", () => video.getVideoDownloadPlan({ model_path: "Lightricks/LTX-2" })],
] as const;

for (const [name, call] of LOADS) {
  test(`${name}: the default sends no override, the setting and a retry send it`, async () => {
    // phase "ready" ends the load notice's own load-progress poll.
    const sent = captureFetch({ entries: [], total_bytes: 0, loaded: false, phase: "ready" });
    await call();
    assert.equal("allow_oversized_load" in sent[0], false, "default must still refuse");
    store.set(ALLOW_OVERSIZED_LOAD_STORAGE_KEY, "true");
    await call();
    assert.equal(sent[1].allow_oversized_load, true);
  });
}

test("a one-shot retry forwards the field without the setting", async () => {
  const sent = captureFetch({ entries: [], total_bytes: 0 });
  await images.getDiffusionDownloadPlan({ model_path: "Qwen/Qwen-Image", allow_oversized_load: true });
  await video.getVideoDownloadPlan({ model_path: "Lightricks/LTX-2", allow_oversized_load: true });
  assert.deepEqual(
    sent.map((b) => b.allow_oversized_load),
    [true, true],
  );
});

test("the plan and load-progress codes are parsed through", async () => {
  captureFetch({
    entries: [],
    total_bytes: 0,
    incompatible_reason: "needs about 56 GB",
    incompatible_code: UNIFIED_MEMORY_LOAD_REFUSAL,
  });
  const plan = await images.getDiffusionDownloadPlan({ model_path: "Qwen/Qwen-Image" });
  assert.equal(plan.incompatible_code, UNIFIED_MEMORY_LOAD_REFUSAL);
  captureFetch({ phase: "error", error: "needs about 56 GB", error_code: UNIFIED_MEMORY_LOAD_REFUSAL });
  assert.equal((await images.getDiffusionLoadProgress()).error_code, UNIFIED_MEMORY_LOAD_REFUSAL);
  assert.equal((await video.getVideoLoadProgress()).error_code, UNIFIED_MEMORY_LOAD_REFUSAL);
});

test("the label matches the setting the backend's refusal names", () => {
  const en = readSrc("i18n/locales/en.ts");
  assert.ok(en.includes(`allowOversizedLoad: "${ALLOW_OVERSIZED_LOAD_LABEL}"`));
  const backend = readText("../../backend/core/inference/diffusion_memory.py");
  assert.ok(backend.includes(ALLOW_OVERSIZED_LOAD_LABEL), "the refusal names a setting the UI does not show");
  assert.ok(backend.includes(`"${UNIFIED_MEMORY_LOAD_REFUSAL}"`));
});

test("both pages offer Load anyway behind a confirmation, and the setting lives in Model memory", () => {
  for (const page of ["features/images/images-page.tsx", "features/video/video-page.tsx"]) {
    const src = readSrc(page);
    assert.match(src, /useLoadAnywayPrompt\(\)/, page);
    assert.match(src, /\{loadAnywayDialog\}/, page);
    assert.match(src, /allow_oversized_load: advanced\.allow_oversized_load/, page);
    assert.match(src, /plan\.incompatible_code/, page);
    assert.match(src, /p\.error_code/, page);
  }
  const dialog = readSrc("components/load-anyway-dialog.tsx");
  // The toast action only opens the confirmation; Confirm is the one place that retries.
  assert.match(dialog, /onClick: \(\) => setPending\(\{ message, retry \}\)/);
  assert.match(dialog, /LOAD_ANYWAY_WARNING/);
  const section = readSrc("features/settings/components/model-memory-section.tsx");
  assert.match(section, /usePersistedToggle\(\s*ALLOW_OVERSIZED_LOAD_STORAGE_KEY/);
});

test("a Load anyway retry from an older pick never starts once a newer pick or load began", () => {
  const { retryWhileCurrent } = refusal;
  const pickSeq = { current: 1 };
  const started: string[] = [];
  const pick = pickSeq.current;
  const stale = retryWhileCurrent(() => pick === pickSeq.current, () => started.push("old"));
  pickSeq.current += 1; // the user picks another model before confirming
  stale();
  assert.equal(started.length, 0, "a stale confirm started the superseded model");
  const newer = pickSeq.current;
  const current = retryWhileCurrent(() => newer === pickSeq.current, () => started.push("new"));
  current();
  assert.deepEqual(started, ["new"], "a current confirm must still retry its pick");
});

test("both pages gate every Load anyway retry on the pick and retire old prompts on a new pick", () => {
  const dialog = readSrc("components/load-anyway-dialog.tsx");
  assert.match(dialog, /return \{ offer, dismiss, dialog \}/);
  for (const page of ["features/images/images-page.tsx", "features/video/video-page.tsx"]) {
    const src = readSrc(page);
    // The plan refusal's retry and the load refusal's retry both re-check the pick.
    assert.match(src, /offerLoadAnyway\(incompatible, retryWhileCurrent\(\(\) => pick === pickSeq\.current,/, page);
    assert.match(src, /retryWhileCurrent\(\s*\(\) => retryPick === pickSeq\.current && startLoad === loadSeq\.current,/, page);
    const beginPick = src.slice(src.indexOf("const beginPick = useCallback"));
    assert.match(beginPick.slice(0, beginPick.indexOf("}, [")), /dismissLoadAnyway\(\);/, page);
    const handleLoad = src.slice(src.indexOf("const handleLoad = useCallback"));
    assert.match(handleLoad.slice(0, handleLoad.indexOf("const settle =")), /dismissLoadAnyway\(\);/, page);
  }
});
