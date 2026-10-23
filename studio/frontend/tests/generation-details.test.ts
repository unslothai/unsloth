// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Live prefill / decode figures, the Generation details setting, and the loaded placement readout.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { formatLiveDecode, formatPrefillDetails, prefillPercent, readGenerationProgress } = await import(
  "../src/features/chat/lib/generation-progress.ts"
);
const { describePlacement, loadedPlacementFrom } = await import(
  "../src/features/chat/lib/partial-offload.ts"
);
const { useChatPreferencesStore } = await import(
  "../src/features/chat/stores/chat-preferences-store.ts"
);
const { normalizeChatGenerationChunkPayload } = await import(
  "../src/features/chat/api/chat-generation-api.ts"
);
const { generationChunkCountsTowardTiming } = await import(
  "../src/features/chat/utils/chat-generation-recovery.ts"
);

test("prefill frames read into store shape, clamped, and give a whole percent", () => {
  const update = readGenerationProgress({
    type: "generation_progress",
    phase: "prefill",
    processed: 6000,
    total: 5316,
    cached: 7000,
    tok_per_s: 1326.4,
  });
  assert.equal(update?.phase, "prefill");
  if (update?.phase !== "prefill") return;
  assert.deepEqual(update.prefill, { processed: 5316, total: 5316, cached: 7000, tokPerS: 1326.4 });
  assert.equal(prefillPercent({ processed: 2048, total: 5316, cached: 0, tokPerS: null }), 38);
  assert.equal(prefillPercent({ processed: 10, total: null, cached: null, tokPerS: null }), null);
});

test("malformed frames are dropped rather than drawn as NaN", () => {
  assert.equal(readGenerationProgress(null), null);
  assert.equal(readGenerationProgress({ phase: "prefill", processed: "12" }), null);
  assert.equal(readGenerationProgress({ phase: "decode", tokens: 0 }), null);
  assert.equal(readGenerationProgress({ phase: "warmup" }), null);
});

test("decode frames keep draft acceptance only when both counts are there", () => {
  const update = readGenerationProgress({ phase: "decode", tokens: 40, tok_per_s: 90, draft_n: 23, draft_accepted: 30 });
  assert.equal(update?.phase, "decode");
  if (update?.phase !== "decode") return;
  assert.deepEqual(update.decode, { tokens: 40, tokPerS: 90, draftN: 23, draftAccepted: 23 });
});

test("the prefill details hide the first sample's noisy rate", () => {
  assert.equal(
    formatPrefillDetails({ processed: 22, total: 18844, cached: 0, tokPerS: 242 }, "en-US"),
    "22 / 18,844 tokens",
  );
  assert.equal(
    formatPrefillDetails({ processed: 4118, total: 18880, cached: 0, tokPerS: 1401.2 }, "en-US"),
    "4,118 / 18,880 tokens · 1,401 tok/s",
  );
});

test("the live line writes every number in the reader's locale", () => {
  const decode = { tokens: 3060, tokPerS: 58.25, draftN: 5030, draftAccepted: 1791 };
  assert.equal(
    formatLiveDecode(decode, "en-US"),
    "3,060 tokens · 58.3 tok/s · draft 1,791/5,030 accepted (36%)",
  );
  // A locale that groups with "." must not get "." as the rate's decimal point too.
  assert.equal(
    formatLiveDecode(decode, "de-DE"),
    "3.060 tokens · 58,3 tok/s · draft 1.791/5.030 accepted (36%)",
  );
});

test("both live lines wrap instead of clipping on a narrow chat column", () => {
  // .generating-glint is nowrap and overflow: hidden; at 390 px the prefill details and the decode line ran past the edge.
  const thread = readSrc("components/assistant-ui/thread.tsx");
  const prefill = thread.slice(thread.indexOf('data-slot="prefill-progress"'), thread.indexOf("Processing prompt"));
  assert.match(prefill, /whitespace-normal!/);
  const line = readSrc("components/assistant-ui/live-generation-line.tsx");
  assert.doesNotMatch(line, /\bshrink-0\b/);
  assert.match(line, /\bmin-w-0\b/);
});

test("Generation details is off by default and wired into Settings > Chat > Display", () => {
  assert.equal(useChatPreferencesStore.getInitialState().showGenerationDetails, false);
  const store = readSrc("features/chat/stores/chat-preferences-store.ts");
  assert.match(store, /showGenerationDetails: saved\?\.showGenerationDetails === true/);
  const tab = readSrc("features/settings/tabs/chat-tab.tsx");
  const times = tab.indexOf('t("settings.chat.showReplyTimes")');
  const details = tab.indexOf('t("settings.chat.showGenerationDetails")');
  assert.ok(times !== -1 && details > times, "the switch sits after Reply times");
  assert.match(readSrc("features/settings/settings-search.ts"), /"settings\.chat\.showGenerationDetailsKeywords"/);
});

test("the live line gates on the preference before any assistant-ui selector", () => {
  const line = readSrc("components/assistant-ui/live-generation-line.tsx");
  const gate = line.slice(line.indexOf("export const LiveGenerationLine"), line.indexOf("const RunningGate"));
  assert.match(gate, /return enabled \? <RunningGate \/> : null;/);
  assert.doesNotMatch(gate, /useAuiState/);
});

test("persisted runs re-tag generation_progress and keep it out of timing", () => {
  const frame = { type: "generation_progress", phase: "prefill", processed: 1, total: 2 };
  const normalized = normalizeChatGenerationChunkPayload(frame) as { _generationProgress?: unknown };
  assert.deepEqual(normalized._generationProgress, frame);
  assert.equal(generationChunkCountsTowardTiming(normalized), false);
});

// The runtime store pulls in the app shell, so like chat-model-residency.test.ts this reads its source.
test("live progress is per thread, transient, and re-keyed with the thread's other run handles", () => {
  const store = readSrc("features/chat/stores/chat-runtime-store.ts");
  // A new prompt pass (the next round of a tool loop) drops the previous round's decode figures.
  assert.match(store, /\[key\]: \{ prefill, decode: null \}/);
  assert.match(store, /move\(state\.liveGenerationByThreadId, "liveGenerationByThreadId"\);/);
  const adapter = readSrc("features/chat/api/chat-adapter.ts");
  assert.match(
    adapter,
    /clearActiveDiffusionCanvasForThread\(cleanupKey\);\s*runtime\.clearLiveGenerationForThread\(cleanupKey\);/,
  );
  // Handled before the first-token stamp, like the other control frames.
  const progress = adapter.indexOf("_generationProgress");
  const firstToken = adapter.indexOf("firstTokenTime = Date.now() - streamStartTime;");
  assert.ok(progress !== -1 && progress < firstToken);
});

test("the placement readout names experts and gives both memory figures", () => {
  assert.deepEqual(
    describePlacement({ offloaded: 31, total: 31, overflowLayers: 8, gpuWeightsMib: 20709.15, hostWeightsMib: 5639.4, expertCacheMib: null }),
    { layers: "All 31 layers on the GPU, 8 with their experts in RAM", memory: "20 GiB in VRAM · 5.5 GiB in RAM" },
  );
  assert.deepEqual(
    describePlacement({ offloaded: 41, total: 41, overflowLayers: 0, gpuWeightsMib: 8161.75, hostWeightsMib: 417.3, expertCacheMib: null }),
    { layers: "All 41 layers on the GPU", memory: "8.0 GiB in VRAM · 0.4 GiB in RAM" },
  );
  assert.deepEqual(
    describePlacement({ offloaded: 20, total: 31, overflowLayers: null, gpuWeightsMib: 12288, hostWeightsMib: 5120, expertCacheMib: null }),
    { layers: "20 of 31 layers on the GPU, 11 on the CPU", memory: "12 GiB in VRAM · 5.0 GiB in RAM" },
  );
  // A Manual load reports its expert offload as n_cpu_moe, not through --fit.
  assert.equal(
    loadedPlacementFrom({ is_gguf: true, offloaded_layers: 31, offload_total_layers: 31, gpu_memory_mode: "manual", n_cpu_moe: 15 })
      ?.overflowLayers,
    15,
  );
  assert.equal(
    loadedPlacementFrom({ is_gguf: true, offloaded_layers: 31, offload_total_layers: 31, gpu_memory_mode: "auto", n_cpu_moe: 15 })
      ?.overflowLayers,
    null,
  );
  assert.equal(loadedPlacementFrom({ is_gguf: false, offload_total_layers: 31 }), null);
  assert.equal(loadedPlacementFrom({ is_gguf: true }), null);
});

test("--moe-cache-mib auto reads as every expert in RAM with a GPU cache", () => {
  // --fit names no layer on this path; the backend reports the model's MoE layer count instead.
  const placement = loadedPlacementFrom({
    is_gguf: true,
    offloaded_layers: 31,
    offload_total_layers: 31,
    overflow_layers: 30,
    gpu_weights_mib: 3900,
    host_weights_mib: 22400,
    expert_cache_mib: 6144,
  });
  assert.ok(placement);
  assert.deepEqual(describePlacement(placement), {
    layers: "All 31 layers on the GPU, all experts in RAM (6.0 GiB cache on the GPU)",
    memory: "3.8 GiB in VRAM · 22 GiB in RAM",
  });
});

test("the prefill label gives way once the prompt is done or decoding started", () => {
  const thread = readSrc("components/assistant-ui/thread.tsx");
  const status = thread.slice(thread.indexOf("const GeneratingStatus"), thread.indexOf("// Placeholder when stop fires"));
  assert.match(status, /percent >= 100 \|\| decoding\)/);
});

test("a followed run (reload, second tab) shows live figures and clears them on exit", () => {
  const provider = readSrc("features/chat/runtime-provider.tsx");
  assert.match(provider, /if \("_generationProgress" in chunk\) \{\s*if \(caughtUp\(update\.run\)\)/);
  assert.match(provider, /\} finally \{\s*useChatRuntimeStore\.getState\(\)\.clearLiveGenerationForThread\(threadId\);/);
});

test("a background load cannot drop the visible model's loaded placement", () => {
  // setCheckpoint clears loadedPlacement when the visible model is restored, so the snapshot must carry it back.
  const adapter = readSrc("features/chat/api/chat-adapter.ts");
  const keys = adapter.slice(
    adapter.indexOf("const VISIBLE_MODEL_RUNTIME_KEYS = ["),
    adapter.indexOf("] as const satisfies"),
  );
  assert.match(keys, /"loadedPlacement"/);
});
