// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Live prefill / decode figures, the Generation details setting, and the loaded placement readout.

import assert from "node:assert/strict";
import test from "node:test";

import type { LiveDecodeProgress, LivePrefillProgress } from "../src/features/chat/stores/chat-runtime-store.ts";
import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const {
  formatLiveDecode,
  formatPrefillDetails,
  formatPrefillTimeLeft,
  prefillLabelShown,
  prefillPercent,
  readGenerationProgress,
} = await import("../src/features/chat/lib/generation-progress.ts");
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

test("a follow-up in a long chat counts only the part the cache did not cover", () => {
  // 29,000 of 30,000 tokens reused: halfway through the new 1,000, not 98% of the whole prompt.
  const followUp = { processed: 29500, total: 30000, cached: 29000, tokPerS: 1000 };
  assert.equal(prefillPercent(followUp), 50);
  assert.equal(formatPrefillDetails(followUp, "en-US"), "500 / 1,000 tokens · 1,000 tok/s");
  // llama.cpp evaluates at least one token, but a cache that covers it all is done, not NaN.
  assert.equal(prefillPercent({ processed: 30000, total: 30000, cached: 30000, tokPerS: null }), 100);
});

test("the time left follows the rate so far and drops out under a second", () => {
  assert.equal(formatPrefillTimeLeft({ processed: 4118, total: 18880, cached: 0, tokPerS: 1401.2 }), "11s left");
  assert.equal(formatPrefillTimeLeft({ processed: 5000, total: 200000, cached: 0, tokPerS: 1500 }), "2m 10s left");
  // The first sample's rate is noise, and at the end there is nothing worth counting down.
  assert.equal(formatPrefillTimeLeft({ processed: 22, total: 18844, cached: 0, tokPerS: 242 }), null);
  assert.equal(formatPrefillTimeLeft({ processed: 18500, total: 18880, cached: 0, tokPerS: 1400 }), null);
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

test("the prefill details give the rate when there is one", () => {
  assert.equal(
    formatPrefillDetails({ processed: 4118, total: 18880, cached: 0, tokPerS: 1401.2 }, "en-US"),
    "4,118 / 18,880 tokens · 1,401 tok/s",
  );
  assert.equal(formatPrefillDetails({ processed: 900, total: null, cached: null, tokPerS: null }, "en-US"), "900 tokens");
});

test("the prefill label shows only for a pass expected to keep it up, then stays for that pass", () => {
  const sample = (processed: number, total: number, tokPerS: number | null, cached = 0) => ({
    processed,
    total,
    cached,
    tokPerS,
  });
  const after = (prefill: LivePrefillProgress, prefillShown: boolean, decode: LiveDecodeProgress | null = null) => ({
    prefill,
    prefillShown,
    decode,
  });
  // llama.cpp's first sample, a few tokens in: its rate is noise, so no estimate yet.
  assert.equal(prefillLabelShown(undefined, sample(22, 18844, 242)), false);
  // One batch into a long prompt: about 10 s left.
  const long = sample(4118, 18880, 1401.2);
  assert.equal(prefillLabelShown(after(sample(22, 18844, 242), false), long), true);
  // Stays up near the end, where the estimate alone would drop it.
  assert.equal(prefillLabelShown(after(long, true), sample(18000, 18880, 1390)), true);
  // A 5,000-token prompt at 2,600 tok/s has about 1 s left after its first batch, so it never shows.
  const fast = sample(2048, 5000, 2600);
  assert.equal(prefillLabelShown(after(sample(0, 5000, null), false), fast), false);
  assert.equal(prefillLabelShown(after(fast, false), sample(4096, 5000, 2600)), false);
  // A cache hit leaves too few fresh tokens for a rate.
  assert.equal(prefillLabelShown(undefined, sample(29000, 30000, 0.5, 29000)), false);
  // The next pass of a tool loop starts over: decode figures came in between.
  const decode = { tokens: 40, tokPerS: 50, draftN: null, draftAccepted: null };
  assert.equal(prefillLabelShown(after(long, true, decode), sample(2048, 5000, 2600)), false);
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
  assert.match(store, /\[key\]: \{ prefill, prefillShown, decode: null \}/);
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

test("the placement readout names expert weights and gives both memory figures", () => {
  // gemma-4-26B-A4B Q8_0 on a 3090: --fit's verdict "31 layers ( 8 overflowing)", the first of them GATE-only.
  assert.deepEqual(
    describePlacement({ offloaded: 31, total: 31, overflowLayers: 8, gpuWeightsMib: 20709.15, hostWeightsMib: 5639.4, expertCacheMib: null }),
    { layers: "All 31 layers on the GPU, expert weights of 8 layers in RAM", memory: "20 GiB in VRAM · 5.5 GiB in RAM" },
  );
  assert.equal(
    describePlacement({ offloaded: 31, total: 31, overflowLayers: 1, gpuWeightsMib: 20000, hostWeightsMib: 900, expertCacheMib: null })
      .layers,
    "All 31 layers on the GPU, expert weights of 1 layer in RAM",
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

test("a GPU build that fell back to the CPU is not reported as GPU layers", () => {
  // CUDA build, no usable device: llama.cpp still logs "offloaded 29/29 layers to GPU", buffers AMX + CPU_Mapped.
  assert.deepEqual(
    describePlacement({ offloaded: 29, total: 29, overflowLayers: null, gpuWeightsMib: 0, hostWeightsMib: 499.8, expertCacheMib: null }),
    { layers: "All 29 layers on the CPU", memory: "0.5 GiB in RAM" },
  );
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
    layers: "All 31 layers on the GPU, all experts in RAM (6.0 GiB expert cache on the GPU)",
    memory: "3.8 GiB in VRAM · 22 GiB in RAM",
  });
});

test("one memory pool gets one figure, and amounts under 0.1 GiB read in MiB", () => {
  // Qwen3.5-4B Q4_K_M under Metal on an M1 Mac mini: MTL0 2,603.5 MiB and CPU 497.3 MiB, one pool.
  const metal = { offloaded: 33, total: 33, overflowLayers: null, gpuWeightsMib: 2603.5, hostWeightsMib: 497.31, expertCacheMib: null };
  assert.deepEqual(describePlacement(metal, "unified memory"), {
    layers: "All 33 layers on the GPU",
    memory: "3.0 GiB in unified memory",
  });
  assert.equal(describePlacement(metal).memory, "2.5 GiB in VRAM · 0.5 GiB in RAM");
  // A small model's token embeddings would otherwise print as "0.0 GiB in RAM".
  assert.equal(
    describePlacement({ offloaded: 31, total: 31, overflowLayers: 0, gpuWeightsMib: 112.6, hostWeightsMib: 28.7, expertCacheMib: null }).memory,
    "0.1 GiB in VRAM · 29 MiB in RAM",
  );
});

test("the prefill label gives way once decoding started, not at 100%", () => {
  const thread = readSrc("components/assistant-ui/thread.tsx");
  const status = thread.slice(thread.indexOf("const GeneratingStatus"), thread.indexOf("// Placeholder when stop fires"));
  assert.match(status, /!labelShown \|\| percent === null \|\| decoding\)/);
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
