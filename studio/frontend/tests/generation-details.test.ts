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
  prefillSecondsLeft,
  withPrefillFit,
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

const prefill = (
  fields: Omit<LivePrefillProgress, "elapsedMs" | "fit"> & Partial<LivePrefillProgress>,
): LivePrefillProgress => ({ elapsedMs: null, fit: null, ...fields });

test("prefill frames read into store shape, clamped, and give a whole percent", () => {
  const update = readGenerationProgress({
    type: "generation_progress",
    phase: "prefill",
    processed: 6000,
    total: 5316,
    cached: 7000,
    tok_per_s: 1326.4,
    time_ms: 3950,
  });
  assert.equal(update?.phase, "prefill");
  if (update?.phase !== "prefill") return;
  assert.deepEqual(update.prefill, prefill({ processed: 5316, total: 5316, cached: 7000, tokPerS: 1326.4, elapsedMs: 3950 }));
  assert.equal(prefillPercent(prefill({ processed: 2048, total: 5316, cached: 0, tokPerS: null })), 38);
  assert.equal(prefillPercent(prefill({ processed: 10, total: null, cached: null, tokPerS: null })), null);
});

test("a follow-up in a long chat counts only the part the cache did not cover", () => {
  // 29,000 of 30,000 tokens reused: halfway through the new 1,000, not 98% of the whole prompt.
  const followUp = prefill({ processed: 29500, total: 30000, cached: 29000, tokPerS: 1000 });
  assert.equal(prefillPercent(followUp), 50);
  assert.equal(formatPrefillDetails(followUp, "en-US"), "500 / 1,000 tokens · 1,000 tok/s");
  // llama.cpp evaluates at least one token, but a cache that covers it all is done, not NaN.
  assert.equal(prefillPercent(prefill({ processed: 30000, total: 30000, cached: 30000, tokPerS: null })), 100);
});

test("the time left follows the rate so far and drops out under a second", () => {
  assert.equal(formatPrefillTimeLeft(prefill({ processed: 4118, total: 18880, cached: 0, tokPerS: 1401.2 })), "11s left");
  assert.equal(formatPrefillTimeLeft(prefill({ processed: 5000, total: 200000, cached: 0, tokPerS: 1500 })), "2m 10s left");
  // The first sample's rate is noise, and at the end there is nothing worth counting down.
  assert.equal(formatPrefillTimeLeft(prefill({ processed: 22, total: 18844, cached: 0, tokPerS: 242 })), null);
  assert.equal(formatPrefillTimeLeft(prefill({ processed: 18500, total: 18880, cached: 0, tokPerS: 1400 })), null);
});

// llama-server's own [processed, time_ms] samples for two prompts. The first, 30,382 tokens through
// llama-server directly, ended at 34,847 ms; the second, 27,268 tokens through Studio, ended at
// 18,229 ms and its first batch was slow (warm-up), which a fit through the first sample misreads.
const LONG_PROMPTS = [
  {
    total: 30382,
    endMs: 34847,
    samples: [
      [0, 0], [1, 6], [2049, 1230], [4097, 2643], [6145, 4180], [8193, 5911], [10241, 7781],
      [12289, 9807], [14337, 11955], [16385, 14277], [18433, 16762], [20481, 19450], [22529, 22273],
      [24577, 25239], [26625, 28324], [28673, 31509], [29866, 33929],
    ],
  },
  {
    total: 27268,
    endMs: 18229,
    samples: [
      [1, 2], [2049, 1122], [4097, 2033], [6145, 3026], [8193, 4029], [10241, 5119], [12289, 6341],
      [14337, 7784], [16385, 9124], [18433, 10551], [20481, 12066], [22529, 13665], [24577, 15436],
      [26625, 17363],
    ],
  },
];

test("the time left allows for later prompt tokens costing more than earlier ones", () => {
  for (const { total, endMs, samples } of LONG_PROMPTS) {
    let previous: { prefill: LivePrefillProgress; prefillShown: boolean; decode: null } | undefined;
    const errors: number[] = [];
    const rateSoFar: number[] = [];
    for (const [processed, time_ms] of samples) {
      const tok_per_s = time_ms > 0 ? (processed / time_ms) * 1000 : 0;
      const update = readGenerationProgress({ phase: "prefill", processed, total, cached: 0, tok_per_s, time_ms });
      assert.equal(update?.phase, "prefill");
      if (update?.phase !== "prefill") return;
      const current = withPrefillFit(previous, update.prefill);
      previous = { prefill: current, prefillShown: false, decode: null };
      const left = prefillSecondsLeft(current);
      const actual = (endMs - time_ms) / 1000;
      // From 30% in (the fit has its samples and full weight by 45%) to the last few seconds.
      if (left === null || processed < 0.3 * total || actual < 3) continue;
      const rateError = ((total - processed) / tok_per_s - actual) / actual;
      const error = (left - actual) / actual;
      rateSoFar.push(rateError);
      assert.ok(Math.abs(error) <= Math.abs(rateError) + 0.01, `${processed}: ${error} vs ${rateError}`);
      if (processed >= 0.45 * total) errors.push(error);
    }
    assert.ok(errors.length >= 6);
    // The rate so far ran 32-45% short at every one of these points; past 45% the fit stays within 20%.
    assert.ok(rateSoFar.every((e) => e < -0.3), String(rateSoFar));
    assert.ok(errors.every((e) => Math.abs(e) < 0.2), String(errors));
  }
});

test("one slow batch early in a long prompt does not read as a slowdown for the rest of it", () => {
  // 2,048 tokens every 500 ms, with batch 4 taking 250 ms longer and the rest back to normal. A curve fit
  // to four samples and extrapolated over the whole prompt read 18 s for 6 s left at 32K and 271 s
  // for 30 s at 131K.
  for (const total of [32768, 131072]) {
    let previous: { prefill: LivePrefillProgress; prefillShown: boolean; decode: null } | undefined;
    let elapsed = 0;
    const totalMs = (total / 2048) * 500 + 250;
    for (let batch = 1; batch * 2048 <= total; batch++) {
      elapsed += 500 + (batch === 4 ? 250 : 0);
      const processed = batch * 2048;
      const update = readGenerationProgress({
        phase: "prefill", processed, total, cached: 0, tok_per_s: (processed / elapsed) * 1000, time_ms: elapsed,
      });
      if (update?.phase !== "prefill") return;
      const current = withPrefillFit(previous, update.prefill);
      previous = { prefill: current, prefillShown: false, decode: null };
      const left = prefillSecondsLeft(current);
      const actual = (totalMs - elapsed) / 1000;
      if (left === null || actual < 2) continue;
      assert.ok(left < actual * 1.6, `${total} at ${processed}: ${left.toFixed(1)} s shown for ${actual.toFixed(1)} s left`);
    }
  }
});

test("the time-left fit sums a pass's timed samples and starts over with the next pass", () => {
  const at = (processed: number, elapsedMs: number, total = 30382) =>
    prefill({ processed, total, cached: 0, tokPerS: 1000, elapsedMs });
  const pass = (p: LivePrefillProgress) => ({ prefill: p, prefillShown: false, decode: null });
  // A few tokens in is noise, so nothing is summed yet.
  const first = withPrefillFit(undefined, at(1, 6));
  assert.equal(first.fit, null);
  const second = withPrefillFit(pass(first), at(2049, 1230));
  const third = withPrefillFit(pass(second), at(4097, 2643));
  assert.equal(third.fit?.sums[0], 2);
  // Until four timed samples, one slow batch swings the fit: the rate so far stands.
  assert.equal(prefillSecondsLeft(third), (30382 - 4097) / 1000);
  // A repeated frame is not a new sample.
  assert.equal(withPrefillFit(pass(third), at(4097, 2643)).fit?.sums[0], 2);
  const fourth = withPrefillFit(pass(third), at(6145, 4180));
  assert.equal(prefillSecondsLeft(fourth), (30382 - 6145) / 1000);
  const fifth = withPrefillFit(pass(fourth), at(8193, 5911));
  assert.equal(fifth.fit?.sums[0], 4);
  assert.equal(fifth.fit?.first, 2049, "the span the curve may be extended by starts at the first timed sample");
  assert.notEqual(prefillSecondsLeft(fifth), (30382 - 8193) / 1000);
  // An older llama-server without time_ms keeps the rate so far.
  const untimed = prefill({ processed: 4097, total: 30382, cached: 0, tokPerS: 1000 });
  assert.equal(withPrefillFit(undefined, untimed).fit, null);
  // The next round of a tool loop (a decode in between, or a new total) starts over.
  const decode = { tokens: 9, tokPerS: 50, draftN: null, draftAccepted: null };
  assert.equal(withPrefillFit({ ...pass(fifth), decode }, at(4097, 900)).fit?.sums[0], 1);
  assert.equal(withPrefillFit(pass(fifth), at(4097, 900, 40000)).fit?.sums[0], 1);
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
    formatPrefillDetails(prefill({ processed: 4118, total: 18880, cached: 0, tokPerS: 1401.2 }), "en-US"),
    "4,118 / 18,880 tokens · 1,401 tok/s",
  );
  assert.equal(formatPrefillDetails(prefill({ processed: 900, total: null, cached: null, tokPerS: null }), "en-US"), "900 tokens");
});

test("the prefill label shows only for a pass expected to keep it up, then stays for that pass", () => {
  const sample = (processed: number, total: number, tokPerS: number | null, cached = 0) =>
    prefill({ processed, total, cached, tokPerS });
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
