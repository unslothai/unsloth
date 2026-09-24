// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  type BenchRun,
  type LoadPayload,
  type RunResult,
  type Variant,
  aggregate,
  baseLine,
  canonicalSpec,
  contextSteps,
  depthSeries,
  familyOf,
  footerLines,
  headline,
  ngramArgs,
  offloadDemand,
  offloadVariants,
  promptsFor,
  rampingRows,
  servedMismatch,
  sweepVariants,
  osShort,
  toMarkdown,
  tuneVerdict,
  userExtraArgs,
  variantLoad,
} from "../src/features/benchmarks/lib/bench-math.ts";

function result(variant: string, rep: number, tps: number | null, extra: Partial<RunResult> = {}): RunResult {
  return {
    variant,
    rep,
    warmup: false,
    promptIndex: 0,
    tps,
    promptTps: null,
    promptTokens: 40,
    genTokens: 256,
    ttftMs: 120,
    wallMs: 5000,
    clientTps: null,
    draftN: null,
    draftAccepted: null,
    loadMs: null,
    at: rep,
    ...extra,
  };
}

const OFF: Variant = { label: "Speculation off", load: { speculative_type: "off", spec_draft_n_max: null } };
const MTP3: Variant = { label: "MTP · 3 draft tokens", load: { speculative_type: "mtp", spec_draft_n_max: 3 } };
const MTP4: Variant = { label: "MTP · 4 draft tokens", load: { speculative_type: "mtp", spec_draft_n_max: 4 } };

test("aggregate orders fastest first, baseline last, and measures against the baseline", () => {
  const rows = aggregate(
    [
      result(OFF.label, 0, 29),
      result(OFF.label, 1, 30),
      result(MTP3.label, 0, 55, { draftN: 100, draftAccepted: 78 }),
      result(MTP3.label, 1, 57, { draftN: 100, draftAccepted: 78 }),
      result(MTP4.label, 0, 33),
    ],
    [OFF, MTP3, MTP4],
    OFF.label,
  );
  assert.deepEqual(
    rows.map((r) => r.label),
    [MTP3.label, MTP4.label, OFF.label],
  );
  assert.equal(rows[0].mean, 56);
  assert.equal(Math.round(rows[0].pct ?? 0), 90);
  assert.equal(rows[0].acceptRate, 0.78);
  assert.equal(rows[2].isBaseline, true);
  assert.equal(rows[2].pct, null);
});

test("warm-ups are left out of the mean but a missing server rate falls back to the client one", () => {
  const rows = aggregate(
    [result(MTP3.label, 0, 10, { warmup: true }), result(MTP3.label, 1, 50), result(OFF.label, 0, null, { clientTps: 25 })],
    [OFF, MTP3],
    null,
  );
  const mtp = rows.find((r) => r.label === MTP3.label);
  const off = rows.find((r) => r.label === OFF.label);
  assert.equal(mtp?.mean, 50);
  assert.equal(mtp?.n, 1);
  assert.equal(off?.mean, 25);
  assert.equal(off?.clientOnly, true);
});

test("a row that ramps across runs is flagged as a cache effect", () => {
  const ngram: Variant = { label: "Ngram", load: { speculative_type: "ngram" } };
  const rows = aggregate(
    [41, 96, 117, 123].map((tps, i) => result(ngram.label, i, tps)).concat([55, 56, 57].map((tps, i) => result(MTP3.label, i, tps))),
    [ngram, MTP3],
    null,
  );
  assert.deepEqual(
    rampingRows(rows).map((r) => r.label),
    ["Ngram"],
  );
});

test("headline names the winner as a multiple of the baseline", () => {
  const rows = aggregate([result(OFF.label, 0, 29.25), result(MTP3.label, 0, 56)], [OFF, MTP3], OFF.label);
  assert.equal(headline(rows), "MTP · 3 draft tokens runs 1.9× speculation off. 1 of 1 settings beat it.");
});

test("families follow the mode, including legacy spellings", () => {
  assert.equal(familyOf({ speculative_type: "draft-mtp" }), "mtp");
  assert.equal(familyOf({ speculative_type: "ngram-mod" }), "ngram");
  assert.equal(familyOf({ speculative_type: "none" }), "off");
  assert.equal(familyOf({ cache_type_kv: "q8_0" }), "other");
  assert.equal(canonicalSpec("draft-mtp,ngram-mod"), "mtp+ngram");
  assert.equal(canonicalSpec(""), "auto");
  assert.equal(canonicalSpec("default"), "auto");
});

test("every preset has unique labels, since a label is a row's identity", () => {
  for (const kind of ["spec", "draft", "ngram", "dspark", "kv", "context", "parallel", "tune"] as const) {
    const labels = sweepVariants(kind, 131072).map((v) => v.label);
    assert.equal(new Set(labels).size, labels.length, kind);
  }
});

test("context steps stop at the model's window and keep at most five rows", () => {
  assert.deepEqual(contextSteps(32768), [4096, 8192, 16384, 32768]);
  assert.deepEqual(contextSteps(40000), [4096, 8192, 16384, 32768, 40000]);
  assert.deepEqual(contextSteps(262144), [16384, 32768, 65536, 131072, 262144]);
  assert.deepEqual(contextSteps(2048), [2048]);
});

test("each variant sends its extra args explicitly, so ngram tuning never leaks into the next row", () => {
  const base: LoadPayload = { model_path: "unsloth/m", llama_extra_args: userExtraArgs(["--threads", "8", ...ngramArgs({ match: 12, min: 8 })]) };
  assert.deepEqual(base.llama_extra_args, ["--threads", "8"]);
  const tuned = variantLoad(base, {
    label: "Ngram · match 16, min 8",
    load: { speculative_type: "ngram", llama_extra_args: ngramArgs({ match: 16, min: 8 }) },
  });
  assert.deepEqual(tuned.llama_extra_args, ["--threads", "8", ...ngramArgs({ match: 16, min: 8 })]);
  assert.equal(tuned.force_reload, true);
  const plain = variantLoad(base, MTP3);
  assert.deepEqual(plain.llama_extra_args, ["--threads", "8"]);
  assert.equal(plain.speculative_type, "mtp");
});

test("a fallen-back or silently kept load is caught before it is measured", () => {
  assert.equal(servedMismatch(MTP3, { speculative_type: "mtp" }), null);
  assert.equal(servedMismatch(MTP3, { speculative_type: "default", spec_fallback_reason: "binary_no_mtp" }), "this llama.cpp build has no MTP support");
  assert.equal(servedMismatch(MTP3, { speculative_type: "ngram" }), "Studio served ngram instead of mtp");
  // A KV row never picked a mode, so an old fallback on the base load is not its problem.
  const kv: Variant = { label: "KV cache q4_0", load: { cache_type_kv: "q4_0" } };
  assert.equal(servedMismatch(kv, { spec_fallback_reason: "binary_no_mtp", cache_type_kv: "q4_0" }), null);
  assert.equal(servedMismatch(kv, { cache_type_kv: "f16" }), "Studio served KV f16 instead of q4_0");
});

test("custom prompts split on --- lines", () => {
  assert.deepEqual(promptsFor("custom", "one\n---\ntwo\n  ---  \nthree"), ["one", "two", "three"]);
  assert.deepEqual(promptsFor("custom", "   "), []);
  assert.equal(promptsFor("code", "").length, 5);
});

test("depth lines leave out tuned ngram rows and families with a single point", () => {
  const tuned: Variant = { label: "MTP + ngram · match 8, min 8", load: { speculative_type: "mtp+ngram", spec_draft_n_max: 3, llama_extra_args: ["--spec-ngram-mod-n-match", "8"] } };
  const rows = aggregate(
    [result(MTP3.label, 0, 56), result(MTP4.label, 0, 33), result(tuned.label, 0, 130)],
    [MTP3, MTP4, tuned],
    null,
  );
  const depth = depthSeries(rows, [MTP3, MTP4, tuned]);
  assert.deepEqual(
    depth.map((s) => [s.family, s.points.map((p) => p.n)]),
    [["mtp", [3, 4]]],
  );
});

test("the markdown export carries provenance and the rows that did not run", () => {
  const run: BenchRun = {
    id: "r",
    createdAt: Date.UTC(2026, 8, 10),
    finishedAt: null,
    model: "unsloth/Qwen3.8-27B-GGUF",
    ggufVariant: "Q4_K_M",
    kv: "q8_0",
    context: 32768,
    config: {
      sweep: "draft",
      variants: [OFF, MTP3],
      baseline: OFF.label,
      promptSet: "chat",
      customPrompt: "",
      rotatePrompts: true,
      maxTokens: 256,
      warmup: 1,
      repetitions: 3,
      temperature: 0.7,
      seed: 3407,
      restoreAfter: true,
    },
    meta: { gpu: "AMD Radeon AI PRO R9700", vramGb: 31.86, backend: "vulkan", llamaTag: "b10840" },
    outcomes: [
      { label: OFF.label, state: "done" },
      { label: "DFlash · 3 draft tokens", state: "skipped", reason: "this model ships no drafter for this mode" },
    ],
    results: [result(OFF.label, 0, 29), result(MTP3.label, 0, 56)],
  };
  assert.deepEqual(footerLines(run), [
    "Qwen3.8-27B Q4_K_M · KV q8_0 · ctx 32K · 3 measured runs per setting (+1 warm-up) · 256 max tokens · prompts rotated",
    "AMD Radeon AI PRO R9700 32 GB · Vulkan",
    "llama.cpp b10840 · 2026-09-10",
  ]);
  const md = toMarkdown(run, aggregate(run.results, run.config.variants, OFF.label));
  assert.match(md, /\| MTP · 3 draft tokens \| 56\.0 tok\/s \| \+93% \|/);
  assert.match(md, /- DFlash · 3 draft tokens: skipped, this model ships no drafter for this mode/);
});

test("chat's settings print in the footer, with the swept one marked varied", () => {
  const base = [
    { field: "cache_type_kv", label: "KV Cache Dtype", value: "q8_0" },
    { field: "speculative_type", label: "Speculative Decoding", value: "MTP+Ngram" },
    { field: "spec_draft_n_max", label: "Draft Tokens", value: "3" },
    { field: "n_parallel", label: "Parallel Slots", value: "2" },
  ];
  assert.equal(baseLine(base, [OFF, MTP3]), "KV Cache Dtype q8_0 · Speculative Decoding varied · Draft Tokens varied · Parallel Slots 2");
  const kv: Variant = { label: "KV cache f16", load: { cache_type_kv: "f16" } };
  assert.equal(baseLine(base, [kv]), "KV Cache Dtype varied · Speculative Decoding MTP+Ngram · Draft Tokens 3 · Parallel Slots 2");
});

test("auto-tune covers off, MTP depths, ngram and the pair, and prefers the simpler row inside the margin", () => {
  const variants = sweepVariants("tune");
  assert.deepEqual(
    variants.map((v) => v.load.speculative_type),
    ["off", "mtp", "mtp", "mtp", "mtp", "ngram", "mtp+ngram", "mtp+ngram"],
  );
  const results: RunResult[] = [];
  const rates: Record<string, number> = {
    "Speculation off": 30,
    "MTP · 1 draft token": 38,
    "MTP · 2 draft tokens": 41.5,
    "MTP · 3 draft tokens": 42, // fastest, but within 3% of the 2-token row
    "MTP · 4 draft tokens": 39,
    "Ngram · match 24, min 48 (stock)": 31,
    "MTP + ngram · 2 draft tokens": 42.5, // fastest of all, one more knob than MTP alone
    "MTP + ngram · 3 draft tokens": 40,
  };
  for (const [label, tps] of Object.entries(rates)) results.push(result(label, 0, tps));
  const verdict = tuneVerdict(aggregate(results, variants, null), variants);
  assert.ok(verdict);
  assert.equal(verdict.pick.label, "MTP · 2 draft tokens");
  assert.equal(verdict.fastest?.label, "MTP + ngram · 2 draft tokens");
  assert.equal(verdict.off?.label, "Speculation off");
  assert.equal(Math.round(verdict.gain ?? 0), 38);
});

test("os strings read as a person would say them", () => {
  assert.equal(osShort("Windows-11-10.0.26200-SP0"), "Windows 11 (10.0.26200)");
  assert.equal(osShort("macOS-15.3-arm64-arm-64bit"), "macOS 15.3 arm64");
  assert.equal(osShort("Linux-6.8.0-45-generic-x86_64-with-glibc2.39"), "Linux 6.8.0");
  assert.equal(osShort(null), "");
});

test("offload rows scale to the MoE layer count and run from the least VRAM to the most", () => {
  const rows = offloadVariants({ layers: 48, moeLayers: 40 });
  assert.deepEqual(
    rows.map((r) => r.label),
    [
      "Studio auto",
      "Experts on CPU · all 40 layers",
      "Experts on CPU · 30 of 40 layers",
      "Experts on CPU · 20 of 40 layers",
      "Experts on CPU · 10 of 40 layers",
      "All on GPU",
    ],
  );
  assert.equal(rows[0].load.gpu_memory_mode, "auto");
  // Every row pins the same context, so auto and manual compare.
  assert.ok(rows.every((r) => r.load.max_seq_length === 8192));
  const demands = rows.slice(1).map((r) => offloadDemand(r.load) ?? -1);
  assert.deepEqual([...demands].sort((a, b) => a - b), demands);
  assert.equal(offloadDemand(rows[0].load), null);
});

test("a dense model gets GPU layer steps, and an unknown shape still gets sensible rows", () => {
  const dense = offloadVariants({ layers: 40, moeLayers: 0 });
  assert.deepEqual(
    dense.map((r) => r.label),
    [
      "Studio auto",
      "GPU layers · 10 of 40",
      "GPU layers · 20 of 40",
      "GPU layers · 30 of 40",
      "All on GPU",
    ],
  );
  assert.ok(dense.every((r) => (r.load.n_cpu_moe ?? 0) === 0));
  const unknown = offloadVariants(null);
  assert.equal(unknown[1].load.n_cpu_moe, 999);
  assert.equal(new Set(unknown.map((r) => r.label)).size, unknown.length);
});

test("a value every row shares prints as pinned, not varied", () => {
  const line = baseLine(
    [
      { field: "max_seq_length", label: "Context Length", value: "Auto" },
      { field: "n_cpu_moe", label: "MoE on CPU", value: "0" },
    ],
    offloadVariants({ layers: 48, moeLayers: 40 }),
  );
  assert.match(line, /Context Length 8192 \(pinned\)/);
  assert.match(line, /MoE on CPU varied/);
});
