// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type { DiffusionGenerateRequest } from "../src/features/images/api.ts";
import {
  availableAxes,
  buildCellRequest,
  cellCoords,
  executionOrder,
  gridNegativePrompt,
  MAX_COMPARE_CELLS,
  planCompare,
  type CompareContext,
  type CompareGridPlan,
} from "../src/features/images/compare/compare-axes.ts";
import { createCompareRun, gridParameters, gridRecipeFields, gridTitles, sliceCells } from "../src/features/images/compare/compare-run.ts";
import { joinAxisList, parseAxisValues, splitAxisList } from "../src/features/images/compare/compare-values.ts";
import { drawCellsInTurn, gridInfotext, gridLayout, MAX_CANVAS_PIXELS, MAX_CANVAS_SIDE, withPngText, wrapText } from "../src/features/images/compare/compose-grid.ts";
import { parseA1111Parameters, pngTextChunks } from "../src/features/images/lib/image-metadata.ts";
import { readSrc } from "./helpers/kit.ts";

const values = (text: string, type: "int" | "float" | "str") => {
  const r = parseAxisValues(text, type);
  assert.ok(r.ok, `expected ${text} to parse: ${r.ok ? "" : r.error}`);
  return r.values;
};

test("A1111 list syntax: commas, newlines, CSV quoting, empty items dropped", () => {
  assert.deepEqual(splitAxisList('euler, "a, b",dpmpp_2m'), ["euler", "a, b", "dpmpp_2m"]);
  assert.deepEqual(splitAxisList('say ""hi""'), ['say ""hi""']);
  assert.deepEqual(splitAxisList('"say ""hi"""'), ['say "hi"']);
  assert.deepEqual(values("1,2\n3,,", "int"), [1, 2, 3]);
  assert.deepEqual(values(" red , blue ", "str"), ["red", "blue"]);
  assert.equal(joinAxisList(["DPM++ 2M", "a, b"]), 'DPM++ 2M, "a, b"');
});

test("integer ranges: a-b, a-b (+s), a-b (-s), a-b [n] (linspace then truncated)", () => {
  assert.deepEqual(values("1-5", "int"), [1, 2, 3, 4, 5]);
  assert.deepEqual(values("1-9 (+2)", "int"), [1, 3, 5, 7, 9]);
  assert.deepEqual(values("10-30 (+10)", "int"), [10, 20, 30]);
  assert.deepEqual(values("1-10(+4)", "int"), [1, 5, 9]);
  // A1111 stops a descending range one short (range(5, 2, -1)); this keeps the end the user typed.
  assert.deepEqual(values("5-1 (-1)", "int"), [5, 4, 3, 2, 1]);
  assert.deepEqual(values("5-1", "int"), [5, 4, 3, 2, 1]);
  // numpy.linspace(1, 10, 4) = 1, 4, 7, 10; linspace(0, 10, 4) = 0, 3.33, 6.67, 10 -> int() truncates.
  assert.deepEqual(values("1-10 [4]", "int"), [1, 4, 7, 10]);
  assert.deepEqual(values("0-10 [4]", "int"), [0, 3, 6, 10]);
  assert.deepEqual(values("-1, 7, 100-102", "int"), [-1, 7, 100, 101, 102]);
  assert.deepEqual(values("20, 1-3 [2]", "int"), [20, 1, 3]);
});

test("decimal ranges: a-b (+s) inclusive via arange, a-b [n] via linspace, 8-decimal rounding", () => {
  assert.deepEqual(values("1-2 (+0.5)", "float"), [1, 1.5, 2]);
  assert.deepEqual(values("0-0.3 (+0.1)", "float"), [0, 0.1, 0.2, 0.3]);
  assert.deepEqual(values("3-7 [3]", "float"), [3, 5, 7]);
  assert.deepEqual(values("0-1 (0.25)", "float"), [0, 0.25, 0.5, 0.75, 1]);
  assert.deepEqual(values("1.5, 2.25, 7", "float"), [1.5, 2.25, 7]);
  assert.deepEqual(values("1-2", "float"), [1, 2]);
});

test("bad values name the item", () => {
  const bad = (text: string, type: "int" | "float") => {
    const r = parseAxisValues(text, type);
    assert.equal(r.ok, false, text);
    return r.ok ? "" : r.error;
  };
  assert.match(bad("abc", "int"), /"abc" is not a number/);
  assert.match(bad("2.5", "int"), /not a whole number/);
  assert.match(bad("1-5 (+0)", "int"), /Bad range step/);
  assert.match(bad("", "int"), /No values/);
  assert.match(bad("0x10", "float"), /not a number/);
});

const SAMPLERS = [
  { id: "default", label: "Model default", schedule_types: ["automatic"] },
  { id: "euler", label: "Euler", schedule_types: ["automatic", "karras"] },
  { id: "euler_a", label: "Euler a", schedule_types: ["automatic"] },
  { id: "dpmpp_2m", label: "DPM++ 2M", schedule_types: ["automatic", "karras", "exponential"] },
];

function sdxl(over: Partial<CompareContext> = {}): CompareContext {
  return {
    prompt: "a red fox in the snow",
    negativePrompt: "blurry",
    negativeCapable: true,
    negativeMinGuidance: 1,
    guidanceFixed: false,
    guidance: 6,
    samplers: SAMPLERS,
    samplerChoice: { sampler: "default", scheduleType: "automatic" },
    shift: null,
    loraCapable: false,
    loras: [],
    sdControls: null,
    controlnetEngaged: false,
    ...over,
  };
}

const zimage = (over: Partial<CompareContext> = {}) =>
  sdxl({ samplers: [], guidanceFixed: true, shift: { node: "sd3", shift: 3 }, negativeCapable: false, ...over });

const ids = (ctx: CompareContext) => availableAxes(ctx).map((a) => a.id);

test("capability gating: the axis list follows what the loaded model exposes", () => {
  assert.deepEqual(ids(sdxl()), ["nothing", "seed", "steps", "guidance", "sampler", "schedule", "prompt_sr"]);
  // A distilled flow model: no sampler picker, guidance fixed, one static shift.
  assert.deepEqual(ids(zimage()), ["nothing", "seed", "steps", "shift", "prompt_sr"]);
  // ComfyUI ModelSamplingFlux: base and max shift.
  assert.ok(ids(zimage({ shift: { node: "flux", base_shift: 0.5, max_shift: 1.15 } })).includes("max_shift"));
  const loras = [
    { id: "a/style", weight: 1, label: "style" },
    { id: "b/detail", weight: 0.5, label: "detail" },
  ];
  assert.ok(!ids(sdxl({ loras })).includes("lora:0"));
  const withLoras = availableAxes(sdxl({ loras, loraCapable: true }));
  assert.deepEqual(
    withLoras.filter((a) => a.id.startsWith("lora:")).map((a) => a.label),
    ["LoRA weight: style", "LoRA weight: detail"],
  );
  // One sampler (the model default only) is no picker, so no axis.
  assert.ok(!ids(sdxl({ samplers: SAMPLERS.slice(0, 1) })).includes("sampler"));
});

test("SD controls axes appear only when status reports them (#14562), CFG rescale not with a ControlNet", () => {
  const sdControls = { clip_skip: { min: 1, max: 12 }, cfg_rescale: { min: 0, max: 1 }, variation_seed: true, vary_strength: 0.2 };
  const all = ids(sdxl({ sdControls }));
  for (const id of ["clip_skip", "cfg_rescale", "variation_seed", "variation_strength"]) assert.ok(all.includes(id), id);
  assert.ok(!ids(sdxl({ sdControls, controlnetEngaged: true })).includes("cfg_rescale"));
  const r = planCompare([{ axisId: "variation_seed", text: "1, 2" }], sdxl({ sdControls }));
  assert.ok(r.ok);
  const req = buildCellRequest(r.plan, 1, base(), sdxl({ sdControls }), { base: 5, keepRandom: false, seedFieldEmpty: false }) as unknown as Record<string, unknown>;
  assert.equal(req.variation_seed, 2);
  assert.equal(req.variation_strength, 0.2);
  const clip = planCompare([{ axisId: "clip_skip", text: "1-3" }], sdxl({ sdControls }));
  assert.ok(clip.ok);
  assert.equal((buildCellRequest(clip.plan, 2, base(), sdxl({ sdControls }), { base: 5, keepRandom: false, seedFieldEmpty: false }) as unknown as Record<string, unknown>).clip_skip, 3);
  assert.equal(planCompare([{ axisId: "clip_skip", text: "13" }], sdxl({ sdControls })).ok, false);
});

test("an axis the model lacks is refused by id, out-of-range values are refused", () => {
  const r = planCompare([{ axisId: "sampler", text: "euler" }], zimage());
  assert.equal(r.ok, false);
  assert.match(r.ok ? "" : r.error, /no sampler axis/);
  assert.equal(planCompare([{ axisId: "steps", text: "0, 5" }], sdxl()).ok, false);
  assert.equal(planCompare([{ axisId: "steps", text: "101" }], sdxl()).ok, false);
  assert.equal(planCompare([{ axisId: "guidance", text: "21" }], sdxl()).ok, false);
  assert.equal(planCompare([{ axisId: "shift", text: "0" }], zimage()).ok, false);
  const unknown = planCompare([{ axisId: "sampler", text: "euler, heun" }], sdxl());
  assert.match(unknown.ok ? "" : unknown.error, /Unknown sampler: heun/);
});

test("sampler values match the id or the A1111 label, case-insensitively", () => {
  const r = planCompare([{ axisId: "sampler", text: "Euler a, dpm++ 2m, euler" }], sdxl());
  assert.ok(r.ok);
  assert.deepEqual(r.plan.axes[0].values, ["euler_a", "dpmpp_2m", "euler"]);
  assert.deepEqual(r.plan.axes[0].labels, ["Euler a", "DPM++ 2M", "Euler"]);
});

test("cell cap: over the limit is refused with the count, at the limit is fine", () => {
  assert.equal(MAX_COMPARE_CELLS, 64);
  const over = planCompare(
    [
      { axisId: "steps", text: "1-5" },
      { axisId: "guidance", text: "1-4" },
      { axisId: "seed", text: "1-4" },
    ],
    sdxl(),
  );
  assert.equal(over.ok, false);
  assert.equal(over.ok ? 0 : over.cellCount, 80);
  assert.match(over.ok ? "" : over.error, /80 cells is over the limit of 64/);
  const at = planCompare(
    [
      { axisId: "steps", text: "1-4" },
      { axisId: "guidance", text: "1-4" },
      { axisId: "seed", text: "1-4" },
    ],
    sdxl(),
  );
  assert.ok(at.ok);
  assert.equal(at.plan.cellCount, 64);
  assert.equal(planCompare([{ axisId: "nothing", text: "" }], sdxl()).ok, false);
  const dup = planCompare([{ axisId: "steps", text: "1" }, { axisId: "steps", text: "2" }], sdxl());
  assert.match(dup.ok ? "" : dup.error, /X and Y are both Steps/);
});

function base(over: Partial<DiffusionGenerateRequest> = {}): DiffusionGenerateRequest {
  return {
    prompt: "a red fox in the snow",
    negative_prompt: "blurry",
    width: 1024,
    height: 1024,
    steps: 25,
    guidance: 6,
    seed: 999,
    batch_size: 4,
    live_preview: true,
    ...over,
  };
}

const plan = (inputs: Array<{ axisId: string; text: string }>, ctx = sdxl(), keepRandom = false) => {
  const r = planCompare(inputs, ctx, { keepRandomSeeds: keepRandom, rng: () => 0.5 });
  assert.ok(r.ok, r.ok ? "" : r.error);
  return r.plan;
};

test("cell requests: batch 1, one shared seed, the cell's steps / sampler / schedule", () => {
  const p = plan([
    { axisId: "sampler", text: "euler, euler_a, dpmpp_2m" },
    { axisId: "steps", text: "10, 20, 30" },
  ]);
  assert.equal(p.cellCount, 9);
  const ctx = sdxl({ samplerChoice: { sampler: "euler", scheduleType: "karras" } });
  const reqs = Array.from({ length: 9 }, (_, i) =>
    buildCellRequest(p, i, base(), ctx, { base: 42, keepRandom: false, seedFieldEmpty: true }),
  );
  assert.ok(reqs.every((r) => r.batch_size === 1 && r.seed === 42));
  // index = ix + iy*3: ix = sampler, iy = steps.
  assert.deepEqual(
    reqs.map((r) => [r.sampler, r.schedule_type, r.steps]),
    [
      ["euler", "karras", 10],
      // Euler a takes no Karras: the form's schedule falls back to automatic (omitted).
      ["euler_a", undefined, 10],
      ["dpmpp_2m", "karras", 10],
      ["euler", "karras", 20],
      ["euler_a", undefined, 20],
      ["dpmpp_2m", "karras", 20],
      ["euler", "karras", 30],
      ["euler_a", undefined, 30],
      ["dpmpp_2m", "karras", 30],
    ],
  );
  // "Model default" is the shipped scheduler: no sampler field at all, as the form sends it.
  const d = plan([{ axisId: "sampler", text: "Model default" }]);
  const dr = buildCellRequest(d, 0, base({ sampler: "euler" }), sdxl(), { base: 1, keepRandom: false, seedFieldEmpty: false });
  assert.equal("sampler" in dr, false);
});

test("seeds: axis values, -1 fixed once per value (A1111 fix_axis_seeds), Keep -1 draws per cell", () => {
  const p = plan([{ axisId: "seed", text: "-1, 7" }, { axisId: "steps", text: "4, 8" }]);
  assert.deepEqual(p.axes[0].values, [2 ** 31, 7]);
  const s = (pl: CompareGridPlan, i: number, keepRandom: boolean, empty: boolean, rng = () => 0.25) =>
    buildCellRequest(pl, i, base(), sdxl(), { base: 42, keepRandom, seedFieldEmpty: empty, rng }).seed;
  assert.deepEqual([0, 1, 2, 3].map((i) => s(p, i, false, true)), [2 ** 31, 7, 2 ** 31, 7]);
  const kept = plan([{ axisId: "seed", text: "-1, 7" }], sdxl(), true);
  assert.equal(kept.axes[0].labels[0], "Seed: random");
  assert.equal(s(kept, 0, true, true), 2 ** 30);
  const steps = plan([{ axisId: "steps", text: "4, 8" }]);
  assert.equal(s(steps, 1, false, true), 42);
  assert.equal(s(steps, 1, true, true), 2 ** 30);
  // A typed seed is not -1, so Keep -1 leaves it alone.
  assert.equal(s(steps, 1, true, false), 42);
});

test("guidance re-derives the negative prompt; Prompt S/R replaces in both prompts and needs a hit", () => {
  const g = plan([{ axisId: "guidance", text: "1, 7" }]);
  const lo = buildCellRequest(g, 0, base(), sdxl(), { base: 1, keepRandom: false, seedFieldEmpty: false });
  const hi = buildCellRequest(g, 1, base(), sdxl(), { base: 1, keepRandom: false, seedFieldEmpty: false });
  assert.equal(lo.negative_prompt, undefined);
  assert.equal(hi.negative_prompt, "blurry");
  const ctx = sdxl({ prompt: "a red fox, red scarf", negativePrompt: "red eyes" });
  const sr = plan([{ axisId: "prompt_sr", text: "red, blue, green" }], ctx);
  const req = (i: number) =>
    buildCellRequest(sr, i, base({ prompt: ctx.prompt, negative_prompt: ctx.negativePrompt }), ctx, { base: 1, keepRandom: false, seedFieldEmpty: false });
  assert.equal(req(0).prompt, "a red fox, red scarf");
  assert.equal(req(1).prompt, "a blue fox, blue scarf");
  assert.equal(req(2).negative_prompt, "green eyes");
  const miss = planCompare([{ axisId: "prompt_sr", text: "purple, blue" }], ctx);
  assert.match(miss.ok ? "" : miss.error, /did not find "purple"/);
});

test("LoRA weight targets one row; 0 drops it; the prompt enhancer is off for every cell", () => {
  const ctx = sdxl({
    loraCapable: true,
    loras: [
      { id: "a/style", weight: 1, label: "style" },
      { id: "b/detail", weight: 0.5, label: "detail" },
    ],
  });
  const p = plan([{ axisId: "lora:1", text: "0, 0.5, 1" }], ctx);
  const req = (i: number) =>
    buildCellRequest(p, i, base({ enhance_prompt: true }), ctx, { base: 1, keepRandom: false, seedFieldEmpty: false });
  assert.deepEqual(req(0).loras, [{ id: "a/style", weight: 1 }]);
  assert.deepEqual(req(2).loras, [
    { id: "a/style", weight: 1 },
    { id: "b/detail", weight: 1 },
  ]);
  assert.equal(req(0).enhance_prompt, false);
  const z = plan([{ axisId: "shift", text: "1, 3, 6" }], zimage());
  const zr = buildCellRequest(z, 2, base(), zimage(), { base: 1, keepRandom: false, seedFieldEmpty: false });
  assert.equal(zr.shift, 6);
  assert.equal("enhance_prompt" in zr, false);
});

test("run order: A1111's z, y, x nesting; costly axes outermost, grouped by shape class", () => {
  // No costly axis: row by row (x fastest), exactly A1111's default loop.
  const p = plan([{ axisId: "steps", text: "1, 2, 3" }, { axisId: "seed", text: "1, 2" }]);
  assert.deepEqual(p.order, [0, 1, 2, 3, 4, 5]);
  // CFG on X crosses the CFG threshold (1): it becomes the outer loop, the two CFG-off values (0, 1) run first
  // even though 7 sits between them, so the denoiser batch changes once, not three times.
  const g = plan([{ axisId: "guidance", text: "0, 7, 1" }, { axisId: "steps", text: "4, 8" }]);
  const ran = g.order.map((i) => {
    const [ix, iy] = cellCoords(g, i);
    return [g.axes[0].values[ix], g.axes[1].values[iy]];
  });
  assert.deepEqual(ran, [
    [0, 4],
    [0, 8],
    [1, 4],
    [1, 8],
    [7, 4],
    [7, 8],
  ]);
  let switches = 0;
  for (let k = 1; k < ran.length; k++) if ((Number(ran[k][0]) > 1) !== (Number(ran[k - 1][0]) > 1)) switches++;
  assert.equal(switches, 1);
  // Prompt S/R outranks guidance: each prompt is entered once.
  const sr = plan(
    [
      { axisId: "guidance", text: "1, 7" },
      { axisId: "prompt_sr", text: "red, blue" },
      { axisId: "steps", text: "4, 8" },
    ],
  );
  const prompts = sr.order.map((i) => cellCoords(sr, i)[1]);
  assert.deepEqual(prompts, [0, 0, 0, 0, 1, 1, 1, 1]);
  assert.deepEqual([...sr.order].sort((a, b) => a - b), [0, 1, 2, 3, 4, 5, 6, 7]);
  assert.deepEqual(executionOrder(sr.axes).length, 8);
});

test("the run object: execution order to cell index, titles, one image per cell, Z slices", () => {
  const p = plan([
    { axisId: "steps", text: "4, 8" },
    { axisId: "nothing", text: "" },
    { axisId: "seed", text: "1, 2" },
  ]);
  assert.deepEqual(gridTitles(p), ["Seed: 1", "Seed: 2"]);
  const run = createCompareRun(p, sdxl(), { keepRandomSeeds: false, seedFieldEmpty: true });
  assert.equal(run.cellCount, 4);
  assert.equal(run.title(0), "Cell 1/4: Steps: 4, Seed: 1");
  for (let i = 0; i < 4; i++) {
    const req = run.request(i, base(), 5);
    run.onImages(i, [{ id: `img${i}`, steps: req.steps, seed: req.seed } as never]);
  }
  assert.deepEqual(
    sliceCells(p, run.results, 1).map((r) => (r as unknown as { seed: number }).seed),
    [2, 2],
  );
  assert.equal(run.requests.filter(Boolean).length, 4);
});

const measure = (text: string, size: number) => text.length * size * 0.5;

test("grid layout: A1111 font size and bands, labels wrapped, cells placed row-major", () => {
  const layout = gridLayout(
    {
      cols: 3,
      rows: 2,
      cellWidth: 512,
      cellHeight: 512,
      colLabels: ["Euler", "Euler a", "DPM++ 2M"],
      rowLabels: ["Steps: 10", "Steps: 20"],
      title: "",
      legend: true,
      margin: 8,
    },
    measure,
  );
  assert.equal(layout.fontSize, Math.round(1024 / 25));
  const padLeft = Math.floor((512 * 3) / 4);
  assert.equal(layout.cells[0].x, padLeft);
  assert.equal(layout.cells[1].x, padLeft + 520);
  assert.equal(layout.cells[3].y, layout.cells[0].y + 520);
  assert.equal(layout.width, padLeft + 3 * 512 + 2 * 8);
  assert.equal(layout.height, layout.cells[0].y + 2 * 512 + 8);
  assert.ok(layout.cells[0].y > 0, "column labels take a band above the cells");
  assert.equal(layout.colText.length, 3);
  assert.equal(layout.rowText[1].x, padLeft / 2);
  assert.equal(layout.title, null);
  const bare = gridLayout({ cols: 2, rows: 1, cellWidth: 64, cellHeight: 32, colLabels: ["a", "b"], rowLabels: [""], title: "Z", legend: false, margin: 0 }, measure);
  assert.deepEqual([bare.width, bare.height, bare.cells[1].x, bare.cells[0].y], [128, 32, 64, 0]);
  const titled = gridLayout({ cols: 1, rows: 1, cellWidth: 256, cellHeight: 256, colLabels: ["x"], rowLabels: [""], title: "Seed: 1", legend: true, margin: 0 }, measure);
  assert.ok(titled.title && titled.colText[0].y > titled.title.y);
  assert.deepEqual(wrapText("one two three four", 40, 10, measure), ["one two", "three", "four"]);
});

test("grid layout shrinks cells to fit a browser canvas", () => {
  const big = gridLayout(
    { cols: 8, rows: 8, cellWidth: 2752, cellHeight: 2752, colLabels: [], rowLabels: [], title: "", legend: false, margin: 0 },
    measure,
  );
  assert.ok(big.scale < 1);
  assert.ok(big.width <= MAX_CANVAS_SIDE && big.height <= MAX_CANVAS_SIDE);
  // Bounded by memory, not just the browser's 2^28 ceiling: an 8x8 grid of 2752 px cells stays under 128 MiB RGBA.
  assert.ok(big.width * big.height <= MAX_CANVAS_PIXELS);
  assert.equal(big.cells.length, 64);
  const labelled = gridLayout(
    { cols: 8, rows: 8, cellWidth: 2752, cellHeight: 2752, colLabels: Array(8).fill("Steps: 20"), rowLabels: Array(8).fill("CFG: 7"), title: "Seed: 1", legend: true, margin: 16 },
    measure,
  );
  assert.ok(labelled.width * labelled.height <= MAX_CANVAS_PIXELS);
});

test("the grid PNG carries an A1111-style parameters chunk the metadata reader parses", () => {
  const png = Uint8Array.from(
    atob("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="),
    (c) => c.charCodeAt(0),
  );
  const info = gridInfotext({
    prompt: "a red fox",
    negativePrompt: "blurry",
    fields: [["Steps", 20], ["Seed", 42], ["Model", null]],
    axes: [{ name: "X", type: "Sampler", text: "euler, dpmpp_2m" }],
  });
  assert.equal(info, 'a red fox\nNegative prompt: blurry\nSteps: 20, Seed: 42, Script: X/Y/Z plot, X Type: Sampler, X Values: "euler, dpmpp_2m"');
  const out = withPngText(png, "parameters", info);
  assert.equal(pngTextChunks(out).parameters, info);
  assert.equal(withPngText(new Uint8Array([1, 2, 3]), "k", "v").length, 3);
});

test("the Images page (its own lazy chunk) renders the panel and loops cells through its own Generate path", () => {
  const page = readSrc("features/images/images-page.tsx");
  assert.match(page, /^import ComparePanel from "\.\/compare\/compare-panel";$/m);
  // The panel and the rest of compare/ are reached only through the Images page, never from the startup graph.
  const root = readSrc("app/routes/__root.tsx");
  assert.match(root, /lazy\(\(\) =>\s*import\("@\/features\/images"\)/);
  assert.match(page, /for \(let i = 0; i < runTotal; i\+\+\)/);
  assert.match(page, /generateDiffusionImage\(compareRequest\(compare, i, baseSeed, \{/);
  assert.match(page, /compare\?\.onImages\(i, res\.images\)/);
  // The lost-response branch also reports the cell, from the records the gallery gained.
  assert.match(page, /compare\?\.onImages\(i, galleryCache\.images\.filter\(\(image\) => !knownIds\.has\(image\.id\)\)\)/);
  assert.match(page, /workflow === "create"/);
});

test("a Schedule type the cell's sampler does not take is refused, not silently run as Automatic", () => {
  const bad = planCompare(
    [
      { axisId: "sampler", text: "euler, euler_a" },
      { axisId: "schedule", text: "karras" },
    ],
    sdxl(),
  );
  assert.equal(bad.ok, false);
  assert.match(bad.ok ? "" : bad.error, /Karras is not available with Euler a/);
  // The form's Model default takes no schedule either.
  const onDefault = planCompare([{ axisId: "schedule", text: "automatic, karras" }], sdxl());
  assert.match(onDefault.ok ? "" : onDefault.error, /Karras is not available with Model default/);
  const good = plan([
    { axisId: "sampler", text: "euler, dpmpp_2m" },
    { axisId: "schedule", text: "automatic, karras" },
  ]);
  const reqs = [0, 1, 2, 3].map((i) =>
    buildCellRequest(good, i, base(), sdxl(), { base: 1, keepRandom: false, seedFieldEmpty: false }),
  );
  assert.deepEqual(
    reqs.map((r) => [r.sampler, r.schedule_type]),
    [
      ["euler", undefined],
      ["dpmpp_2m", undefined],
      ["euler", "karras"],
      ["dpmpp_2m", "karras"],
    ],
  );
});

test("Prompt S/R counts a negative-prompt match only where some cell's CFG engages the negative prompt", () => {
  const ctx = sdxl({ prompt: "a fox", negativePrompt: "red eyes", guidance: 1 });
  const off = planCompare([{ axisId: "prompt_sr", text: "red, blue" }], ctx);
  assert.equal(off.ok, false);
  assert.match(off.ok ? "" : off.error, /did not find "red" in cell "red"/);
  // A CFG axis that raises guidance over the threshold engages it for those cells.
  // Per cell: a CFG 1 cell would not use the negative prompt, so its substitution would do nothing.
  const mixed = planCompare([{ axisId: "prompt_sr", text: "red, blue" }, { axisId: "guidance", text: "1, 7" }], ctx);
  assert.match(mixed.ok ? "" : mixed.error, /did not find "red" in cell "red, CFG: 1"/);
  const withCfg = planCompare([{ axisId: "prompt_sr", text: "red, blue" }, { axisId: "guidance", text: "3, 7" }], ctx);
  assert.ok(withCfg.ok);
  const req = buildCellRequest(withCfg.ok ? withCfg.plan : (null as never), 3, base({ prompt: "a fox", negative_prompt: undefined, guidance: 1 }), ctx, {
    base: 1,
    keepRandom: false,
    seedFieldEmpty: false,
  });
  assert.equal(req.negative_prompt, "blue eyes");
  assert.equal(planCompare([{ axisId: "prompt_sr", text: "red, blue" }], sdxl({ ...ctx, guidance: 6, negativeCapable: false })).ok, false);
  assert.ok(planCompare([{ axisId: "prompt_sr", text: "red, blue" }], sdxl({ ...ctx, guidance: 6 })).ok);
});

test("LoRA weight cells: every weighted cell runs before any adapter-free one, across all axes", () => {
  const ctx = sdxl({ loraCapable: true, loras: [{ id: "a/style", weight: 1, label: "style" }], prompt: "a red fox" });
  const p = plan(
    [
      { axisId: "lora:0", text: "0, 0.5, 1" },
      { axisId: "prompt_sr", text: "red, blue" },
      { axisId: "steps", text: "4, 8" },
    ],
    ctx,
  );
  const hasLora = p.order.map((i) =>
    Boolean(buildCellRequest(p, i, base(), ctx, { base: 1, keepRandom: false, seedFieldEmpty: false }).loras),
  );
  const firstOff = hasLora.indexOf(false);
  assert.ok(firstOff > 0);
  assert.ok(hasLora.slice(firstOff).every((x) => !x), "a LoRA cell ran after an adapter-free one");
  // The backend's deferred compile rule: with LoRA-free runs back to back, a later LoRA request would be refused.
  let attached = true;
  let compiled = false;
  for (const lora of hasLora) {
    assert.ok(!(lora && compiled), "a LoRA cell would hit a compiled load");
    if (!lora && !attached) compiled = true;
    attached = lora;
  }
});

test("a lost-response recovery hands the cell its own record, matched by seed", () => {
  const p = plan([{ axisId: "seed", text: "11, 12" }]);
  const run = createCompareRun(p, sdxl(), { keepRandomSeeds: false, seedFieldEmpty: false });
  run.request(0, base(), 5);
  run.request(1, base(), 5);
  run.onImages(1, [{ id: "other", seed: 99 } as never, { id: "mine", seed: 12 } as never]);
  assert.equal((run.results[1] as unknown as { id: string }).id, "mine");
  run.onImages(0, [{ id: "a", seed: 1 } as never, { id: "b", seed: 2 } as never]);
  assert.equal(run.results[0], null, "an ambiguous set without the cell's seed is not guessed");
  run.onImages(0, [{ id: "only", seed: 11 } as never]);
  assert.equal((run.results[0] as unknown as { id: string }).id, "only");
});

test("grid metadata names the sampler and schedule whenever no axis varies them", () => {
  const steps = plan([{ axisId: "steps", text: "10, 20" }]);
  const first = base({ sampler: "dpmpp_2m", schedule_type: "karras" });
  const fields = Object.fromEntries(gridRecipeFields(steps, [first], { steps: 25, guidance: 6, width: 1024, height: 1024, model: "m" }));
  assert.equal(fields.Sampler, "DPM++ 2M");
  assert.equal(fields["Schedule type"], "Karras");
  const samplers = plan([{ axisId: "sampler", text: "euler, dpmpp_2m" }]);
  const varied = Object.fromEntries(gridRecipeFields(samplers, [first], { steps: 25, guidance: 6, width: 1024, height: 1024, model: "m" }));
  assert.equal("Sampler" in varied, false);
  assert.equal(varied["Schedule type"], "Karras");
  const plain = Object.fromEntries(gridRecipeFields(steps, [base()], { steps: 25, guidance: 6, width: 1024, height: 1024, model: "m" }));
  assert.equal(plain.Sampler, "Model default");
  assert.equal("Schedule type" in plain, false);
});

test("grid metadata writes a schedule or shift only when every cell ran it", () => {
  const form = { steps: 25, guidance: 6, width: 1024, height: 1024, model: "m" };
  // Euler a takes no Karras: its cell fell back to Automatic, so Karras is not the grid's schedule.
  const samplers = plan([{ axisId: "sampler", text: "euler, euler_a" }]);
  const mixed = Object.fromEntries(
    gridRecipeFields(samplers, [base({ sampler: "euler", schedule_type: "karras" }), base({ sampler: "euler_a" })], form),
  );
  assert.equal("Schedule type" in mixed, false);
  const same = Object.fromEntries(
    gridRecipeFields(samplers, [base({ sampler: "euler", schedule_type: "karras" }), null, base({ sampler: "euler_a", schedule_type: "karras" })], form),
  );
  assert.equal(same["Schedule type"], "Karras");
  // A custom static shift every cell shares survives; a Shift axis keeps it out of the shared fields.
  const steps = plan([{ axisId: "steps", text: "10, 20" }]);
  const shifted = Object.fromEntries(gridRecipeFields(steps, [base({ shift: 3 }), base({ shift: 3, steps: 20 })], form));
  assert.equal(shifted.Shift, 3);
  const none = Object.fromEntries(gridRecipeFields(steps, [base(), base({ steps: 20 })], form));
  assert.equal(none.Shift, undefined);
});

test("grid metadata: one every-cell-agrees rule for Flux shift, fitted size, seeds and variation strength", () => {
  const form = { steps: 25, guidance: 6, width: 2048, height: 2048, model: "m" };
  const steps = plan([{ axisId: "steps", text: "10, 20" }]);
  const fields = Object.fromEntries(
    gridRecipeFields(
      steps,
      [
        base({ steps: 10, width: 1024, height: 1024, base_shift: 0.5, max_shift: 1.15, ...({ variation_strength: 0.15 } as object) }),
        base({ steps: 20, width: 1024, height: 1024, base_shift: 0.5, max_shift: 1.15, ...({ variation_strength: 0.15 } as object) }),
      ],
      form,
    ),
  );
  assert.equal(fields["Base shift"], 0.5);
  assert.equal(fields["Max shift"], 1.15);
  // The size the cells were fitted to, not the 2048 form value.
  assert.equal(fields.Size, "1024x1024");
  assert.equal(fields.Seed, 999);
  assert.equal(fields["Variation seed strength"], 0.15);
  // A varied axis is never written as shared, even if only one cell ran.
  assert.equal(fields.Steps, undefined);
  // Keep -1 with an empty seed box: each cell drew its own seed, so none is shared.
  const random = Object.fromEntries(gridRecipeFields(steps, [base({ seed: 1 }), base({ seed: 2 })], form));
  assert.equal(random.Seed, undefined);
  // The implicit Vary strength a Variation seed axis adds reaches the grid's parameters and parses back.
  const ctx = sdxl({ sdControls: { variation_seed: true, vary_strength: 0.2 } as never });
  const vp = plan([{ axisId: "variation_seed", text: "1, 2" }], ctx);
  const reqs = vp.order.map((cell) => buildCellRequest(vp, cell, base(), ctx, { base: 5, keepRandom: false, seedFieldEmpty: false }));
  const info = gridParameters(vp, ctx, reqs, { ...form, keepRandomSeeds: false });
  assert.match(info, /Variation seed strength: 0\.2/);
  // The importer restores the Flux shift pair the grid writes.
  const recipe = parseA1111Parameters(gridInfotext({ prompt: "p", fields: [["Steps", 4], ["Base shift", 0.5], ["Max shift", 1.15]], axes: [] }));
  assert.equal(recipe?.base_shift, 0.5);
  assert.equal(recipe?.max_shift, 1.15);
});

test("grid cells are decoded and drawn one at a time; missing and failed cells keep the placeholder", async () => {
  let live = 0;
  let peak = 0;
  const drawn: number[] = [];
  const count = await drawCellsInTurn(
    ["a", null, "fail", "d"],
    async (r) => {
      live++;
      peak = Math.max(peak, live);
      await new Promise((resolve) => setTimeout(resolve, 2));
      if (r === "fail") {
        live--;
        throw new Error("decode");
      }
      return r;
    },
    (_bitmap, i) => {
      drawn.push(i);
      live--;
    },
  );
  assert.equal(peak, 1);
  assert.equal(count, 2);
  assert.deepEqual(drawn, [0, 3]);
  const panel = readSrc("features/images/compare/compare-panel.tsx");
  assert.doesNotMatch(panel, /Promise\.all/);
  assert.match(panel, /resizeWidth: rect\.w/);
});

test("Prompt S/R is validated per cell after the earlier axes' substitutions, in X, Y, Z order", () => {
  const ctx = sdxl({ prompt: "red fox" });
  // X=blue removes "red" before Y runs, so the Y=green cells would be mislabelled duplicates.
  const removed = planCompare([{ axisId: "prompt_sr", text: "red, blue" }, { axisId: "prompt_sr", text: "red, green" }], ctx);
  assert.match(removed.ok ? "" : removed.error, /Y: Prompt S\/R did not find "red" in cell "blue, red"/);
  // X introduces Y's term in every cell: accepted, and the chain applies in order.
  const added = plan([{ axisId: "prompt_sr", text: "fox, fox cub" }, { axisId: "prompt_sr", text: "fox, wolf" }], ctx);
  const req = (i: number) => buildCellRequest(added, i, base({ prompt: "red fox" }), ctx, { base: 1, keepRandom: false, seedFieldEmpty: false }).prompt;
  assert.deepEqual([0, 1, 2, 3].map(req), ["red fox", "red fox cub", "red wolf", "red wolf cub"]);
});

test("numbers reject whitespace inside a literal; a space after a sign is A1111 syntax", () => {
  for (const text of ["1 2", "1 e2", "2. 5", "1 0-20"]) {
    assert.equal(parseAxisValues(text, "float").ok, false, text);
    assert.equal(parseAxisValues(text, "int").ok, false, text);
  }
  assert.deepEqual(values(" 12 , - 3, 1 - 3", "int"), [12, -3, 1, 2, 3]);
  assert.deepEqual(values("1e2, + 0.5", "float"), [100, 0.5]);
});

test("grid parameters: every LoRA by exact id at the first cell's weights, negative prompt when any cell engages it", () => {
  const ctx = sdxl({
    loraCapable: true,
    loras: [
      { id: "krea/Krea-2-LoRA-retro", weight: 0.8, label: "retro" },
      { id: "hub:odd/id", weight: 1, label: "odd" },
    ],
    prompt: "a fox",
    negativePrompt: "blurry",
    guidance: 1,
  });
  const p = plan([{ axisId: "guidance", text: "1, 7" }, { axisId: "steps", text: "4, 8" }], ctx);
  const first = buildCellRequest(p, p.order[0], base({ prompt: "a fox", negative_prompt: undefined, guidance: 1, loras: [{ id: "krea/Krea-2-LoRA-retro", weight: 0.8 }, { id: "hub:odd/id", weight: 1 }] }), ctx, { base: 1, keepRandom: false, seedFieldEmpty: false });
  assert.equal(first.negative_prompt, undefined, "the first cell runs at CFG 1 without the negative prompt");
  const info = gridParameters(p, ctx, [first], { steps: 4, guidance: 1, width: 64, height: 64, model: "m", keepRandomSeeds: false });
  const lines = info.split("\n");
  assert.equal(lines[0], "a fox");
  assert.equal(lines[1], "Negative prompt: blurry");
  assert.match(lines[2], /LoRAs: "krea\/Krea-2-LoRA-retro: 0\.8, hub:odd\/id: 1"/);
  // Hub ids restore from the field, not through the local-only prompt-tag matcher.
  assert.deepEqual(parseA1111Parameters(info)?.loras, ["krea/Krea-2-LoRA-retro:0.8", "hub:odd/id:1"]);
  // No cell engages it: not recorded.
  const flat = plan([{ axisId: "steps", text: "4, 8" }], ctx);
  assert.doesNotMatch(gridParameters(flat, ctx, [first], { steps: 4, guidance: 1, width: 64, height: 64, model: "m", keepRandomSeeds: false }), /Negative prompt/);
});

test("a lost response keeps the seed the compare cell was sent for Reuse last seed", () => {
  const p = plan([{ axisId: "seed", text: "11, 12" }]);
  const run = createCompareRun(p, sdxl(), { keepRandomSeeds: false, seedFieldEmpty: false });
  run.request(1, base(), 5);
  assert.equal(run.sentSeed(1), 12);
  assert.equal(run.sentSeed(0), undefined);
  const page = readSrc("features/images/images-page.tsx");
  assert.match(page, /setSessionSeed\(compare\?\.sentSeed\(i\) \?\? runBaseSeed\)/);
});

test("compare wiring: busy through composition, gated on preset hydration, no batch offset seed guard", () => {
  const panel = readSrc("features/images/compare/compare-panel.tsx");
  // `running` clears only in the finally that also wraps composition, and the button honours it.
  const body = panel.slice(panel.indexOf("const run = async"), panel.indexOf("const download ="));
  assert.ok(body.indexOf("composeGrid(") < body.indexOf("setRunning(null)"));
  assert.match(panel, /disabled=\{busy \|\| running !== null \|\| !preview\.ok\}/);
  const page = readSrc("features/images/images-page.tsx");
  assert.match(page, /busy=\{busy !== null \|\| compareRun !== null \|\| !imagePresets\.hydrated\}/);
  assert.match(page, /if \(busy !== null \|\| !imagePresets\.hydrated\) return;\n\s*compareOnce\.current = run;/);
  assert.match(page, /if \(!compare && baseSeed > Number\.MAX_SAFE_INTEGER - \(runs \* batchSize - 1\)\)/);
});

test("grid metadata records a ControlNet every cell shares, without its image", () => {
  const form = { steps: 25, guidance: 6, width: 1024, height: 1024, model: "m" };
  const steps = plan([{ axisId: "steps", text: "10, 20" }]);
  const controlnet = { id: "xinsir/canny", image: "data:image/png;base64,AAAA", control_type: "canny", strength: 0.7 };
  const reqs = steps.order.map((cell) => buildCellRequest(steps, cell, base({ controlnet }), sdxl(), { base: 1, keepRandom: false, seedFieldEmpty: false }));
  const fields = Object.fromEntries(gridRecipeFields(steps, reqs, form));
  assert.equal(fields["ControlNet 0"], "Module: canny, Model: xinsir/canny, Weight: 0.7");
  const mixed = Object.fromEntries(gridRecipeFields(steps, [base({ controlnet }), base()], form));
  assert.equal("ControlNet 0" in mixed, false);
  const none = Object.fromEntries(gridRecipeFields(steps, [base(), base()], form));
  assert.equal("ControlNet 0" in none, false);
});

test("a compare cell refused by the memory estimate offers no Generate anyway retry", () => {
  const page = readSrc("features/images/images-page.tsx");
  // The queued retry would run after the grid ended, as a plain form generation.
  assert.match(page, /if \(offerAnyway && compare\) \{\n\s*toast\.error\(MEMORY_REFUSAL_TITLE, \{\n\s*description: [^\n]*ALLOW_OVERSIZED_LABEL/);
  const compareBranch = page.slice(page.indexOf("if (offerAnyway && compare)"), page.indexOf("} else if (offerAnyway)"));
  assert.doesNotMatch(compareBranch, /setOversizedRetryQueued/);
});

test("grid metadata, every field x shared / varied / axis / absent: written only when every cell agrees", () => {
  const form = { steps: 25, guidance: 6, width: 1024, height: 1024, model: "m" };
  const axesOf = (ids: string[]) => ({ axes: ids.map((id) => ({ def: { id } })) }) as unknown as CompareGridPlan;
  const cn = (strength: number) => ({ id: "xinsir/canny", image: "data:image/png;base64,AAAA", control_type: "canny", strength });
  // [request key, axis id, field, value a, value b, the field's text when every cell sent a, its text when absent]
  const table: Array<[string, string, string, unknown, unknown, string | number, string | number | undefined]> = [
    ["steps", "steps", "Steps", 20, 30, 20, undefined],
    ["sampler", "sampler", "Sampler", "euler", "dpmpp_2m", "Euler", "Model default"],
    ["schedule_type", "schedule", "Schedule type", "karras", "exponential", "Karras", undefined],
    ["shift", "shift", "Shift", 3, 6, 3, undefined],
    ["base_shift", "base_shift", "Base shift", 0.5, 0.75, 0.5, undefined],
    ["max_shift", "max_shift", "Max shift", 1.15, 1.5, 1.15, undefined],
    ["guidance", "guidance", "CFG scale", 6, 7, 6, undefined],
    ["seed", "seed", "Seed", 1, 2, 1, undefined],
    ["variation_seed", "variation_seed", "Variation seed", 5, 6, 5, undefined],
    ["variation_strength", "variation_strength", "Variation seed strength", 0.1, 0.3, 0.1, undefined],
    ["width", "width", "Size", 1024, 768, "1024x1024", undefined],
    ["clip_skip", "clip_skip", "Clip skip", 2, 3, 2, undefined],
    ["cfg_rescale", "cfg_rescale", "CFG rescale", 0.7, 0.35, 0.7, undefined],
    ["prompt_weighting", "prompt_weighting", "Emphasis", false, true, "None", undefined],
    ["enhance_prompt", "enhance_prompt", "Prompt enhancement", false, true, "off", undefined],
    // Each cell's unit is a fresh object: compared by content.
    ["controlnet", "controlnet", "ControlNet 0", cn(0.7), cn(0.5), "Module: canny, Model: xinsir/canny, Weight: 0.7", undefined],
  ];
  const cell = (key: string, value: unknown): DiffusionGenerateRequest => {
    const req: Record<string, unknown> = { prompt: "p", batch_size: 1, height: 1024, ...(key === "width" ? {} : { width: 1024 }) };
    if (value !== undefined) req[key] = key === "controlnet" ? { ...(value as object) } : value;
    return req as unknown as DiffusionGenerateRequest;
  };
  const fieldOf = (plan: CompareGridPlan, cells: DiffusionGenerateRequest[], name: string) =>
    Object.fromEntries(gridRecipeFields(plan, cells, form))[name];
  for (const [key, axisId, name, a, b, sharedText, absentText] of table) {
    const none = axesOf([]);
    assert.equal(fieldOf(none, [cell(key, a), cell(key, a), null as never, cell(key, a)], name), sharedText, `${name} shared`);
    assert.equal(fieldOf(none, [cell(key, a), cell(key, b)], name), undefined, `${name} varied across cells`);
    assert.equal(fieldOf(none, [cell(key, a), cell(key, undefined)], name), undefined, `${name} on some cells only`);
    assert.equal(fieldOf(axesOf([axisId]), [cell(key, a), cell(key, a)], name), undefined, `${name} on an axis`);
    assert.equal(fieldOf(none, [cell(key, undefined), cell(key, undefined)], name), absentText, `${name} absent`);
  }
});

test("a Var. strength axis runs every cell on one variation seed; the grid restores the enhancer off", () => {
  const sdControls = { clip_skip: null, cfg_rescale: null, variation_seed: true, vary_strength: 0.2 } as never;
  const ctx = sdxl({ sdControls });
  const p = plan([{ axisId: "variation_strength", text: "0.1, 0.3, 0.5" }], ctx);
  let draws = 0;
  const run = createCompareRun(p, ctx, { keepRandomSeeds: false, seedFieldEmpty: false, rng: () => (++draws, 0.25) });
  const seeds = p.order.map((_, i) => (run.request(i, base(), 7) as unknown as Record<string, unknown>).variation_seed);
  assert.deepEqual(seeds, [2 ** 30, 2 ** 30, 2 ** 30]);
  assert.equal(draws, 1, "drawn once per run, not per cell");
  // Keep -1 for seeds with an empty seed box: the variation seed stays random as well.
  const keep = createCompareRun(p, ctx, { keepRandomSeeds: true, seedFieldEmpty: true, rng: () => 0.25 });
  assert.equal((keep.request(0, base(), 7) as unknown as Record<string, unknown>).variation_seed, undefined);
  // The enhancer compare/ forces off is recorded and parsed back as off.
  const off = buildCellRequest(p, 0, base({ enhance_prompt: true }), ctx, { base: 1, keepRandom: false, seedFieldEmpty: false });
  assert.equal(off.enhance_prompt, false);
  const info = gridParameters(plan([{ axisId: "steps", text: "4, 8" }]), sdxl(), [off, off], { steps: 4, guidance: 1, width: 64, height: 64, model: "m", keepRandomSeeds: false });
  assert.match(info, /Prompt enhancement: off/);
  assert.equal(parseA1111Parameters(info)?.prompt_enhancement, "off");
});

test("the SD controls axes reach the backend's request fields, the form's variation seed unshifted per cell", async () => {
  const { readFileSync } = await import("node:fs");
  const backend = readFileSync(new URL("../../backend/models/inference.py", import.meta.url), "utf8");
  const sdControls = { clip_skip: { min: 1, max: 12 }, cfg_rescale: { min: 0, max: 1 }, variation_seed: true, vary_strength: 0.2 };
  const ctx = sdxl({ sdControls });
  for (const [axisId, text] of [["variation_seed", "1, 2"], ["variation_strength", "0.1, 0.3"], ["clip_skip", "1, 2"], ["cfg_rescale", "0, 0.7"]]) {
    const p = plan([{ axisId, text }], ctx);
    const run = createCompareRun(p, ctx, { keepRandomSeeds: false, seedFieldEmpty: false, rng: () => 0.5 });
    const req = run.request(1, base(), 7) as unknown as Record<string, unknown>;
    assert.notEqual(req[axisId], undefined, axisId);
    // Every key a cell sends is a field of the backend's generate request.
    for (const key of ["variation_seed", "variation_strength", "clip_skip", "cfg_rescale"]) {
      if (req[key] !== undefined) assert.match(backend, new RegExp(`^\\s+${key}: Optional\\[`, "m"), key);
    }
  }
  // A Var. seed axis cell carries the implicit strength the backend needs to apply it.
  const vs = plan([{ axisId: "variation_seed", text: "1, 2" }], ctx);
  const cell = createCompareRun(vs, ctx, { keepRandomSeeds: false, seedFieldEmpty: false }).request(0, base(), 7) as unknown as Record<string, unknown>;
  assert.equal(cell.variation_strength, 0.2);
  // The page sends a compare grid the form's variation seed without the per-run offset.
  const page = readSrc("features/images/images-page.tsx");
  assert.match(page, /baseVariationSeed !== null \? \(baseVariationSeed \+ \(compare \? 0 : i \* batchSize\)\) % 2 \*\* 32 : null/);
  assert.match(page, /const compareSdControls = sdCaps;/);
});

test("variation seed table: a Var. strength sweep shares one seed across every cell, 0 included", () => {
  const sdControls = { clip_skip: null, cfg_rescale: null, variation_seed: true, vary_strength: 0.2 } as never;
  const ctx = sdxl({ sdControls });
  const p = plan([{ axisId: "variation_strength", text: "0, 0.3, 0.6" }], ctx);
  // [typed Variation seed, Keep -1 for seeds, seed box empty, expected per-cell variation seeds]
  const rows: Array<[number | null, boolean, boolean, Array<number | undefined>]> = [
    [1234, false, false, [1234, 1234, 1234]], // typed with the form's strength at 0: the typed seed, not a fresh draw
    [1234, true, true, [1234, 1234, 1234]], // typed wins over Keep -1 too
    [null, false, false, [2 ** 30, 2 ** 30, 2 ** 30]], // empty: one draw for the run
    [null, true, true, [undefined, undefined, undefined]], // Keep -1 and an empty seed box: random per cell
  ];
  for (const [typed, keepRandomSeeds, seedFieldEmpty, expected] of rows) {
    const run = createCompareRun(p, ctx, { keepRandomSeeds, seedFieldEmpty, rng: () => 0.25, variationSeed: typed });
    const reqs = p.order.map((_, i) => run.request(i, base(), 7));
    const label = `typed ${typed}, keep ${keepRandomSeeds}, empty ${seedFieldEmpty}`;
    assert.deepEqual(reqs.map((r) => (r as unknown as Record<string, unknown>).variation_seed), expected, label);
    const fields = Object.fromEntries(gridRecipeFields(p, run.requests, { steps: 25, guidance: 6, width: 1024, height: 1024, model: "m" }));
    assert.equal(fields["Variation seed"], expected[0], label);
  }
  const panel = readSrc("features/images/compare/compare-panel.tsx");
  assert.match(panel, /variationSeed: props\.variationSeed,/);
  const page = readSrc("features/images/images-page.tsx");
  assert.match(page, /variationSeed=\{sdCaps\?\.variation_seed \? parseVariationSeed\(sdChoice\.variationSeed\) : null\}/);
});

test("canvas table: the final, reflowed layout fits the side and pixel limits", () => {
  const long = Array.from({ length: 400 }, (_, i) => `word${i}`).join(" ");
  const wordy = Array.from({ length: 1000 }, (_, i) => `${"x".repeat(11)}${i % 10}`).join(" ");
  const huge = Array.from({ length: 5000 }, (_, i) => `w${i}`).join(" ");
  // [cols, rows, cell side, column label, row label]
  const rows: Array<[number, number, number, string, string]> = [
    [8, 8, 2752, "", ""],
    [1, 64, 2752, long, ""],
    // Narrower cells rewrap labels onto more lines: one shrink pass leaves this at 97x19218 (over the side limit).
    [1, 64, 512, wordy, ""],
    [64, 1, 512, wordy, ""],
    // So long that even 1 px cells with 10 px text cannot hold it: cut with an ellipsis, not drawn past the limits.
    [5, 1, 1024, huge, ""],
    [1, 64, 2752, huge, huge],
    [64, 1, 2752, "", long],
    [8, 8, 2752, long, long],
    [2, 2, 512, "a", "b"], // under the limits: unscaled
  ];
  for (const [cols, rows_, side, col, row] of rows) {
    const layout = gridLayout(
      {
        cols,
        rows: rows_,
        cellWidth: side,
        cellHeight: side,
        colLabels: Array(cols).fill(col),
        rowLabels: Array(rows_).fill(row),
        title: "",
        legend: true,
        margin: 0,
      },
      measure,
    );
    const label = `${cols}x${rows_} ${col ? "col label" : ""} ${row ? "row label" : ""}`;
    assert.ok(layout.width <= MAX_CANVAS_SIDE && layout.height <= MAX_CANVAS_SIDE, `${label}: ${layout.width}x${layout.height}`);
    assert.ok(layout.width * layout.height <= MAX_CANVAS_PIXELS, label);
    assert.equal(layout.cells.length, cols * rows_);
  }
});

test("Prompt S/R table: empty replacements kept, an empty search value refused, other text axes drop empties", () => {
  // [input, Prompt S/R values or an error pattern]
  const rows: Array<[string, string[] | RegExp]> = [
    ['red, ""', ["red", ""]],
    ["red,", ["red", ""]],
    ['red, "", blue', ["red", "", "blue"]],
    ["red\nblue\n", ["red", "blue"]], // a trailing newline is a blank line, not an empty value
    ['"", red', /first value/],
    ["", /No values/],
  ];
  for (const [text, expected] of rows) {
    const parsed = parseAxisValues(text, "str", { keepEmpty: true });
    if (expected instanceof RegExp) assert.match(parsed.ok ? "" : parsed.error, expected, text);
    else assert.deepEqual(parsed.ok ? parsed.values : parsed.error, expected, text);
  }
  assert.deepEqual(values('euler, ""', "str"), ["euler"]);
  // End to end: the empty cell removes the searched text from the prompt.
  const ctx = sdxl({ prompt: "a red fox" });
  const p = plan([{ axisId: "prompt_sr", text: 'red, ""' }], ctx);
  assert.deepEqual(
    [0, 1].map((i) => buildCellRequest(p, i, base({ prompt: "a red fox" }), ctx, { base: 1, keepRandom: false, seedFieldEmpty: false }).prompt),
    ["a red fox", "a  fox"],
  );
});

test("sampler x schedule table: every cell must take its effective schedule, the form's included", () => {
  // [form sampler, form schedule, axes, ok or an error pattern]
  const rows: Array<[string, string, Array<{ axisId: string; text: string }>, true | RegExp]> = [
    ["euler", "karras", [{ axisId: "sampler", text: "euler, euler_a" }], /Euler a does not take the form's Karras schedule/],
    ["euler", "karras", [{ axisId: "sampler", text: "euler, dpmpp_2m" }], true],
    ["euler", "automatic", [{ axisId: "sampler", text: "euler, euler_a" }], true],
    // No Sampler axis: the form's own pair, already valid in the form.
    ["euler", "karras", [{ axisId: "steps", text: "10, 20" }], true],
    ["euler", "karras", [{ axisId: "sampler", text: "euler, euler_a" }, { axisId: "schedule", text: "automatic" }], true],
    ["euler", "automatic", [{ axisId: "sampler", text: "euler_a" }, { axisId: "schedule", text: "automatic, karras" }], /Schedule type Karras is not available with Euler a/],
  ];
  for (const [sampler, scheduleType, axes, expected] of rows) {
    const r = planCompare(axes, sdxl({ samplerChoice: { sampler, scheduleType } }));
    const label = `${sampler}/${scheduleType} ${axes.map((a) => `${a.axisId}=${a.text}`).join(" ")}`;
    if (expected === true) assert.ok(r.ok, `${label}: ${r.ok ? "" : r.error}`);
    else assert.match(r.ok ? "" : r.error, expected, label);
  }
});

test("compare wiring: the request's ControlNet predicate gates CFG rescale; a failed save is reported", () => {
  const page = readSrc("features/images/images-page.tsx");
  assert.match(page, /\n\s+controlnetEngaged,\n\s+\}\),/);
  assert.match(page, /hiresEngagesNegative: hiresEngagesNegative\(hires, hiresCapability, workflow, negativeMinGuidance\),/);
  assert.doesNotMatch(page, /controlnetEngaged: Boolean\(controlnetCapable && controlnetId && controlImage\)/);
  // A strength-0 ControlNet is not engaged, so the axis stays (the axes only read the flag).
  const sdControls = { clip_skip: null, cfg_rescale: { min: 0, max: 1 }, variation_seed: false } as never;
  assert.ok(availableAxes(sdxl({ sdControls, controlnetEngaged: false })).some((d) => d.id === "cfg_rescale"));
  assert.ok(!availableAxes(sdxl({ sdControls, controlnetEngaged: true })).some((d) => d.id === "cfg_rescale"));
  const panel = readSrc("features/images/compare/compare-panel.tsx");
  assert.match(panel, /if \(!isDownloadCancelled\(err\)\) \{\n\s+toast\.error\(/);
  assert.doesNotMatch(panel, /\.catch\(\(\) => \{\}\)/);
});

test("LoRA ids with commas or quotes round-trip through the grid's LoRAs field", () => {
  // [ids, expected restored entries]
  const rows: Array<[string[], string[]]> = [
    [["style, blue"], ["style, blue:0.5"]],
    [['say "hi"', "plain"], ['say "hi":0.5', "plain:0.5"]],
    [["a, b", "hub:odd/id", "c"], ["a, b:0.5", "hub:odd/id:0.5", "c:0.5"]],
  ];
  for (const [ids, expected] of rows) {
    const ctx = sdxl({ loraCapable: true, loras: ids.map((id) => ({ id, weight: 0.5, label: id })) });
    const first = base({ loras: ids.map((id) => ({ id, weight: 0.5 })) });
    const info = gridParameters(plan([{ axisId: "steps", text: "4, 8" }], ctx), ctx, [first], { steps: 4, guidance: 1, width: 64, height: 64, model: "m", keepRandomSeeds: false });
    assert.deepEqual(parseA1111Parameters(info)?.loras, expected, ids.join(" | "));
  }
});

test("an invalid typed Variation seed is refused for a compare run at any form strength", () => {
  const page = readSrc("features/images/images-page.tsx");
  assert.match(page, /const checkVariation = varying \|\| \(compareOnce\.current !== null && Boolean\(sdCaps\?\.variation_seed\)\);\n\s+if \(checkVariation && runChoice\.variationSeed\.trim\(\) && parseVariationSeed\(runChoice\.variationSeed\) === null\) \{\n\s+toast\.error/);
});

test("Hires CFG keeps the negative prompt on a CFG cell whose base guidance leaves it off, as Generate does", () => {
  const hires = { scale: 1.5, denoise: 0.5, steps: null, upscaler: "latent", guidance: 7 };
  const g = plan([{ axisId: "guidance", text: "1, 7" }]);
  const seeds = { base: 1, keepRandom: false, seedFieldEmpty: false };
  const ctx = sdxl({ hiresEngagesNegative: true });
  assert.equal(buildCellRequest(g, 0, base({ hires }), ctx, seeds).negative_prompt, "blurry");
  assert.equal(buildCellRequest(g, 1, base({ hires }), ctx, seeds).negative_prompt, "blurry");
  assert.equal(buildCellRequest(g, 0, base(), sdxl(), seeds).negative_prompt, undefined);
  // A negative-only S/R term at base CFG 1 is read by the refine pass, so not refused.
  const sr = sdxl({ prompt: "a fox", negativePrompt: "red eyes", guidance: 1 });
  assert.equal(planCompare([{ axisId: "prompt_sr", text: "red, blue" }], sr).ok, false);
  assert.ok(planCompare([{ axisId: "prompt_sr", text: "red, blue" }], { ...sr, hiresEngagesNegative: true }).ok);
  const off = plan([{ axisId: "steps", text: "10, 20" }], { ...sr, negativePrompt: "blurry" });
  assert.equal(gridNegativePrompt(off, { ...sr, negativePrompt: "blurry" }), undefined);
  assert.equal(gridNegativePrompt(off, { ...sr, negativePrompt: "blurry", hiresEngagesNegative: true }), "blurry");
});

test("grid metadata records the Hires fix every cell ran, in A1111 names, and drops it when cells differ", () => {
  const steps = plan([{ axisId: "steps", text: "10, 20" }]);
  const form = { steps: 25, guidance: 6, width: 1024, height: 1024, model: "m" };
  const hires = { scale: 1.5, denoise: 0.5, steps: 12, upscaler: "latent", guidance: 7 };
  const fields = Object.fromEntries(gridRecipeFields(steps, [base({ hires }), base({ hires: { ...hires } })], form));
  assert.equal(fields.Size, "1024x1024");
  assert.equal(fields["Denoising strength"], 0.5);
  assert.equal(fields["Hires upscale"], 1.5);
  assert.equal(fields["Hires steps"], 12);
  assert.equal(fields["Hires upscaler"], "Latent (nearest-exact)");
  assert.equal(fields["Hires CFG Scale"], 7);
  const parsed = parseA1111Parameters(gridParameters(steps, sdxl(), [base({ hires })], { ...form, keepRandomSeeds: false }));
  assert.equal(parsed?.hires_scale, 1.5);
  assert.equal(parsed?.hires_guidance, 7);
  const mixed = Object.fromEntries(gridRecipeFields(steps, [base({ hires }), base()], form));
  assert.equal("Hires upscale" in mixed, false);
  assert.equal("Hires upscale" in Object.fromEntries(gridRecipeFields(steps, [base()], form)), false);
});
