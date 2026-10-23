// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Compare grid (AUTOMATIC1111's X/Y/Z plot): which axes the loaded model can take, the cell plan, the order the
 *  cells run in, and each cell's generate request. Pure, so tests drive it without a page. */

import type { DiffusionGenerateRequest, LoraSpecInput } from "../api.ts";
import {
  AUTOMATIC_SCHEDULE as AUTOMATIC,
  SCHEDULE_LABELS,
  type SamplerChoice,
  type SamplerOption,
} from "../lib/sampler-choice.ts";
import { SHIFT_MAX, type ShiftControl } from "../lib/shift-choice.ts";
import { parseAxisValues, roundValue, type AxisValueType } from "./compare-values.ts";

export const MAX_COMPARE_CELLS = 64;
const STEPS_MAX = 100;
const GUIDANCE_MAX = 20;
const LORA_WEIGHT_MAX = 2;
const SEED_MAX = Number.MAX_SAFE_INTEGER;

/** PR-gated SD controls, read structurally so axes appear once a backend reports them. */
export interface CompareSdControls {
  clip_skip?: { min: number; max: number } | null;
  cfg_rescale?: { min: number; max: number } | null;
  variation_seed?: boolean;
  vary_strength?: number;
}

/** What the page knows about the loaded model and the form. Compare runs only on a loaded local model. */
export interface CompareContext {
  prompt: string;
  negativePrompt: string;
  negativeCapable: boolean;
  // Guidance a negative prompt (and so the unconditional pass) needs; null = always engaged.
  negativeMinGuidance: number | null;
  guidanceFixed: boolean;
  // The form's effective guidance, which every cell runs at unless a CFG axis sets it.
  guidance: number;
  samplers: readonly SamplerOption[];
  samplerChoice: SamplerChoice;
  shift: ShiftControl | null;
  loraCapable: boolean;
  // The form's active LoRA rows, in order; a weight axis targets one of them.
  loras: ReadonlyArray<{ id: string; weight: number; label: string }>;
  sdControls?: CompareSdControls | null;
  controlnetEngaged: boolean;
  // Hires CFG makes the refine pass read the negative prompt at any base CFG; no axis varies Hires, so it is grid-wide.
  hiresEngagesNegative?: boolean;
}

type Value = number | string;

/** Everything a cell can override, folded into the request once every axis has applied. */
export interface CellState {
  seed?: number | "random";
  steps?: number;
  guidance?: number;
  sampler?: string;
  schedule?: string;
  shift?: number;
  baseShift?: number;
  maxShift?: number;
  loraWeights: Record<number, number>;
  replacements: Array<[string, string]>;
  extra: Record<string, number>;
}

export interface AxisDef {
  id: string;
  label: string;
  type: AxisValueType;
  placeholder: string;
  choices?: ReadonlyArray<{ value: string; label: string }>;
  // A1111 AxisOption.cost: costly axes run in the outer loop. Here cost = may change the denoiser's work shape.
  cost: number;
  resolve(value: Value): { ok: true; value: Value } | { ok: false; error: string };
  apply(cell: CellState, value: Value, values: readonly Value[]): void;
  format(value: Value): string;
  // Values with the same key run with the same tensor shapes; the plan keeps each key's cells together.
  shapeKey?(value: Value): string;
  // Keys that must run first, in this order; others follow in first-seen order.
  shapeOrder?: readonly string[];
}

export const NOTHING = "nothing";

const ok = (value: Value) => ({ ok: true as const, value });
const fail = (error: string) => ({ ok: false as const, error });

function numberAxis(
  id: string,
  label: string,
  type: "int" | "float",
  min: number,
  max: number,
  placeholder: string,
  apply: (cell: CellState, v: number) => void,
  extra: Partial<AxisDef> = {},
  minExclusive = false,
): AxisDef {
  return {
    id,
    label,
    type,
    placeholder,
    cost: 0,
    resolve: (v) => {
      const n = Number(v);
      if (!Number.isFinite(n)) return fail(`${label}: "${v}" is not a number`);
      if (minExclusive ? n <= min : n < min) return fail(`${label} ${n} is below ${minExclusive ? "or at " : ""}${min}`);
      if (n > max) return fail(`${label} ${n} is above ${max}`);
      return ok(type === "int" ? Math.trunc(n) : roundValue(n));
    },
    apply: (cell, v) => apply(cell, Number(v)),
    format: (v) => `${label}: ${v}`,
    ...extra,
  };
}

function choiceAxis(
  id: string,
  label: string,
  choices: ReadonlyArray<{ value: string; label: string }>,
  apply: (cell: CellState, v: string) => void,
): AxisDef {
  return {
    id,
    label,
    type: "str",
    placeholder: choices.slice(0, 3).map((c) => c.label).join(", "),
    choices,
    cost: 0,
    resolve: (v) => {
      const want = String(v).trim().toLowerCase();
      const hit = choices.find((c) => c.value.toLowerCase() === want || c.label.toLowerCase() === want);
      return hit ? ok(hit.value) : fail(`Unknown ${label.toLowerCase()}: ${v}`);
    },
    apply: (cell, v) => apply(cell, String(v)),
    format: (v) => choices.find((c) => c.value === v)?.label ?? String(v),
  };
}

/** The axes the loaded model can take, in menu order. "Nothing" first, as in A1111. */
export function availableAxes(ctx: CompareContext): AxisDef[] {
  const axes: AxisDef[] = [];
  axes.push({
    id: NOTHING,
    label: "Nothing",
    type: "str",
    placeholder: "",
    cost: 0,
    resolve: ok,
    apply: () => {},
    format: () => "",
  });
  axes.push(
    numberAxis("seed", "Seed", "int", -1, SEED_MAX, "-1, 11, 12 or 100-103", (c, v) => {
      c.seed = v < 0 ? "random" : v;
    }),
  );
  axes.push(numberAxis("steps", "Steps", "int", 1, STEPS_MAX, "10, 20, 30 or 10-30 (+10)", (c, v) => (c.steps = v)));
  if (!ctx.guidanceFixed) {
    const min = ctx.negativeMinGuidance;
    axes.push(
      numberAxis("guidance", "CFG", "float", 0, GUIDANCE_MAX, "3, 5, 7 or 3-7 [3]", (c, v) => (c.guidance = v), {
        // Crossing the threshold adds or drops the unconditional pass (a batch-size change), so those cells group.
        cost: 1,
        shapeKey: (v) => (min === null || Number(v) > min ? "cfg" : "nocfg"),
      }),
    );
  }
  if (ctx.samplers.length > 1) {
    axes.push(
      choiceAxis(
        "sampler",
        "Sampler",
        ctx.samplers.map((s) => ({ value: s.id, label: s.label })),
        (c, v) => (c.sampler = v),
      ),
    );
    const schedules = [...new Set(ctx.samplers.flatMap((s) => s.schedule_types))];
    if (schedules.length > 1) {
      axes.push(
        choiceAxis(
          "schedule",
          "Schedule type",
          schedules.map((t) => ({ value: t, label: SCHEDULE_LABELS[t] ?? t })),
          (c, v) => (c.schedule = v),
        ),
      );
    }
  }
  if (ctx.shift?.node === "sd3") {
    axes.push(numberAxis("shift", "Shift", "float", 0, SHIFT_MAX, "1, 3, 6", (c, v) => (c.shift = v), {}, true));
  } else if (ctx.shift?.node === "flux") {
    axes.push(numberAxis("base_shift", "Base shift", "float", 0, SHIFT_MAX, "0.25, 0.5, 1", (c, v) => (c.baseShift = v)));
    axes.push(numberAxis("max_shift", "Max shift", "float", 0, SHIFT_MAX, "0.75, 1.15, 1.5", (c, v) => (c.maxShift = v)));
  }
  if (ctx.loraCapable) {
    ctx.loras.forEach((lora, index) => {
      const name = ctx.loras.length > 1 ? `LoRA weight: ${lora.label}` : "LoRA weight";
      axes.push(
        numberAxis(`lora:${index}`, name, "float", 0, LORA_WEIGHT_MAX, "0, 0.5, 1", (c, v) => (c.loraWeights[index] = v), {
          // Weight 0 drops the adapter from the request. Every weighted cell runs before any of those, outermost
          // of all axes: Speed Auto compiles on an adapter-free run, and a compiled load refuses LoRA after that.
          cost: 4,
          shapeKey: (v) => (Number(v) === 0 ? "off" : "on"),
          shapeOrder: ["on", "off"],
          format: (v) => `${name}: ${v}`,
        }),
      );
    });
  }
  const sd = ctx.sdControls;
  if (sd?.clip_skip) {
    const { min, max } = sd.clip_skip;
    axes.push(numberAxis("clip_skip", "Clip skip", "int", min, max, "1, 2", (c, v) => (c.extra.clip_skip = v)));
  }
  if (sd?.cfg_rescale && !ctx.controlnetEngaged) {
    const { min, max } = sd.cfg_rescale;
    axes.push(numberAxis("cfg_rescale", "CFG rescale", "float", min, max, "0, 0.35, 0.7", (c, v) => (c.extra.cfg_rescale = v)));
  }
  if (sd?.variation_seed) {
    axes.push(
      numberAxis("variation_seed", "Var. seed", "int", 0, 2 ** 32 - 1, "1, 2, 3", (c, v) => (c.extra.variation_seed = v)),
    );
    axes.push(
      numberAxis("variation_strength", "Var. strength", "float", 0, 1, "0, 0.1, 0.3", (c, v) => (c.extra.variation_strength = v)),
    );
  }
  axes.push({
    id: "prompt_sr",
    label: "Prompt S/R",
    type: "str",
    placeholder: "red, blue, green (first value is searched for)",
    // A new text length can retrace a dynamic-shape denoiser: outermost, so each prompt is entered once.
    cost: 3,
    resolve: (v) => ok(String(v)),
    apply: (cell, v, values) => cell.replacements.push([String(values[0]), String(v)]),
    format: (v) => String(v),
    shapeKey: (v) => String(v),
  });
  return axes;
}

export interface AxisInput {
  axisId: string;
  text: string;
}

export interface ResolvedAxis {
  def: AxisDef;
  text: string;
  values: Value[];
  labels: string[];
}

export interface CompareGridPlan {
  axes: [ResolvedAxis, ResolvedAxis, ResolvedAxis];
  cellCount: number;
  // Linear cell indexes (ix + iy*nx + iz*nx*ny) in the order they run.
  order: number[];
}

export type PlanResult = { ok: true; plan: CompareGridPlan } | { ok: false; error: string; cellCount?: number };

const AXIS_NAMES = ["X", "Y", "Z"] as const;

export function cellCoords(plan: CompareGridPlan, index: number): [number, number, number] {
  const nx = plan.axes[0].values.length;
  const ny = plan.axes[1].values.length;
  return [index % nx, Math.floor(index / nx) % ny, Math.floor(index / (nx * ny))];
}

/** Costliest axis outermost (A1111), and a costly axis's values grouped by shape key so the model switches shape
 *  at most once per key. Cells are placed by index, not run order. */
export function executionOrder(axes: readonly ResolvedAxis[]): number[] {
  const n = axes.map((a) => a.values.length);
  const total = n[0] * n[1] * n[2];
  const priority = [2, 1, 0].sort((a, b) => axes[b].def.cost - axes[a].def.cost);
  const rank = axes.map((axis) => {
    const keys: string[] = [...(axis.def.shapeOrder ?? [])];
    return axis.values.map((v) => {
      const key = axis.def.shapeKey ? axis.def.shapeKey(v) : "";
      let at = keys.indexOf(key);
      if (at < 0) at = keys.push(key) - 1;
      return at;
    });
  });
  const cells = Array.from({ length: total }, (_, i) => i);
  const coord = (i: number) => [i % n[0], Math.floor(i / n[0]) % n[1], Math.floor(i / (n[0] * n[1]))];
  // Shape keys of every axis first (outer to inner), then the nested-loop position.
  const key = (i: number) => {
    const c = coord(i);
    return [...priority.map((a) => rank[a][c[a]]), ...priority.map((a) => c[a])];
  };
  const keys = cells.map(key);
  return cells.sort((a, b) => {
    const ka = keys[a];
    const kb = keys[b];
    for (let j = 0; j < ka.length; j++) if (ka[j] !== kb[j]) return ka[j] - kb[j];
    return 0;
  });
}

/** Parse, validate and order the three axes. `rng` draws the fixed seeds for -1 seed values (A1111 fix_axis_seeds). */
export function planCompare(
  inputs: readonly AxisInput[],
  ctx: CompareContext,
  options: { keepRandomSeeds: boolean; maxCells?: number; rng?: () => number } = { keepRandomSeeds: false },
): PlanResult {
  const defs = availableAxes(ctx);
  const nothing = defs[0];
  const resolved: ResolvedAxis[] = [];
  const seen = new Map<string, string>();
  for (let a = 0; a < 3; a++) {
    const input = inputs[a] ?? { axisId: NOTHING, text: "" };
    const def = defs.find((d) => d.id === input.axisId);
    if (!def) return { ok: false, error: `${AXIS_NAMES[a]}: this model has no ${input.axisId.replace(/_/g, " ")} axis` };
    if (def.id === NOTHING) {
      resolved.push({ def: nothing, text: "", values: [""], labels: [""] });
      continue;
    }
    if (def.id !== "prompt_sr" && seen.has(def.id)) {
      return { ok: false, error: `${seen.get(def.id)} and ${AXIS_NAMES[a]} are both ${def.label}` };
    }
    seen.set(def.id, AXIS_NAMES[a]);
    const parsed = parseAxisValues(input.text, def.type, { keepEmpty: def.id === "prompt_sr" });
    if (!parsed.ok) return { ok: false, error: `${AXIS_NAMES[a]} (${def.label}): ${parsed.error}` };
    const values: Value[] = [];
    for (const raw of parsed.values) {
      const r = def.resolve(raw);
      if (!r.ok) return { ok: false, error: `${AXIS_NAMES[a]}: ${r.error}` };
      values.push(r.value);
    }
    if (def.id === "seed" && !options.keepRandomSeeds) {
      const rng = options.rng ?? Math.random;
      for (let i = 0; i < values.length; i++) if (values[i] === -1) values[i] = Math.floor(rng() * 2 ** 32);
    }
    resolved.push({
      def,
      text: input.text,
      values,
      labels: values.map((v) => (def.id === "seed" && v === -1 ? "Seed: random" : def.format(v))),
    });
  }
  const cellCount = resolved.reduce((n, a) => n * a.values.length, 1);
  const max = options.maxCells ?? MAX_COMPARE_CELLS;
  if (resolved.every((a) => a.def.id === NOTHING)) return { ok: false, error: "Pick at least one axis", cellCount: 1 };
  if (cellCount > max) {
    return {
      ok: false,
      error: `${cellCount} cells is over the limit of ${max}. Use fewer values or drop an axis.`,
      cellCount,
    };
  }
  const axes = resolved as [ResolvedAxis, ResolvedAxis, ResolvedAxis];
  const plan: CompareGridPlan = { axes, cellCount, order: executionOrder(axes) };
  const problem = cellProblem(plan, ctx);
  return problem ? { ok: false, error: problem } : { ok: true, plan };
}

/** Combinations no cell may run: a Prompt S/R term a cell's text lacks once the earlier axes' substitutions ran
 *  (A1111 apply_prompt raises per cell; a negative prompt counts only where that cell's CFG, or Hires CFG, engages it), or a
 *  Schedule type, from an axis or the form, the cell's sampler does not take (the backend would run Automatic under a
 *  Karras label). */
function cellProblem(plan: CompareGridPlan, ctx: CompareContext): string | null {
  const srAxes = plan.axes.flatMap((axis, a) => (axis.def.id === "prompt_sr" ? [a] : []));
  for (let i = 0; i < plan.cellCount; i++) {
    const cell = cellState(plan, i);
    let prompt = ctx.prompt;
    let negative = negativeEngaged(ctx, cell.guidance ?? ctx.guidance) ? ctx.negativePrompt : "";
    for (let k = 0; k < cell.replacements.length; k++) {
      const [search, replace] = cell.replacements[k];
      if (!prompt.includes(search) && !negative.includes(search)) {
        return `${AXIS_NAMES[srAxes[k]]}: Prompt S/R did not find "${search}" in cell "${cellTitle(plan, i)}" (its prompt after earlier substitutions, or a negative prompt its CFG uses)`;
      }
      prompt = prompt.split(search).join(replace);
      negative = negative.split(search).join(replace);
    }
    // The form's schedule counts too once a Sampler axis picks the sampler: falling back to Automatic for one cell
    // would vary the schedule along with the labelled sampler.
    const schedule = cell.schedule ?? (cell.sampler !== undefined ? ctx.samplerChoice.scheduleType : undefined);
    if (schedule !== undefined && schedule !== AUTOMATIC) {
      const sampler = cell.sampler ?? ctx.samplerChoice.sampler;
      const option = ctx.samplers.find((s) => s.id === sampler);
      if (!option?.schedule_types.includes(schedule)) {
        const label = SCHEDULE_LABELS[schedule] ?? schedule;
        return cell.schedule !== undefined
          ? `Schedule type ${label} is not available with ${option?.label ?? sampler}. Drop it, or compare only samplers that take it.`
          : `${option?.label ?? sampler} does not take the form's ${label} schedule. Set Schedule type to Automatic, or compare only samplers that take it.`;
      }
    }
  }
  return null;
}

function negativeEngaged(ctx: CompareContext, guidance: number): boolean {
  return (
    ctx.negativeCapable &&
    (ctx.negativeMinGuidance === null || guidance > ctx.negativeMinGuidance || ctx.hiresEngagesNegative === true)
  );
}

/** The negative prompt the grid's recipe records: the form's text when any cell runs with it engaged. */
export function gridNegativePrompt(plan: CompareGridPlan, ctx: CompareContext): string | undefined {
  const text = ctx.negativePrompt.trim();
  if (!text) return undefined;
  for (let i = 0; i < plan.cellCount; i++) {
    if (negativeEngaged(ctx, cellState(plan, i).guidance ?? ctx.guidance)) return text;
  }
  return undefined;
}

/** The cell's overrides, every axis applied in X, Y, Z order (so two Prompt S/R axes chain as in A1111). */
export function cellState(plan: CompareGridPlan, index: number): CellState {
  const cell: CellState = { loraWeights: {}, replacements: [], extra: {} };
  const c = cellCoords(plan, index);
  plan.axes.forEach((axis, a) => axis.def.apply(cell, axis.values[c[a]], axis.values));
  return cell;
}

export function cellTitle(plan: CompareGridPlan, index: number): string {
  const c = cellCoords(plan, index);
  return plan.axes
    .map((axis, a) => (axis.def.id === NOTHING ? "" : axis.labels[c[a]]))
    .filter(Boolean)
    .join(", ");
}

/** `base` is the request Generate would send for the form (text-to-image). The cell runs it at batch size 1 with
 *  the prompt enhancer off (a rewrite per cell would change the thing being compared). */
export function buildCellRequest(
  plan: CompareGridPlan,
  index: number,
  base: DiffusionGenerateRequest,
  ctx: CompareContext,
  // `variation`: the run's one variation seed, for cells that set a strength without one.
  seeds: { base: number; keepRandom: boolean; seedFieldEmpty: boolean; rng?: () => number; variation?: number },
): DiffusionGenerateRequest {
  const cell = cellState(plan, index);
  const rng = seeds.rng ?? Math.random;
  const req: DiffusionGenerateRequest & Record<string, unknown> = { ...base, batch_size: 1 };
  if (req.enhance_prompt !== undefined) req.enhance_prompt = false;
  // Seed: the axis value, else one shared seed for every cell (A1111 fix_seed), unless "Keep -1 for seeds" with an
  // empty seed box, when every cell draws its own.
  if (cell.seed === "random") req.seed = Math.floor(rng() * 2 ** 32);
  else if (typeof cell.seed === "number") req.seed = cell.seed;
  else req.seed = seeds.keepRandom && seeds.seedFieldEmpty ? Math.floor(rng() * 2 ** 32) : seeds.base;
  if (cell.steps !== undefined) req.steps = cell.steps;
  if (cell.guidance !== undefined) {
    req.guidance = cell.guidance;
    req.negative_prompt = negativeEngaged(ctx, cell.guidance) ? ctx.negativePrompt.trim() || undefined : undefined;
  }
  if (cell.sampler !== undefined || cell.schedule !== undefined) {
    const sampler = cell.sampler ?? ctx.samplerChoice.sampler;
    const option = ctx.samplers.find((s) => s.id === sampler);
    const schedule = cell.schedule ?? ctx.samplerChoice.scheduleType;
    delete req.sampler;
    delete req.schedule_type;
    if (sampler !== "default") req.sampler = sampler;
    // planCompare refuses a Schedule axis value the sampler lacks; only the form's schedule can fall back here.
    if (schedule !== AUTOMATIC && option?.schedule_types.includes(schedule)) req.schedule_type = schedule;
  }
  if (cell.shift !== undefined) req.shift = cell.shift;
  if (cell.baseShift !== undefined) req.base_shift = cell.baseShift;
  if (cell.maxShift !== undefined) req.max_shift = cell.maxShift;
  if (Object.keys(cell.loraWeights).length > 0) {
    const loras: LoraSpecInput[] = ctx.loras
      .map((l, i) => ({ id: l.id.trim(), weight: cell.loraWeights[i] ?? l.weight }))
      .filter((l) => l.id && l.weight > 0);
    req.loras = loras.length ? loras : undefined;
  }
  if (cell.replacements.length > 0) {
    let prompt = req.prompt;
    let negative = req.negative_prompt;
    for (const [search, replace] of cell.replacements) {
      prompt = prompt.split(search).join(replace);
      if (negative !== undefined) negative = negative.split(search).join(replace);
    }
    req.prompt = prompt;
    req.negative_prompt = negative;
  }
  for (const [key, value] of Object.entries(cell.extra)) req[key] = value;
  // Variation seed without a strength means nothing to the backend: use the model's Vary strength.
  if (cell.extra.variation_seed !== undefined && cell.extra.variation_strength === undefined && !req.variation_strength) {
    req.variation_strength = ctx.sdControls?.vary_strength ?? 0.15;
  }
  // A strength with no variation seed draws a random one per request: fix one for the run (A1111 fix_seed on the
  // subseed), so a Var. strength axis compares strengths over one noise source.
  // Strength 0 cells get it too (the backend ignores it there), so the grid's recipe still shares one seed.
  if (typeof req.variation_strength === "number" && req.variation_seed === undefined && seeds.variation !== undefined) {
    req.variation_seed = seeds.variation;
  }
  return req;
}
