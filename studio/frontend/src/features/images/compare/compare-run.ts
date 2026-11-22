// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** One compare run as the page's Generate loop sees it: N runs, each with its own request, reporting its records
 *  back. The page keeps owning progress polling, Stop, the gallery merge and error toasts. */

import type { ControlNetSpecInput, DiffusionGenerateRequest, GalleryImage } from "../api.ts";
import { HIRES_UPSCALER_LABELS, type HiresRequest } from "../lib/hires-choice.ts";
import { SAMPLER_LABELS, SCHEDULE_LABELS } from "../lib/sampler-choice.ts";
import { gridInfotext } from "./compose-grid.ts";
import { joinAxisList } from "./compare-values.ts";
import {
  buildCellRequest,
  gridNegativePrompt,
  cellTitle,
  NOTHING,
  type CompareContext,
  type CompareGridPlan,
} from "./compare-axes.ts";

export interface CompareRun {
  readonly cellCount: number;
  request(run: number, base: DiffusionGenerateRequest, baseSeed: number): DiffusionGenerateRequest;
  title(done: number): string;
  onImages(run: number, images: readonly GalleryImage[]): void;
  // The seed run `run` was actually sent with (a Seed axis or Keep -1 replaces the form's).
  sentSeed(run: number): number | undefined;
}

export interface CompareRunState extends CompareRun {
  readonly plan: CompareGridPlan;
  // By cell index (ix + iy*nx + iz*nx*ny), null until that cell lands.
  readonly results: Array<GalleryImage | null>;
  readonly requests: Array<DiffusionGenerateRequest | null>;
}

export function createCompareRun(
  plan: CompareGridPlan,
  ctx: CompareContext,
  // `variationSeed`: the form's typed Variation seed (null = the field is empty), used even when the form's strength
  // is 0 because a Var. strength axis supplies it.
  options: { keepRandomSeeds: boolean; seedFieldEmpty: boolean; rng?: () => number; variationSeed?: number | null },
): CompareRunState {
  const results: Array<GalleryImage | null> = Array.from({ length: plan.cellCount }, () => null);
  const requests: Array<DiffusionGenerateRequest | null> = Array.from({ length: plan.cellCount }, () => null);
  // One variation seed per run: the typed one, else drawn up front; "Keep -1 for seeds" with an empty seed box keeps
  // it random per cell too.
  const variation =
    options.variationSeed ??
    (options.keepRandomSeeds && options.seedFieldEmpty ? undefined : Math.floor((options.rng ?? Math.random)() * 2 ** 32));
  return {
    plan,
    results,
    requests,
    cellCount: plan.cellCount,
    request(run, base, baseSeed) {
      const cell = plan.order[run];
      const req = buildCellRequest(plan, cell, base, ctx, {
        base: baseSeed,
        keepRandom: options.keepRandomSeeds,
        seedFieldEmpty: options.seedFieldEmpty,
        rng: options.rng,
        variation,
      });
      requests[cell] = req;
      return req;
    },
    title(done) {
      const cell = plan.order[Math.min(done, plan.cellCount - 1)];
      const label = cellTitle(plan, cell);
      return `Cell ${Math.min(done + 1, plan.cellCount)}/${plan.cellCount}${label ? `: ${label}` : ""}`;
    },
    sentSeed(run) {
      const cell = plan.order[run];
      return cell === undefined ? undefined : requests[cell]?.seed;
    },
    onImages(run, images) {
      const cell = plan.order[run];
      if (cell === undefined) return;
      // A lost-response recovery passes every record the gallery gained: the cell's own carries its seed.
      const seed = requests[cell]?.seed;
      const own = images.find((image) => image.seed === seed) ?? (images.length === 1 ? images[0] : undefined);
      if (own) results[cell] = own;
    },
  };
}

/** Cells of one Z slice, row-major (ix + iy * nx). */
export function sliceCells<T>(plan: CompareGridPlan, items: readonly T[], iz: number): T[] {
  const nx = plan.axes[0].values.length;
  const ny = plan.axes[1].values.length;
  return items.slice(iz * nx * ny, (iz + 1) * nx * ny);
}

export function gridTitles(plan: CompareGridPlan): string[] {
  const z = plan.axes[2];
  return z.def.id === NOTHING ? [""] : z.labels;
}

export function completedCells(run: CompareRunState): number {
  return run.results.filter(Boolean).length;
}

/** The value every sent cell agrees on for `key`, or none when an axis varies it or any two cells differ (a
 *  schedule a sampler fell back from, a per-cell random seed, a Seed axis). `{ value: undefined }` = every cell
 *  omitted it. Cells are compared through `same` (each cell's request is built afresh, so objects by content). */
function sharedValue(
  sent: ReadonlyArray<Readonly<Record<string, unknown>>>,
  varied: ReadonlySet<string>,
  key: string,
  axisId: string = key,
  same: (a: unknown, b: unknown) => boolean = Object.is,
): { value: unknown } | null {
  if (varied.has(axisId) || sent.length === 0) return null;
  const value = sent[0][key];
  return sent.every((r) => same(r[key], value)) ? { value } : null;
}

// A ControlNet unit by every field it is sent with, the control image included.
const sameControlNet = (a: unknown, b: unknown): boolean => {
  const x = a as ControlNetSpecInput | undefined;
  const y = b as ControlNetSpecInput | undefined;
  if (!x || !y) return x === y;
  return (
    x.id === y.id &&
    x.image === y.image &&
    x.control_type === y.control_type &&
    x.strength === y.strength &&
    x.guidance_start === y.guidance_start &&
    x.guidance_end === y.guidance_end
  );
};

const scalar = (value: unknown): string | number | undefined =>
  typeof value === "string" || typeof value === "number" ? value : undefined;

/** sd-webui-controlnet's `ControlNet 0` unit text, without the control image (supplied again on import). */
function controlNetUnit(value: unknown): string | undefined {
  const cn = value as ControlNetSpecInput | undefined;
  if (!cn?.id) return undefined;
  const parts = [`Module: ${cn.control_type}`, `Model: ${cn.id}`, `Weight: ${cn.strength}`];
  if (cn.guidance_start !== undefined) parts.push(`Guidance Start: ${cn.guidance_start}`);
  if (cn.guidance_end !== undefined) parts.push(`Guidance End: ${cn.guidance_end}`);
  return parts.join(", ");
}

const sameHires = (a: unknown, b: unknown): boolean => JSON.stringify(a ?? null) === JSON.stringify(b ?? null);

/** Hires fields in A1111 names (diffusion_hires.infotext_fields); Size stays the base pass. */
function hiresFields(hires: HiresRequest | undefined): Array<[string, string | number | undefined]> {
  if (!hires) return [];
  const upscaler = hires.upscaler ? (HIRES_UPSCALER_LABELS[hires.upscaler] ?? hires.upscaler.replace(/^local:/, "")) : undefined;
  return [
    ["Denoising strength", hires.denoise],
    ["Hires upscale", hires.scale],
    ["Hires steps", hires.steps ? hires.steps : undefined],
    ["Hires upscaler", upscaler],
    ["Hires CFG Scale", hires.guidance ?? undefined],
  ];
}

/** The grid's shared recipe for its `parameters` chunk, from the requests the cells were sent with: a field is
 *  written only when every cell ran the same value, so the PNG never claims one cell's setting for the grid. */
export function gridRecipeFields(
  plan: CompareGridPlan,
  requests: ReadonlyArray<DiffusionGenerateRequest | null | undefined>,
  form: { steps: number; guidance: number; width: number; height: number; model: string | null },
): Array<[string, string | number | null | undefined]> {
  const varied = new Set(plan.axes.map((a) => a.def.id));
  const sent = requests.filter((r): r is DiffusionGenerateRequest => Boolean(r)) as unknown as Array<
    Record<string, unknown>
  >;
  const shared = (key: string, axisId?: string) => scalar(sharedValue(sent, varied, key, axisId)?.value);
  const sampler = sharedValue(sent, varied, "sampler");
  const schedule = shared("schedule_type", "schedule");
  const width = shared("width");
  const height = shared("height");
  const fields: Array<[string, string | number | null | undefined]> = [
    ["Steps", sent.length ? shared("steps") : form.steps],
    ["Sampler", sampler ? (SAMPLER_LABELS[String(sampler.value ?? "default")] ?? scalar(sampler.value)) : undefined],
    ["Schedule type", schedule ? (SCHEDULE_LABELS[String(schedule)] ?? schedule) : undefined],
    ["Shift", shared("shift")],
    ["Base shift", shared("base_shift")],
    ["Max shift", shared("max_shift")],
    ["CFG scale", sent.length ? shared("guidance") : form.guidance],
    ["Seed", shared("seed")],
    ["Variation seed", shared("variation_seed")],
    ["Variation seed strength", shared("variation_strength")],
    // The backend's own infotext names (core/inference/image_gallery.py).
    ["Clip skip", shared("clip_skip")],
    ["CFG rescale", shared("cfg_rescale")],
    ["Emphasis", sharedValue(sent, varied, "prompt_weighting")?.value === false ? "None" : undefined],
    // The fitted size the cells ran at, not the form's (Generate may shrink it to the model's pixel limit).
    ["Size", sent.length ? (width && height ? `${width}x${height}` : undefined) : `${form.width}x${form.height}`],
    ["Model", form.model],
    // Every cell is conditioned the same way; the control image itself is not embedded.
    ["ControlNet 0", controlNetUnit(sharedValue(sent, varied, "controlnet", "controlnet", sameControlNet)?.value)],
    // Every cell runs with the prompt enhancer off (compare/ forces it), so a restore must not rewrite the prompt.
    ["Prompt enhancement", sharedValue(sent, varied, "enhance_prompt")?.value === false ? "off" : undefined],
    ...hiresFields(sharedValue(sent, varied, "hires", "hires", sameHires)?.value as HiresRequest | undefined),
  ];
  return fields.filter(([, value]) => value !== undefined);
}

/** The grid's active LoRAs as one `LoRAs: id: weight, ...` field at the first cell's weights. Studio ids, Hub ones
 *  included, restore exactly from it; A1111 `<lora:...>` prompt tags would match local adapters only. */
export function gridLoraField(
  ctx: Pick<CompareContext, "loraCapable" | "loras">,
  first: DiffusionGenerateRequest | null | undefined,
): string | null {
  if (!ctx.loraCapable) return null;
  const items: string[] = [];
  for (const row of ctx.loras) {
    const id = row.id.trim();
    if (!id) continue;
    items.push(`${id}: ${first?.loras?.find((l) => l.id === id)?.weight ?? 0}`);
  }
  // CSV-quoted where an id holds a comma or quote, so every id splits back out exactly.
  return items.length ? joinAxisList(items) : null;
}

const AXIS_NAMES = ["X", "Y", "Z"] as const;

/** The grid PNG's A1111-style `parameters`: the prompt, the negative prompt when any
 *  cell engages it (the first cell may not), the shared recipe, then the axes. */
export function gridParameters(
  plan: CompareGridPlan,
  ctx: CompareContext,
  requests: ReadonlyArray<DiffusionGenerateRequest | null | undefined>,
  form: { steps: number; guidance: number; width: number; height: number; model: string | null; keepRandomSeeds: boolean },
): string {
  const fields = gridRecipeFields(plan, requests, form);
  const first = requests.find(Boolean);
  const loras = gridLoraField(ctx, first);
  if (loras) fields.push(["LoRAs", loras]);
  const axes = plan.axes
    .map((a, i) => ({ a, name: AXIS_NAMES[i] }))
    .filter(({ a }) => a.def.id !== NOTHING)
    .map(({ a, name }) => ({
      name,
      type: a.def.label,
      text: a.text,
      fixed: a.def.id === "seed" && !form.keepRandomSeeds ? a.values : undefined,
    }));
  const prompt = ctx.prompt.trim();
  return gridInfotext({
    prompt,
    negativePrompt: gridNegativePrompt(plan, ctx),
    fields,
    axes,
  });
}
