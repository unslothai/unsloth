// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** One compare run as the page's Generate loop sees it: N runs, each with its own request, reporting its records
 *  back. The page keeps owning progress polling, Stop, the gallery merge and error toasts. */

import type { DiffusionGenerateRequest, GalleryImage } from "../api.ts";
import { SAMPLER_LABELS, SCHEDULE_LABELS } from "../lib/sampler-choice.ts";
import { gridInfotext } from "./compose-grid.ts";
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
  options: { keepRandomSeeds: boolean; seedFieldEmpty: boolean; rng?: () => number },
): CompareRunState {
  const results: Array<GalleryImage | null> = Array.from({ length: plan.cellCount }, () => null);
  const requests: Array<DiffusionGenerateRequest | null> = Array.from({ length: plan.cellCount }, () => null);
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
 *  omitted it. */
function sharedValue(
  sent: ReadonlyArray<Readonly<Record<string, unknown>>>,
  varied: ReadonlySet<string>,
  key: string,
  axisId: string = key,
): { value: string | number | undefined } | null {
  if (varied.has(axisId) || sent.length === 0) return null;
  const value = sent[0][key];
  if (!sent.every((r) => r[key] === value)) return null;
  return { value: typeof value === "string" || typeof value === "number" ? value : undefined };
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
  const shared = (key: string, axisId?: string) => sharedValue(sent, varied, key, axisId)?.value;
  const sampler = sharedValue(sent, varied, "sampler");
  const schedule = shared("schedule_type", "schedule");
  const width = shared("width");
  const height = shared("height");
  const fields: Array<[string, string | number | null | undefined]> = [
    ["Steps", sent.length ? shared("steps") : form.steps],
    ["Sampler", sampler ? (SAMPLER_LABELS[String(sampler.value ?? "default")] ?? sampler.value) : undefined],
    ["Schedule type", schedule ? (SCHEDULE_LABELS[String(schedule)] ?? schedule) : undefined],
    ["Shift", shared("shift")],
    ["Base shift", shared("base_shift")],
    ["Max shift", shared("max_shift")],
    ["CFG scale", sent.length ? shared("guidance") : form.guidance],
    ["Seed", shared("seed")],
    ["Variation seed", shared("variation_seed")],
    ["Variation seed strength", shared("variation_strength")],
    // The fitted size the cells ran at, not the form's (Generate may shrink it to the model's pixel limit).
    ["Size", sent.length ? (width && height ? `${width}x${height}` : undefined) : `${form.width}x${form.height}`],
    ["Model", form.model],
  ];
  return fields.filter(([, value]) => value !== undefined);
}

/** The grid's active LoRAs as A1111 `<lora:name:weight>` prompt tags (what the Images importer restores), at the
 *  first cell's weights; an id a tag cannot carry goes in a `LoRAs` field instead. */
export function gridLoraTags(
  ctx: Pick<CompareContext, "loraCapable" | "loras">,
  first: DiffusionGenerateRequest | null | undefined,
): { tags: string; field: string | null } {
  if (!ctx.loraCapable) return { tags: "", field: null };
  const tags: string[] = [];
  const plain: string[] = [];
  for (const row of ctx.loras) {
    const id = row.id.trim();
    if (!id) continue;
    const weight = first?.loras?.find((l) => l.id === id)?.weight ?? 0;
    if (/[:<>]/.test(id)) plain.push(`${id}: ${weight}`);
    else tags.push(`<lora:${id}:${weight}>`);
  }
  return { tags: tags.join(" "), field: plain.length ? plain.join(", ") : null };
}

const AXIS_NAMES = ["X", "Y", "Z"] as const;

/** The grid PNG's A1111-style `parameters`: the prompt with the active LoRAs as tags, the negative prompt when any
 *  cell engages it (the first cell may not), the shared recipe, then the axes. */
export function gridParameters(
  plan: CompareGridPlan,
  ctx: CompareContext,
  requests: ReadonlyArray<DiffusionGenerateRequest | null | undefined>,
  form: { steps: number; guidance: number; width: number; height: number; model: string | null; keepRandomSeeds: boolean },
): string {
  const fields = gridRecipeFields(plan, requests, form);
  const first = requests.find(Boolean);
  const loras = gridLoraTags(ctx, first);
  if (loras.field) fields.push(["LoRAs", loras.field]);
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
    prompt: loras.tags ? `${prompt} ${loras.tags}` : prompt,
    negativePrompt: gridNegativePrompt(plan, ctx),
    fields,
    axes,
  });
}
