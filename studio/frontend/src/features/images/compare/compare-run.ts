// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** One compare run as the page's Generate loop sees it: N runs, each with its own request, reporting its records
 *  back. The page keeps owning progress polling, Stop, the gallery merge and error toasts. */

import type { DiffusionGenerateRequest, GalleryImage } from "../api.ts";
import { SAMPLER_LABELS, SCHEDULE_LABELS } from "../lib/sampler-choice.ts";
import {
  buildCellRequest,
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

/** The grid's shared recipe for its `parameters` chunk. Sampler and schedule are written whenever an axis does not
 *  vary them, so the PNG still names the scheduler its cells ran. */
export function gridRecipeFields(
  plan: CompareGridPlan,
  first: DiffusionGenerateRequest | null | undefined,
  form: { steps: number; guidance: number; width: number; height: number; model: string | null },
): Array<[string, string | number | null | undefined]> {
  const varied = new Set(plan.axes.map((a) => a.def.id));
  const fields: Array<[string, string | number | null | undefined]> = [
    ["Steps", first?.steps ?? form.steps],
  ];
  if (!varied.has("sampler")) fields.push(["Sampler", SAMPLER_LABELS[first?.sampler ?? "default"] ?? first?.sampler]);
  if (!varied.has("schedule") && first?.schedule_type) {
    fields.push(["Schedule type", SCHEDULE_LABELS[first.schedule_type] ?? first.schedule_type]);
  }
  fields.push(
    ["CFG scale", first?.guidance ?? form.guidance],
    ["Seed", first?.seed],
    ["Size", `${form.width}x${form.height}`],
    ["Model", form.model],
  );
  return fields;
}
