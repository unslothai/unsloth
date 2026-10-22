// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** One compare run as the page's Generate loop sees it: N runs, each with its own request, reporting its records
 *  back. The page keeps owning progress polling, Stop, the gallery merge and error toasts. */

import type { DiffusionGenerateRequest, GalleryImage } from "../api.ts";
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
  // Overlay title while run `done` (0-based) is in flight.
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
      if (cell !== undefined && images[0]) results[cell] = images[0];
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
