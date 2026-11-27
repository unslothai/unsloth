// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TranslationKey } from "@/i18n";
import {
  PARALLEL_MODES,
  type ParallelCapabilities,
  type ParallelGpu,
  type ParallelMode,
  type ParallelModeAvailability,
  type ParallelModeChoice,
  type ParallelOverride,
  type ParallelPlan,
  type ParallelPlanError,
  type ParallelRequestFields,
  type RunParallelInfo,
} from "../types/parallel.ts";

type Rec = Record<string, unknown>;

const isRec = (v: unknown): v is Rec =>
  typeof v === "object" && v !== null && !Array.isArray(v);

const num = (v: unknown): number | null =>
  typeof v === "number" && Number.isFinite(v) ? v : null;

const str = (v: unknown): string | null =>
  typeof v === "string" && v.trim() !== "" ? v : null;

const intList = (v: unknown): number[] =>
  Array.isArray(v)
    ? v.filter((x): x is number => Number.isInteger(x) && x >= 0)
    : [];

export function isParallelMode(v: unknown): v is ParallelMode {
  return (PARALLEL_MODES as readonly unknown[]).includes(v);
}

const isModeChoice = (v: unknown): v is ParallelModeChoice =>
  v === "auto" || isParallelMode(v);

function availabilityList(raw: unknown): ParallelModeAvailability[] {
  if (!Array.isArray(raw)) return [];
  return raw.filter(isRec).flatMap((m) =>
    isModeChoice(m.mode)
      ? [
          {
            mode: m.mode,
            available: m.available === true,
            reason: str(m.reason),
            minGpus: num(m.min_gpus) ?? 1,
          },
        ]
      : [],
  );
}

function planError(raw: unknown): ParallelPlanError | null {
  if (typeof raw === "string") {
    return str(raw) ? { code: null, message: raw, alternative: null } : null;
  }
  if (!isRec(raw)) return null;
  const message = str(raw.message);
  return message
    ? { code: str(raw.code), message, alternative: str(raw.alternative) }
    : null;
}

/** `GET /api/train/parallel/capabilities`; null when unreadable. */
export function normalizeParallelCapabilities(
  raw: unknown,
): ParallelCapabilities | null {
  if (!isRec(raw) || !Array.isArray(raw.gpus)) return null;
  const gpus: ParallelGpu[] = raw.gpus.filter(isRec).flatMap((g) => {
    const index = num(g.index);
    if (index === null || !Number.isInteger(index) || index < 0) return [];
    return [
      {
        index,
        name: str(g.name) ?? `GPU ${index}`,
        totalGb: num(g.total_gb),
        freeGb: num(g.free_gb),
      },
    ];
  });
  return {
    multiProcess: raw.multi_process === true,
    multiProcessReason: str(raw.multi_process_reason),
    gpus,
    modes: availabilityList(raw.modes),
  };
}

/** `POST /api/train/parallel/plan`; anything unreadable is "no plan", never a crash. */
export function normalizeParallelPlan(raw: unknown): ParallelPlan | null {
  if (!isRec(raw) || !isParallelMode(raw.mode)) return null;
  const perGpu =
    isRec(raw.estimate) && isRec(raw.estimate.per_gpu_gb)
      ? num(raw.estimate.per_gpu_gb[raw.mode])
      : null;
  return {
    mode: raw.mode,
    worldSize: Math.max(num(raw.world_size) ?? 1, 1),
    gpuIds: intList(raw.gpu_ids),
    spec: str(raw.spec),
    reason: str(raw.reason),
    perGpuGb: perGpu,
    warnings: Array.isArray(raw.warnings)
      ? raw.warnings.filter((w): w is string => str(w) !== null)
      : [],
    notes: Array.isArray(raw.notes)
      ? raw.notes.filter((n): n is string => str(n) !== null)
      : [],
    modes: availabilityList(raw.modes),
    error: planError(raw.error),
  };
}

/** `parallel` on the training status / history config; null for legacy and single-process runs. */
export function normalizeRunParallel(raw: unknown): RunParallelInfo | null {
  if (!isRec(raw) || !isParallelMode(raw.mode)) return null;
  return {
    mode: raw.mode,
    worldSize: Math.max(num(raw.world_size) ?? 1, 1),
    gpuIds: intList(raw.gpu_ids),
    spec: str(raw.spec),
  };
}

/** Two or more GPUs training can use; one GPU (or none) keeps the Train page as it was. */
export function showsGpuSection(gpuCount: number): boolean {
  return gpuCount >= 2;
}

export const PARALLEL_MODE_LABEL_KEYS = {
  auto: "studio.gpus.mode.auto",
  single: "studio.gpus.mode.single",
  layer_split: "studio.gpus.mode.layerSplit",
  ddp: "studio.gpus.mode.ddp",
  fsdp2: "studio.gpus.mode.fsdp2",
  tp: "studio.gpus.mode.tp",
  pp: "studio.gpus.mode.pp",
  cp: "studio.gpus.mode.cp",
  ep: "studio.gpus.mode.ep",
  custom: "studio.gpus.mode.custom",
} as const satisfies Record<ParallelModeChoice, TranslationKey>;

const MODE_SUMMARY_KEYS = {
  single: "studio.gpus.summary.single",
  layer_split: "studio.gpus.summary.layerSplit",
  ddp: "studio.gpus.summary.ddp",
  fsdp2: "studio.gpus.summary.fsdp2",
  tp: "studio.gpus.summary.tp",
  pp: "studio.gpus.summary.pp",
  cp: "studio.gpus.summary.cp",
  ep: "studio.gpus.summary.ep",
  custom: "studio.gpus.summary.custom",
} as const satisfies Record<ParallelMode, TranslationKey>;

/** single and layer_split are one process, so world_size is 1 there: count the GPUs instead. An auto
 * layer split leaves gpu_ids null (picked at load), so `hostGpuCount` stands in. */
export function runGpuCount(
  run: {
    mode: ParallelMode;
    worldSize: number;
    gpuIds: readonly number[];
  },
  hostGpuCount = 1,
): number {
  if (run.mode === "single") return 1;
  if (run.mode === "layer_split") {
    return Math.max(run.gpuIds.length || hostGpuCount, 1);
  }
  return Math.max(run.worldSize, run.gpuIds.length, 1);
}

export function planSummary(
  plan: Pick<ParallelPlan, "mode" | "worldSize" | "gpuIds" | "spec">,
  formatCount: (n: number) => string = String,
  hostGpuCount = 1,
): { key: TranslationKey; params: { count: string; spec: string } } {
  return {
    key: MODE_SUMMARY_KEYS[plan.mode],
    params: {
      count: formatCount(runGpuCount(plan, hostGpuCount)),
      spec: plan.spec ?? "",
    },
  };
}

/** Model-aware availability from the plan first, host-level from capabilities second. */
export function modeAvailability(
  mode: ParallelModeChoice,
  plan: ParallelPlan | null,
  capabilities: ParallelCapabilities | null,
): ParallelModeAvailability | null {
  return (
    plan?.modes.find((m) => m.mode === mode) ??
    capabilities?.modes.find((m) => m.mode === mode) ??
    null
  );
}

export type OverrideIssueKey =
  | "studio.gpus.error.noGpu"
  | "studio.gpus.error.singleNeedsOne"
  | "studio.gpus.error.needsTwo"
  | "studio.gpus.error.customSpec";

/** What the browser can answer without a round trip (`parallel_gpu_ids`); the backend checks the rest. */
export function overrideIssue(
  override: ParallelOverride,
  allGpuIds: readonly number[],
): OverrideIssueKey | null {
  if (!override.enabled) return null;
  const count = (override.gpuIds ?? allGpuIds).length;
  if (count === 0) return "studio.gpus.error.noGpu";
  // With every GPU ticked the backend trains on the freest one; only an explicit set of several is refused.
  if (
    override.mode === "single" &&
    override.gpuIds !== null &&
    override.gpuIds.length > 1
  ) {
    return "studio.gpus.error.singleNeedsOne";
  }
  if (override.mode !== "auto" && override.mode !== "single" && count < 2) {
    return "studio.gpus.error.needsTwo";
  }
  if (override.mode === "custom" && override.spec.trim() === "") {
    return "studio.gpus.error.customSpec";
  }
  return null;
}

/** Mirrors parallel_plan._spec_example. */
export function layoutExample(gpuCount: number): string {
  return `dp=${Math.max(gpuCount, 1)}`;
}

/** null when the selection becomes every GPU again, so "all" survives a card being added later. */
export function toggleGpuSelection(
  selected: readonly number[] | null,
  index: number,
  allGpuIds: readonly number[],
): number[] | null {
  const current = new Set(selected ?? allGpuIds);
  if (current.has(index)) current.delete(index);
  else current.add(index);
  const next = allGpuIds.filter((id) => current.has(id));
  return next.length === allGpuIds.length ? null : next;
}

/** The fields /plan and /start carry. Nothing on a 0 or 1 GPU host, or when the backend has no
 * multi-GPU routes, so those requests stay byte-identical to before. A multi-GPU host sends "auto",
 * so the run does what the GPUs section showed. */
export function parallelRequestFields(
  override: ParallelOverride,
  capabilities: ParallelCapabilities | null,
): ParallelRequestFields {
  if (!capabilities || !showsGpuSection(capabilities.gpus.length)) return {};
  if (!override.enabled) return { parallel_mode: "auto" };
  const known = new Set(capabilities.gpus.map((g) => g.index));
  const fields: ParallelRequestFields = { parallel_mode: override.mode };
  if (override.mode === "custom") fields.parallel_spec = override.spec.trim();
  if (override.gpuIds !== null) {
    // A GPU that disappeared since the choice was saved is dropped, not sent.
    fields.gpu_ids = override.gpuIds.filter((id) => known.has(id));
  }
  return fields;
}
