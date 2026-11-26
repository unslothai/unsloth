// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** `parallel_mode` on /api/train/start, minus "auto". Order = the override menu's order. */
export const PARALLEL_MODES = [
  "single",
  "layer_split",
  "ddp",
  "fsdp2",
  "tp",
  "pp",
  "cp",
  "ep",
  "custom",
] as const;

export type ParallelMode = (typeof PARALLEL_MODES)[number];

export type ParallelModeChoice = ParallelMode | "auto";

export interface ParallelModeAvailability {
  mode: ParallelModeChoice;
  available: boolean;
  reason: string | null;
  minGpus: number;
}

export interface ParallelGpu {
  index: number;
  name: string;
  totalGb: number | null;
  freeGb: number | null;
}

export interface ParallelCapabilities {
  multiProcess: boolean;
  multiProcessReason: string | null;
  gpus: ParallelGpu[];
  modes: ParallelModeAvailability[];
}

export interface ParallelPlanError {
  code: string | null;
  message: string;
  alternative: string | null;
}

export interface ParallelPlan {
  mode: ParallelMode;
  worldSize: number;
  gpuIds: number[];
  spec: string | null;
  reason: string | null;
  perGpuGb: number | null;
  warnings: string[];
  /** Host facts (mixed GPUs), shown muted; warnings are risks of this run. */
  notes: string[];
  modes: ParallelModeAvailability[];
  error: ParallelPlanError | null;
}

/** `GET /api/train/status` `parallel`, and history `config.parallel`. */
export interface RunParallelInfo {
  mode: ParallelMode;
  worldSize: number;
  gpuIds: number[];
  spec: string | null;
}

export interface ParallelOverride {
  enabled: boolean;
  mode: ParallelModeChoice;
  /** null = every detected GPU. */
  gpuIds: number[] | null;
  /** Only sent with "custom", e.g. "dp=2,tp=2". */
  spec: string;
}

export interface ParallelRequestFields {
  parallel_mode?: ParallelModeChoice;
  parallel_spec?: string;
  gpu_ids?: number[];
}
