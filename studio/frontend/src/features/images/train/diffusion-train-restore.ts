// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { DiffusionTrainingStatus } from "../api.ts";

type Status = Pick<DiffusionTrainingStatus, "active" | "status" | "job_id" | "family" | "base_model">;
type Family = { name: string; base_repos: string[] };

type Run = { family?: string | null; base_model?: string | null };
export type TrainingRunSelection = { familyName: string; baseModel: string | null };

// The family and base a run was started from; family stays null until the model loads, so fall back to the family
// whose bases list base_model.
export function trainingRunSelection(
  run: Run | null | undefined,
  families: readonly Family[],
): TrainingRunSelection | null {
  if (!run) return null;
  const base = run.base_model?.trim() || null;
  const family =
    (run.family ? families.find((f) => f.name === run.family) : undefined) ??
    (base ? families.find((f) => f.base_repos.includes(base)) : undefined);
  return family ? { familyName: family.name, baseModel: base } : null;
}

// The form a running job was started from, so a reload does not show the page default beside it.
export function runningTrainingSelection(
  status: Status | null | undefined,
  families: readonly Family[],
): TrainingRunSelection | null {
  if (!status || !(status.active || status.status === "running")) return null;
  return trainingRunSelection(status, families);
}

const SHOWN_TERMINAL = new Set(["completed", "stopped", "error"]);

// A finished run the panel still shows (the backend keeps the terminal status until the next start), so a reload does
// not show the FLUX.1-dev default beside a completed Z-Image run. Dismissed runs no longer show.
export function shownTrainingSelection(
  status: Status | null | undefined,
  families: readonly Family[],
  dismissedJobId: string | null,
): TrainingRunSelection | null {
  const running = runningTrainingSelection(status, families);
  if (running || !status) return running;
  if (!SHOWN_TERMINAL.has(status.status ?? "") || !status.job_id || status.job_id === dismissedJobId) return null;
  return trainingRunSelection(status, families);
}
