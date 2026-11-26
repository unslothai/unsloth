// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { DiffusionTrainingStatus } from "../api.ts";

type Status = Pick<DiffusionTrainingStatus, "active" | "status" | "job_id" | "family" | "base_model">;
type Family = { name: string; base_repos: string[] };

const TERMINAL_STATUSES = new Set(["completed", "stopped", "error"]);

// The form a running job was started from, so a reload does not show the page default beside it.
// family stays null until the model loads, so fall back to the family whose bases list base_model.
// A finished run (completed, stopped, error) also seeds the form, but only while the user has not picked a model:
// the backend keeps that status until the next start, and the panel beside the form describes that run.
export function runningTrainingSelection(
  status: Status | null | undefined,
  families: readonly Family[],
  opts: { formUntouched?: boolean } = {},
): { familyName: string; baseModel: string | null } | null {
  if (!status) return null;
  const running = Boolean(status.active) || status.status === "running";
  const terminal = !running && TERMINAL_STATUSES.has(status.status);
  if (!running && !(terminal && opts.formUntouched)) return null;
  return trainingRunSelection(status, families);
}

// The family and base a run record names (status or run detail), when the family is one this build can train.
export function trainingRunSelection(
  run: { family?: string | null; base_model?: string | null } | null | undefined,
  families: readonly Family[],
): { familyName: string; baseModel: string | null } | null {
  if (!run) return null;
  const base = run.base_model?.trim() || null;
  const family =
    (run.family ? families.find((f) => f.name === run.family) : undefined) ??
    (base ? families.find((f) => f.base_repos.includes(base)) : undefined);
  return family ? { familyName: family.name, baseModel: base } : null;
}
