// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { DiffusionTrainingStatus } from "../api.ts";

type Status = Pick<DiffusionTrainingStatus, "active" | "status" | "job_id" | "family" | "base_model">;
type Family = { name: string; base_repos: string[] };

// The form a running job was started from, so a reload does not show the page default beside it.
// family stays null until the model loads, so fall back to the family whose bases list base_model.
export function runningTrainingSelection(
  status: Status | null | undefined,
  families: readonly Family[],
): { familyName: string; baseModel: string | null } | null {
  if (!status || !(status.active || status.status === "running")) return null;
  const base = status.base_model?.trim() || null;
  const family =
    (status.family ? families.find((f) => f.name === status.family) : undefined) ??
    (base ? families.find((f) => f.base_repos.includes(base)) : undefined);
  return family ? { familyName: family.name, baseModel: base } : null;
}
