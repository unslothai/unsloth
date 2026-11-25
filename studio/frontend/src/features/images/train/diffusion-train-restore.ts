// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { DiffusionTrainingStatus } from "../api.ts";

type Status = Pick<DiffusionTrainingStatus, "active" | "status" | "job_id" | "family" | "base_model">;
type Family = { name: string; base_repos: string[] };

// Job's own form, so reloads skip page defaults; family is null until load: match base_model.
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
