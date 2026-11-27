// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** `/api/system` `repair_recommendation`: an existing Windows home older than the managed torch. */
export interface RepairRecommendation {
  reason:
    | "torch_older_than_managed"
    | "driver_older_than_managed"
    | string;
  torch?: string | null;
  managed_torch?: string | null;
}

type Translate = (key: never, values?: Record<string, string>) => string;

/** The hint under the Repair row, or null when nothing is recommended or the reason is unknown. */
export function repairRecommendationText(
  rec: RepairRecommendation | null | undefined,
  t: Translate,
): string | null {
  if (!rec) return null;
  const values = {
    torch: rec.torch ?? "",
    managedTorch: rec.managed_torch ?? "",
  };
  if (rec.reason === "torch_older_than_managed") {
    return t("settings.general.repairInstall.recommendedTorch" as never, values);
  }
  if (rec.reason === "driver_older_than_managed") {
    return t("settings.general.repairInstall.recommendedDriver" as never, values);
  }
  return null;
}
