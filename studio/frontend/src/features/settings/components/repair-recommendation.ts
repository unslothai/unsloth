// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** `/api/system` `repair_recommendation`: an existing Windows home older than the managed torch / xFormers. */
export interface RepairRecommendation {
  reason: "torch_older_than_managed" | "xformers_older_than_managed" | string;
  torch?: string | null;
  xformers?: string | null;
  managed_torch?: string | null;
  managed_xformers?: string | null;
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
    xformers: rec.xformers ?? "",
    managedTorch: rec.managed_torch ?? "",
    managedXformers: rec.managed_xformers ?? "",
  };
  if (rec.reason === "torch_older_than_managed") {
    return t("settings.general.repairInstall.recommendedTorch" as never, values);
  }
  if (rec.reason === "xformers_older_than_managed") {
    return t("settings.general.repairInstall.recommendedXformers" as never, values);
  }
  return null;
}
