// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

export const ML_WARMUP_FEATURES = [
  "hardware",
  "text",
  "datasets",
  "media",
  "quantization",
] as const;

export type MlWarmupFeature = (typeof ML_WARMUP_FEATURES)[number];

export type MlWarmupSettings = {
  features: MlWarmupFeature[];
  effectiveFeatures: MlWarmupFeature[];
  disabledByEnv: boolean;
};

type ApiMlWarmupSettings = {
  features: MlWarmupFeature[];
  // biome-ignore lint/style/useNamingConvention: API schema
  effective_features: MlWarmupFeature[];
  // biome-ignore lint/style/useNamingConvention: API schema
  disabled_by_env: boolean;
};

function fromApi(settings: ApiMlWarmupSettings): MlWarmupSettings {
  return {
    features: settings.features,
    effectiveFeatures: settings.effective_features,
    disabledByEnv: settings.disabled_by_env,
  };
}

export async function loadMlWarmupSettings(): Promise<MlWarmupSettings> {
  const res = await authFetch("/api/settings/ml-warmup");
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to load startup warmup settings"),
    );
  }
  return fromApi(await res.json());
}

export async function updateMlWarmupSettings(
  features: readonly MlWarmupFeature[],
): Promise<MlWarmupSettings> {
  const res = await authFetch("/api/settings/ml-warmup", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ features }),
  });
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to save startup warmup settings"),
    );
  }
  return fromApi(await res.json());
}

/** Keep hardware's explicit preference intact when a dependent selection is removed. */
export function updateWarmupSelection(
  selected: readonly MlWarmupFeature[],
  feature: MlWarmupFeature,
  enabled: boolean,
): MlWarmupFeature[] {
  const next = new Set(selected);
  if (enabled) next.add(feature);
  else next.delete(feature);
  return ML_WARMUP_FEATURES.filter((item) => next.has(item));
}

export function hardwareWarmupRequired(features: readonly MlWarmupFeature[]): boolean {
  return features.some(
    (feature) => feature === "text" || feature === "media" || feature === "quantization",
  );
}
