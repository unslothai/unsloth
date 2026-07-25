// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

export type CheckpointLocationSettings = {
  path: string;
  source: "default" | "studio" | "environment" | "colab" | "kaggle";
  editable: boolean;
  isCustom: boolean;
  environmentVariable: string | null;
};

type ApiSettings = Omit<
  CheckpointLocationSettings,
  "isCustom" | "environmentVariable"
> & {
  // biome-ignore lint/style/useNamingConvention: API schema
  is_custom: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  environment_variable: string | null;
};

function fromApi(value: ApiSettings): CheckpointLocationSettings {
  return {
    ...value,
    isCustom: value.is_custom,
    environmentVariable: value.environment_variable,
  };
}

export async function loadCheckpointLocation() {
  const response = await authFetch("/api/settings/checkpoint-location");
  if (!response.ok)
    throw new Error(
      await readFastApiError(response, "Failed to load checkpoint location"),
    );
  return fromApi(await response.json());
}

export async function updateCheckpointLocation(path: string | null) {
  const response = await authFetch("/api/settings/checkpoint-location", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  });
  if (!response.ok)
    throw new Error(
      await readFastApiError(response, "Failed to update checkpoint location"),
    );
  return fromApi(await response.json());
}
