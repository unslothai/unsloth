// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";
import { create } from "zustand";

export type Q4nxExportSettings = {
  enabled: boolean;
  npuSupported: boolean;
};

type ApiQ4nxExportSettings = {
  enabled: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  npu_supported: boolean;
};

const fromApi = (s: ApiQ4nxExportSettings): Q4nxExportSettings => ({
  enabled: s.enabled,
  npuSupported: s.npu_supported,
});

/** Shared so the export page sees a Settings change without a reload. */
export const useQ4nxExportStore = create<{
  settings: Q4nxExportSettings | null;
}>(() => ({ settings: null }));

// Bumped by each save, so a read that started before it cannot overwrite its answer.
let saves = 0;

export async function loadQ4nxExportSettings(): Promise<Q4nxExportSettings> {
  const started = saves;
  const res = await authFetch("/api/settings/q4nx-export");
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to load Q4NX export setting"),
    );
  }
  const settings = fromApi(await res.json());
  if (started !== saves) {
    return useQ4nxExportStore.getState().settings ?? settings;
  }
  useQ4nxExportStore.setState({ settings });
  return settings;
}

export async function updateQ4nxExportSettings(
  enabled: boolean,
): Promise<Q4nxExportSettings> {
  saves += 1;
  const res = await authFetch("/api/settings/q4nx-export", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ enabled }),
  });
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to update Q4NX export setting"),
    );
  }
  const settings = fromApi(await res.json());
  useQ4nxExportStore.setState({ settings });
  return settings;
}
