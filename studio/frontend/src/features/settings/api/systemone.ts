// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

export type SystemOneDevice = "cpu" | "gpu";

export type SystemOneModel = {
  name: string;
  description: string;
  downloadBytes: number;
};

export type SystemOneSettings = {
  enabled: boolean;
  enabledLocked: boolean;
  model: string;
  modelLocked: boolean;
  device: SystemOneDevice;
  deviceLocked: boolean;
  gpuAvailable: boolean;
  models: SystemOneModel[];
  loadedModel: string | null;
  loadedDevice: string | null;
  loadingModel: string | null;
  installing: boolean;
  error: string | null;
};

export type SystemOneDownloadPlan = {
  repo: string | null;
  files: string[];
  sizeBytes: number;
  cached: boolean;
  error: string | null;
};

type ApiSystemOneSettings = {
  enabled: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  enabled_locked: boolean;
  model: string;
  // biome-ignore lint/style/useNamingConvention: API schema
  model_locked: boolean;
  device: SystemOneDevice;
  // biome-ignore lint/style/useNamingConvention: API schema
  device_locked: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  gpu_available: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  models: { name: string; description: string; download_bytes: number }[];
  // biome-ignore lint/style/useNamingConvention: API schema
  loaded_model: string | null;
  // biome-ignore lint/style/useNamingConvention: API schema
  loaded_device: string | null;
  // biome-ignore lint/style/useNamingConvention: API schema
  loading_model: string | null;
  installing: boolean;
  error: string | null;
};

type ApiSystemOneDownloadPlan = {
  repo: string | null;
  files: string[];
  // biome-ignore lint/style/useNamingConvention: API schema
  size_bytes: number;
  cached: boolean;
  error: string | null;
};

const SETTINGS_PATH = "/api/settings/systemone";

function fromApi(settings: ApiSystemOneSettings): SystemOneSettings {
  return {
    enabled: settings.enabled,
    enabledLocked: settings.enabled_locked,
    model: settings.model,
    modelLocked: settings.model_locked,
    device: settings.device,
    deviceLocked: settings.device_locked,
    gpuAvailable: settings.gpu_available,
    models: settings.models.map((m) => ({
      name: m.name,
      description: m.description,
      downloadBytes: m.download_bytes,
    })),
    loadedModel: settings.loaded_model,
    loadedDevice: settings.loaded_device,
    loadingModel: settings.loading_model,
    installing: settings.installing,
    error: settings.error,
  };
}

async function readSettings(res: Response, fallback: string) {
  if (!res.ok) {
    throw new Error(await readFastApiError(res, fallback));
  }
  return fromApi((await res.json()) as ApiSystemOneSettings);
}

export async function loadSystemOneSettings(): Promise<SystemOneSettings> {
  return readSettings(
    await authFetch(SETTINGS_PATH),
    "Failed to load Decision API settings",
  );
}

export async function updateSystemOneSettings(patch: {
  enabled?: boolean;
  model?: string;
  device?: SystemOneDevice;
}): Promise<SystemOneSettings> {
  return readSettings(
    await authFetch(SETTINGS_PATH, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    }),
    "Failed to save Decision API settings",
  );
}

export async function unloadSystemOneModel(): Promise<SystemOneSettings> {
  return readSettings(
    await authFetch(`${SETTINGS_PATH}/unload`, { method: "POST" }),
    "Failed to unload the Decision API model",
  );
}

export async function resolveSystemOneDownload(): Promise<SystemOneDownloadPlan> {
  const res = await authFetch(`${SETTINGS_PATH}/resolve`);
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to check the Decision API model"),
    );
  }
  const plan = (await res.json()) as ApiSystemOneDownloadPlan;
  return {
    repo: plan.repo,
    files: plan.files,
    sizeBytes: plan.size_bytes,
    cached: plan.cached,
    error: plan.error,
  };
}
