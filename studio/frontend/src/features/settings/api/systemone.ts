// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

export type SystemOneDevice = "cpu" | "gpu";
export type SystemOneBackend = "auto" | "llama.cpp" | "pytorch";

export type SystemOneModel = {
  name: string;
  description: string;
  downloadBytes: number;
  kind: "catalog" | "fine_tune";
  label: string | null;
  available: boolean;
  unavailableReason: string | null;
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
  backend: SystemOneBackend;
  effectiveBackend: string | null;
  loadedBackend: string | null;
  fallbackReason: string | null;
  inputModalities: string[];
  loadingModel: string | null;
  installing: boolean;
  error: string | null;
  mcpUrl: string;
};

export type SystemOneConnection = {
  name: string;
  providerId: string;
  provider: string;
  model: string;
};

export type SystemOneDownloadPlan = {
  repo: string | null;
  revision?: string | null;
  files: string[];
  sizeBytes: number;
  cached: boolean;
  error: string | null;
};

export type SystemOneSettingsPatch = {
  enabled?: boolean;
  model?: string;
  device?: SystemOneDevice;
  backend?: SystemOneBackend;
  expectedEnabled?: boolean;
  expectedModel?: string;
  expectedBackend?: SystemOneBackend;
};

type ApiSystemOneConnection = {
  name: string;
  // biome-ignore lint/style/useNamingConvention: API schema
  provider_id: string;
  provider: string;
  model: string;
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
  models: {
    name: string;
    description: string;
    // biome-ignore lint/style/useNamingConvention: API schema
    download_bytes: number;
    kind?: "catalog" | "fine_tune";
    label?: string | null;
    available?: boolean;
    // biome-ignore lint/style/useNamingConvention: API schema
    unavailable_reason?: string | null;
  }[];
  // biome-ignore lint/style/useNamingConvention: API schema
  loaded_model: string | null;
  // biome-ignore lint/style/useNamingConvention: API schema
  loaded_device: string | null;
  backend: SystemOneBackend;
  // biome-ignore lint/style/useNamingConvention: API schema
  effective_backend: string | null;
  // biome-ignore lint/style/useNamingConvention: API schema
  loaded_backend: string | null;
  // biome-ignore lint/style/useNamingConvention: API schema
  fallback_reason: string | null;
  // biome-ignore lint/style/useNamingConvention: API schema
  input_modalities: string[];
  // biome-ignore lint/style/useNamingConvention: API schema
  loading_model: string | null;
  installing: boolean;
  error: string | null;
  // biome-ignore lint/style/useNamingConvention: API schema
  mcp_url: string;
};

type ApiSystemOneDownloadPlan = {
  repo: string | null;
  revision?: string | null;
  files: string[];
  // biome-ignore lint/style/useNamingConvention: API schema
  size_bytes: number;
  cached: boolean;
  error: string | null;
};

const SETTINGS_PATH = "/api/settings/systemone";
const SYSTEMONE_SETTINGS_EVENT = "unsloth-systemone-settings-change";

export function subscribeSystemOneSettings(
  listener: (settings: SystemOneSettings) => void,
) {
  const handleChange = (event: Event) => {
    listener((event as CustomEvent<SystemOneSettings>).detail);
  };
  window.addEventListener(SYSTEMONE_SETTINGS_EVENT, handleChange);
  return () =>
    window.removeEventListener(SYSTEMONE_SETTINGS_EVENT, handleChange);
}

function publishSystemOneSettings(settings: SystemOneSettings) {
  window.dispatchEvent(
    new CustomEvent(SYSTEMONE_SETTINGS_EVENT, { detail: settings }),
  );
  return settings;
}

function toApiPatch(patch: SystemOneSettingsPatch) {
  const { expectedEnabled, expectedModel, expectedBackend, ...settings } = patch;
  return {
    ...settings,
    ...(expectedEnabled !== undefined && {
      expected_enabled: expectedEnabled,
    }),
    ...(expectedModel !== undefined && { expected_model: expectedModel }),
    ...(expectedBackend !== undefined && { expected_backend: expectedBackend }),
  };
}

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
      kind: m.kind ?? "catalog",
      label: m.label ?? null,
      available: m.available ?? true,
      unavailableReason: m.unavailable_reason ?? null,
    })),
    loadedModel: settings.loaded_model,
    loadedDevice: settings.loaded_device,
    backend: settings.backend,
    effectiveBackend: settings.effective_backend,
    loadedBackend: settings.loaded_backend,
    fallbackReason: settings.fallback_reason,
    inputModalities: settings.input_modalities,
    loadingModel: settings.loading_model,
    installing: settings.installing,
    error: settings.error,
    mcpUrl: settings.mcp_url,
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

export async function updateSystemOneSettings(
  patch: SystemOneSettingsPatch,
): Promise<SystemOneSettings> {
  return publishSystemOneSettings(
    await readSettings(
      await authFetch(SETTINGS_PATH, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(toApiPatch(patch)),
      }),
      "Failed to save Decision API settings",
    ),
  );
}

export async function validateSystemOneSettings(
  patch: SystemOneSettingsPatch,
): Promise<void> {
  const res = await authFetch(`${SETTINGS_PATH}/validate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(toApiPatch(patch)),
  });
  if (!res.ok) {
    throw new Error(
      await readFastApiError(
        res,
        "Couldn't validate the Decision API setting.",
      ),
    );
  }
}

export async function unloadSystemOneModel(): Promise<SystemOneSettings> {
  return readSettings(
    await authFetch(`${SETTINGS_PATH}/unload`, { method: "POST" }),
    "Failed to unload the Decision API model",
  );
}

export async function loadSystemOneConnections(): Promise<
  SystemOneConnection[]
> {
  const res = await authFetch(`${SETTINGS_PATH}/connections`);
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to load Decision API connections"),
    );
  }
  const options = (await res.json()) as ApiSystemOneConnection[];
  return options.map((option) => ({
    name: option.name,
    providerId: option.provider_id,
    provider: option.provider,
    model: option.model,
  }));
}

export async function resolveSystemOneDownload(
  model?: string,
  backend?: SystemOneBackend,
): Promise<SystemOneDownloadPlan> {
  const params = new URLSearchParams();
  if (model) params.set("model", model);
  if (backend) params.set("backend", backend);
  const query = params.size ? `?${params}` : "";
  const res = await authFetch(`${SETTINGS_PATH}/resolve${query}`);
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to check the Decision API model"),
    );
  }
  const plan = (await res.json()) as ApiSystemOneDownloadPlan;
  return {
    repo: plan.repo,
    revision: plan.revision ?? null,
    files: plan.files,
    sizeBytes: plan.size_bytes,
    cached: plan.cached,
    error: plan.error,
  };
}
