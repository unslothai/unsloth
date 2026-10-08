// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import {
  formatFastApiDetail,
  readFastApiError,
} from "@/lib/format-fastapi-error";
import type { DecisionResponse } from "../lib/decision-request";

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
  llamaCppOnly: boolean;
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
  mcpUrl: string;
  backend: SystemOneBackend;
  nativeCtx: number;
  effectiveBackend: string | null;
  loadedBackend: string | null;
  fallbackReason: string | null;
  inputModalities: string[];
  layout: string | null;
};

export type SystemOneConnection = {
  name: string;
  providerId: string;
  provider: string;
  model: string;
};

export type SystemOneDownloadPlan = {
  repo: string | null;
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
  nativeCtx?: number;
  expectedEnabled?: boolean;
  expectedModel?: string;
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
    // biome-ignore lint/style/useNamingConvention: API schema
    llama_cpp_only?: boolean;
  }[];
  // biome-ignore lint/style/useNamingConvention: API schema
  loaded_model: string | null;
  // biome-ignore lint/style/useNamingConvention: API schema
  loaded_device: string | null;
  // biome-ignore lint/style/useNamingConvention: API schema
  loading_model: string | null;
  installing: boolean;
  error: string | null;
  // biome-ignore lint/style/useNamingConvention: API schema
  mcp_url: string;
  backend?: SystemOneBackend;
  // biome-ignore lint/style/useNamingConvention: API schema
  native_ctx?: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  effective_backend?: string | null;
  // biome-ignore lint/style/useNamingConvention: API schema
  loaded_backend?: string | null;
  // biome-ignore lint/style/useNamingConvention: API schema
  fallback_reason?: string | null;
  // biome-ignore lint/style/useNamingConvention: API schema
  input_modalities?: string[];
  layout?: string | null;
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
  const { expectedEnabled, expectedModel, nativeCtx, ...settings } = patch;
  return {
    ...settings,
    ...(nativeCtx !== undefined && { native_ctx: nativeCtx }),
    ...(expectedEnabled !== undefined && {
      expected_enabled: expectedEnabled,
    }),
    ...(expectedModel !== undefined && { expected_model: expectedModel }),
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
      llamaCppOnly: m.llama_cpp_only ?? false,
    })),
    loadedModel: settings.loaded_model,
    loadedDevice: settings.loaded_device,
    loadingModel: settings.loading_model,
    installing: settings.installing,
    error: settings.error,
    mcpUrl: settings.mcp_url,
    backend: settings.backend ?? "auto",
    nativeCtx: settings.native_ctx ?? 16384,
    effectiveBackend: settings.effective_backend ?? null,
    loadedBackend: settings.loaded_backend ?? null,
    fallbackReason: settings.fallback_reason ?? null,
    inputModalities: settings.input_modalities ?? ["text"],
    layout: settings.layout ?? null,
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
    files: plan.files,
    sizeBytes: plan.size_bytes,
    cached: plan.cached,
    error: plan.error,
  };
}

const LOAD_WAIT_MS = 10 * 60 * 1000;

export class DecisionError extends Error {
  readonly status: number | null;

  constructor(message: string, status: number | null) {
    super(message);
    this.status = status;
  }
}

function wait(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve) => {
    const timer = window.setTimeout(resolve, ms);
    signal.addEventListener(
      "abort",
      () => {
        window.clearTimeout(timer);
        resolve();
      },
      { once: true },
    );
  });
}

export async function runDecision(
  body: unknown,
  signal: AbortSignal,
  onWaiting: (message: string) => void,
): Promise<{ response: DecisionResponse; latencyMs: number }> {
  const deadline = Date.now() + LOAD_WAIT_MS;
  for (;;) {
    const started = performance.now();
    const res = await authFetch("/v1/systemone", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal,
    });
    if (res.ok) {
      const response = (await res.json()) as DecisionResponse;
      return { response, latencyMs: Math.round(performance.now() - started) };
    }
    const data = (await res.json().catch(() => null)) as {
      detail?: unknown;
    } | null;
    const detail = data?.detail;
    const fields =
      detail && typeof detail === "object"
        ? (detail as Record<string, unknown>)
        : {};
    const message =
      (typeof fields.message === "string" ? fields.message : null) ??
      formatFastApiDetail(detail) ??
      res.statusText;
    const loading = res.status === 503 && fields.error_type === "model_loading";
    if (!loading || Date.now() > deadline || signal.aborted) {
      throw new DecisionError(message, res.status);
    }
    onWaiting(message);
    const retryAfter = Number(res.headers.get("Retry-After")) || 5;
    await wait(Math.min(retryAfter * 1000, deadline - Date.now()), signal);
    if (signal.aborted || Date.now() >= deadline) {
      throw new DecisionError(message, res.status);
    }
  }
}
