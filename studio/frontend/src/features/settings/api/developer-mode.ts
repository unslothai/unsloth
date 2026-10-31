// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

export type DeveloperModeSettings = {
  /** Effective answer: the env override when set, else the saved setting. */
  videoCpu: boolean;
  videoCpuSetting: boolean;
  defaultVideoCpu: boolean;
  /** true / false when UNSLOTH_VIDEO_ALLOW_CPU decides, null when the setting does. */
  videoCpuEnvOverride: boolean | null;
  /** Whether the opt-in changes anything on this host (a CPU-only host, nothing else wrong). */
  videoCpuApplicable: boolean;
  hostRamBytes: number | null;
  minRamBytes: number;
};

type ApiDeveloperModeSettings = {
  // biome-ignore lint/style/useNamingConvention: API schema
  video_cpu: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  video_cpu_setting: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  default_video_cpu: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  video_cpu_env_override: boolean | null;
  // biome-ignore lint/style/useNamingConvention: API schema
  video_cpu_applicable: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  host_ram_bytes: number | null;
  // biome-ignore lint/style/useNamingConvention: API schema
  min_ram_bytes: number;
};

export function developerModeFromApi(
  settings: ApiDeveloperModeSettings,
): DeveloperModeSettings {
  return {
    videoCpu: settings.video_cpu,
    videoCpuSetting: settings.video_cpu_setting,
    defaultVideoCpu: settings.default_video_cpu,
    videoCpuEnvOverride: settings.video_cpu_env_override ?? null,
    videoCpuApplicable: settings.video_cpu_applicable,
    hostRamBytes: settings.host_ram_bytes ?? null,
    minRamBytes: settings.min_ram_bytes,
  };
}

export async function loadDeveloperModeSettings(): Promise<DeveloperModeSettings> {
  const res = await authFetch("/api/settings/developer-mode");
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to load Developer mode settings"),
    );
  }
  return developerModeFromApi(await res.json());
}

export async function updateDeveloperModeSettings(
  videoCpu: boolean,
): Promise<DeveloperModeSettings> {
  const res = await authFetch("/api/settings/developer-mode", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    // biome-ignore lint/style/useNamingConvention: API schema
    body: JSON.stringify({ video_cpu: videoCpu }),
  });
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to update Developer mode settings"),
    );
  }
  return developerModeFromApi(await res.json());
}
