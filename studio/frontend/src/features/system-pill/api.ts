// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { pillSetConfig, pillStatus } from "@/lib/pill-native";
import type { PillModelOption, PillSettings } from "./types";

// These run at startup, racing the desktop auth handshake: a 401 here means
// "not signed in yet", not "no". Retry briefly before reporting the failure.
async function authFetchBootTolerant(path: string): Promise<Response> {
  let response = await authFetch(path);
  for (let attempt = 0; response.status === 401 && attempt < 5; attempt++) {
    await new Promise((resolve) => setTimeout(resolve, 1500));
    response = await authFetch(path);
  }
  return response;
}

export async function fetchPillSettings(): Promise<PillSettings> {
  const response = await authFetchBootTolerant("/api/pill/settings");
  if (!response.ok) throw new Error(`Failed to load settings (${response.status})`);
  return (await response.json()) as PillSettings;
}

export async function updatePillSettings(
  update: Partial<PillSettings>,
): Promise<PillSettings> {
  const response = await authFetch("/api/pill/settings", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(update),
  });
  if (!response.ok) throw new Error(`Failed to save settings (${response.status})`);
  return (await response.json()) as PillSettings;
}

// Pushes enabled/excludedApps down to the Rust layer, preserving its hotkey.
export async function syncNativePillConfig(settings: PillSettings): Promise<void> {
  const status = await pillStatus();
  if (!status.supported) return;
  if (
    settings.enabled !== status.enabled ||
    settings.excludedApps.join("\n") !== status.excludedApps.join("\n")
  ) {
    await pillSetConfig({
      enabled: settings.enabled,
      hotkey: status.hotkey,
      excludedApps: settings.excludedApps,
    });
  }
}

type LoRAInfo = {
  display_name: string;
  adapter_path: string;
  source: "training" | "exported";
  export_type: "lora" | "merged" | "gguf";
};

type CachedGgufEntry = {
  repo_id?: string;
};

export async function fetchPillModelOptions(): Promise<PillModelOption[]> {
  const options: PillModelOption[] = [];
  try {
    const response = await authFetchBootTolerant("/api/models/loras");
    if (response.ok) {
      const body = (await response.json()) as { loras: LoRAInfo[] };
      for (const lora of body.loras) {
        if (lora.export_type === "gguf") {
          options.push({
            id: lora.adapter_path,
            label: lora.display_name,
            source: "exported",
          });
        }
      }
    }
  } catch {
    // exported models are optional
  }
  try {
    const response = await authFetchBootTolerant("/api/models/cached-gguf");
    if (response.ok) {
      const body = (await response.json()) as { cached: CachedGgufEntry[] };
      for (const entry of body.cached) {
        if (entry.repo_id) {
          options.push({ id: entry.repo_id, label: entry.repo_id, source: "cached" });
        }
      }
    }
  } catch {
    // cached models are optional
  }
  return options;
}
