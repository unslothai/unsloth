// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { pillSetConfig, pillStatus } from "@/lib/pill-native";
import type { PillModelOption, PillSettings } from "./types";

// Startup races the desktop auth handshake: retry 401s briefly.
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

// Settings tab and startup sync take turns so read+apply is atomic; a marker check raced the IPC awaits.
let nativeApplyChain: Promise<unknown> = Promise.resolve();

export function withNativeApplyLock<T>(run: () => Promise<T>): Promise<T> {
  const next = nativeApplyChain.then(run, run);
  nativeApplyChain = next.then(
    () => undefined,
    () => undefined,
  );
  return next;
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

export async function syncNativePillConfig(settings: PillSettings): Promise<void> {
  const status = await pillStatus();
  if (!status.supported) return;
  if (settings.enabled !== status.enabled) {
    await pillSetConfig({ enabled: settings.enabled });
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
  load_id?: string | null;
  task?: string | null;
};

// Same rule as isChattableCachedRepo: no routing step here, so diffusion rows are excluded.
const IMAGE_OR_VIDEO_TASKS: ReadonlySet<string> = new Set([
  "text-to-image",
  "text-to-video",
  "image-diffusion-unsupported",
]);

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
  }
  try {
    const response = await authFetchBootTolerant("/api/models/cached-gguf");
    if (response.ok) {
      const body = (await response.json()) as { cached: CachedGgufEntry[] };
      for (const entry of body.cached) {
        if (entry.repo_id && !IMAGE_OR_VIDEO_TASKS.has(entry.task ?? "")) {
          options.push({
            // load_id: repo outside the active hub cache; load that snapshot.
            id: entry.load_id || entry.repo_id,
            label: entry.repo_id,
            source: "cached",
          });
        }
      }
    }
  } catch {
  }
  return options;
}
