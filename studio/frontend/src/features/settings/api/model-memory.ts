// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

import { SettingsRouteAbsentError } from "./settings-route-absent";
import { invalidateOpenAIAutoSwitchSettings } from "./openai-auto-switch";

const MODEL_MEMORY_EVENT = "unsloth-model-memory-change";

export type ModelMemorySettings = {
  keepResident: boolean;
  noRamReserve: boolean;
  defaultKeepResident: boolean;
  defaultNoRamReserve: boolean;
  mlockActive: boolean;
  /** False when the model is fully on a discrete GPU, so nothing in host RAM to lock. */
  mlockApplicable: boolean;
  reloadRequired: boolean;
  /** Soft RLIMIT_MEMLOCK; null means unlimited or N/A. */
  memlockLimitBytes: number | null;
};

type ApiModelMemorySettings = {
  // biome-ignore lint/style/useNamingConvention: API schema
  keep_resident: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  no_ram_reserve: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  default_keep_resident: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  default_no_ram_reserve: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  mlock_active: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  mlock_applicable?: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  reload_required: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  memlock_limit_bytes: number | null;
};

let inFlightModelMemory: Promise<ModelMemorySettings> | null = null;
// Bumped by every forced read so a displaced read stops publishing to other subscribers.
let modelMemoryGeneration = 0;

export function subscribeModelMemorySettings(
  listener: (settings: ModelMemorySettings) => void,
) {
  const handleChange = (event: Event) => {
    listener((event as CustomEvent<ModelMemorySettings>).detail);
  };
  window.addEventListener(MODEL_MEMORY_EVENT, handleChange);
  return () => window.removeEventListener(MODEL_MEMORY_EVENT, handleChange);
}

function fromApi(settings: ApiModelMemorySettings): ModelMemorySettings {
  return {
    keepResident: settings.keep_resident,
    noRamReserve: settings.no_ram_reserve,
    defaultKeepResident: settings.default_keep_resident,
    defaultNoRamReserve: settings.default_no_ram_reserve,
    mlockActive: settings.mlock_active,
    // Older backends omit this; do not claim nothing is lockable.
    mlockApplicable: settings.mlock_applicable ?? true,
    reloadRequired: settings.reload_required,
    memlockLimitBytes: settings.memlock_limit_bytes,
  };
}

// No cache: reloadRequired and memlockLimitBytes go stale on any load or swap.
function publishModelMemory(settings: ModelMemorySettings) {
  window.dispatchEvent(
    new CustomEvent(MODEL_MEMORY_EVENT, { detail: settings }),
  );
  return settings;
}

async function fetchModelMemorySettings(): Promise<ModelMemorySettings> {
  const res = await authFetch("/api/settings/model-memory");
  if (res.status === 404) {
    // A caller deciding whether to skip a load treats "no such setting" and "could not ask" oppositely.
    throw new SettingsRouteAbsentError("/api/settings/model-memory");
  }
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to load model memory settings"),
    );
  }
  return fromApi(await res.json());
}

/** Always refetches; concurrent calls share one request unless `force`, which a caller deciding
 * whether to reload needs so it never gets a read that predates a save or model change. */
export async function loadModelMemorySettings(
  options: { force?: boolean } = {},
) {
  if (options.force) {
    inFlightModelMemory = null;
    modelMemoryGeneration += 1;
  }
  const generation = modelMemoryGeneration;
  inFlightModelMemory ??= fetchModelMemorySettings()
    .then((settings) =>
      // A displaced read is already known stale; publishing it would repaint subscribers out of order.
      generation === modelMemoryGeneration
        ? publishModelMemory(settings)
        : settings,
    )
    .finally(() => {
      // Only the current request owns the slot, or the next caller opens a third request.
      if (generation === modelMemoryGeneration) {
        inFlightModelMemory = null;
      }
    });
  return inFlightModelMemory;
}

export async function updateModelMemorySettings(
  patch: Partial<Pick<ModelMemorySettings, "keepResident" | "noRamReserve">>,
): Promise<ModelMemorySettings> {
  const body: Record<string, boolean> = {};
  if (patch.keepResident !== undefined) {
    body.keep_resident = patch.keepResident;
  }
  if (patch.noRamReserve !== undefined) {
    body.no_ram_reserve = patch.noRamReserve;
  }
  const res = await authFetch("/api/settings/model-memory", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to update model memory settings"),
    );
  }
  // Residency vetoes the idle-unload TTL, so the auto-switch cache's idleUnloadActive is stale.
  invalidateOpenAIAutoSwitchSettings();
  return publishModelMemory(fromApi(await res.json()));
}
