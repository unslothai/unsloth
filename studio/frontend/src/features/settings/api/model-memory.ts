// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

const MODEL_MEMORY_EVENT = "unsloth-model-memory-change";

export type ModelMemorySettings = {
  keepResident: boolean;
  noRamReserve: boolean;
  defaultKeepResident: boolean;
  defaultNoRamReserve: boolean;
  /** Whether --mlock applies; false when noRamReserve vetoes it. */
  mlockActive: boolean;
  /** A model is loaded whose --mlock state differs from the saved one. */
  reloadRequired: boolean;
  /** Soft RLIMIT_MEMLOCK when finite; null means unlimited or N/A. */
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
  reload_required: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  memlock_limit_bytes: number | null;
};

let inFlightModelMemory: Promise<ModelMemorySettings> | null = null;

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
    reloadRequired: settings.reload_required,
    memlockLimitBytes: settings.memlock_limit_bytes,
  };
}

// No read-through cache on purpose: the response carries runtime state
// (reloadRequired, memlockLimitBytes) that goes stale as soon as a model is
// loaded or swapped. This only fans the latest value out to subscribers.
function publishModelMemory(settings: ModelMemorySettings) {
  window.dispatchEvent(
    new CustomEvent(MODEL_MEMORY_EVENT, { detail: settings }),
  );
  return settings;
}

async function fetchModelMemorySettings(): Promise<ModelMemorySettings> {
  const res = await authFetch("/api/settings/model-memory");
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to load model memory settings"),
    );
  }
  return fromApi(await res.json());
}

/**
 * Always refetches: `reloadRequired` and `memlockLimitBytes` describe the
 * currently loaded process, so a cached copy goes stale as soon as a model is
 * loaded or swapped. Concurrent calls still share one request.
 */
export async function loadModelMemorySettings() {
  inFlightModelMemory ??= fetchModelMemorySettings()
    .then(publishModelMemory)
    .finally(() => {
      inFlightModelMemory = null;
    });
  return inFlightModelMemory;
}

/** Partial update: omitted fields keep their stored value. */
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
  return publishModelMemory(fromApi(await res.json()));
}
