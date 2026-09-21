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
// Bumped by every forced read, so a displaced one can tell it is no longer the current
// answer. It still resolves for its own caller; it just stops speaking for everyone else.
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
  if (res.status === 404) {
    // Told apart from a failed read: a caller deciding whether it may skip a load has to
    // treat "this backend has no such setting" and "could not ask" oppositely.
    throw new SettingsRouteAbsentError("/api/settings/model-memory");
  }
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
 *
 * `force` drops that sharing, as the VRAM budget's reader does: a read that started
 * before a save or a model transition answers about the state being replaced, and a
 * caller deciding whether to reload for a policy change must not be handed it.
 */
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
      // A displaced read describes the state its replacement was issued because of, so
      // publishing it would repaint every subscriber with the answer that was already
      // known to be stale, and in whichever order the two land.
      generation === modelMemoryGeneration
        ? publishModelMemory(settings)
        : settings,
    )
    .finally(() => {
      // Only the current request owns the slot. Clearing it from a displaced one drops
      // the newer promise's sharing handle while it is still in flight, so the next
      // caller opens a third request rather than joining the second.
      if (generation === modelMemoryGeneration) {
        inFlightModelMemory = null;
      }
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
  // Residency vetoes the idle-unload TTL, so the auto-switch endpoint's
  // idleUnloadActive changed too and its own cache is now stale.
  invalidateOpenAIAutoSwitchSettings();
  return publishModelMemory(fromApi(await res.json()));
}
