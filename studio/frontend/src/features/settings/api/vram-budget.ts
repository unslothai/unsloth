// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

const VRAM_BUDGET_EVENT = "unsloth-vram-budget-change";

export type VramBudgetSettings = {
  /** Fraction of each GPU a load may claim, e.g. 0.97. */
  fraction: number;
  /** False when inherited from UNSLOTH_VRAM_FRACTION or the built-in default. */
  isStored: boolean;
  defaultFraction: number;
  minFraction: number;
  maxFraction: number;
  /** A model is loaded that was sized against a different budget. */
  reloadRequired: boolean;
};

type ApiVramBudgetSettings = {
  fraction: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  is_stored: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  default_fraction: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  min_fraction: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  max_fraction: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  reload_required: boolean;
};

let inFlightVramBudget: Promise<VramBudgetSettings> | null = null;

// Held here rather than in the row, because the row unmounts on Run and on the
// Advanced toggle, and the load has to be able to flush the edit that unmount
// would otherwise carry away with it.
let stagedVramBudgetFraction: number | null = null;

/** Record a fraction a debounced save has not sent yet. */
export function stageVramBudgetSave(fraction: number | null) {
  stagedVramBudgetFraction = fraction;
}

/**
 * Send a staged fraction now, ahead of whatever the caller does next.
 *
 * Returns null when nothing is staged, so a caller that only has to wait in the
 * rare case can keep its synchronous path in the common one.
 */
export function flushVramBudgetSave(): Promise<VramBudgetSettings> | null {
  const fraction = stagedVramBudgetFraction;
  stagedVramBudgetFraction = null;
  return fraction === null ? null : updateVramBudgetSettings(fraction);
}

export function subscribeVramBudgetSettings(
  listener: (settings: VramBudgetSettings) => void,
) {
  const handleChange = (event: Event) => {
    listener((event as CustomEvent<VramBudgetSettings>).detail);
  };
  window.addEventListener(VRAM_BUDGET_EVENT, handleChange);
  return () => window.removeEventListener(VRAM_BUDGET_EVENT, handleChange);
}

function fromApi(settings: ApiVramBudgetSettings): VramBudgetSettings {
  return {
    fraction: settings.fraction,
    isStored: settings.is_stored,
    defaultFraction: settings.default_fraction,
    minFraction: settings.min_fraction,
    maxFraction: settings.max_fraction,
    reloadRequired: settings.reload_required,
  };
}

// No read-through cache: reloadRequired describes the running process and goes
// stale as soon as a model is loaded or swapped. This only fans the latest value
// out to subscribers.
function publishVramBudget(settings: VramBudgetSettings) {
  window.dispatchEvent(
    new CustomEvent(VRAM_BUDGET_EVENT, { detail: settings }),
  );
  return settings;
}

async function fetchVramBudgetSettings(): Promise<VramBudgetSettings> {
  const res = await authFetch("/api/settings/vram-budget");
  if (!res.ok) {
    throw new Error(await readFastApiError(res, "Failed to load VRAM budget"));
  }
  return fromApi(await res.json());
}

/**
 * Always refetches, since `reloadRequired` describes the currently loaded
 * process. Concurrent calls still share one request.
 *
 * Returns null rather than throwing when the endpoint is absent, so a newer UI
 * talking to an older backend hides the control instead of erroring.
 */
export async function loadVramBudgetSettings(): Promise<VramBudgetSettings | null> {
  inFlightVramBudget ??= fetchVramBudgetSettings()
    .then(publishVramBudget)
    .finally(() => {
      inFlightVramBudget = null;
    });
  try {
    return await inFlightVramBudget;
  } catch {
    return null;
  }
}

/** `null` clears the stored budget so the env var or the default applies again. */
export async function updateVramBudgetSettings(
  fraction: number | null,
): Promise<VramBudgetSettings> {
  const res = await authFetch("/api/settings/vram-budget", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ fraction }),
  });
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to update VRAM budget"),
    );
  }
  return publishVramBudget(fromApi(await res.json()));
}
