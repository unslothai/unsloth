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

// Held here, not in the row: the row unmounts on Run and on the Advanced toggle,
// and the load must still be able to flush that edit.
let stagedVramBudgetFraction: number | null = null;

/** Record a fraction a debounced save has not sent yet. */
export function stageVramBudgetSave(fraction: number | null) {
  stagedVramBudgetFraction = fraction;
}

/**
 * Send a staged fraction now. Returns null when nothing is staged, so callers keep
 * their synchronous path in the common case.
 */
export function flushVramBudgetSave(): Promise<VramBudgetSettings> | null {
  const fraction = stagedVramBudgetFraction;
  stagedVramBudgetFraction = null;
  return fraction === null ? null : updateVramBudgetSettings(fraction);
}

/**
 * Everything the next load must wait for: a staged fraction, or a debounced save
 * still in flight. A user who pauses past the 400 ms debounce and clicks Load has
 * nothing staged but an open PUT, and the load would otherwise use the fraction
 * that request replaces. The chain swallows rejections the debounced save reported.
 */
export function settleVramBudgetSave(): Promise<unknown> | null {
  return (
    flushVramBudgetSave() ??
    (vramBudgetWritesOpen > 0 ? vramBudgetWriteChain : null)
  );
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
// stale on any load or swap. This only fans the latest value out to subscribers.
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
 * Always refetches, since `reloadRequired` describes the loaded process;
 * concurrent calls share one request. Returns null rather than throwing when the
 * endpoint is absent, so a newer UI on an older backend hides the control.
 */
export async function loadVramBudgetSettings(): Promise<VramBudgetSettings | null> {
  // Read behind any open write: a row remounting right after a flushed drag can
  // otherwise GET the old fraction before the PUT commits and answer after it,
  // repainting the control with the value the server just replaced. The
  // subscription cannot untangle that, since only the order is wrong.
  const pendingWrites =
    vramBudgetWritesOpen > 0 ? vramBudgetWriteChain : Promise.resolve();
  inFlightVramBudget ??= pendingWrites
    .then(fetchVramBudgetSettings)
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

async function putVramBudget(
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
  return fromApi(await res.json());
}

// One drag can outrun its own saves: on a slow link a second PUT starts while the
// first is open, and either order of apply or response could let the older edit win
// the database row and the published value. Chaining serialises the writes; the
// generation lets only the newest publish, so a late response cannot repaint.
let vramBudgetWriteChain: Promise<unknown> = Promise.resolve();
let vramBudgetWriteGeneration = 0;
// Issued but unsettled writes, so a load can tell whether to wait.
let vramBudgetWritesOpen = 0;

/** `null` clears the stored budget so the env var or the default applies again. */
export function updateVramBudgetSettings(
  fraction: number | null,
): Promise<VramBudgetSettings> {
  vramBudgetWriteGeneration += 1;
  const generation = vramBudgetWriteGeneration;
  vramBudgetWritesOpen += 1;
  const write = vramBudgetWriteChain
    .then(
      () => putVramBudget(fraction),
      () => putVramBudget(fraction),
    )
    .finally(() => {
      vramBudgetWritesOpen -= 1;
    });
  // The chain must survive a rejection, or one failed save strands all later ones.
  vramBudgetWriteChain = write.catch(() => undefined);
  return write.then(
    (settings) =>
      generation === vramBudgetWriteGeneration
        ? publishVramBudget(settings)
        : settings,
    (error: unknown) => {
      // Put a failed edit back for the next flush or Run, but only while it is
      // still the newest intent: a later edit may already be staged, or already
      // sent past this one, and resending would undo it.
      if (
        generation === vramBudgetWriteGeneration &&
        stagedVramBudgetFraction === null
      ) {
        stagedVramBudgetFraction = fraction;
      }
      throw error;
    },
  );
}
