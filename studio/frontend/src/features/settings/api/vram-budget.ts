// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

import { SettingsRouteAbsentError } from "./settings-route-absent";

const VRAM_BUDGET_EVENT = "unsloth-vram-budget-change";
const VRAM_BUDGET_LOCK_EVENT = "unsloth-vram-budget-lock";

export type VramBudgetSettings = {
  fraction: number;
  /** False when inherited from UNSLOTH_VRAM_FRACTION or the default. */
  isStored: boolean;
  defaultFraction: number;
  minFraction: number;
  maxFraction: number;
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

// Held here because the row unmounts on Run and on the Advanced toggle.
let stagedVramBudgetFraction: number | null = null;

// Tells a re-staged failed retry apart from a newer edit.
let stagedVramBudgetSequence = 0;
let retryVramBudgetSequence = -1;

export function stageVramBudgetSave(fraction: number | null) {
  stagedVramBudgetFraction = fraction;
  stagedVramBudgetSequence += 1;
}

/** Drop a re-staged retry before a load, or teardown would flush it and race the load. */
export function dropVramBudgetRetry() {
  if (stagedVramBudgetSequence === retryVramBudgetSequence) {
    stagedVramBudgetFraction = null;
  }
}

/** Null when nothing is staged, so callers keep their synchronous path. */
export function flushVramBudgetSave(): Promise<VramBudgetSettings> | null {
  const fraction = stagedVramBudgetFraction;
  stagedVramBudgetFraction = null;
  return fraction === null ? null : updateVramBudgetSettings(fraction);
}

/** A staged fraction or a debounced PUT still in flight; the chain swallows rejections. */
export function settleVramBudgetSave(): Promise<unknown> | null {
  // The newest write, not the swallowing chain; writes settle in order, so it covers all of them.
  return (
    flushVramBudgetSave() ??
    (vramBudgetWritesOpen > 0 ? vramBudgetNewestWrite : null)
  );
}

// Locks the control while a load waits on the budget, so an edit cannot race the load request.
let vramBudgetLocked = false;

export function setVramBudgetLocked(locked: boolean) {
  vramBudgetLocked = locked;
  window.dispatchEvent(
    new CustomEvent(VRAM_BUDGET_LOCK_EVENT, { detail: locked }),
  );
}

export function isVramBudgetLocked() {
  return vramBudgetLocked;
}

export function subscribeVramBudgetLock(listener: (locked: boolean) => void) {
  const handleChange = (event: Event) => {
    listener((event as CustomEvent<boolean>).detail);
  };
  window.addEventListener(VRAM_BUDGET_LOCK_EVENT, handleChange);
  return () => window.removeEventListener(VRAM_BUDGET_LOCK_EVENT, handleChange);
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

// No cache: reloadRequired goes stale on any load or swap.
function publishVramBudget(settings: VramBudgetSettings) {
  window.dispatchEvent(
    new CustomEvent(VRAM_BUDGET_EVENT, { detail: settings }),
  );
  return settings;
}

async function fetchVramBudgetSettings(): Promise<VramBudgetSettings> {
  const res = await authFetch("/api/settings/vram-budget");
  if (res.status === 404) {
    throw new SettingsRouteAbsentError("/api/settings/vram-budget");
  }
  if (!res.ok) {
    throw new Error(await readFastApiError(res, "Failed to load VRAM budget"));
  }
  return fromApi(await res.json());
}

/** Always refetches with shared concurrent calls; null when the endpoint is absent. */
export async function loadVramBudgetSettings(
  options: { force?: boolean; rethrow?: boolean } = {},
): Promise<VramBudgetSettings | null> {
  // Read behind open writes, or a remount could GET the old fraction before the PUT commits.
  const pendingWrites =
    vramBudgetWritesOpen > 0 ? vramBudgetWriteChain : Promise.resolve();
  // A save issued while this GET is in the air can publish first; this answer must not repaint it.
  const generationAtRead = vramBudgetWriteGeneration;
  if (options.force) {
    // A read started before a load finished describes the replaced child, so do not share it.
    inFlightVramBudget = null;
  }
  if (!inFlightVramBudget) {
    const read: Promise<VramBudgetSettings> = pendingWrites
      .then(fetchVramBudgetSettings)
      .then((settings) => {
        // Refuse a displaced or overtaken answer; the caller applies the return value by hand.
        if (
          inFlightVramBudget !== read ||
          generationAtRead !== vramBudgetWriteGeneration
        ) {
          throw new Error("superseded");
        }
        return publishVramBudget(settings);
      })
      .finally(() => {
        // Identity-checked so a forced read's handle is not dropped.
        if (inFlightVramBudget === read) {
          inFlightVramBudget = null;
        }
      });
    inFlightVramBudget = read;
  }
  try {
    return await inFlightVramBudget;
  } catch (error) {
    // Null means "no usable answer"; `rethrow` is for the caller deciding whether to skip a save.
    if (options.rethrow) {
      throw error;
    }
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

// Chain writes and let only the newest generation publish, so an older edit cannot win.
let vramBudgetWriteChain: Promise<unknown> = Promise.resolve();
let vramBudgetNewestWrite: Promise<unknown> = Promise.resolve();
let vramBudgetWriteGeneration = 0;
let vramBudgetWritesOpen = 0;

/** `null` clears the stored budget so the env var or the default applies. */
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
  vramBudgetNewestWrite = write;
  return write.then(
    (settings) =>
      generation === vramBudgetWriteGeneration
        ? publishVramBudget(settings)
        : settings,
    (error: unknown) => {
      // Re-stage a failed edit only while it is still the newest intent.
      if (
        generation === vramBudgetWriteGeneration &&
        stagedVramBudgetFraction === null
      ) {
        stageVramBudgetSave(fraction);
        retryVramBudgetSequence = stagedVramBudgetSequence;
      }
      throw error;
    },
  );
}
