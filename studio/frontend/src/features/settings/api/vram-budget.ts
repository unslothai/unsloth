// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

const VRAM_BUDGET_EVENT = "unsloth-vram-budget-change";
const VRAM_BUDGET_LOCK_EVENT = "unsloth-vram-budget-lock";

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

// Bumped on every stage, so a retry put back by a failed write can be told apart
// from a newer edit the user staged over it.
let stagedVramBudgetSequence = 0;
let retryVramBudgetSequence = -1;

/** Record a fraction a debounced save has not sent yet. */
export function stageVramBudgetSave(fraction: number | null) {
  stagedVramBudgetFraction = fraction;
  stagedVramBudgetSequence += 1;
}

/**
 * Drop a retry a failed write put back, unless something newer is staged over it.
 * For a caller that is about to start a load: the retry would otherwise be flushed
 * by the teardown and race the load request.
 */
export function dropVramBudgetRetry() {
  if (stagedVramBudgetSequence === retryVramBudgetSequence) {
    stagedVramBudgetFraction = null;
  }
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
  // The newest write, not the chain: the chain swallows rejections so one failed
  // save cannot strand those behind it, and a caller waiting on it would be told
  // the save succeeded. Only the newest can have re-staged a retry, and writes
  // settle in order, so it still covers every open write.
  return (
    flushVramBudgetSave() ??
    (vramBudgetWritesOpen > 0 ? vramBudgetNewestWrite : null)
  );
}

// A load waits on the budget it is about to launch against, so an edit made in
// that window is flushed by the teardown alongside the load request and either
// fraction could size the child. Settling in a loop only shrinks that window;
// closing the control closes it. Held here because the row unmounts and the load
// does not.
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
export async function loadVramBudgetSettings(
  options: { force?: boolean } = {},
): Promise<VramBudgetSettings | null> {
  // Read behind any open write: a row remounting right after a flushed drag can
  // otherwise GET the old fraction before the PUT commits and answer after it,
  // repainting the control with the value the server just replaced. The
  // subscription cannot untangle that, since only the order is wrong.
  const pendingWrites =
    vramBudgetWritesOpen > 0 ? vramBudgetWriteChain : Promise.resolve();
  // Waiting behind the writes open now says nothing about a save issued while the
  // GET is in the air: that PUT can publish first, and this answer would repaint
  // the slider, and the Reload state, with what the server held before it. The
  // post-load refresh is the way in, since the control is live while the load runs.
  const generationAtRead = vramBudgetWriteGeneration;
  if (options.force) {
    // reloadRequired describes the running child, so a read that started before a
    // load finished answers about the child being replaced. Sharing it would
    // republish that stale answer as if it described the new one.
    inFlightVramBudget = null;
  }
  if (!inFlightVramBudget) {
    const read: Promise<VramBudgetSettings> = pendingWrites
      .then(fetchVramBudgetSettings)
      .then((settings) => {
        // Answer only while this is still the current read and nothing newer has
        // been written. A displaced read describes the child being replaced, an
        // overtaken one the fraction a save replaced; either restores state the
        // newer answer just corrected. Refused, not merely unpublished: the caller
        // applies the return value by hand and would put the same answer back.
        if (
          inFlightVramBudget !== read ||
          generationAtRead !== vramBudgetWriteGeneration
        ) {
          throw new Error("superseded");
        }
        return publishVramBudget(settings);
      })
      .finally(() => {
        // Identity-checked: a forced read displaces this one, and clearing blindly
        // would drop the newer request's handle and leave it unshared.
        if (inFlightVramBudget === read) {
          inFlightVramBudget = null;
        }
      });
    inFlightVramBudget = read;
  }
  try {
    return await inFlightVramBudget;
  } catch {
    // Null is already the "no usable answer" contract here: an older backend with
    // no such route reads the same way, and every caller keeps what it has.
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
// The same write, unswallowed, for callers that need to hear about a failure.
let vramBudgetNewestWrite: Promise<unknown> = Promise.resolve();
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
  vramBudgetNewestWrite = write;
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
        stageVramBudgetSave(fraction);
        retryVramBudgetSequence = stagedVramBudgetSequence;
      }
      throw error;
    },
  );
}
