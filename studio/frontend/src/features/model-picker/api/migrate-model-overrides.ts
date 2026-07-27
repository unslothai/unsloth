// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// One-time backfill of per-model settings into the server override map.
//
// Settings used to live only in this browser, so on upgrade the server knows
// nothing about models the user already configured. Without this they keep
// showing as remembered in the UI while an API load quietly uses app defaults,
// which is the exact bug the server-side map exists to fix.

import {
  isDefaultConfig,
  listPerModelConfigs,
} from "../model-config/per-model-config";
import {
  fetchModelOverrides,
  modelOverrideKey,
  putModelOverride,
} from "./model-overrides";

const DONE_FLAG = "unsloth_model_overrides_backfilled_v1";

function alreadyRan(): boolean {
  try {
    return window.localStorage.getItem(DONE_FLAG) === "1";
  } catch {
    // Storage denied: treat as done rather than re-running on every mount.
    return true;
  }
}

function markRan(): void {
  try {
    window.localStorage.setItem(DONE_FLAG, "1");
  } catch {
    // Nothing to do; the backfill is idempotent anyway.
  }
}

/**
 * Push local configs the server has never seen. Never deletes and never
 * overwrites: an entry already on the server is the newer authority, and losing
 * a setting here would be worse than leaving one unmigrated.
 */
export async function backfillModelOverrides(): Promise<void> {
  if (alreadyRan()) {
    return;
  }
  const local = listPerModelConfigs().filter(
    (entry) => !isDefaultConfig(entry.config),
  );
  if (local.length === 0) {
    markRan();
    return;
  }

  let existing: Awaited<ReturnType<typeof fetchModelOverrides>>;
  try {
    existing = await fetchModelOverrides();
  } catch {
    // Offline or not authenticated yet. Leave the flag unset so the next start
    // tries again rather than silently skipping the migration forever.
    return;
  }

  let failed = false;
  for (const entry of local) {
    const key = modelOverrideKey(entry.modelId, entry.ggufVariant);
    if (existing[key]) {
      continue;
    }
    try {
      await putModelOverride(entry.modelId, entry.ggufVariant, entry.config);
    } catch {
      failed = true;
    }
  }
  if (!failed) {
    markRan();
  }
}
