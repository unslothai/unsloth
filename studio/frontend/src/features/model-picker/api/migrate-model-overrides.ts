// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// One-time backfill of per-model settings into the server override map.
//
// Settings used to live only in this browser, so on upgrade the server knows
// nothing about models the user already configured. Without this they keep
// showing as remembered in the UI while an API load quietly uses app defaults,
// which is the exact bug the server-side map exists to fix.

import {
  normalizeGgufVariantIdentity,
  normalizeModelIdentity,
} from "../model-config/model-identity";
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
 * A server key under the same identity this browser stores.
 *
 * `app_settings` has no schema version and holds whatever id was current when
 * the row was written, so an install that predates identity normalization has
 * keys like `Unsloth/Repo-GGUF:Q4_K_M` while this browser only ever produces
 * the folded form. The backend resolves the two to one model, so an exact
 * property lookup would report "not on the server" for a row that is, and the
 * backfill would overwrite it. Variants never contain a colon, so the last one
 * splits the key; a repo id folds and a POSIX path deliberately does not.
 */
function normalizedOverrideKey(key: string): string {
  const separator = key.lastIndexOf(":");
  if (separator < 0) {
    return modelOverrideKey(normalizeModelIdentity(key));
  }
  return modelOverrideKey(
    normalizeModelIdentity(key.slice(0, separator)),
    normalizeGgufVariantIdentity(key.slice(separator + 1)),
  );
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
    // A quant means it is a GGUF, which is the only thing API auto-switch
    // resolves. Backfilling a safetensors config would claim an API behaviour
    // that does not exist.
    (entry) => entry.ggufVariant != null && !isDefaultConfig(entry.config),
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

  const known = new Set(Object.keys(existing).map(normalizedOverrideKey));

  let failed = false;
  for (const entry of local) {
    // Folded on this side too: a v2 storage key already holds the normalized
    // identity, but the older `id::variant` keys this browser still reads back
    // hold whatever casing was typed.
    const key = normalizedOverrideKey(
      modelOverrideKey(entry.modelId, entry.ggufVariant),
    );
    if (known.has(key)) {
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
