// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// One-time backfill of per-model settings into the server override map.
//
// Settings used to live only in this browser, so on upgrade the server knows
// nothing about models already configured: they still show as remembered while an
// API load uses app defaults, the exact bug the server-side map exists to fix.

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
 * `app_settings` has no schema version and holds whatever id was current when the
 * row was written, so an old install has keys like `Unsloth/Repo-GGUF:Q4_K_M` while
 * this browser only produces the folded form. The backend resolves both to one
 * model, so an exact lookup would report "not on the server" and let the backfill
 * overwrite it. Variants never contain a colon, so the last one splits the key; a
 * repo id folds and a POSIX path deliberately does not.
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
 * Push local configs the server has never seen. Never deletes and never overwrites:
 * an entry already there is the newer authority, and losing a setting would be
 * worse than leaving one unmigrated.
 */
export async function backfillModelOverrides(): Promise<void> {
  if (alreadyRan()) {
    return;
  }
  const local = listPerModelConfigs().filter(
    // A quant means GGUF, the only thing API auto-switch resolves, so backfilling a
    // safetensors config would claim behaviour that does not exist. A standalone
    // .gguf has no quant to select between and is stored with a null variant, so it
    // needs the extra test or its settings stay browser-only for good.
    (entry) =>
      (entry.ggufVariant != null ||
        entry.modelId.toLowerCase().endsWith(".gguf")) &&
      !isDefaultConfig(entry.config),
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
    // retries rather than skipping the migration forever.
    return;
  }

  const known = new Set(Object.keys(existing).map(normalizedOverrideKey));

  let failed = false;
  for (const entry of local) {
    // Folded here too: a v2 storage key holds the normalized identity, but the
    // older `id::variant` keys this browser still reads hold the typed casing.
    const key = normalizedOverrideKey(
      modelOverrideKey(entry.modelId, entry.ggufVariant),
    );
    if (known.has(key)) {
      continue;
    }
    // Re-read rather than trusting the snapshot from before the fetch: this write
    // is queued behind the interactive one and commits last, so a save or forget
    // during the round trip would be undone by it.
    const current = listPerModelConfigs().find(
      (candidate) =>
        normalizedOverrideKey(
          modelOverrideKey(candidate.modelId, candidate.ggufVariant),
        ) === key,
    );
    if (!current || isDefaultConfig(current.config)) {
      continue;
    }
    try {
      await putModelOverride(
        current.modelId,
        current.ggufVariant,
        current.config,
      );
    } catch {
      failed = true;
    }
  }
  if (!failed) {
    markRan();
  }
}
