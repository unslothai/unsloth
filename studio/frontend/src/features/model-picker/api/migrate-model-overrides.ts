// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// One-time backfill of per-model settings into the server override map.
//
// Settings used to live only in this browser, so on upgrade the server knows
// nothing about models already configured: they still show as remembered while an
// API load uses app defaults, the exact bug the server-side map exists to fix.

import {
  isNativeFileLabel,
  isOllamaLinkPath,
  normalizeGgufVariantIdentity,
  normalizeModelIdentity,
  splitQuantSuffix,
} from "../model-config/model-identity";
import {
  isDefaultConfig,
  listPerModelConfigs,
} from "../model-config/per-model-config";
import type { ApiModelOverride } from "./model-overrides";
import {
  fetchModelOverrides,
  modelOverrideKey,
  putModelOverride,
  toApiOverride,
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
 * `app_settings` has no schema version, so an old install holds keys like
 * `Unsloth/Repo-GGUF:Q4_K_M` that the backend resolves to the same model as this
 * browser's folded form; an exact lookup would call it missing and let the backfill
 * overwrite it. The quant-aware split folds repo ids and leaves POSIX paths alone.
 */
function normalizedOverrideKey(key: string): string {
  const split = splitQuantSuffix(key);
  if (!split) {
    return modelOverrideKey(normalizeModelIdentity(key));
  }
  return modelOverrideKey(
    normalizeModelIdentity(split[0]),
    normalizeGgufVariantIdentity(split[1]),
  );
}

/**
 * The fields *config* would contribute that the stored entry does not hold.
 *
 * A malformed entry (nothing constrains what an older install wrote into
 * app_settings) counts as holding nothing, so the migration still runs.
 */
function absentFields(
  stored: ApiModelOverride,
  config: Parameters<typeof toApiOverride>[0],
): string[] {
  const fields = Object.keys(toApiOverride(config));
  if (typeof stored !== "object" || stored === null) {
    return fields;
  }
  return fields.filter((field) => !(field in stored));
}

/**
 * Push local settings the server does not hold. Never deletes and never overwrites:
 * a value already there is the newer authority, and losing a setting would be worse
 * than leaving one unmigrated.
 *
 * Field by field, not entry by entry. The override map shipped before this browser
 * mirror did, storing only llama_extra_args and max_seq_length, so an upgraded
 * install can hold an entry for a model whose context, KV cache, speculative and GPU
 * settings live only here. Treating the key as done would skip exactly the settings
 * this migration exists to carry and then mark it complete.
 */
export async function backfillModelOverrides(): Promise<void> {
  if (alreadyRan()) {
    return;
  }
  const local = listPerModelConfigs().filter(
    // A quant means GGUF, the only thing API auto-switch resolves, so backfilling a
    // safetensors config would claim behaviour that does not exist. A standalone
    // .gguf has no quant to select between and is stored with a null variant, so it
    // needs the extra test or its settings stay browser-only for good. An Ollama
    // blob is GGUF but reached through a link dir the resolver skips, so it is not
    // auto-switchable either, and a bare file name is a dropped/picked file's label,
    // which the resolver never keys.
    (entry) =>
      (entry.ggufVariant != null ||
        entry.modelId.toLowerCase().endsWith(".gguf")) &&
      !isOllamaLinkPath(entry.modelId) &&
      !isNativeFileLabel(entry.modelId) &&
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

  const known = new Map<string, ApiModelOverride>();
  for (const [storedKey, storedEntry] of Object.entries(existing)) {
    known.set(normalizedOverrideKey(storedKey), storedEntry);
  }

  let failed = false;
  for (const entry of local) {
    // Folded here too: a v2 storage key holds the normalized identity, but the
    // older `id::variant` keys this browser still reads hold the typed casing.
    const key = normalizedOverrideKey(
      modelOverrideKey(entry.modelId, entry.ggufVariant),
    );
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
    const stored = known.get(key);
    // Nothing this browser could add, so skip the round trip entirely.
    if (stored && absentFields(stored, current.config).length === 0) {
      continue;
    }
    try {
      // Fills the gaps only. `known` is a snapshot from before this loop started, so
      // a save by another tab during the pass is invisible here; the server reads and
      // writes together rather than this re-fetching once per model.
      await putModelOverride(
        current.modelId,
        current.ggufVariant,
        current.config,
        { fillAbsentFields: true },
      );
    } catch {
      failed = true;
    }
  }
  if (!failed) {
    markRan();
  }
}
