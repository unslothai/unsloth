// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// One-time backfill of browser-only per-model settings into the server override map.

import {
  cachedRepoConfigId,
  isNativeFileLabel,
  isOllamaLinkPath,
  normalizeGgufVariantIdentity,
  normalizeModelIdentity,
  splitQuantSuffix,
} from "../model-config/model-identity";
import {
  adoptCachedRepoConfig,
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

// Bump when the filter admits more, so a completed pass reruns.
const DONE_FLAG = "unsloth_model_overrides_backfilled_v3";

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

/** Folds keys like the backend: repo ids fold, POSIX paths keep their case. */
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

/** A malformed entry from an older install counts as holding nothing. */
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

/** Never deletes or overwrites: server values are the newer authority. Field by field. */
export async function backfillModelOverrides(): Promise<void> {
  if (alreadyRan()) {
    return;
  }
  // Uploaded under the path, an older cached-repo record would outrank the repo's.
  for (const entry of listPerModelConfigs()) {
    adoptCachedRepoConfig(entry.modelId, entry.ggufVariant);
  }
  const local = listPerModelConfigs().filter(
    (entry) =>
      !isOllamaLinkPath(entry.modelId) &&
      !isNativeFileLabel(entry.modelId) &&
      cachedRepoConfigId(entry.modelId, entry.ggufVariant) === null &&
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
    // Leave the flag unset so the next start retries.
    return;
  }

  const known = new Map<string, ApiModelOverride>();
  for (const [storedKey, storedEntry] of Object.entries(existing)) {
    known.set(normalizedOverrideKey(storedKey), storedEntry);
  }

  let failed = false;
  for (const entry of local) {
    // Older `id::variant` keys keep the typed casing.
    const key = normalizedOverrideKey(
      modelOverrideKey(entry.modelId, entry.ggufVariant),
    );
    // Re-read rather than trusting the pre-fetch snapshot: this write commits last.
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
    if (stored && absentFields(stored, current.config).length === 0) {
      continue;
    }
    try {
      // `known` predates this loop, so another tab's save is invisible here.
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
