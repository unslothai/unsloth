


// One-time backfill of per-model settings into the server override map. Settings used to live
// only in this browser, so on upgrade an already-configured model still shows as remembered
// while an API load uses app defaults.

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
 * A server key under the same identity this browser stores. `app_settings` has no schema
 * version, so an old install holds keys the backend resolves to this model while an exact lookup
 * calls them missing and overwrites them. The split folds repo ids and leaves POSIX paths alone.
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
 * The fields *config* would contribute that the stored entry does not hold. A malformed entry
 * (nothing constrains what an older install wrote) counts as holding nothing.
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
 * Push local settings the server does not hold. Never deletes and never overwrites: a value
 * already there is the newer authority. Field by field, not entry by entry, since a legacy entry
 * holds only llama_extra_args and max_seq_length, so treating the key as done would skip exactly
 * the settings this migration exists to carry.
 */
export async function backfillModelOverrides(): Promise<void> {
  if (alreadyRan()) {
    return;
  }
  const local = listPerModelConfigs().filter(
    // A quant means GGUF, the only thing auto-switch resolves. A standalone .gguf is
    // stored with a null variant, so it needs the extra test or stays browser-only. An
    // Ollama blob sits behind a link dir the resolver skips, and a bare file name is a
    // dropped file's label, which the resolver never keys.
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
    // Offline or not authenticated yet: leave the flag unset so the next start retries.
    return;
  }

  const known = new Map<string, ApiModelOverride>();
  for (const [storedKey, storedEntry] of Object.entries(existing)) {
    known.set(normalizedOverrideKey(storedKey), storedEntry);
  }

  let failed = false;
  for (const entry of local) {
    // Folded here too: the older `id::variant` keys hold the casing that was typed.
    const key = normalizedOverrideKey(
      modelOverrideKey(entry.modelId, entry.ggufVariant),
    );
    // Re-read rather than trusting the pre-fetch snapshot: this write commits last, so a
    // save or forget during the round trip would be undone by it.
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
      // Fills the gaps only. `known` predates this loop, so another tab's save is
      // invisible here; the server reads and writes together instead.
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
