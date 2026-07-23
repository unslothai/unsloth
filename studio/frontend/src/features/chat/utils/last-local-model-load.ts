// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type LastLocalModelKind = "gguf" | "model";

/**
 * Where the remembered model came from:
 *   - "hf_cache": a managed Hugging Face cache repo (active or inactive cache),
 *     resolved through the cached-gguf / cached-models inventory.
 *   - "models_dir" / "lmstudio" / "custom": a backend-indexed local inventory
 *     row, resolved through the /api/hub/local inventory.
 *   - "local": an indexed local row whose exact scan source was not known at
 *     record time (interactive picker loads); resolved like the other local
 *     sources.
 *
 * Native file-picker selections are never recorded here: their access depends
 * on a signed, expiring native path lease that must not be bypassed by a raw
 * remembered path (the caller skips recording when a lease token is present).
 */
export type LastLocalModelSource =
  | "hf_cache"
  | "models_dir"
  | "lmstudio"
  | "custom"
  | "local";

export type LastLocalModelLoad = {
  /** Display / repository identity (HF repo id, or a local path for indexed rows). */
  id: string;
  kind: LastLocalModelKind;
  /** Managed-cache GGUF quant. Null is valid when the load target itself
   *  identifies the actual GGUF (a local file or directory). */
  ggufVariant: string | null;
  /** Backend-provided load target when it differs from `id` (e.g. an
   *  inactive-cache row's load_id or an indexed local path). */
  loadId: string | null;
  /** Stable backend inventory row identity, when available. */
  inventoryId: string | null;
  source: LastLocalModelSource;
  loadedAt: number;
};

// Kept at v1: new fields are parsed backward-compatibly, so existing records
// (which predate loadId/inventoryId/source) keep resolving as managed-cache
// entries without a migration pass.
const STORAGE_KEY = "unsloth.last-local-model-load.v1";

function storage(): Storage | null {
  try {
    return typeof localStorage === "undefined" ? null : localStorage;
  } catch {
    return null;
  }
}

function isLastLocalModelKind(value: unknown): value is LastLocalModelKind {
  return value === "gguf" || value === "model";
}

const LOCAL_MODEL_SOURCES: readonly LastLocalModelSource[] = [
  "hf_cache",
  "models_dir",
  "lmstudio",
  "custom",
  "local",
];

function isLastLocalModelSource(
  value: unknown,
): value is LastLocalModelSource {
  return LOCAL_MODEL_SOURCES.includes(value as LastLocalModelSource);
}

export function isManagedCacheSource(source: LastLocalModelSource): boolean {
  return source === "hf_cache";
}

function normalizeOptionalString(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value : null;
}

export function readLastLocalModelLoad(): LastLocalModelLoad | null {
  try {
    const raw = storage()?.getItem(STORAGE_KEY);
    if (!raw) {
      return null;
    }
    const parsed = JSON.parse(raw) as Partial<LastLocalModelLoad>;
    if (
      typeof parsed.id !== "string" ||
      !parsed.id.trim() ||
      !isLastLocalModelKind(parsed.kind) ||
      typeof parsed.loadedAt !== "number"
    ) {
      return null;
    }
    // Legacy v1 records carry no source; they were only ever written for
    // managed-cache repos.
    const source = isLastLocalModelSource(parsed.source)
      ? parsed.source
      : "hf_cache";
    const ggufVariant = normalizeOptionalString(parsed.ggufVariant);
    // A managed-cache GGUF loads by repo + quant, so the quant is required.
    // An indexed local GGUF's load target identifies the file itself, so a
    // null variant stays valid.
    if (parsed.kind === "gguf" && source === "hf_cache" && !ggufVariant) {
      return null;
    }
    return {
      id: parsed.id,
      kind: parsed.kind,
      ggufVariant,
      loadId: normalizeOptionalString(parsed.loadId),
      inventoryId: normalizeOptionalString(parsed.inventoryId),
      source,
      loadedAt: parsed.loadedAt,
    };
  } catch {
    return null;
  }
}

export function recordLastLocalModelLoad(input: {
  id: string;
  kind: LastLocalModelKind;
  ggufVariant?: string | null;
  loadId?: string | null;
  inventoryId?: string | null;
  source?: LastLocalModelSource;
}): void {
  const id = input.id.trim();
  if (!id) {
    return;
  }
  const source = input.source ?? "hf_cache";
  const ggufVariant = input.ggufVariant?.trim() || null;
  if (input.kind === "gguf" && source === "hf_cache" && !ggufVariant) {
    return;
  }
  try {
    // Only inventory identity is stored: never tokens, native path leases, or
    // security approvals.
    storage()?.setItem(
      STORAGE_KEY,
      JSON.stringify({
        id,
        kind: input.kind,
        ggufVariant: input.kind === "gguf" ? ggufVariant : null,
        loadId: input.loadId?.trim() || null,
        inventoryId: input.inventoryId?.trim() || null,
        source,
        loadedAt: Date.now(),
      } satisfies LastLocalModelLoad),
    );
  } catch {
    // Ignore disabled storage / quota errors; auto-load falls back to size order.
  }
}
