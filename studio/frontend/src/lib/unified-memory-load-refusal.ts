// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Unified-memory load refusal: sent as a code (`incompatible_code` / `error_code`) or the
// X-Unsloth-Refusal header on a 400; lifted by `allow_oversized_load`.

import { REFUSAL_HEADER, refusalHeadersOf } from "@/lib/multi-model-refusal";

/** Backend: diffusion_memory.UNIFIED_OVERSIZE_REFUSAL_CODE. */
export const UNIFIED_MEMORY_LOAD_REFUSAL = "unified-memory-load";
export const ALLOW_OVERSIZED_LOAD_STORAGE_KEY = "unsloth_allow_oversized_load";
/** Quoted by the backend refusal text; keep identical. */
export const ALLOW_OVERSIZED_LOAD_LABEL = "Allow oversized model loads";
export const UNIFIED_MEMORY_LOAD_TITLE = "Not enough memory to load this model";
export const LOAD_ANYWAY_LABEL = "Load anyway";
export const LOAD_ANYWAY_CONFIRM_TITLE = "Load this model anyway?";
export const LOAD_ANYWAY_WARNING =
  "Unsloth estimated that this model's weights do not fit the memory this device can safely give it after reserving room for the operating system. The check is an estimate and can be wrong, but loading past it can fail, use up all memory, make the computer swap heavily or stop responding, or stop Unsloth outright. Save your work in other applications first.";

export function readAllowOversizedLoad(): boolean {
  try {
    return localStorage.getItem(ALLOW_OVERSIZED_LOAD_STORAGE_KEY) === "true";
  } catch {
    return false;
  }
}

export function allowOversizedLoadField(
  persistedSetting: boolean,
  oneShot: boolean,
): true | undefined {
  return persistedSetting || oneShot ? true : undefined;
}

export function withAllowOversizedLoad<
  T extends { allow_oversized_load?: boolean },
>(body: T): T {
  if (body.allow_oversized_load !== undefined) return body;
  const field = allowOversizedLoadField(readAllowOversizedLoad(), false);
  return field ? { ...body, allow_oversized_load: field } : body;
}

export function isUnifiedMemoryLoadRefusal(input: {
  code?: string | null;
  failure?: unknown;
}): boolean {
  if ((input.code ?? "").trim().toLowerCase() === UNIFIED_MEMORY_LOAD_REFUSAL)
    return true;
  const headers = refusalHeadersOf(input.failure);
  if (!headers) return false;
  let value: string | null = null;
  if (typeof (headers as Headers).get === "function") {
    value = (headers as Headers).get(REFUSAL_HEADER);
  } else {
    for (const [name, v] of Object.entries(headers)) {
      if (name.toLowerCase() === REFUSAL_HEADER && typeof v === "string")
        value = v;
    }
  }
  return (value ?? "").trim().toLowerCase() === UNIFIED_MEMORY_LOAD_REFUSAL;
}

/** Not when the load already sent the override: then the refusal means something else changed. */
export function shouldOfferLoadAnyway(input: {
  code?: string | null;
  failure?: unknown;
  overrideSent: boolean;
}): boolean {
  return !input.overrideSent && isUnifiedMemoryLoadRefusal(input);
}

/** Runs only while its pick is still current: the toast outlives the refusal and must not supersede a newer pick. */
export function retryWhileCurrent(
  isCurrent: () => boolean,
  retry: () => void,
): () => void {
  return () => {
    if (isCurrent()) {
      retry();
    }
  };
}
