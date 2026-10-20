// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// An image or video load refused because its weights are estimated not to fit unified memory
// (Apple Silicon, integrated GPUs). The backend sends it as a code (download-plan
// `incompatible_code`, load-progress `error_code`) or as the X-Unsloth-Refusal header on a 400,
// and honours `allow_oversized_load` on the next request. Images and Video both surface it.

import { REFUSAL_HEADER, refusalHeadersOf } from "@/lib/multi-model-refusal";

/** Backend: diffusion_memory.UNIFIED_OVERSIZE_REFUSAL_CODE. */
export const UNIFIED_MEMORY_LOAD_REFUSAL = "unified-memory-load";
/** Persisted with usePersistedToggle: only "true" is stored, so the default stays off. */
export const ALLOW_OVERSIZED_LOAD_STORAGE_KEY = "unsloth_allow_oversized_load";
/** The backend quotes this label in the refusal text; keep the two identical. */
export const ALLOW_OVERSIZED_LOAD_LABEL = "Allow oversized model loads";
export const UNIFIED_MEMORY_LOAD_TITLE = "Not enough memory to load this model";
export const LOAD_ANYWAY_LABEL = "Load anyway";
export const LOAD_ANYWAY_CONFIRM_TITLE = "Load this model anyway?";
export const LOAD_ANYWAY_WARNING =
  "Studio estimated that this model's weights do not fit the memory this device can safely give it after reserving room for the operating system. The check is an estimate and can be wrong, but loading past it can fail, use up all memory, make the computer swap heavily or stop responding, or stop Studio outright. Save your work in other applications first.";

export function readAllowOversizedLoad(): boolean {
  try {
    return localStorage.getItem(ALLOW_OVERSIZED_LOAD_STORAGE_KEY) === "true";
  } catch {
    return false;
  }
}

/** `true` when the persisted setting or a one-shot Load anyway asks for it, else omitted. */
export function allowOversizedLoadField(
  persistedSetting: boolean,
  oneShot: boolean,
): true | undefined {
  return persistedSetting || oneShot ? true : undefined;
}

/** A request body with the persisted setting applied when the caller left the field unset. */
export function withAllowOversizedLoad<
  T extends { allow_oversized_load?: boolean },
>(body: T): T {
  if (body.allow_oversized_load !== undefined) return body;
  const field = allowOversizedLoadField(readAllowOversizedLoad(), false);
  return field ? { ...body, allow_oversized_load: field } : body;
}

/** From the response code, or from the refusal header a failed load POST carries. */
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
