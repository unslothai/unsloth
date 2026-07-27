// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  normalizeGgufVariantIdentity,
  normalizeModelIdentity,
} from "@/features/hub";

export {
  normalizeGgufVariantIdentity,
  normalizeModelIdentity,
} from "@/features/hub";

const MODEL_STORAGE_KEY_PREFIX = "v2:";

// Mirror the backend's _classify_diffusion_gguf (routes/inference.py): with no header,
// strip non-alphanumerics and look for the DiffusionGemma name so staged Load agrees.
export function looksLikeDiffusionGemma(
  ...parts: (string | null | undefined)[]
): boolean {
  return parts.some((part) =>
    (part ?? "")
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "")
      .includes("diffusiongemma"),
  );
}

// Tri-state, mirroring the backend's _preflight_is_diffusion (core/inference/llama_cpp.py):
// null means no GGUF header has been read yet, so nothing is classified.
export type DiffusionClassification = boolean | null;

// Resolve one diffusion answer from the backend's header classification plus the name.
// The header wins whenever it exists: it is what /validate and /load themselves decide
// on, so an ordinary GGUF that merely carries DiffusionGemma in its id keeps the controls
// those endpoints accept for it. The name check is only the stand-in for a staged pick,
// whose header the backend has not read yet either.
export function resolveIsDiffusion(
  loaded: DiffusionClassification,
  ...parts: (string | null | undefined)[]
): boolean {
  return loaded ?? looksLikeDiffusionGemma(...parts);
}

type ParsedModelStorageKey = {
  modelId: string;
  ggufVariant: string;
};

function parseVersionedModelStorageKey(
  key: string,
): ParsedModelStorageKey | null {
  if (!key.startsWith(MODEL_STORAGE_KEY_PREFIX)) {
    return null;
  }
  try {
    const parsed = JSON.parse(key.slice(MODEL_STORAGE_KEY_PREFIX.length));
    if (
      !Array.isArray(parsed) ||
      parsed.length !== 2 ||
      typeof parsed[0] !== "string" ||
      typeof parsed[1] !== "string"
    ) {
      return null;
    }
    return { modelId: parsed[0], ggufVariant: parsed[1] };
  } catch {
    return null;
  }
}

export function modelStorageKey(
  modelId: string,
  ggufVariant?: string | null,
): string {
  return `${MODEL_STORAGE_KEY_PREFIX}${JSON.stringify([
    normalizeModelIdentity(modelId),
    normalizeGgufVariantIdentity(ggufVariant),
  ])}`;
}

export function modelIdFromStorageKey(key: string): string | null {
  const parsed = parseVersionedModelStorageKey(key);
  if (parsed) {
    return parsed.modelId;
  }
  const separator = key.lastIndexOf("::");
  return separator >= 0 ? key.slice(0, separator) : null;
}

export function ggufVariantFromStorageKey(key: string): string | null {
  const parsed = parseVersionedModelStorageKey(key);
  if (parsed) {
    return parsed.ggufVariant;
  }
  const separator = key.lastIndexOf("::");
  return separator >= 0 ? key.slice(separator + 2) : null;
}
