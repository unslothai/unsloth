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

// Mirror the backend's pre-download classifier (_classify_diffusion_gguf in
// routes/inference.py): with no header to read, strip non-alphanumerics and look for
// the DiffusionGemma family name, so staged Load agrees with /validate and /load.
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
