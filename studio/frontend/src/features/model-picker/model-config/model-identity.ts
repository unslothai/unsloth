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

// Mirrors split_quant_suffix in studio/backend/utils/openai_auto_switch_settings.py.
// A quant label may carry a bits-per-weight modifier ("IQ4_XS-3.53bpw"), and the
// two backend label helpers disagree on whether to keep it, so both forms parse.
const BPW_SUFFIX = /-[0-9]+(?:\.[0-9]+)?bpw$/i;
const KNOWN_QUANT =
  /^(UD-)?(MXFP[0-9]+(?:_[A-Z0-9]+)*|IQ[0-9]+_[A-Z]+(?:_[A-Z0-9]+)?|TQ[0-9]+_[0-9]+|Q[0-9]+_K_[A-Z]+|Q[0-9]+_[0-9]+|Q[0-9]+_K|BF16|F16|F32)$/i;
const MAX_QUANT_SUFFIX_LEN = 64;

/**
 * `[head, quant]` for a `head:QUANT` key, or null when the colon is not one.
 *
 * The suffix has to look like a real quant, so an ordinary colon inside a POSIX
 * filename is left alone ("/models/foo:bar.gguf" is one valid filename) and a
 * Windows drive letter is never mistaken for a model id.
 */
export function splitQuantSuffix(value: string): [string, string] | null {
  const separator = value.lastIndexOf(":");
  if (separator <= 0 || separator === value.length - 1) {
    return null;
  }
  const head = value.slice(0, separator);
  const tail = value.slice(separator + 1);
  if (tail.includes("/") || tail.includes("\\")) {
    return null;
  }
  if (
    tail.length <= MAX_QUANT_SUFFIX_LEN &&
    KNOWN_QUANT.test(tail.replace(BPW_SUFFIX, ""))
  ) {
    return [head, tail];
  }
  // A .gguf with no recognizable quant token is labelled by its stem, so keys
  // like "/models/CustomModel.gguf:custommodel" exist. Requiring the head to be
  // a .gguf keeps an ordinary colon out: "/models/foo:bar.gguf" splits to a head
  // that is not one.
  return head.toLowerCase().endsWith(".gguf") ? [head, tail] : null;
}
