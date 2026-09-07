// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// eslint-disable-next-line no-restricted-imports -- Avoid the hub barrel's React and download-manager exports.
import {
  normalizeGgufVariantIdentity,
  normalizeModelIdentity,
} from "@/features/hub/lib/model-identity";
import { looksLikeLocalPath } from "@/lib/local-path";

// eslint-disable-next-line no-restricted-imports -- Avoid the hub barrel's React and download-manager exports.
export {
  isNativeFileLabel,
  isOllamaLinkPath,
  isStandaloneGgufPath,
  modelDisplayName,
  normalizeGgufVariantIdentity,
  normalizeModelIdentity,
  publicModelId,
} from "@/features/hub/lib/model-identity";

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

// Mirrors split_quant_suffix in openai_auto_switch_settings.py. The bpw modifier
// ("IQ4_XS-3.53bpw") is optional: the backend label helpers disagree.
const BPW_SUFFIX = /-[0-9]+(?:\.[0-9]+)?bpw$/i;
// One source for the anchored test and the scan below. Mirrors _GGUF_QUANT_RE in gguf.py.
const QUANT_TOKEN_SOURCE =
  "(UD-)?(MXFP[0-9]+(?:_[A-Z0-9]+)*|IQ[0-9]+_[A-Z]+(?:_[A-Z0-9]+)?|TQ[0-9]+_[0-9]+|Q[0-9]+_K_[A-Z]+|Q[0-9]+_[0-9]+|Q[0-9]+_K|BF16|F16|F32)";
const KNOWN_QUANT = new RegExp(`^${QUANT_TOKEN_SOURCE}$`, "i");
const QUANT_TOKEN = new RegExp(QUANT_TOKEN_SOURCE, "gi");
const MAX_QUANT_SUFFIX_LEN = 64;
// Mirrors _GGUF_SPLIT_SUFFIX_RE in studio/backend/hub/utils/gguf.py.
const GGUF_SPLIT_SUFFIX = /-[0-9]{3,}-of-[0-9]{3,}/gi;
const BACKSLASHES = /\\/g;
// A float precision labels a file only when nothing sharper does, as _select_quant_match.
const FLOAT_PRECISION_QUANTS: ReadonlySet<string> = new Set([
  "BF16",
  "F16",
  "F32",
]);

/** Mirrors _gguf_stem in studio/backend/hub/utils/gguf.py, for a bare filename. */
function ggufStem(filename: string): string {
  const dot = filename.lastIndexOf(".");
  const withoutExtension = dot >= 0 ? filename.slice(0, dot) : filename;
  return withoutExtension.replace(GGUF_SPLIT_SUFFIX, "").trim();
}

/** Mirrors extract_quant_label in gguf.py, for a bare filename. The parent-directory pass cannot
 *  fire on a basename, so this is the stem's quant token or the stem itself. */
export function ggufQuantLabel(filename: string): string {
  const stem = ggufStem(filename);
  let fallback: RegExpExecArray | null = null;
  for (const match of stem.matchAll(QUANT_TOKEN)) {
    if (FLOAT_PRECISION_QUANTS.has(match[2].toUpperCase())) {
      fallback ??= match;
      continue;
    }
    return `${match[1] ?? ""}${match[2]}`;
  }
  if (fallback) {
    return `${fallback[1] ?? ""}${fallback[2]}`;
  }
  return stem || "gguf";
}

/** `[head, quant]` for a `head:QUANT` key, or null when the colon is not one. The suffix must look
 *  like a real quant, so an ordinary colon in a POSIX filename and a drive letter are left alone. */
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
  // A .gguf with no quant is labelled by its stem; a non-.gguf head is a plain colon.
  if (!head.toLowerCase().endsWith(".gguf")) {
    return null;
  }
  // Exactly that label, as the backend requires: a colon is legal in a POSIX filename, so reading
  // the suffix as a variant folds two real files onto one lowercased key.
  const filename = head.replace(BACKSLASHES, "/").split("/").pop() ?? head;
  return tail.toLowerCase() === ggufQuantLabel(filename).toLowerCase()
    ? [head, tail]
    : null;
}

/**
* `[modelId, variant]` for an override key, or `[value, null]` when it names no variant.
*
* splitQuantSuffix answers for a bare quant token, which is what a stored key usually spells.
* A qualified variant is not one: it can name a directory (`distilled/model-Q6_K`) or a whole
* filename stem, and both are refused there. A repo id carries no colon, so for anything that
* is not a local path the last colon is the separator whatever the tail spells. A path is left
* whole for the reason the backend's resolver leaves it whole: a colon is legal in a POSIX
* filename, so "/models/foo:bar/baz.gguf" is one name and splitting it would answer for a
* different model. */
export function splitModelOverrideKey(value: string): [string, string | null] {
  const quant = splitQuantSuffix(value);
  if (quant) {
    return quant;
  }
  const separator = value.lastIndexOf(":");
  if (
    separator <= 0 ||
    separator === value.length - 1 ||
    looksLikeLocalPath(value)
  ) {
    return [value, null];
  }
  return [value.slice(0, separator), value.slice(separator + 1)];
}
