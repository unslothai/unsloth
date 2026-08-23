// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  normalizeGgufVariantIdentity,
  normalizeModelIdentity,
} from "@/lib/model-identity";

export { normalizeGgufVariantIdentity, normalizeModelIdentity };

// Mirrors core/inference/model_ids.py _looks_like_path. These identities belong to
// model selection/configuration; Hub only needs the generic normalizers above.
const PUBLIC_ID_PATH_PREFIX_RE = /^(?:[/\\]|\.{1,2}[\\/]|~)/;
const GGUF_SUFFIX_RE = /\.gguf$/i;
const BACKSLASHES_RE = /\\/g;
const TRAILING_SLASHES_RE = /\/+$/;

export function looksLikeModelPath(identifier: string): boolean {
  if (GGUF_SUFFIX_RE.test(identifier)) {
    return true;
  }
  if (PUBLIC_ID_PATH_PREFIX_RE.test(identifier)) {
    return true;
  }
  if (identifier.length >= 2 && identifier[1] === ":") {
    return true;
  }
  return identifier.split("/").length - 1 >= 2 || identifier.includes("\\");
}

/** `.../models--org--name/snapshots/<sha>` -> `org/name`, else null. */
function hfCacheRepoId(path: string): string | null {
  const parts = path.replace(BACKSLASHES_RE, "/").split("/");
  for (let index = 0; index < parts.length; index += 1) {
    const part = parts[index];
    if (part.startsWith("models--") && parts[index + 1] === "snapshots") {
      return part.slice("models--".length).replaceAll("--", "/");
    }
  }
  return null;
}

/** Mirrors hf_cache_repo_id in core/inference/model_ids.py. */
export function isHfCacheSnapshotPath(
  identifier: string | null | undefined,
): boolean {
  return identifier != null && hfCacheRepoId(identifier) !== null;
}

/**
 * The public identity the backend reports for a path-loaded model: an HF cache
 * snapshot becomes its repository id and another local GGUF becomes its filename stem.
 * Mirrors public_model_id in core/inference/model_ids.py, which is what
 * /api/inference/status puts in active_model.
 */
export function publicModelId(identifier: string): string {
  const trimmed = identifier.trim();
  if (!(trimmed && looksLikeModelPath(trimmed))) {
    return trimmed;
  }
  const repoId = hfCacheRepoId(trimmed);
  if (repoId) {
    return repoId;
  }
  const slashPath = trimmed
    .replace(BACKSLASHES_RE, "/")
    .replace(TRAILING_SLASHES_RE, "");
  const name = slashPath.slice(slashPath.lastIndexOf("/") + 1);
  return name.replace(GGUF_SUFFIX_RE, "") || trimmed;
}

/** `org/name`, including Hub repositories named `org/name.gguf`. */
function isHubRepoId(identifier: string): boolean {
  if (identifier.split("/").length - 1 !== 1) {
    return false;
  }
  return !looksLikeModelPath(identifier.replace(GGUF_SUFFIX_RE, ""));
}

/** Mirrors display_model_name in core/inference/model_ids.py: the public id's
 * trailing segment. Splitting the raw id instead would leak the host layout on
 * Windows, where `C:\Users\...` holds no `/`. */
export function modelDisplayName(identifier: string): string {
  const trimmed = identifier.trim();
  if (isHubRepoId(trimmed)) {
    return trimmed.slice(trimmed.indexOf("/") + 1);
  }
  const clean = publicModelId(trimmed);
  return clean.slice(clean.lastIndexOf("/") + 1) || clean;
}

// Ollama's blobs reach the picker through a symlink directory that the local model
// resolver refuses to index, so they cannot advertise a loadable API identity.
const OLLAMA_LINK_SEGMENTS = new Set([".studio_links", "ollama_links"]);
const OLLAMA_MANIFEST_REF_PREFIX = "ollama-manifest:";

export function isOllamaLinkPath(modelId: string | null | undefined): boolean {
  if (!modelId) {
    return false;
  }
  if (modelId.startsWith(OLLAMA_MANIFEST_REF_PREFIX)) {
    return true;
  }
  return modelId
    .replace(BACKSLASHES_RE, "/")
    .split("/")
    .some((segment) => OLLAMA_LINK_SEGMENTS.has(segment));
}

// A dropped/file-picked GGUF is identified by its bare label because the lease-backed
// status cannot expose the host path. Anything indexed as a local model has a separator.
const NATIVE_FILE_LABEL_RE = /^[^/\\]+\.gguf$/i;

export function isNativeFileLabel(modelId: string | null | undefined): boolean {
  return modelId != null && NATIVE_FILE_LABEL_RE.test(modelId);
}

/** A scanned standalone GGUF path has no separately selectable variant identity. */
export function isStandaloneGgufPath(
  modelId: string | null | undefined,
): boolean {
  if (modelId == null || !GGUF_SUFFIX_RE.test(modelId)) {
    return false;
  }
  return (
    PUBLIC_ID_PATH_PREFIX_RE.test(modelId) ||
    // A drive letter, or more separators than the single one a repo id carries.
    (modelId.length >= 2 && modelId[1] === ":") ||
    modelId.includes("\\") ||
    modelId.split("/").length - 1 >= 2 ||
    isNativeFileLabel(modelId)
  );
}

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
// The bpw modifier ("IQ4_XS-3.53bpw") is optional: the backend label helpers disagree.
const BPW_SUFFIX = /-[0-9]+(?:\.[0-9]+)?bpw$/i;
// One source for the anchored test and the scan below. Mirrors _GGUF_QUANT_RE in gguf.py.
const QUANT_TOKEN_SOURCE =
  "(UD-)?(MXFP[0-9]+(?:_[A-Z0-9]+)*|IQ[0-9]+_[A-Z]+(?:_[A-Z0-9]+)?|TQ[0-9]+_[0-9]+|Q[0-9]+_K_[A-Z]+|Q[0-9]+_[0-9]+|Q[0-9]+_K|BF16|F16|F32)";
const KNOWN_QUANT = new RegExp(`^${QUANT_TOKEN_SOURCE}$`, "i");
const QUANT_TOKEN = new RegExp(QUANT_TOKEN_SOURCE, "gi");
const MAX_QUANT_SUFFIX_LEN = 64;
// Mirrors _GGUF_SPLIT_SUFFIX_RE in studio/backend/hub/utils/gguf.py.
const GGUF_SPLIT_SUFFIX = /-[0-9]{3,}-of-[0-9]{3,}/gi;
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

/**
 * Mirrors extract_quant_label in gguf.py, for a bare filename. The parent-directory pass
 * cannot fire on a basename, so this is the stem's quant token or the stem itself. */
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

/**
 * `[head, quant]` for a `head:QUANT` key, or null when the colon is not one. The suffix must
 * look like a real quant, so an ordinary colon in a POSIX filename and a drive letter are left alone. */
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
  // Exactly that label, as the backend requires: a colon is legal in a POSIX filename, so
  // reading the suffix as a variant folds two real files onto one lowercased key.
  const filename = head.replace(BACKSLASHES_RE, "/").split("/").pop() ?? head;
  return tail.toLowerCase() === ggufQuantLabel(filename).toLowerCase()
    ? [head, tail]
    : null;
}
