// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { looksLikeLocalPath } from "./local-path.ts";

const WINDOWS_DRIVE_PATH_RE = /^[A-Za-z]:[\\/]/;
const WSL_DRIVE_PATH_RE = /^\/mnt\/[A-Za-z](?:\/|$)/;

function trimTrailingSeparators(path: string, minLength: number): string {
  let end = path.length;
  while (end > minLength && path.charCodeAt(end - 1) === 47) {
    end -= 1;
  }
  return end === path.length ? path : path.slice(0, end);
}

function normalizeCaseInsensitivePath(path: string, minLength: number): string {
  return trimTrailingSeparators(
    path.replace(/\\/g, "/"),
    minLength,
  ).toLowerCase();
}

export function normalizeModelIdentity(modelId: string): string {
  const trimmed = modelId.trim();
  if (!looksLikeLocalPath(trimmed)) {
    return trimmed.toLowerCase();
  }
  const slashPath = trimmed.replace(/\\/g, "/");
  if (WINDOWS_DRIVE_PATH_RE.test(trimmed)) {
    return normalizeCaseInsensitivePath(trimmed, 3);
  }
  if (slashPath.startsWith("//")) {
    return normalizeCaseInsensitivePath(trimmed, 2);
  }
  if (WSL_DRIVE_PATH_RE.test(slashPath)) {
    return normalizeCaseInsensitivePath(trimmed, 6);
  }
  return trimTrailingSeparators(slashPath, 1);
}

export function normalizeGgufVariantIdentity(
  ggufVariant?: string | null,
): string {
  return ggufVariant?.trim().toLowerCase() ?? "";
}

export function modelIdsMatch(
  left: string | null | undefined,
  right: string | null | undefined,
): boolean {
  if (!(left && right)) {
    return false;
  }
  return normalizeModelIdentity(left) === normalizeModelIdentity(right);
}

export function ggufVariantsMatch(
  left: string | null | undefined,
  right: string | null | undefined,
): boolean {
  return (
    normalizeGgufVariantIdentity(left) === normalizeGgufVariantIdentity(right)
  );
}

// Mirrors core/inference/model_ids.py _looks_like_path.
const PUBLIC_ID_PATH_PREFIX_RE = /^(?:[/\\]|\.{1,2}[\\/]|~)/;
const GGUF_SUFFIX_RE = /\.gguf$/i;
const BACKSLASHES_RE = /\\/g;
const TRAILING_SLASHES_RE = /\/+$/;

function looksLikeModelPath(identifier: string): boolean {
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

/**
 * The clean id the backend reports for a model loaded by path. Mirrors ``public_model_id`` in
 * core/inference/model_ids.py, which is what ``/api/inference/status`` puts in ``active_model``:
 * an HF cache snapshot becomes its repo id, any other local GGUF its filename stem, and a repo
 * id or clean name comes back unchanged.
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

/**
 * Whether the model the backend reports as loaded is one of *candidates*.
 *
 * A GGUF from an inactive HF cache loads by path, so a caller holding only the public id would
 * read an exact comparison as "not loaded": candidates are compared literally first, then by
 * public id. That second pass only accepts an identity naming one model, since an HF snapshot
 * collapses onto its unique repo id while every other path collapses onto a stem two models can
 * share, and accepting a stem would save one model's live config under another's key.
 */
export function residentModelIdMatches(
  activeModelId: string | null | undefined,
  ...candidates: (string | null | undefined)[]
): boolean {
  if (candidates.some((candidate) => modelIdsMatch(activeModelId, candidate))) {
    return true;
  }
  const active = activeModelId?.trim();
  // A path-shaped active id is the raw identifier, which the literal pass covered.
  if (!active || looksLikeModelPath(active)) {
    return false;
  }
  return candidates.some((candidate) => {
    const trimmed = candidate?.trim();
    if (!trimmed) {
      return false;
    }
    const publicId = publicModelId(trimmed);
    // Unambiguous only when the collapse produced a namespaced repo id.
    return publicId.includes("/") && modelIdsMatch(active, publicId);
  });
}

// Ollama's blobs reach the picker through a symlink dir that local_model_resolver.py
// refuses to index, so the API can never load one and mirroring its settings would
// advertise a load that cannot happen.
const OLLAMA_LINK_SEGMENTS = new Set([".studio_links", "ollama_links"]);

export function isOllamaLinkPath(modelId: string | null | undefined): boolean {
  if (!modelId) {
    return false;
  }
  return modelId
    .replace(BACKSLASHES_RE, "/")
    .split("/")
    .some((segment) => OLLAMA_LINK_SEGMENTS.has(segment));
}

// A dropped or file-picked GGUF is the API's second unreachable identity: /status withholds the
// host path for a lease-backed load, so the browser keys settings by the bare file name echoed
// back, which _build_index (path and stem only) never uses as an index key. Anything the API can
// load is keyed by a path or a repo id, both of which carry a separator.
const NATIVE_FILE_LABEL_RE = /^[^/\\]+\.gguf$/i;

export function isNativeFileLabel(modelId: string | null | undefined): boolean {
  return modelId != null && NATIVE_FILE_LABEL_RE.test(modelId);
}

// A scanned standalone .gguf, keyed by its on-disk path with no variant: it has no quant to
// choose between, and adopting the filename label would key one file's config two ways
// (settings-identity.ts applies the same rule to a Hub row). The suffix alone does not say that:
// repo ids ending in .gguf are real on the Hub, an iMat repo among them, and those hold every
// quant, so reading one as a single file would collapse Q4 and Q8 onto the same key. The id has
// to name something on this machine too, which a repo id never does.
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
