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
  return trimmed;
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
 * The clean id the backend reports for a model loaded by path.
 *
 * Mirrors ``public_model_id`` in studio/backend/core/inference/model_ids.py, which
 * is what ``/api/inference/status`` puts in ``active_model``: an HF cache snapshot
 * becomes its repo id and any other local GGUF becomes its filename stem. Repo ids
 * and already-clean names come back unchanged.
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
 * A GGUF loaded from an inactive HF cache is loaded by path, but a caller holding
 * only the public id would read an exact comparison against the catalog row's path
 * as "not loaded" and fall back to saved or default values instead of the live
 * launch config. Candidates are compared literally first, then by the public id.
 *
 * That second pass only accepts an identity that can name one model: an HF cache
 * snapshot collapses onto its repo id, which is globally unique, while every other
 * path collapses onto a filename or directory stem that two models can share
 * (`/models/alpha/model.gguf` and `/models/beta/model.gguf` are both "model").
 * Accepting a stem would mark the wrong row resident, seeding its editor with
 * another model's live config and saving it under this model's key. Callers with
 * the loadable identifier (`/status`'s `model_identifier`) pass it as the active
 * id, and the literal pass answers exactly.
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

// Ollama's blobs reach the picker through a ".studio_links"/"ollama_links" symlink
// directory. core/inference/local_model_resolver.py refuses to index anything under
// those (the scanner that creates them runs off the request path), so the API can
// never load one and mirroring its settings would advertise a load that cannot happen.
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

// A drag-dropped or file-picked GGUF is the API's second unreachable identity.
// /api/inference/status reports model_identifier as null for a lease-backed load
// (routes/inference.py withholds the host path), so the checkpoint the browser
// keys settings by is the bare file name the backend echoes back. _build_index
// keys a standalone GGUF by its on-disk path and by its .gguf-stripped stem, so
// that name is never an index key and no auto-switch load can read an override
// stored under it. Anything the API can load is keyed by a path or a repo id,
// both of which carry a separator.
const NATIVE_FILE_LABEL_RE = /^[^/\\]+\.gguf$/i;

export function isNativeFileLabel(modelId: string | null | undefined): boolean {
  return modelId != null && NATIVE_FILE_LABEL_RE.test(modelId);
}

// A scanned standalone .gguf, keyed by its on-disk path. Its settings identity
// carries no variant: it has no quant to choose between, while the loader and the
// inventory both label it from its filename, so adopting that label would key one
// file's config two ways. settings-identity.ts applies the same rule to a Hub row.
export function isStandaloneGgufPath(
  modelId: string | null | undefined,
): boolean {
  return modelId != null && modelId.toLowerCase().endsWith(".gguf");
}
