// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export interface ActiveLlamaServerArgumentsResponse {
  /** Preferred explicit effective-load identity for newer backend snapshots. */
  effective_load_identifier?: string | null;
  /** Compatibility name used by the initial Studio-only endpoint. */
  model_identifier?: string | null;
  gguf_variant: string | null;
  runtime_revision: string | null;
  llama_extra_args: string[];
}

export interface ActiveLlamaArgumentsIdentity {
  effectiveLoadIdentifier: string | null;
  ggufVariant: string | null;
  runtimeRevision: string | null;
}

export function activeLlamaArgumentsResponseLoadIdentity(
  response: ActiveLlamaServerArgumentsResponse,
): string | null {
  return (
    response.effective_load_identifier ?? response.model_identifier ?? null
  );
}

export function currentEffectiveLlamaLoadIdentity({
  activeLoadId,
  residentCheckpoint,
  selectedCheckpoint,
}: {
  activeLoadId: string | null;
  residentCheckpoint: string | null | undefined;
  selectedCheckpoint: string | null;
}): string | null {
  return activeLoadId ?? residentCheckpoint ?? selectedCheckpoint;
}

const TRAILING_SLASHES = /\/+$/;
const CASE_SENSITIVE_PATH_PREFIX = /^(\/|\.{1,2}\/|~\/)/;
const WINDOWS_DRIVE_PREFIX = /^[A-Za-z]:\//;
const WSL_MOUNT_PREFIX = /^\/mnt\/[A-Za-z](?:\/|$)/;

function normalizeModelIdentity(value: string | null): string {
  if (value == null) {
    return "";
  }
  const slashPath = value
    .trim()
    .replaceAll("\\", "/")
    .replace(TRAILING_SLASHES, "");
  const caseInsensitive =
    !CASE_SENSITIVE_PATH_PREFIX.test(slashPath) ||
    WINDOWS_DRIVE_PREFIX.test(slashPath) ||
    slashPath.startsWith("//") ||
    WSL_MOUNT_PREFIX.test(slashPath);
  return caseInsensitive ? slashPath.toLowerCase() : slashPath;
}

function normalizeGgufVariant(value: string | null): string {
  return value?.trim().toLowerCase() ?? "";
}

/** Apply only to the exact process that initiated the hydration request. */
export function activeLlamaArgumentsHydrationMatches(
  response: ActiveLlamaServerArgumentsResponse,
  requested: ActiveLlamaArgumentsIdentity,
  current: ActiveLlamaArgumentsIdentity,
): boolean {
  if (
    response.runtime_revision == null ||
    requested.runtimeRevision == null ||
    current.runtimeRevision == null
  ) {
    return false;
  }
  const responseLoadIdentity =
    activeLlamaArgumentsResponseLoadIdentity(response);
  return (
    normalizeModelIdentity(responseLoadIdentity) !== "" &&
    normalizeModelIdentity(responseLoadIdentity) ===
      normalizeModelIdentity(requested.effectiveLoadIdentifier) &&
    normalizeModelIdentity(responseLoadIdentity) ===
      normalizeModelIdentity(current.effectiveLoadIdentifier) &&
    normalizeGgufVariant(response.gguf_variant) ===
      normalizeGgufVariant(requested.ggufVariant) &&
    normalizeGgufVariant(response.gguf_variant) ===
      normalizeGgufVariant(current.ggufVariant) &&
    response.runtime_revision === requested.runtimeRevision &&
    response.runtime_revision === current.runtimeRevision
  );
}
