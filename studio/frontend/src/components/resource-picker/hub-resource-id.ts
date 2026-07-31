// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const HUB_RESOURCE_ID_SEGMENT_RE =
  /^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$/;
const MAX_HUB_RESOURCE_ID_SEGMENT_LENGTH = 96;

export type HubResourceIdValidationResult =
  | { ok: true; id: string }
  | { ok: false };

export function validateHubResourceId(
  value: string,
): HubResourceIdValidationResult {
  const id = value.trim();
  if (!id || id.endsWith(".git") || id.includes("--") || id.includes("..")) {
    return { ok: false };
  }
  const segments = id.split("/");
  if (
    (segments.length !== 1 && segments.length !== 2) ||
    segments.some(
      (segment) =>
        segment.length > MAX_HUB_RESOURCE_ID_SEGMENT_LENGTH ||
        !HUB_RESOURCE_ID_SEGMENT_RE.test(segment),
    )
  ) {
    return { ok: false };
  }
  return { ok: true, id };
}

export function isValidHubResourceId(value: string): boolean {
  return validateHubResourceId(value).ok;
}

export function hubResourceIdsEqual(
  first: string | null | undefined,
  second: string | null | undefined,
): boolean {
  const normalizedFirst = first?.trim().toLowerCase();
  const normalizedSecond = second?.trim().toLowerCase();
  return Boolean(
    normalizedFirst && normalizedSecond && normalizedFirst === normalizedSecond,
  );
}

export function findCanonicalHubResourceId(
  query: string,
  ids: readonly string[],
): string | undefined {
  return ids.find((id) => hubResourceIdsEqual(id, query));
}
