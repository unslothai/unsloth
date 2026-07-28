// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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
