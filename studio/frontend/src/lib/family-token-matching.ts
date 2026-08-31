// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const REGEX_META = /[.*+?^${}()|[\]\\]/g;

/** Match a family token as a whole identifier segment with flexible -, _, or . punctuation. */
export function familyTokenMatches(token: string, identifier: string): boolean {
  const parts = token
    .toLowerCase()
    .split(/[-_.]+/)
    .filter(Boolean)
    .map((part) => part.replace(REGEX_META, "\\$&"));
  if (parts.length === 0) return false;
  const inner = parts.join("[-_.]+");
  return new RegExp(`(?:^|[-_./\\\\])${inner}(?:$|[-_./\\\\])`).test(
    identifier.toLowerCase(),
  );
}
