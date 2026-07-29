// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type DevicePickerItemResolution<T> =
  | { kind: "match"; item: T }
  | { kind: "ambiguous"; firstItem: T }
  | { kind: "none" };

function normalizeDeviceItemTitle(value: string): string {
  return value.trim().normalize("NFC").toLowerCase();
}

export function resolveDevicePickerItem<T>({
  query,
  items,
  canonicalMatch,
  title,
}: {
  query: string;
  items: readonly T[];
  canonicalMatch: (item: T, query: string) => boolean;
  title: (item: T) => string;
}): DevicePickerItemResolution<T> {
  const canonicalItem = items.find((item) => canonicalMatch(item, query));
  if (canonicalItem) {
    return { kind: "match", item: canonicalItem };
  }

  const normalizedQuery = normalizeDeviceItemTitle(query);
  if (!normalizedQuery) {
    return { kind: "none" };
  }

  const titleMatches = items.filter(
    (item) => normalizeDeviceItemTitle(title(item)) === normalizedQuery,
  );
  if (titleMatches.length === 1) {
    return { kind: "match", item: titleMatches[0] };
  }
  return titleMatches.length > 1
    ? { kind: "ambiguous", firstItem: titleMatches[0] }
    : { kind: "none" };
}
