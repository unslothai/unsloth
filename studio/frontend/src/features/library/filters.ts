// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { LibraryItem, LibrarySource } from "./api";
import { type LibraryTypeFilter, TYPE_FILTER_KINDS, fileKind } from "./file-kind";

export interface LibraryFilters {
  sources: Set<LibrarySource>;
  types: Set<LibraryTypeFilter>;
}

export const EMPTY_FILTERS: LibraryFilters = { sources: new Set(), types: new Set() };

export function filtersActive(filters: LibraryFilters): boolean {
  return filters.sources.size > 0 || filters.types.size > 0;
}

export function matchesFilters(
  item: LibraryItem,
  filters: LibraryFilters,
  typesApply: boolean,
): boolean {
  if (filters.sources.size > 0 && !filters.sources.has(item.source)) return false;
  if (!typesApply || filters.types.size === 0) return true;
  const kind = fileKind(item);
  return [...filters.types].some((type) => TYPE_FILTER_KINDS[type].includes(kind));
}
