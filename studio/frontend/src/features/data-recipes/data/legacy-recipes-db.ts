// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Low-level migration reader; intentionally outside the feature UI API.
// eslint-disable-next-line no-restricted-imports
import { readLegacyStorePage } from "@/features/user-assets/legacy-indexeddb";
import type { RecipeRecord } from "../types";

const DATABASE_NAME = "unsloth-data-recipes";
const DEFAULT_PAGE_SIZE = 100;

export type LegacyRecipePage = {
  items: RecipeRecord[];
  nextCursor: string | null;
};

export async function readLegacyRecipes(
  cursor: string | null = null,
  limit = DEFAULT_PAGE_SIZE,
): Promise<LegacyRecipePage> {
  const pageSize = Math.max(1, Math.min(DEFAULT_PAGE_SIZE, Math.floor(limit)));
  const page = await readLegacyStorePage<RecipeRecord>({
    databaseName: DATABASE_NAME,
    storeName: "recipes",
    cursor,
    limit: pageSize,
  });
  const items = page.items.map((row) => ({
      ...row,
      revision: row.revision ?? 0,
  }));
  return { items, nextCursor: page.nextCursor };
}
