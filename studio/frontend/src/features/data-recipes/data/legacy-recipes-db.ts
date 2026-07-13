// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import Dexie, { type EntityTable } from "dexie";
import type { RecipeRecord } from "../types";

const DATABASE_NAME = "unsloth-data-recipes";
const DEFAULT_PAGE_SIZE = 100;

export type LegacyRecipePage = {
  items: RecipeRecord[];
  nextCursor: string | null;
};

async function databaseExists(name: string): Promise<boolean> {
  if (typeof indexedDB === "undefined" || !("databases" in indexedDB)) {
    return false;
  }
  const databases = await indexedDB.databases();
  return databases.some((database) => database.name === name);
}

export async function readLegacyRecipes(
  cursor: string | null = null,
  limit = DEFAULT_PAGE_SIZE,
): Promise<LegacyRecipePage> {
  if (!(await databaseExists(DATABASE_NAME)))
    return { items: [], nextCursor: null };
  const pageSize = Math.max(1, Math.min(DEFAULT_PAGE_SIZE, Math.floor(limit)));
  const db = new Dexie(DATABASE_NAME) as Dexie & {
    recipes: EntityTable<RecipeRecord, "id">;
  };
  db.version(1).stores({ recipes: "id, name, updatedAt, createdAt" });
  try {
    const rows = await (cursor
      ? db.recipes.where("id").above(cursor)
      : db.recipes.orderBy("id")
    )
      .limit(pageSize + 1)
      .toArray();
    const hasMore = rows.length > pageSize;
    const items = rows.slice(0, pageSize).map((row) => ({
      ...row,
      revision: row.revision ?? 0,
    }));
    return {
      items,
      nextCursor: hasMore ? (items.at(-1)?.id ?? null) : null,
    };
  } finally {
    db.close();
  }
}
