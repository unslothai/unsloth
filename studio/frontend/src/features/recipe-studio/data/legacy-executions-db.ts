// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import Dexie, { type EntityTable } from "dexie";
import type { RecipeExecutionRecord } from "../execution-types";
import { serializeExecutionMetadata } from "./executions-db";

const DATABASE_NAME = "unsloth-data-recipe-executions";
const DEFAULT_PAGE_SIZE = 100;

export type LegacyRecipeExecutionPage = {
  items: ReturnType<typeof serializeExecutionMetadata>[];
  nextCursor: string | null;
};

async function databaseExists(name: string): Promise<boolean> {
  if (typeof indexedDB === "undefined" || !("databases" in indexedDB))
    return false;
  const databases = await indexedDB.databases();
  return databases.some((database) => database.name === name);
}

export async function readLegacyRecipeExecutions(
  cursor: string | null = null,
  limit = DEFAULT_PAGE_SIZE,
): Promise<LegacyRecipeExecutionPage> {
  if (!(await databaseExists(DATABASE_NAME)))
    return { items: [], nextCursor: null };
  const pageSize = Math.max(1, Math.min(DEFAULT_PAGE_SIZE, Math.floor(limit)));
  const db = new Dexie(DATABASE_NAME) as Dexie & {
    executions: EntityTable<RecipeExecutionRecord, "id">;
  };
  db.version(1).stores({ executions: "id, recipeId, kind, status, createdAt" });
  db.version(2).stores({
    executions: "id, recipeId, kind, status, createdAt, finishedAt, jobId",
  });
  try {
    const records = await (cursor
      ? db.executions.where("id").above(cursor)
      : db.executions.orderBy("id")
    )
      .limit(pageSize + 1)
      .toArray();
    const hasMore = records.length > pageSize;
    const items = records.slice(0, pageSize).map(serializeExecutionMetadata);
    return {
      items,
      nextCursor: hasMore ? (items.at(-1)?.id ?? null) : null,
    };
  } finally {
    db.close();
  }
}
