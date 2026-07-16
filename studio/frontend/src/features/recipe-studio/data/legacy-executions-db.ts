// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Low-level migration reader; intentionally outside the feature UI API.
// eslint-disable-next-line no-restricted-imports
import { readLegacyStorePage } from "@/features/user-assets/legacy-indexeddb";
import type { RecipeExecutionRecord } from "../execution-types";
import { serializeExecutionMetadata } from "./executions-db";

const DATABASE_NAME = "unsloth-data-recipe-executions";
const DEFAULT_PAGE_SIZE = 100;

export type LegacyRecipeExecutionPage = {
  items: ReturnType<typeof serializeExecutionMetadata>[];
  nextCursor: string | null;
};

export async function readLegacyRecipeExecutions(
  cursor: string | null = null,
  limit = DEFAULT_PAGE_SIZE,
): Promise<LegacyRecipeExecutionPage> {
  const pageSize = Math.max(1, Math.min(DEFAULT_PAGE_SIZE, Math.floor(limit)));
  const page = await readLegacyStorePage<RecipeExecutionRecord>({
    databaseName: DATABASE_NAME,
    storeName: "executions",
    cursor,
    limit: pageSize,
  });
  return {
    items: page.items.map(serializeExecutionMetadata),
    nextCursor: page.nextCursor,
  };
}
