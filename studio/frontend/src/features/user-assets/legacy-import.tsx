// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type ReactElement, useEffect, useRef } from "react";
import {
  type LegacyImportItemResult,
  type UserAssetsBootstrap,
  bootstrapUserAssets,
  importLegacyUserAssets,
} from "./api";
import { MAX_LEGACY_BATCH_JSON_BYTES } from "./persistence-policy";

const SOURCE = "recipe-indexeddb-v1";
const LEGACY_PAGE_SIZE = 100;
const utf8Encoder = new TextEncoder();

type LegacyPage<T> = {
  items: T[];
  nextCursor: string | null;
};

type LegacyPageReader<T> = (
  cursor?: string | null,
  limit?: number,
) => Promise<LegacyPage<T>>;

type LegacyRecipe = { id: string; revision?: number } & object;
type LegacyExecution = { id: string } & object;

type LegacyImportCoordinatorProps = {
  onImported: () => void;
  readRecipes: LegacyPageReader<LegacyRecipe>;
  readExecutions: LegacyPageReader<LegacyExecution>;
};

type LegacyBatchPlan<T> = {
  batches: T[][];
};

function withoutRevision(recipe: LegacyRecipe) {
  return { ...recipe, revision: undefined };
}

function splitByLegacyBatchBytes<T extends { id: string }>(
  items: T[],
  kind: "recipes" | "executions",
  confirmSubject: string,
): LegacyBatchPlan<T> {
  const batches: T[][] = [];
  let current: T[] = [];
  const emptyEnvelope = {
    source: SOURCE,
    confirmSubject,
    recipes: [] as T[],
    executions: [] as T[],
  };
  const baseBytes = utf8Encoder.encode(
    JSON.stringify(emptyEnvelope),
  ).byteLength;
  let currentBytes = baseBytes;

  for (const item of items) {
    const itemBytes = utf8Encoder.encode(JSON.stringify(item)).byteLength;
    const singleItemBytes = baseBytes + itemBytes;
    if (singleItemBytes > MAX_LEGACY_BATCH_JSON_BYTES) {
      continue;
    }
    const addedBytes = itemBytes + (current.length > 0 ? 1 : 0);
    if (
      current.length > 0 &&
      currentBytes + addedBytes > MAX_LEGACY_BATCH_JSON_BYTES
    ) {
      batches.push(current);
      current = [item];
      currentBytes = singleItemBytes;
    } else {
      current.push(item);
      currentBytes += addedBytes;
    }
  }
  if (current.length > 0) {
    batches.push(current);
  }

  for (const batch of batches) {
    const envelope = {
      ...emptyEnvelope,
      recipes: kind === "recipes" ? batch : [],
      executions: kind === "executions" ? batch : [],
    };
    if (
      utf8Encoder.encode(JSON.stringify(envelope)).byteLength >
      MAX_LEGACY_BATCH_JSON_BYTES
    ) {
      throw new Error(
        "Legacy import batch accounting exceeded its byte limit.",
      );
    }
  }
  return { batches };
}

async function importRecipePages(
  bootstrap: UserAssetsBootstrap,
  readRecipes: LegacyPageReader<LegacyRecipe>,
  signal: AbortSignal,
): Promise<void> {
  const importedIds = new Set(bootstrap.importLedger.recipes);
  let cursor: string | null = null;
  do {
    if (signal.aborted) {
      return;
    }
    const page = await readRecipes(cursor, LEGACY_PAGE_SIZE);
    const recipes = page.items
      .filter((item) => !importedIds.has(item.id))
      .map(withoutRevision);
    const plan = splitByLegacyBatchBytes(recipes, "recipes", bootstrap.subject);
    for (const recipeBatch of plan.batches) {
      await importLegacyUserAssets(
        {
          source: SOURCE,
          confirmSubject: bootstrap.subject,
          recipes: recipeBatch,
          executions: [],
        },
        { signal },
      );
    }
    if (cursor !== null && page.nextCursor === cursor) {
      throw new Error("Legacy recipe cursor did not advance.");
    }
    cursor = page.nextCursor;
  } while (cursor);
}

function effectiveRetryIds(results: LegacyImportItemResult[]): Set<string> {
  return new Set(
    results
      .filter((item) => item.outcome === "missing_parent" && item.id)
      .map((item) => item.id as string),
  );
}

async function importExecutionPages(
  bootstrap: UserAssetsBootstrap,
  readExecutions: LegacyPageReader<LegacyExecution>,
  signal: AbortSignal,
): Promise<void> {
  const importedIds = new Set(bootstrap.importLedger.executions);
  let cursor: string | null = null;
  do {
    if (signal.aborted) {
      return;
    }
    const page = await readExecutions(cursor, LEGACY_PAGE_SIZE);
    const executions = page.items.filter((item) => !importedIds.has(item.id));
    const plan = splitByLegacyBatchBytes(
      executions,
      "executions",
      bootstrap.subject,
    );
    for (const executionBatch of plan.batches) {
      const result = await importLegacyUserAssets(
        {
          source: SOURCE,
          confirmSubject: bootstrap.subject,
          recipes: [],
          executions: executionBatch,
        },
        { signal },
      );
      const retryIds = effectiveRetryIds(result.executions);
      if (retryIds.size > 0) {
        await importLegacyUserAssets(
          {
            source: SOURCE,
            confirmSubject: bootstrap.subject,
            recipes: [],
            executions: executionBatch.filter((item) => retryIds.has(item.id)),
          },
          { signal },
        );
      }
    }
    if (cursor !== null && page.nextCursor === cursor) {
      throw new Error("Legacy execution cursor did not advance.");
    }
    cursor = page.nextCursor;
  } while (cursor);
}

export function LegacyImportCoordinator({
  onImported,
  readRecipes,
  readExecutions,
}: LegacyImportCoordinatorProps): ReactElement | null {
  const onImportedRef = useRef(onImported);
  useEffect(() => {
    onImportedRef.current = onImported;
  }, [onImported]);

  useEffect(() => {
    const controller = new AbortController();

    bootstrapUserAssets()
      .then(async (bootstrap) => {
        await importRecipePages(bootstrap, readRecipes, controller.signal);
        await importExecutionPages(
          bootstrap,
          readExecutions,
          controller.signal,
        );
        if (!controller.signal.aborted) {
          onImportedRef.current();
        }
      })
      .catch(() => undefined);

    return () => controller.abort();
  }, [readExecutions, readRecipes]);

  return null;
}
