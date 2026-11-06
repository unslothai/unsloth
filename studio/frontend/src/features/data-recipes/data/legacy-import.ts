// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { accountDatabaseName } from "@/lib/account-transition";
import { normalizeNonEmptyName } from "@/utils";
import Dexie from "dexie";
import { RecipeApiError, recipeRequest } from "./recipes-api";

// One-time copy into studio.db; the server import is insert-only, so retries are safe.
const RECIPES_DB = "unsloth-data-recipes";
const EXECUTIONS_DB = "unsloth-data-recipe-executions";
const BATCH_BYTES = 8 * 1024 * 1024;

type Row = Record<string, unknown>;
type ImportKind = "recipes" | "executions";

let pending: Promise<void> | null = null;

function doneKey(): string {
  return `${accountDatabaseName(RECIPES_DB)}:server-import.v1`;
}

export function importLegacyRecipes(): Promise<void> {
  if (localStorage.getItem(doneKey())) return Promise.resolve();
  pending ??= runImport().catch((error) => {
    pending = null;
    // biome-ignore lint/suspicious/noConsole: retried on the next load
    console.error("Legacy Data Recipe import failed:", error);
  });
  return pending;
}

async function readStore(name: string, table: string): Promise<Row[]> {
  const dbName = accountDatabaseName(name);
  if (!(await Dexie.exists(dbName))) return [];
  const db = new Dexie(dbName);
  try {
    await db.open();
    return db.tables.some((t) => t.name === table)
      ? await db.table<Row>(table).toArray()
      : [];
  } finally {
    db.close();
  }
}

const isId = (value: unknown): value is string =>
  typeof value === "string" && value.length > 0 && value.length <= 128;
const toTime = (value: unknown): number =>
  typeof value === "number" && Number.isFinite(value) ? Math.round(value) : 0;

function toRecipe(row: Row): Row | null {
  if (!isId(row.id) || !row.payload || typeof row.payload !== "object") {
    return null;
  }
  const createdAt = toTime(row.createdAt);
  return {
    id: row.id,
    name: normalizeNonEmptyName(typeof row.name === "string" ? row.name : ""),
    payload: row.payload,
    createdAt,
    updatedAt: toTime(row.updatedAt) || createdAt,
    learningRecipeId: isId(row.learningRecipeId)
      ? row.learningRecipeId
      : undefined,
    learningRecipeTitle:
      typeof row.learningRecipeTitle === "string"
        ? row.learningRecipeTitle
        : undefined,
  };
}

function toExecution(row: Row): Row | null {
  if (!isId(row.id) || !isId(row.recipeId)) return null;
  return { ...row, createdAt: toTime(row.createdAt) };
}

function* batches(rows: Row[]): Generator<Row[]> {
  let batch: Row[] = [];
  let size = 0;
  for (const row of rows) {
    const rowSize = JSON.stringify(row).length;
    if (batch.length > 0 && size + rowSize > BATCH_BYTES) {
      yield batch;
      batch = [];
      size = 0;
    }
    batch.push(row);
    size += rowSize;
  }
  if (batch.length > 0) yield batch;
}

// Statuses that describe the records themselves; auth, throttling and server errors retry later.
const RECORD_REJECTED = new Set([400, 413, 422]);

async function submit(
  account: string,
  kind: ImportKind,
  rows: Row[],
): Promise<void> {
  // A same-tab account switch must not push one account's browser data into the next.
  if (accountDatabaseName(RECIPES_DB) !== account) {
    throw new Error("Account changed during legacy Data Recipe import");
  }
  try {
    await recipeRequest("/import", {
      method: "POST",
      body: JSON.stringify({ [kind]: rows }),
    });
  } catch (error) {
    if (
      !(error instanceof RecipeApiError) ||
      !RECORD_REJECTED.has(error.status)
    ) {
      throw error;
    }
    if (rows.length === 1) {
      // biome-ignore lint/suspicious/noConsole: the record stays in IndexedDB
      console.warn(`Skipped legacy Data Recipe ${kind} record:`, error.message);
      return;
    }
    for (const row of rows) await submit(account, kind, [row]);
  }
}

async function runImport(): Promise<void> {
  const account = accountDatabaseName(RECIPES_DB);
  const key = doneKey();
  const recipes = (await readStore(RECIPES_DB, "recipes"))
    .map(toRecipe)
    .filter((row): row is Row => row !== null);
  const executions = recipes.length
    ? (await readStore(EXECUTIONS_DB, "executions"))
        .map(toExecution)
        .filter((row): row is Row => row !== null)
    : [];
  // Recipes first: the server drops runs whose recipe it does not hold.
  for (const batch of batches(recipes)) {
    await submit(account, "recipes", batch);
  }
  for (const batch of batches(executions)) {
    await submit(account, "executions", batch);
  }
  if (accountDatabaseName(RECIPES_DB) === account) {
    localStorage.setItem(key, String(Date.now()));
  }
}
