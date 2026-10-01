// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { accountDatabaseName } from "@/lib/account-transition";
import { normalizeNonEmptyName } from "@/utils";
import Dexie from "dexie";
import { RecipeApiError, recipeRequest } from "./recipes-api";

// One-time copy of the browser-only stores into studio.db. The server import is insert-only and
// honours deletions, so a retry after a partial run (or a cleared flag) cannot clobber or revive
// anything; the IndexedDB stores are left in place.
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
const optionalText = (value: unknown, max: number): string | undefined =>
  typeof value === "string" ? value.slice(0, max) : undefined;

function toRecipe(row: Row): Row | null {
  if (!isId(row.id) || !row.payload || typeof row.payload !== "object") {
    return null;
  }
  const createdAt = toTime(row.createdAt);
  return {
    id: row.id,
    name: normalizeNonEmptyName(
      typeof row.name === "string" ? row.name : "",
    ).slice(0, 500),
    payload: row.payload,
    createdAt,
    updatedAt: toTime(row.updatedAt) || createdAt,
    learningRecipeId: isId(row.learningRecipeId)
      ? row.learningRecipeId
      : undefined,
    learningRecipeTitle: optionalText(row.learningRecipeTitle, 500),
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

async function submit(kind: ImportKind, rows: Row[]): Promise<void> {
  try {
    await recipeRequest("/import", {
      method: "POST",
      body: JSON.stringify({ [kind]: rows }),
    });
  } catch (error) {
    // A 4xx is about the records themselves: isolate it so one bad row does not block the rest.
    if (!(error instanceof RecipeApiError) || error.status >= 500) throw error;
    if (rows.length === 1) {
      // biome-ignore lint/suspicious/noConsole: the record stays in IndexedDB
      console.warn(`Skipped legacy Data Recipe ${kind} record:`, error.message);
      return;
    }
    for (const row of rows) await submit(kind, [row]);
  }
}

async function runImport(): Promise<void> {
  const recipes = (await readStore(RECIPES_DB, "recipes"))
    .map(toRecipe)
    .filter((row): row is Row => row !== null);
  const executions = recipes.length
    ? (await readStore(EXECUTIONS_DB, "executions"))
        .map(toExecution)
        .filter((row): row is Row => row !== null)
    : [];
  // Recipes first: the server drops runs whose recipe it does not hold.
  for (const batch of batches(recipes)) await submit("recipes", batch);
  for (const batch of batches(executions)) await submit("executions", batch);
  localStorage.setItem(doneKey(), String(Date.now()));
}
