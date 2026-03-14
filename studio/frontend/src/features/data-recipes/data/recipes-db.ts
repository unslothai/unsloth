// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createEmptyRecipePayload } from "@/features/recipe-studio";
import { normalizeNonEmptyName } from "@/utils";
import Dexie, { type EntityTable, liveQuery } from "dexie";
import { useEffect, useState } from "react";
import type { RecipeRecord, SaveRecipeInput } from "../types";

const db = new Dexie("unsloth-data-recipes") as Dexie & {
  recipes: EntityTable<RecipeRecord, "id">;
};

db.version(1).stores({
  recipes: "id, name, updatedAt, createdAt",
});

const recentRecipeCache = new Map<string, RecipeRecord>();

export function listRecipes(): Promise<RecipeRecord[]> {
  return db.recipes.orderBy("updatedAt").reverse().toArray();
}

export function getRecipe(id: string): Promise<RecipeRecord | undefined> {
  return db.recipes.get(id);
}

function writeRecipeCache(record: RecipeRecord): void {
  recentRecipeCache.set(record.id, record);
}

export function getCachedRecipe(id: string): RecipeRecord | null {
  return recentRecipeCache.get(id) ?? null;
}

export function primeRecipeCache(record: RecipeRecord): void {
  writeRecipeCache(record);
}

export async function saveRecipe(
  input: SaveRecipeInput,
): Promise<RecipeRecord> {
  const now = Date.now();
  const id = input.id ?? crypto.randomUUID();
  const existing = input.id ? await db.recipes.get(input.id) : undefined;
  const record: RecipeRecord = {
    id,
    name: normalizeNonEmptyName(input.name),
    payload: input.payload,
    createdAt: existing?.createdAt ?? now,
    updatedAt: now,
    learningRecipeId: input.learningRecipeId ?? existing?.learningRecipeId,
    learningRecipeTitle:
      input.learningRecipeTitle ?? existing?.learningRecipeTitle,
  };
  await db.recipes.put(record);
  writeRecipeCache(record);
  return record;
}

export async function deleteRecipe(id: string): Promise<void> {
  await db.recipes.delete(id);
  recentRecipeCache.delete(id);
}

export function createRecipeDraft(): Promise<RecipeRecord> {
  return saveRecipe({
    name: "Unnamed",
    payload: createEmptyRecipePayload(),
  });
}

export function createRecipeFromLearningRecipe(input: {
  templateId: string;
  templateTitle: string;
  payload: RecipeRecord["payload"];
}): Promise<RecipeRecord> {
  return saveRecipe({
    name: input.templateTitle,
    payload: input.payload,
    learningRecipeId: input.templateId,
    learningRecipeTitle: input.templateTitle,
  });
}

export function useRecipes(): {
  recipes: RecipeRecord[];
  ready: boolean;
} {
  const [recipes, setRecipes] = useState<RecipeRecord[]>([]);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    const sub = liveQuery(() => listRecipes()).subscribe({
      next: (value) => {
        for (const recipe of value) {
          writeRecipeCache(recipe);
        }
        setRecipes(value);
        setReady(true);
      },
      error: (error) => {
        console.error("data-recipes liveQuery:", error);
        setReady(true);
      },
    });
    return () => sub.unsubscribe();
  }, []);

  return { recipes, ready };
}
