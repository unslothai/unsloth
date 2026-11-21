// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createEmptyRecipePayload } from "@/features/recipe-studio";
import { normalizeNonEmptyName } from "@/utils";
import { useEffect, useState } from "react";
import type { RecipeRecord, SaveRecipeInput } from "../types";
import { importLegacyRecipes } from "./legacy-import";
import { RecipeApiError, recipeRequest } from "./recipes-api";

const recentRecipeCache = new Map<string, RecipeRecord>();
const listeners = new Set<(recipes: RecipeRecord[]) => void>();
let cachedRecipeList: RecipeRecord[] = [];
let recipeListReady = false;
let recipeListRequest: Promise<RecipeRecord[]> | null = null;
let mutationVersion = 0;

function publishRecipeList(recipes: RecipeRecord[]): RecipeRecord[] {
  cachedRecipeList = [...recipes].sort((a, b) => b.updatedAt - a.updatedAt);
  recipeListReady = true;
  // The list is authoritative: a recipe deleted elsewhere must not reopen from the cache.
  recentRecipeCache.clear();
  for (const recipe of cachedRecipeList) {
    recentRecipeCache.set(recipe.id, recipe);
  }
  for (const listener of listeners) {
    listener(cachedRecipeList);
  }
  return cachedRecipeList;
}

async function fetchRecipeList(): Promise<RecipeRecord[]> {
  await importLegacyRecipes();
  for (;;) {
    // A save or delete that lands while the GET is in flight makes its answer stale.
    const version = mutationVersion;
    const { recipes } = await recipeRequest<{ recipes: RecipeRecord[] }>("");
    if (version === mutationVersion) return publishRecipeList(recipes);
  }
}

export function listRecipes(): Promise<RecipeRecord[]> {
  recipeListRequest ??= fetchRecipeList().finally(() => {
    recipeListRequest = null;
  });
  return recipeListRequest;
}

export function preloadRecipes(): Promise<RecipeRecord[]> {
  return recipeListReady ? Promise.resolve(cachedRecipeList) : listRecipes();
}

export async function getRecipe(id: string): Promise<RecipeRecord | undefined> {
  await importLegacyRecipes();
  try {
    const record = await recipeRequest<RecipeRecord>(
      `/${encodeURIComponent(id)}`,
    );
    recentRecipeCache.set(id, record);
    return record;
  } catch (error) {
    if (error instanceof RecipeApiError && error.status === 404) {
      return undefined;
    }
    throw error;
  }
}

export function getCachedRecipe(id: string): RecipeRecord | null {
  return recentRecipeCache.get(id) ?? null;
}

export function primeRecipeCache(record: RecipeRecord): void {
  recentRecipeCache.set(record.id, record);
}

export async function saveRecipe(
  input: SaveRecipeInput,
): Promise<RecipeRecord> {
  const now = Date.now();
  const id = input.id ?? crypto.randomUUID();
  const existing = input.id
    ? (recentRecipeCache.get(input.id) ?? (await getRecipe(input.id)))
    : undefined;
  let record: RecipeRecord;
  try {
    record = await recipeRequest<RecipeRecord>(`/${encodeURIComponent(id)}`, {
      method: "PUT",
      body: JSON.stringify({
        id,
        name: normalizeNonEmptyName(input.name),
        payload: input.payload,
        createdAt: existing?.createdAt ?? now,
        updatedAt: now,
        learningRecipeId: input.learningRecipeId ?? existing?.learningRecipeId,
        learningRecipeTitle:
          input.learningRecipeTitle ?? existing?.learningRecipeTitle,
        baseUpdatedAt: input.baseUpdatedAt ?? existing?.updatedAt,
      }),
    });
  } catch (error) {
    if (error instanceof RecipeApiError && error.status === 409) {
      recentRecipeCache.delete(id);
    }
    throw error;
  }
  mutationVersion += 1;
  recentRecipeCache.set(id, record);
  if (recipeListReady) {
    publishRecipeList([
      record,
      ...cachedRecipeList.filter((recipe) => recipe.id !== id),
    ]);
  }
  return record;
}

export async function deleteRecipe(id: string): Promise<void> {
  await recipeRequest<void>(`/${encodeURIComponent(id)}`, {
    method: "DELETE",
  });
  mutationVersion += 1;
  recentRecipeCache.delete(id);
  if (recipeListReady) {
    publishRecipeList(cachedRecipeList.filter((recipe) => recipe.id !== id));
  }
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

export function useRecipes(enabled = true): {
  recipes: RecipeRecord[];
  ready: boolean;
} {
  const [recipes, setRecipes] = useState<RecipeRecord[]>(cachedRecipeList);
  const [ready, setReady] = useState(recipeListReady);

  useEffect(() => {
    if (!enabled) return;
    const listener = (value: RecipeRecord[]) => {
      setRecipes(value);
      setReady(true);
    };
    listeners.add(listener);
    listRecipes().catch((error) => {
      // biome-ignore lint/suspicious/noConsole: the page keeps its cached list
      console.error("Load data recipes failed:", error);
      setReady(true);
    });
    return () => {
      listeners.delete(listener);
    };
  }, [enabled]);

  return { recipes, ready };
}
