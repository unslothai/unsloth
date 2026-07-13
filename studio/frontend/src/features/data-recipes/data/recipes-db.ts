// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createEmptyRecipePayload } from "@/features/recipe-studio";
import {
  createServerRecipe,
  deleteServerRecipe,
  getServerRecipe,
  listServerRecipes,
  updateServerRecipe,
} from "@/features/user-assets";
import { normalizeNonEmptyName } from "@/utils";
import { useCallback, useEffect, useState } from "react";
import type { RecipeRecord, SaveRecipeInput } from "../types";

const recentRecipeCache = new Map<string, RecipeRecord>();
const repositoryListeners = new Set<() => void>();

function notifyRepositoryChanged(): void {
  for (const listener of repositoryListeners) listener();
}

export function listRecipes(): Promise<RecipeRecord[]> {
  return listServerRecipes<RecipeRecord["payload"]>();
}

export async function getRecipe(id: string): Promise<RecipeRecord | undefined> {
  try {
    const record = await getServerRecipe<RecipeRecord["payload"]>(id);
    primeRecipeCache(record);
    return record;
  } catch (error) {
    if (error instanceof Error && "status" in error && error.status === 404) {
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
  const name = normalizeNonEmptyName(input.name);
  const record =
    input.id && input.revision !== undefined
      ? await updateServerRecipe({
          id: input.id,
          name,
          payload: input.payload,
          revision: input.revision,
          learningRecipeId: input.learningRecipeId,
          learningRecipeTitle: input.learningRecipeTitle,
        })
      : await createServerRecipe({
          id: input.id ?? crypto.randomUUID(),
          name,
          payload: input.payload,
          learningRecipeId: input.learningRecipeId,
          learningRecipeTitle: input.learningRecipeTitle,
        });
  primeRecipeCache(record);
  notifyRepositoryChanged();
  return record;
}

export async function deleteRecipe(
  id: string,
  revision: number,
): Promise<void> {
  await deleteServerRecipe(id, revision);
  recentRecipeCache.delete(id);
  notifyRepositoryChanged();
}

export function createRecipeDraft(): Promise<RecipeRecord> {
  return saveRecipe({ name: "Unnamed", payload: createEmptyRecipePayload() });
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
  error: Error | null;
  refresh: () => void;
} {
  const [recipes, setRecipes] = useState<RecipeRecord[]>([]);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [refreshVersion, setRefreshVersion] = useState(0);
  const refresh = useCallback(
    () => setRefreshVersion((value) => value + 1),
    [],
  );

  useEffect(() => {
    repositoryListeners.add(refresh);
    return () => {
      repositoryListeners.delete(refresh);
    };
  }, [refresh]);

  useEffect(() => {
    let active = true;
    listRecipes()
      .then((records) => {
        if (!active) return;
        for (const recipe of records) primeRecipeCache(recipe);
        setRecipes(records);
        setError(null);
        setReady(true);
      })
      .catch((caught: unknown) => {
        if (!active) return;
        setRecipes([]);
        setError(
          caught instanceof Error
            ? caught
            : new Error("Failed to load recipes."),
        );
        setReady(true);
      });
    return () => {
      active = false;
    };
  }, [refreshVersion]);

  return { recipes, ready, error, refresh };
}
