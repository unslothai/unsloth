// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createEmptyRecipePayload } from "@/features/recipe-studio";
import { reserveKeepaliveBody } from "@/lib/keepalive-budget";
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
    uncertainRecipes.delete(id);
    return record;
  } catch (error) {
    if (error instanceof RecipeApiError && error.status === 404) {
      return undefined;
    }
    throw error;
  }
}

export function getCachedRecipe(id: string): RecipeRecord | null {
  return uncertainRecipes.has(id) ? null : (recentRecipeCache.get(id) ?? null);
}

export function primeRecipeCache(record: RecipeRecord): void {
  recentRecipeCache.set(record.id, record);
}

const pendingSaves = new Map<string, Set<Promise<unknown>>>();
// A save lost to the network may have landed: the cached record cannot seed an editor until reread.
// Maps id to the newest version a lost save may have written.
const uncertainRecipes = new Map<string, number>();

/** Resolves once no save for `id` is pending; null when none is. */
export function settleRecipeSaves(id: string): Promise<void> | null {
  if (!pendingSaves.get(id)?.size) return null;
  return (async () => {
    // A save deferred behind another is sent from that one's callbacks: yield a task and recheck.
    for (let pending = pendingSaves.get(id); pending?.size; pending = pendingSaves.get(id)) {
      await Promise.allSettled([...pending]);
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
  })();
}

export function saveRecipe(input: SaveRecipeInput): Promise<RecipeRecord> {
  const save = putRecipe(input);
  if (input.id) {
    const id = input.id;
    const set = pendingSaves.get(id) ?? new Set();
    pendingSaves.set(id, set);
    set.add(save);
    void save.catch(() => undefined).finally(() => {
      set.delete(save);
      if (!set.size) pendingSaves.delete(id);
    });
  }
  return save;
}

async function putRecipe(input: SaveRecipeInput): Promise<RecipeRecord> {
  const now = input.updatedAt ?? Date.now();
  const id = input.id ?? crypto.randomUUID();
  // `known` lets an unloading page skip this read, which the browser would cancel.
  const existing = input.id
    ? (recentRecipeCache.get(input.id) ?? input.known ?? (await getRecipe(input.id)))
    : undefined;
  const body = JSON.stringify({
    id,
    name: normalizeNonEmptyName(input.name),
    payload: input.payload,
    createdAt: existing?.createdAt ?? now,
    updatedAt: now,
    learningRecipeId: input.learningRecipeId ?? existing?.learningRecipeId,
    learningRecipeTitle:
      input.learningRecipeTitle ?? existing?.learningRecipeTitle,
    baseUpdatedAt: input.baseUpdatedAt ?? existing?.updatedAt,
    alsoBaseUpdatedAt: input.alsoBaseUpdatedAt?.length ? input.alsoBaseUpdatedAt : undefined,
  });
  const release = input.keepalive ? reserveKeepaliveBody(body) : null;
  let record: RecipeRecord;
  try {
    record = await recipeRequest<RecipeRecord>(`/${encodeURIComponent(id)}`, {
      method: "PUT",
      body,
      keepalive: input.keepalive ? release !== null : undefined,
    });
    // A 200 whose body was cut off still committed: treat it like a lost answer.
    if (typeof record?.updatedAt !== "number") {
      throw new TypeError("The recipe save answer could not be read.");
    }
  } catch (error) {
    if (!(error instanceof RecipeApiError)) {
      uncertainRecipes.set(id, Math.max(uncertainRecipes.get(id) ?? 0, now));
    }
    // A late 409 must not evict a save at or past this version that already landed.
    if (
      error instanceof RecipeApiError &&
      error.status === 409 &&
      (recentRecipeCache.get(id)?.updatedAt ?? -1) < now
    ) {
      recentRecipeCache.delete(id);
    }
    throw error;
  } finally {
    release?.();
  }
  mutationVersion += 1;
  if ((uncertainRecipes.get(id) ?? 0) <= record.updatedAt) uncertainRecipes.delete(id);
  // Saves can overlap when a closing page sends its last edit; a late reply must not roll back.
  if ((recentRecipeCache.get(id)?.updatedAt ?? -1) > record.updatedAt) {
    return record;
  }
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
