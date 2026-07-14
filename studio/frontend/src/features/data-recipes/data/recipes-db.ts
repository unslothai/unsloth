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
// Shared persistence policy is infrastructure, not a feature UI dependency.
// eslint-disable-next-line no-restricted-imports
import { DENIED_SECRET_KEYS } from "@/features/user-assets/persistence-policy";
import { normalizeNonEmptyName } from "@/utils";
import { useCallback, useEffect, useState } from "react";
import type { RecipeRecord, SaveRecipeInput } from "../types";

const recentRecipeCache = new Map<string, RecipeRecord>();
const repositoryListeners = new Set<() => void>();
function normalizeSecretKey(key: string): string {
  return key
    .replace(/(.)([A-Z][a-z]+)/g, "$1_$2")
    .replace(/([a-z0-9])([A-Z])/g, "$1_$2")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
}

function hasValue(value: unknown): boolean {
  if (value === null || value === undefined) {
    return false;
  }
  if (typeof value === "string") {
    return value.trim().length > 0;
  }
  if (Array.isArray(value)) {
    return value.length > 0;
  }
  if (typeof value === "object") {
    return Object.keys(value).length > 0;
  }
  return true;
}

function isSecretField(path: string[], key: string, value: unknown): boolean {
  if (!hasValue(value)) {
    return false;
  }
  const normalizedKey = normalizeSecretKey(key);
  if (normalizedKey === "api_key_env") {
    return false;
  }
  if (
    normalizedKey === "token" &&
    path.slice(-2).join(".") === "seed_config.source"
  ) {
    return true;
  }
  if (
    DENIED_SECRET_KEYS.has(normalizedKey) ||
    [
      "_api_key",
      "_token",
      "_password",
      "_secret",
      "_credential",
      "_credentials",
      "_private_key",
      "_access_key",
      "_access_key_id",
    ].some((suffix) => normalizedKey.endsWith(suffix))
  ) {
    return true;
  }
  const inMcpEnv =
    path.at(-1) === "env" &&
    path.slice(0, -1).some((part) => part.includes("mcp"));
  return (
    inMcpEnv &&
    ["secret", "token", "password", "credential"].some((part) =>
      normalizedKey.includes(part),
    )
  );
}

function sanitizeRecipeForPersistence(
  value: unknown,
  path: string[] = [],
  removedPaths: string[] = [],
): unknown {
  if (Array.isArray(value)) {
    return value.map((entry, index) =>
      sanitizeRecipeForPersistence(entry, [...path, String(index)], removedPaths),
    );
  }
  if (!value || typeof value !== "object") {
    return value;
  }

  const output: Record<string, unknown> = {};
  for (const [key, entry] of Object.entries(value)) {
    const normalizedKey = normalizeSecretKey(key);
    if (isSecretField(path, key, entry)) {
      removedPaths.push([...path, normalizedKey].join("."));
      continue;
    }
    output[key] = sanitizeRecipeForPersistence(
      entry,
      [...path, normalizedKey],
      removedPaths,
    );
  }
  return output;
}

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
  const removedCredentialPaths: string[] = [];
  const payload = sanitizeRecipeForPersistence(
    input.payload,
    [],
    removedCredentialPaths,
  ) as RecipeRecord["payload"];
  const record =
    input.id && input.revision !== undefined
      ? await updateServerRecipe({
          id: input.id,
          name,
          payload,
          revision: input.revision,
          learningRecipeId: input.learningRecipeId,
          learningRecipeTitle: input.learningRecipeTitle,
        })
      : await createServerRecipe({
          id: input.id ?? crypto.randomUUID(),
          name,
          payload,
          learningRecipeId: input.learningRecipeId,
          learningRecipeTitle: input.learningRecipeTitle,
        });
  const authoritativeRecord = { ...record, removedCredentialPaths };
  primeRecipeCache(authoritativeRecord);
  notifyRepositoryChanged();
  return authoritativeRecord;
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
