// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  getAuthSubjectKey,
  subscribeAuthSubject,
} from "@/features/auth";
import { createEmptyRecipePayload } from "@/features/recipe-studio";
import {
  createServerRecipe,
  deleteServerRecipe,
  getServerRecipe,
  listServerRecipes,
  updateServerRecipe,
} from "@/features/user-assets";
// Persistence policy is infrastructure, not a feature UI dependency.
// eslint-disable-next-line no-restricted-imports
import {
  DENIED_SECRET_KEYS,
  MCP_ENV_DENIED_EXACT_KEYS,
  MCP_ENV_DENIED_KEY_PARTS,
  MCP_ENV_DENIED_KEY_SUFFIXES,
  SAFE_SECRET_LOOKING_KEYS,
} from "@/features/user-assets/persistence-policy";
import { normalizeNonEmptyName } from "@/utils";
import {
  useCallback,
  useEffect,
  useState,
  useSyncExternalStore,
} from "react";
import type { RecipeRecord, SaveRecipeInput } from "../types";

const recentRecipeCache = new Map<string, RecipeRecord>();
const recipeRecordSubjects = new WeakMap<RecipeRecord, string>();
const repositoryListeners = new Set<() => void>();

function recipeCacheKey(subject: string, id: string): string {
  return `${subject}\u0000${id}`;
}

function assertSubjectUnchanged(subject: string): void {
  if (getAuthSubjectKey() !== subject) {
    throw new Error("Recipe persistence account changed.");
  }
}

function primeRecipeCacheForSubject(
  subject: string,
  record: RecipeRecord,
): void {
  recipeRecordSubjects.set(record, subject);
  recentRecipeCache.set(recipeCacheKey(subject, record.id), record);
}

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
  if (SAFE_SECRET_LOOKING_KEYS.has(normalizedKey)) {
    return false;
  }
  // JSON Schema may define credential-like fields such as
  // `api_key` or `password`;
  // exempt only valid output_format.properties definitions.
  // Scalars still undergo secret removal.
  const isStructuredOutputPropertyDefinition =
    path.at(-1) === "properties" &&
    path.includes("output_format") &&
    (typeof value === "boolean" ||
      (Boolean(value) && typeof value === "object" && !Array.isArray(value)));
  if (isStructuredOutputPropertyDefinition) {
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
  return Boolean(
    inMcpEnv &&
      (MCP_ENV_DENIED_EXACT_KEYS.has(normalizedKey) ||
        MCP_ENV_DENIED_KEY_SUFFIXES.some((suffix) =>
          normalizedKey.endsWith(suffix),
        ) ||
        MCP_ENV_DENIED_KEY_PARTS.some((part) =>
          normalizedKey.includes(part),
        )),
  );
}

function sanitizeRecipeForPersistence(
  value: unknown,
  path: string[] = [],
  removedPaths: string[] = [],
): unknown {
  if (Array.isArray(value)) {
    return value.map((entry, index) =>
      sanitizeRecipeForPersistence(
        entry,
        [...path, String(index)],
        removedPaths,
      ),
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

export async function listRecipes(): Promise<RecipeRecord[]> {
  const subject = getAuthSubjectKey();
  const records = await listServerRecipes<RecipeRecord["payload"]>();
  assertSubjectUnchanged(subject);
  return records;
}

export async function getRecipe(
  id: string,
  options: { signal?: AbortSignal } = {},
): Promise<RecipeRecord | undefined> {
  const subject = getAuthSubjectKey();
  try {
    const record = await getServerRecipe<RecipeRecord["payload"]>(id, options);
    assertSubjectUnchanged(subject);
    primeRecipeCacheForSubject(subject, record);
    return record;
  } catch (error) {
    assertSubjectUnchanged(subject);
    if (error instanceof Error && "status" in error && error.status === 404) {
      return undefined;
    }
    throw error;
  }
}

export function getCachedRecipe(id: string): RecipeRecord | null {
  return recentRecipeCache.get(recipeCacheKey(getAuthSubjectKey(), id)) ?? null;
}

export function primeRecipeCache(record: RecipeRecord): void {
  const subject = getAuthSubjectKey();
  const recordSubject = recipeRecordSubjects.get(record);
  if (recordSubject && recordSubject !== subject) {
    return;
  }
  primeRecipeCacheForSubject(subject, record);
}

export async function saveRecipe(
  input: SaveRecipeInput,
): Promise<RecipeRecord> {
  const subject = getAuthSubjectKey();
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
  assertSubjectUnchanged(subject);
  const authoritativeRecord = { ...record, removedCredentialPaths };
  primeRecipeCacheForSubject(subject, authoritativeRecord);
  notifyRepositoryChanged();
  return authoritativeRecord;
}

export async function deleteRecipe(
  id: string,
  revision: number,
): Promise<void> {
  const subject = getAuthSubjectKey();
  await deleteServerRecipe(id, revision);
  assertSubjectUnchanged(subject);
  recentRecipeCache.delete(recipeCacheKey(subject, id));
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
  const subject = useSyncExternalStore(
    subscribeAuthSubject,
    getAuthSubjectKey,
    getAuthSubjectKey,
  );
  const [loadState, setLoadState] = useState<{
    subject: string;
    recipes: RecipeRecord[];
    ready: boolean;
    error: Error | null;
  }>(() => ({ subject, recipes: [], ready: false, error: null }));
  const [refreshVersion, setRefreshVersion] = useState(0);
  const refresh = useCallback(() => {
    const currentSubject = getAuthSubjectKey();
    setLoadState((current) =>
      current.subject === currentSubject
        ? { ...current, ready: false, error: null }
        : current,
    );
    setRefreshVersion((value) => value + 1);
  }, []);

  useEffect(() => {
    repositoryListeners.add(refresh);
    return () => {
      repositoryListeners.delete(refresh);
    };
  }, [refresh]);

  // refreshVersion reloads manually;
  // the subject key prevents cross-account reuse.
  // biome-ignore lint/correctness/useExhaustiveDependencies: see above
  useEffect(() => {
    let active = true;
    listRecipes()
      .then((records) => {
        if (!active || getAuthSubjectKey() !== subject) {
          return;
        }
        for (const recipe of records) {
          primeRecipeCacheForSubject(subject, recipe);
        }
        setLoadState({ subject, recipes: records, error: null, ready: true });
      })
      .catch((caught: unknown) => {
        if (!active || getAuthSubjectKey() !== subject) {
          return;
        }
        setLoadState({
          subject,
          recipes: [],
          error:
            caught instanceof Error
              ? caught
              : new Error("Failed to load recipes."),
          ready: true,
        });
      });
    return () => {
      active = false;
    };
  }, [refreshVersion, subject]);

  if (loadState.subject !== subject) {
    return { recipes: [], ready: false, error: null, refresh };
  }
  return { ...loadState, refresh };
}
