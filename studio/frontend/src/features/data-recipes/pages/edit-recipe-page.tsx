// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  getAuthSubjectKey,
  subscribeAuthSubject,
} from "@/features/auth";
import {
  RecipeStudioPage,
  readLegacyRecipeExecutions,
  type RecipePayload,
} from "@/features/recipe-studio";
import { importLegacyUserAssetsFromIndexedDb } from "@/features/user-assets";
import { useNavigate } from "@tanstack/react-router";
import type { ReactElement } from "react";
import {
  useCallback,
  useEffect,
  useState,
  useSyncExternalStore,
} from "react";
import { readLegacyRecipes } from "../data/legacy-recipes-db";
import {
  getCachedRecipe,
  getRecipe,
  primeRecipeCache,
  saveRecipe,
} from "../data/recipes-db";
import type { RecipeRecord } from "../types";

type EditRecipePageProps = {
  recipeId: string;
};

type LoadState = {
  subject: string;
  recipeId: string;
  reloadVersion: number;
} &
  (
    | { status: "loading" }
    | { status: "missing" }
    | { status: "error"; error: Error }
    | { status: "ready"; record: RecipeRecord }
  );

function RecipeLoadState({
  title,
  description,
  onBack,
  onRetry,
}: {
  title: string;
  description: string;
  onBack: () => void;
  onRetry?: () => void;
}): ReactElement {
  return (
    <div className="min-h-[calc(100dvh-var(--studio-titlebar-height,0px))] bg-background">
      <main className="mx-auto flex min-h-[70vh] w-full max-w-4xl items-center justify-center px-6 py-8">
        <div className="w-full rounded-2xl border bg-card p-8 text-center">
          <h1 className="text-lg font-semibold">{title}</h1>
          <p className="mt-2 text-sm text-muted-foreground">{description}</p>
          <div className="mt-5 flex justify-center gap-2">
            {onRetry ? (
              <Button type="button" onClick={onRetry}>
                Try Again
              </Button>
            ) : null}
            <Button type="button" variant="outline" onClick={onBack}>
              Back to Recipes
            </Button>
          </div>
        </div>
      </main>
    </div>
  );
}

export function EditRecipePage({ recipeId }: EditRecipePageProps): ReactElement {
  const navigate = useNavigate();
  const subject = useSyncExternalStore(
    subscribeAuthSubject,
    getAuthSubjectKey,
    getAuthSubjectKey,
  );
  const [reloadVersion, setReloadVersion] = useState(0);
  const [loadState, setLoadState] = useState<LoadState>(() => {
    const cachedRecipe = getCachedRecipe(recipeId);
    if (cachedRecipe) {
      return {
        status: "ready",
        subject,
        recipeId,
        reloadVersion,
        record: cachedRecipe,
      };
    }
    return { status: "loading", subject, recipeId, reloadVersion };
  });

  useEffect(() => {
    let active = true;
    const controller = new AbortController();

    void (async () => {
      try {
        let record = await getRecipe(recipeId, {
          signal: controller.signal,
        });
        if (!record) {
          try {
            await importLegacyUserAssetsFromIndexedDb({
              readRecipes: readLegacyRecipes,
              readExecutions: readLegacyRecipeExecutions,
              signal: controller.signal,
            });
          } catch (error) {
            if (!controller.signal.aborted) {
              console.warn("Legacy recipe import failed:", error);
            }
          }
          if (!controller.signal.aborted) {
            record = await getRecipe(recipeId, {
              signal: controller.signal,
            });
          }
        }
        if (!active || controller.signal.aborted) return;
        if (!record) {
          setLoadState({ status: "missing", subject, recipeId, reloadVersion });
          return;
        }
        primeRecipeCache(record);
        setLoadState({
          status: "ready",
          subject,
          recipeId,
          reloadVersion,
          record,
        });
      } catch (error) {
        if (!active || controller.signal.aborted) return;
        setLoadState({
          status: "error",
          subject,
          recipeId,
          reloadVersion,
          error:
            error instanceof Error
              ? error
              : new Error("Failed to load this recipe."),
        });
      }
    })();
    return () => {
      active = false;
      controller.abort();
    };
  }, [recipeId, reloadVersion, subject]);

  const handlePersist = useCallback(
    async (input: {
      id: string | null;
      name: string;
      payload: RecipePayload;
      revision?: number;
    }) => {
      if (
        loadState.status !== "ready" ||
        loadState.subject !== subject ||
        loadState.recipeId !== recipeId ||
        loadState.reloadVersion !== reloadVersion
      ) {
        throw new Error("Recipe persistence account changed. Please try again.");
      }
      const record = await saveRecipe({
        id: input.id ?? recipeId,
        name: input.name,
        payload: input.payload,
        revision: input.revision,
        learningRecipeId: loadState.record.learningRecipeId,
        learningRecipeTitle: loadState.record.learningRecipeTitle,
      });
      primeRecipeCache(record);
      // This acknowledges an older snapshot; using it as initial state would
      // reset edits made while the save was in flight.
      return {
        id: record.id,
        updatedAt: record.updatedAt,
        revision: record.revision,
        payload: record.payload,
        removedCredentialPaths: record.removedCredentialPaths ?? [],
      };
    },
    [loadState, recipeId, reloadVersion, subject],
  );

  const currentLoadState: LoadState =
    loadState.subject === subject &&
    loadState.recipeId === recipeId &&
    loadState.reloadVersion === reloadVersion
      ? loadState
      : { status: "loading", subject, recipeId, reloadVersion };

  if (currentLoadState.status === "loading") {
    return (
      <RecipeLoadState
        title="Loading recipe..."
        description="Please wait while we load your recipe."
        onBack={() => void navigate({ to: "/data-recipes" })}
      />
    );
  }

  if (currentLoadState.status === "missing") {
    return (
      <RecipeLoadState
        title="Recipe not found"
        description="This recipe may have been deleted."
        onBack={() => void navigate({ to: "/data-recipes" })}
      />
    );
  }

  if (currentLoadState.status === "error") {
    return (
      <RecipeLoadState
        title="Couldn't load recipe"
        description={currentLoadState.error.message}
        onRetry={() => setReloadVersion((value) => value + 1)}
        onBack={() => void navigate({ to: "/data-recipes" })}
      />
    );
  }

  return (
    <RecipeStudioPage
      key={`${currentLoadState.subject}:${currentLoadState.record.id}`}
      recipeId={currentLoadState.record.id}
      initialRecipeName={currentLoadState.record.name}
      initialPayload={currentLoadState.record.payload}
      initialSavedAt={currentLoadState.record.updatedAt}
      initialRevision={currentLoadState.record.revision}
      onPersistRecipe={handlePersist}
    />
  );
}
