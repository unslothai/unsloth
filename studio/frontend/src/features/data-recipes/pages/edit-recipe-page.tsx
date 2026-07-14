// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { type RecipePayload, RecipeStudioPage } from "@/features/recipe-studio";
import { useNavigate } from "@tanstack/react-router";
import type { ReactElement } from "react";
import { useCallback, useEffect, useState } from "react";
import { getRecipe, primeRecipeCache, saveRecipe } from "../data/recipes-db";
import type { RecipeRecord } from "../types";

type EditRecipePageProps = {
  recipeId: string;
};

type LoadState =
  | { recipeId: string; status: "loading" }
  | { recipeId: string; status: "missing" }
  | { recipeId: string; status: "error"; message: string }
  | { recipeId: string; status: "ready"; record: RecipeRecord };

function RecipeLoadState({
  title,
  description,
  onBack,
}: {
  title: string;
  description: string;
  onBack: () => void;
}): ReactElement {
  return (
    <div className="min-h-[calc(100dvh-var(--studio-titlebar-height,0px))] bg-background">
      <main className="mx-auto flex min-h-[70vh] w-full max-w-4xl items-center justify-center px-6 py-8">
        <div className="w-full rounded-2xl border bg-card p-8 text-center">
          <h1 className="text-lg font-semibold">{title}</h1>
          <p className="mt-2 text-sm text-muted-foreground">{description}</p>
          <Button
            type="button"
            variant="outline"
            className="mt-5"
            onClick={onBack}
          >
            Back to Recipes
          </Button>
        </div>
      </main>
    </div>
  );
}

export function EditRecipePage({
  recipeId,
}: EditRecipePageProps): ReactElement {
  const navigate = useNavigate();
  const [loadState, setLoadState] = useState<LoadState>({
    recipeId,
    status: "loading",
  });

  useEffect(() => {
    let active = true;
    void getRecipe(recipeId)
      .then((record) => {
        if (!active) return;
        if (!record) {
          setLoadState({ recipeId, status: "missing" });
          return;
        }
        primeRecipeCache(record);
        setLoadState({ recipeId, status: "ready", record });
      })
      .catch((error: unknown) => {
        if (!active) return;
        setLoadState({
          recipeId,
          status: "error",
          message:
            error instanceof Error ? error.message : "Failed to load recipe.",
        });
      });
    return () => {
      active = false;
    };
  }, [recipeId]);

  const handlePersist = useCallback(
    async (input: {
      id: string | null;
      name: string;
      payload: RecipePayload;
      revision?: number;
    }) => {
      const record = await saveRecipe({
        id: input.id,
        name: input.name,
        payload: input.payload,
        revision: input.revision,
        learningRecipeId:
          loadState.recipeId === recipeId && loadState.status === "ready"
            ? loadState.record.learningRecipeId
            : undefined,
        learningRecipeTitle:
          loadState.recipeId === recipeId && loadState.status === "ready"
            ? loadState.record.learningRecipeTitle
            : undefined,
      });
      primeRecipeCache(record);
      if (record.id !== recipeId) {
        await navigate({
          to: "/data-recipes/$recipeId",
          params: { recipeId: record.id },
        });
      }
      return {
        id: record.id,
        updatedAt: record.updatedAt,
        revision: record.revision,
        payload: record.payload,
        removedCredentialPaths: record.removedCredentialPaths ?? [],
      };
    },
    [loadState, navigate, recipeId],
  );

  const handleReload = useCallback(async () => {
    const record = await getRecipe(recipeId);
    if (!record) return null;
    primeRecipeCache(record);
    return record;
  }, [recipeId]);

  if (loadState.recipeId !== recipeId || loadState.status === "loading") {
    return (
      <RecipeLoadState
        title="Loading recipe..."
        description="Please wait while we load your recipe."
        onBack={() => void navigate({ to: "/data-recipes" })}
      />
    );
  }

  if (loadState.status === "missing") {
    return (
      <RecipeLoadState
        title="Recipe not found"
        description="This recipe may have been deleted."
        onBack={() => void navigate({ to: "/data-recipes" })}
      />
    );
  }

  if (loadState.status === "error") {
    return (
      <RecipeLoadState
        title="Could not load recipe"
        description={loadState.message}
        onBack={() => void navigate({ to: "/data-recipes" })}
      />
    );
  }

  return (
    <RecipeStudioPage
      key={loadState.record.id}
      recipeId={loadState.record.id}
      initialRecipeName={loadState.record.name}
      initialPayload={loadState.record.payload}
      initialSavedAt={loadState.record.updatedAt}
      initialRevision={loadState.record.revision}
      onPersistRecipe={handlePersist}
      onReloadRecipe={handleReload}
    />
  );
}
