// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { RecipeStudioPage, type RecipePayload } from "@/features/recipe-studio";
import { useNavigate } from "@tanstack/react-router";
import type { ReactElement } from "react";
import { useCallback, useEffect, useState } from "react";
import { getCachedRecipe, getRecipe, primeRecipeCache, saveRecipe } from "../data/recipes-db";
import type { RecipeRecord } from "../types";

type EditRecipePageProps = {
  recipeId: string;
};

type LoadState =
  | { status: "loading" }
  | { status: "missing" }
  | { status: "ready"; record: RecipeRecord };

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
    <div className="min-h-screen bg-background">
      <main className="mx-auto flex min-h-[70vh] w-full max-w-4xl items-center justify-center px-6 py-8">
        <div className="w-full rounded-2xl border bg-card p-8 text-center">
          <h1 className="text-lg font-semibold">{title}</h1>
          <p className="mt-2 text-sm text-muted-foreground">{description}</p>
          <Button type="button" variant="outline" className="mt-5" onClick={onBack}>
            Back to Recipes
          </Button>
        </div>
      </main>
    </div>
  );
}

export function EditRecipePage({ recipeId }: EditRecipePageProps): ReactElement {
  const navigate = useNavigate();
  const [loadState, setLoadState] = useState<LoadState>(() => {
    const cachedRecipe = getCachedRecipe(recipeId);
    if (cachedRecipe) {
      return { status: "ready", record: cachedRecipe };
    }
    return { status: "loading" };
  });

  useEffect(() => {
    let active = true;
    const cachedRecipe = getCachedRecipe(recipeId);
    if (cachedRecipe) {
      setLoadState({ status: "ready", record: cachedRecipe });
    } else {
      setLoadState({ status: "loading" });
    }

    void getRecipe(recipeId).then((record) => {
      if (!active) {
        return;
      }
      if (!record) {
        setLoadState({ status: "missing" });
        return;
      }
      primeRecipeCache(record);
      setLoadState({ status: "ready", record });
    });
    return () => {
      active = false;
    };
  }, [recipeId]);

  const handlePersist = useCallback(
    async (input: { id: string | null; name: string; payload: RecipePayload }) => {
      const record = await saveRecipe({
        id: input.id ?? recipeId,
        name: input.name,
        payload: input.payload,
      });
      primeRecipeCache(record);
      return { id: record.id, updatedAt: record.updatedAt };
    },
    [recipeId],
  );

  if (loadState.status === "loading") {
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

  return (
    <RecipeStudioPage
      key={loadState.record.id}
      recipeId={loadState.record.id}
      initialRecipeName={loadState.record.name}
      initialPayload={loadState.record.payload}
      initialSavedAt={loadState.record.updatedAt}
      onPersistRecipe={handlePersist}
    />
  );
}
