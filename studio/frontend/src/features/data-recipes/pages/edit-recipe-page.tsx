// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useAppShellReadySignal } from "@/components/app-readiness";
import { Button } from "@/components/ui/button";
import { RecipeStudioPage, type RecipePayload } from "@/features/recipe-studio";
import { useNavigate } from "@tanstack/react-router";
import type { ReactElement } from "react";
import { useCallback, useEffect, useRef, useState } from "react";
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
    <div className="min-h-[calc(100dvh-var(--studio-titlebar-height,0px))] bg-background">
      <main className="mx-auto flex min-h-[70dvh] w-full max-w-4xl items-center justify-center px-6 py-8">
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
  const signalReady = useAppShellReadySignal();
  const navigate = useNavigate();
  const reloadReadySent = useRef(false);
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
      // A later server read would replace edits already made in the open editor.
      setLoadState({ status: "ready", record: cachedRecipe });
      return;
    }
    setLoadState({ status: "loading" });

    getRecipe(recipeId)
      .then((record) => {
        if (!active) {
          return;
        }
        if (!record) {
          setLoadState({ status: "missing" });
          return;
        }
        primeRecipeCache(record);
        setLoadState({ status: "ready", record });
      })
      .catch((error) => {
        // biome-ignore lint/suspicious/noConsole: the load state below is what the user sees
        console.error("Load recipe failed:", error);
        if (active) setLoadState({ status: "missing" });
      });
    return () => {
      active = false;
    };
  }, [recipeId]);

  useEffect(() => {
    if (loadState.status === "loading" || reloadReadySent.current) {
      return;
    }
    reloadReadySent.current = true;
    signalReady();
  }, [loadState.status, signalReady]);

  // The version this editor is built on, so a save over another window's newer copy is refused.
  const editedVersion = useRef<number | undefined>(undefined);
  const loadedRecord = loadState.status === "ready" ? loadState.record : null;
  useEffect(() => {
    editedVersion.current = loadedRecord?.updatedAt;
  }, [loadedRecord]);

  const handlePersist = useCallback(
    async (input: { id: string | null; name: string; payload: RecipePayload }) => {
      const record = await saveRecipe({
        id: input.id ?? recipeId,
        name: input.name,
        payload: input.payload,
        baseUpdatedAt: editedVersion.current,
      });
      editedVersion.current = record.updatedAt;
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
