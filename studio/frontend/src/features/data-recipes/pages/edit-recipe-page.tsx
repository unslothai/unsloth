// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useAppShellReadySignal } from "@/components/app-readiness";
import { Button } from "@/components/ui/button";
import { RecipeStudioPage, type RecipePayload } from "@/features/recipe-studio";
import { useNavigate } from "@tanstack/react-router";
import type { ReactElement } from "react";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  getCachedRecipe,
  getRecipe,
  primeRecipeCache,
  saveRecipe,
  settleRecipeSaves,
} from "../data/recipes-db";
import { RecipeApiError } from "../data/recipes-api";
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

const ALSO_BASE_LIMIT = 500;

export function EditRecipePage({ recipeId }: EditRecipePageProps): ReactElement {
  const signalReady = useAppShellReadySignal();
  const navigate = useNavigate();
  const reloadReadySent = useRef(false);
  const [loadState, setLoadState] = useState<LoadState>(() => {
    const cachedRecipe = getCachedRecipe(recipeId);
    if (cachedRecipe && !settleRecipeSaves(recipeId)) {
      return { status: "ready", record: cachedRecipe };
    }
    return { status: "loading" };
  });

  useEffect(() => {
    let active = true;
    const pending = settleRecipeSaves(recipeId);
    const cachedRecipe = getCachedRecipe(recipeId);
    if (cachedRecipe && !pending) {
      // A later server read would replace edits already made in the open editor.
      setLoadState({ status: "ready", record: cachedRecipe });
      return;
    }
    setLoadState({ status: "loading" });

    // Reopened while the last editor's exit save is in flight: start from the version it writes.
    (pending ?? Promise.resolve())
      .then(() => getCachedRecipe(recipeId) ?? getRecipe(recipeId))
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

  const editedVersion = useRef<number | undefined>(undefined);
  const loadedRecord = loadState.status === "ready" ? loadState.record : null;
  const loadedRef = useRef(loadedRecord);
  useEffect(() => {
    editedVersion.current = loadedRecord?.updatedAt;
    if (loadedRecord) loadedRef.current = loadedRecord;
  }, [loadedRecord]);

  const inFlightVersion = useRef<number | undefined>(undefined);
  // Versions of saves lost to the network that may have landed, newest first.
  const unansweredVersions = useRef<number[]>([]);
  const handlePersist = useCallback(
    async (input: {
      id: string | null;
      name: string;
      payload: RecipePayload;
      keepalive?: boolean;
      recover?: boolean;
    }) => {
      // The server keeps the updatedAt sent, so a save builds on one in flight; a recovering
      // one also names every version an unanswered save may have left.
      const confirmed = editedVersion.current;
      const inFlight = inFlightVersion.current;
      const candidates = input.recover ? unansweredVersions.current : [];
      // A recovering exit also keeps the confirmed base: an in-flight save may be cancelled unsent.
      const bases = input.recover
        ? [...new Set([...(inFlight === undefined ? [] : [inFlight]), ...candidates, confirmed])]
        : [inFlight !== undefined ? inFlight : confirmed];
      // Past every attempt that may have landed, even ones this save does not name as a base.
      const updatedAt = Math.max(
        Date.now(),
        Math.max(confirmed ?? 0, inFlight ?? 0, ...unansweredVersions.current) + 1,
      );
      inFlightVersion.current = updatedAt;
      try {
        // One conditional write: the server admits it on any listed base in a single statement.
        const [baseUpdatedAt, ...rest] = bases;
        const alsoBase = rest.filter((v): v is number => v !== undefined);
        // The API takes 500; past that (hundreds of failed saves) keep the newest and the confirmed.
        if (alsoBase.length > ALSO_BASE_LIMIT) {
          alsoBase.splice(ALSO_BASE_LIMIT - 1, alsoBase.length - ALSO_BASE_LIMIT);
        }
        const record = await saveRecipe({
          id: input.id ?? recipeId,
          name: input.name,
          payload: input.payload,
          baseUpdatedAt,
          alsoBaseUpdatedAt: alsoBase,
          updatedAt,
          keepalive: input.keepalive,
          known: loadedRef.current ?? undefined,
        });
        // Only this save or an older one is settled; a newer unanswered save stays a candidate.
        unansweredVersions.current = unansweredVersions.current.filter(
          (v) => v > record.updatedAt,
        );
        if (record.updatedAt >= (editedVersion.current ?? 0)) {
          editedVersion.current = record.updatedAt;
        }
        return { id: record.id, updatedAt: record.updatedAt };
      } catch (error) {
        const kept = unansweredVersions.current.filter((v) => v !== updatedAt);
        if (!(error instanceof RecipeApiError)) {
          unansweredVersions.current = [updatedAt, ...kept].sort((a, b) => b - a);
        } else if (error.status === 409) {
          // No listed base matched, so none of them is the stored version.
          unansweredVersions.current = kept.filter((v) => !bases.includes(v));
        }
        throw error;
      } finally {
        if (inFlightVersion.current === updatedAt) {
          inFlightVersion.current = undefined;
        }
      }
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
