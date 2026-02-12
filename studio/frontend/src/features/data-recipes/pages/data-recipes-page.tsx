import { Button } from "@/components/ui/button";
import {
  Empty,
  EmptyContent,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from "@/components/ui/empty";
import { CookBookIcon, Delete02Icon, PlusSignIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useNavigate } from "@tanstack/react-router";
import type { ReactElement } from "react";
import { useState } from "react";
import { createRecipeDraft, deleteRecipe, useRecipes } from "../data/recipes-db";

function formatRelativeTime(value: number): string {
  const now = Date.now();
  const diffMs = Math.max(0, now - value);
  const minute = 60 * 1000;
  const hour = 60 * minute;
  const day = 24 * hour;
  const week = 7 * day;

  if (diffMs < minute) {
    return "just now";
  }
  if (diffMs < hour) {
    const minutes = Math.floor(diffMs / minute);
    return `${minutes} minute${minutes === 1 ? "" : "s"} ago`;
  }
  if (diffMs < day) {
    const hours = Math.floor(diffMs / hour);
    return `${hours} hour${hours === 1 ? "" : "s"} ago`;
  }
  if (diffMs < week) {
    const days = Math.floor(diffMs / day);
    return `${days} day${days === 1 ? "" : "s"} ago`;
  }
  const weeks = Math.floor(diffMs / week);
  return `${weeks} week${weeks === 1 ? "" : "s"} ago`;
}

export function DataRecipesPage(): ReactElement {
  const navigate = useNavigate();
  const recipes = useRecipes();
  const [creatingRecipe, setCreatingRecipe] = useState(false);

  async function openNewRecipe(): Promise<void> {
    if (creatingRecipe) {
      return;
    }
    setCreatingRecipe(true);
    try {
      const recipe = await createRecipeDraft();
      await navigate({
        to: "/data-recipes/$recipeId",
        params: { recipeId: recipe.id },
      });
    } finally {
      setCreatingRecipe(false);
    }
  }

  function openRecipe(recipeId: string): void {
    void navigate({
      to: "/data-recipes/$recipeId",
      params: { recipeId },
    });
  }

  async function handleDeleteRecipe(recipeId: string): Promise<void> {
    await deleteRecipe(recipeId);
  }

  return (
    <div className="min-h-screen bg-background">
      <main className="mx-auto w-full max-w-7xl px-6 py-8">
        <div className="flex items-center justify-between gap-4">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">Data Recipes</h1>
            <p className="mt-1 text-sm text-muted-foreground">
              Create and manage local recipe workflows.
            </p>
          </div>
          <Button
            type="button"
            onClick={() => {
              void openNewRecipe();
            }}
            disabled={creatingRecipe}
          >
            <HugeiconsIcon icon={PlusSignIcon} className="size-4" />
            New Recipe
          </Button>
        </div>

        {recipes.length === 0 ? (
          <Empty className="mt-8 border border-dashed border-border/70">
            <EmptyHeader>
              <EmptyMedia variant="icon">
                <HugeiconsIcon icon={CookBookIcon} className="size-5" />
              </EmptyMedia>
              <EmptyTitle>No recipes yet</EmptyTitle>
              <EmptyDescription>
                Create your first recipe to start building workflows.
              </EmptyDescription>
            </EmptyHeader>
            <EmptyContent>
              <Button
                type="button"
                onClick={() => {
                  void openNewRecipe();
                }}
                disabled={creatingRecipe}
              >
                <HugeiconsIcon icon={PlusSignIcon} className="size-4" />
                Create Recipe
              </Button>
            </EmptyContent>
          </Empty>
        ) : (
          <div className="mt-8 space-y-2">
            {recipes.map((recipe) => (
              <div
                key={recipe.id}
                className="flex items-center gap-3 rounded-xl border bg-card px-4 py-3"
              >
                <button
                  type="button"
                  className="flex min-w-0 flex-1 items-center gap-3 text-left"
                  onClick={() => openRecipe(recipe.id)}
                >
                  <div className="flex size-9 shrink-0 items-center justify-center rounded-lg border border-border/70 bg-muted/20">
                    <HugeiconsIcon icon={CookBookIcon} className="size-4 text-muted-foreground" />
                  </div>
                  <div className="min-w-0">
                    <p className="truncate text-sm font-medium">{recipe.name}</p>
                    <p className="text-xs text-muted-foreground">
                      Last updated {formatRelativeTime(recipe.updatedAt)} | Created{" "}
                      {formatRelativeTime(recipe.createdAt)}
                    </p>
                  </div>
                </button>
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  className="size-8"
                  onClick={() => void handleDeleteRecipe(recipe.id)}
                  aria-label={`Delete ${recipe.name}`}
                >
                  <HugeiconsIcon icon={Delete02Icon} className="size-4" />
                </Button>
              </div>
            ))}
          </div>
        )}
      </main>
    </div>
  );
}
