


import { useEffect, useState } from "react";
import { liveQuery } from "dexie";
import { listRecipes } from "../data/recipes-db";
import type { RecipeRecord } from "../types";

export function useRecipeSidebarItems(enabled: boolean) {
  const [recipes, setRecipes] = useState<RecipeRecord[]>([]);

  useEffect(() => {
    if (!enabled) return;
    const sub = liveQuery(() => listRecipes()).subscribe({
      next: (value) => setRecipes(value),
      error: (err) => console.error("recipe sidebar liveQuery:", err),
    });
    return () => sub.unsubscribe();
  }, [enabled]);

  return recipes;
}
