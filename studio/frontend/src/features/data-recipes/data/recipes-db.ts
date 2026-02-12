import Dexie, { type EntityTable, liveQuery } from "dexie";
import type { RecipePayload } from "@/features/recipe-studio";
import { useEffect, useState } from "react";
import type { RecipeRecord, SaveRecipeInput } from "../types";

const db = new Dexie("unsloth-data-recipes") as Dexie & {
  recipes: EntityTable<RecipeRecord, "id">;
};

db.version(1).stores({
  recipes: "id, name, updatedAt, createdAt",
});

function normalizeRecipeName(name: string): string {
  const trimmed = name.trim();
  return trimmed.length > 0 ? trimmed : "Unnamed";
}

function createEmptyPayload(): RecipePayload {
  return {
    recipe: {
      // biome-ignore lint/style/useNamingConvention: api schema
      model_providers: [],
      // biome-ignore lint/style/useNamingConvention: api schema
      mcp_providers: [],
      // biome-ignore lint/style/useNamingConvention: api schema
      model_configs: [],
      // biome-ignore lint/style/useNamingConvention: api schema
      tool_configs: [],
      columns: [],
      processors: [],
    },
    run: {
      rows: 5,
      preview: true,
      // biome-ignore lint/style/useNamingConvention: api schema
      output_formats: ["jsonl"],
    },
    ui: {
      nodes: [],
      edges: [],
    },
  };
}

export async function listRecipes(): Promise<RecipeRecord[]> {
  return db.recipes.orderBy("updatedAt").reverse().toArray();
}

export async function getRecipe(id: string): Promise<RecipeRecord | undefined> {
  return db.recipes.get(id);
}

export async function saveRecipe(input: SaveRecipeInput): Promise<RecipeRecord> {
  const now = Date.now();
  const id = input.id ?? crypto.randomUUID();
  const existing = input.id ? await db.recipes.get(input.id) : undefined;
  const record: RecipeRecord = {
    id,
    name: normalizeRecipeName(input.name),
    payload: input.payload,
    createdAt: existing?.createdAt ?? now,
    updatedAt: now,
  };
  await db.recipes.put(record);
  return record;
}

export async function deleteRecipe(id: string): Promise<void> {
  await db.recipes.delete(id);
}

export async function createRecipeDraft(): Promise<RecipeRecord> {
  return saveRecipe({
    name: "Unnamed",
    payload: createEmptyPayload(),
  });
}

export function useRecipes(): RecipeRecord[] {
  const [recipes, setRecipes] = useState<RecipeRecord[]>([]);

  useEffect(() => {
    const sub = liveQuery(() => listRecipes()).subscribe({
      next: (value) => setRecipes(value),
      error: (error) => console.error("data-recipes liveQuery:", error),
    });
    return () => sub.unsubscribe();
  }, []);

  return recipes;
}
