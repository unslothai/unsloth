// SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
// Copyright © 2025 Unsloth AI

import type { RecipePayload } from "@/features/recipe-studio";

export type RecipeRecord = {
  id: string;
  name: string;
  payload: RecipePayload;
  createdAt: number;
  updatedAt: number;
  learningRecipeId?: string;
  learningRecipeTitle?: string;
};

export type SaveRecipeInput = {
  id?: string | null;
  name: string;
  payload: RecipePayload;
  learningRecipeId?: string;
  learningRecipeTitle?: string;
};
