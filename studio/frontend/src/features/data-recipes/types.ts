// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { RecipePayload } from "@/features/recipe-studio";

export type RecipeRecord = {
  id: string;
  name: string;
  payload: RecipePayload;
  revision: number;
  createdAt: number;
  updatedAt: number;
  learningRecipeId?: string | null;
  learningRecipeTitle?: string | null;
  removedCredentialPaths?: string[];
};

export type SaveRecipeInput = {
  id?: string | null;
  name: string;
  payload: RecipePayload;
  revision?: number;
  learningRecipeId?: string | null;
  learningRecipeTitle?: string | null;
};
