// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useRecipes } from "../data/recipes-db";
import type { RecipeRecord } from "../types";

export function useRecipeSidebarItems(enabled: boolean): RecipeRecord[] {
  return useRecipes(enabled).recipes;
}
