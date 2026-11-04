// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createContext, useContext } from "react";
import type { NodeConfig } from "../types";
import { useRecipeStudioStore } from "./recipe-studio";

/** Lets a host outside Recipe Studio (a Flows recipe group) hand the block dialogs its own configs. */
export const RecipeConfigsContext = createContext<Record<
  string,
  NodeConfig
> | null>(null);

export function useRecipeConfigs(): Record<string, NodeConfig> {
  const hosted = useContext(RecipeConfigsContext);
  const own = useRecipeStudioStore((state) => state.configs);
  return hosted ?? own;
}
