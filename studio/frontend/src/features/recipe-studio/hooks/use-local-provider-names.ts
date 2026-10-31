// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useMemo } from "react";
import { useRecipeStudioStore } from "../stores/recipe-studio";
import { localProviderNamesKey } from "../utils/node-connection-status";

/** The same Set until the local provider names change, so typing in a node re-renders no other node for it. */
export function useLocalProviderNames(): Set<string> {
  const key = useRecipeStudioStore((state) =>
    localProviderNamesKey(state.configs),
  );
  return useMemo(() => new Set<string>(JSON.parse(key) as string[]), [key]);
}
