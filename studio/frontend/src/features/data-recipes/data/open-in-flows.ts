// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { importRecipe } from "@/features/flows/api";
import { toastError } from "@/shared/toast";
import type { useNavigate } from "@tanstack/react-router";
import { toast } from "sonner";

/** The saved recipe as a Flows recipe group (the same flow again while the recipe is unchanged). */
export async function openRecipeInFlows(
  recipeId: string,
  navigate: ReturnType<typeof useNavigate>,
): Promise<void> {
  try {
    const { flow, report } = await importRecipe({ recipe_id: recipeId });
    for (const notice of report.notices) toast.info(notice);
    await navigate({ to: "/flows/$flowId", params: { flowId: flow.id } });
  } catch (error) {
    toastError(
      "Could not open the recipe in Flows",
      error instanceof Error ? error.message : undefined,
    );
  }
}
