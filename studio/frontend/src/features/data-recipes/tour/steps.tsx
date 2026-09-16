// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TourStep } from "@/features/tour";

/** The list and the learning-recipe cards are two branches of the same slot, so only one exists. */
export function buildDataRecipesTourSteps({
  hasRecipes,
}: {
  hasRecipes: boolean;
}): TourStep[] {
  return [
    {
      id: "new",
      target: "recipes-new",
      title: "Build a dataset",
      body: (
        <>
          A recipe turns raw documents into training data. Start empty, or start
          from a learning recipe that is already wired up for a common job.
        </>
      ),
    },
    hasRecipes
      ? {
          id: "list",
          target: "recipes-list",
          title: "Your recipes",
          body: (
            <>
              Open one to edit its steps or rerun it. Recipes are saved on this
              device, and the dataset a run produces is ready to pick on the
              Train page.
            </>
          ),
        }
      : {
          id: "templates",
          target: "recipes-templates",
          title: "Learning recipes",
          body: (
            <>
              Worked examples you can open and run as they are. The fastest way
              to see how nodes connect before you build one yourself.
            </>
          ),
        },
  ];
}
