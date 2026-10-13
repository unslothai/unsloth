// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TourStep } from "@/features/tour";

/** Without an Easy form the tabs read Editor and Runs, so neither "three" nor "Advanced" fits. */
function viewsStep(supportsEasyMode: boolean): TourStep {
  return {
    id: "views",
    target: "recipe-views",
    title: supportsEasyMode ? "Three views" : "Two views",
    body: supportsEasyMode ? (
      <>
        Easy is a plain form, Advanced is the node graph, and Runs keeps every
        execution with the dataset it produced.
      </>
    ) : (
      <>
        Editor is the node graph. Runs keeps every execution with the dataset it
        produced.
      </>
    ),
  };
}

const saveStep: TourStep = {
  id: "save",
  target: "recipe-save",
  title: "Save",
  body: <>Recipes are stored on this device. Save before a long run.</>,
};

/** Only the graph view mounts the canvas and its controls, and only once the recipe has loaded. */
export function buildRecipeEditorTourSteps({
  isGraphView,
  supportsEasyMode,
}: {
  isGraphView: boolean;
  supportsEasyMode: boolean;
}): TourStep[] {
  if (!isGraphView) {
    return [viewsStep(supportsEasyMode), saveStep];
  }

  return [
    viewsStep(supportsEasyMode),
    {
      id: "canvas",
      target: "recipe-canvas",
      title: "The graph",
      body: (
        <>
          Each node is one step, and the wires carry rows forward. A source node
          in, a dataset you can train on out.
        </>
      ),
    },
    {
      id: "add-step",
      target: "recipe-add-step",
      title: "Add a step",
      body: (
        <>
          Source data, samplers, model calls, validators and expressions. Import
          is where a PDF or a folder of text enters the graph.
        </>
      ),
    },
    {
      id: "run",
      target: "recipe-run",
      title: "Check and run",
      body: (
        <>
          Check reports broken wires and missing settings without spending
          tokens. Run lands the result in Runs, ready for the Train page.
        </>
      ),
    },
  ];
}
