// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TourStep } from "@/features/tour";

const viewsStep: TourStep = {
  id: "views",
  target: "recipe-views",
  title: "Three views",
  body: (
    <>
      Easy is a plain form for simple recipes, where the recipe offers one.
      Advanced is the node graph. Runs keeps every execution with a preview of
      the dataset it produced.
    </>
  ),
};

const saveStep: TourStep = {
  id: "save",
  target: "recipe-save",
  title: "Save",
  body: (
    <>
      Recipes are stored on this device. Save before a long run so you can come
      back to the same graph.
    </>
  ),
};

/** Only the graph view mounts the canvas and its floating controls. */
export function buildRecipeEditorTourSteps({
  isGraphView,
}: {
  isGraphView: boolean;
}): TourStep[] {
  if (!isGraphView) {
    return [viewsStep, saveStep];
  }

  return [
    viewsStep,
    {
      id: "canvas",
      target: "recipe-canvas",
      title: "The graph",
      body: (
        <>
          Each node is one step, and the wires between them carry rows forward.
          Data starts at a source node and comes out the other end as a dataset
          you can train on.
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
          brings in your own documents, so this is where a PDF or a folder of
          text enters the graph.
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
          tokens. Run executes the graph, and the result lands in Runs, ready to
          use on the Train page.
        </>
      ),
    },
    saveStep,
  ];
}
