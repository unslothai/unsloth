// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TourStep } from "@/features/tour";

export function buildChatTourSteps({
  canCompare,
  openModelSelector,
  closeModelSelector,
  openSettings,
  closeSettings,
  enterCompare,
  exitCompare,
}: {
  canCompare: boolean;
  openModelSelector: () => void;
  closeModelSelector: () => void;
  openSettings: () => void;
  closeSettings: () => void;
  enterCompare: () => void;
  exitCompare: () => void;
}): TourStep[] {
  const steps: TourStep[] = [
    {
      id: "model",
      target: "chat-model-selector",
      title: "Pick a model",
      body: (
        <>
          This selects what’s loaded for inference. Hub = base models. Fine-tuned
          = trained Studio outputs, including LoRA adapters and full finetunes.
        </>
      ),
    },
    {
      id: "model-tabs",
      target: "chat-model-selector-popover",
      title: "Two tabs",
      body: (
        <>
          Hub: search Hugging Face models. Fine-tuned: local Studio outputs you’ve
          trained or exported. If results look off, compare base vs fine-tuned
          outputs to see what changed.
        </>
      ),
      onEnter: openModelSelector,
      onExit: closeModelSelector,
    },
    {
      id: "settings",
      target: "chat-settings",
      title: "Settings sidebar",
      body: (
        <>
          Sampling (temperature/top-p/top-k) + system prompt live here. If you
          want more deterministic outputs, lower temperature first.
        </>
      ),
      onEnter: openSettings,
      onExit: closeSettings,
    },
  ];

  if (canCompare) {
    // Compare now lives in the + menu, so there is no sidebar button to anchor
    // to; the view step enters compare on its own and explains it.
    steps.push({
      id: "compare-view",
      target: "chat-compare-view",
      title: "Side-by-side threads",
      body: (
        <>
          Compare any two models side-by-side, available from the + menu. Same
          prompt, 2 threads. If LoRA is worse than base, it’s usually data
          formatting, too many epochs, or a bad checkpoint choice.
        </>
      ),
      onEnter: enterCompare,
      onExit: exitCompare,
    });
  }

  return steps;
}
