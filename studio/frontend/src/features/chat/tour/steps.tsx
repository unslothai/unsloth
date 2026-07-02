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
          Selects what’s loaded for inference. Recommended is Unsloth’s curated
          base models; On Device is your downloads and fine-tuned outputs (LoRA
          adapters and full finetunes).
        </>
      ),
    },
    {
      id: "model-tabs",
      target: "chat-model-selector-popover",
      title: "Find a model",
      body: (
        <>
          Search Unsloth’s models, or hit Search Hub for all of Hugging Face.
          Switch Recommended and On Device, filter by format, and sort by
          trending or recent. An OOM tag means it won’t fit in your VRAM.
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
    {
      id: "plus-menu",
      target: "chat-plus-menu",
      title: "The + menu",
      body: (
        <>
          Everything else lives here: attach photos and files, reuse saved
          prompts, toggle tools and MCP, start a side-by-side compare, and
          export the chat.
        </>
      ),
    },
  ];

  if (canCompare) {
    // Compare lives in the + menu (no sidebar button to anchor to); this step
    // enters compare on its own and explains it.
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
