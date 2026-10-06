// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TourStep } from "@/features/tour";

export function buildChatTourSteps({
  canShowNav,
  canCompare,
  openModelSelector,
  closeModelSelector,
  openSettings,
  closeSettings,
  enterCompare,
  exitCompare,
}: {
  canShowNav: boolean;
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
          Loads a model for inference. Runs local GGUF, safetensors and your own
          LoRA adapters, plus any cloud provider you add in Settings, such as
          Gemini, OpenAI, Anthropic or OpenRouter.
        </>
      ),
    },
    {
      id: "model-tabs",
      target: "chat-model-selector-popover",
      title: "Find a model",
      body: (
        <>
          Recommended is Unsloth's curated list, On Device is your downloads and
          finetunes. Search Hub reaches all of Hugging Face. An OOM tag means it
          will not fit in your VRAM.
        </>
      ),
      onEnter: openModelSelector,
      onExit: closeModelSelector,
    },
    {
      id: "plus-menu",
      target: "chat-plus-menu",
      title: "Tools and attachments",
      body: (
        <>
          Open this to attach PDFs, images, audio and code, or to switch on web
          search, the sandboxed Bash and Python tools, MCP servers and skills.
        </>
      ),
    },
    {
      id: "settings",
      target: "chat-settings",
      title: "Run settings",
      body: (
        <>
          Temperature, top-p, top-k, system prompt and the chat template. Lower
          temperature first when you want steadier answers.
        </>
      ),
      onEnter: openSettings,
      onExit: closeSettings,
    },
  ];

  if (canShowNav) {
    // The mobile sidebar is a closed sheet, so there is nothing to spotlight there.
    steps.unshift({
      id: "nav",
      target: "navbar",
      title: "Where everything lives",
      body: (
        <>
          Chat runs models. Train fine-tunes them, Recipes turns documents into
          datasets, and Export packages the result. Images, Video and Audio are
          their own workspaces, and Model hub manages what is on this device.
        </>
      ),
    });
  }

  if (canCompare) {
    // Compare lives in the + menu, with no sidebar button to anchor to; this step enters compare on
    // its own and explains it.
    steps.push({
      id: "compare-view",
      target: "chat-compare-view",
      title: "Compare two models",
      body: (
        <>
          One prompt, two threads, side by side. The quickest way to check a
          finetune against its base model. If yours is worse, suspect dataset
          formatting, too many epochs or the wrong checkpoint.
        </>
      ),
      onEnter: enterCompare,
      onExit: exitCompare,
    });
  }

  return steps;
}
