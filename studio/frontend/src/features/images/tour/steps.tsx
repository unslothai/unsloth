// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TourStep } from "@/features/tour";

const modeStep: TourStep = {
  id: "mode",
  target: "images-mode",
  title: "Create or Train",
  body: (
    <>
      Create generates images. Train fine-tunes a diffusion LoRA on your own
      pictures. The workflows in the sidebar, such as edit, inpaint, extend and
      upscale, all run in Create.
    </>
  ),
};

/** Create and Train swap the whole page body, so each mode gets the anchors it actually has. */
export function buildImagesTourSteps({
  pageMode,
}: {
  pageMode: "create" | "train";
}): TourStep[] {
  if (pageMode === "train") {
    return [
      modeStep,
      {
        id: "train-dataset",
        target: "images-train-dataset",
        title: "Your images",
        body: (
          <>
            Drop in a folder, pick a dataset, or start from an example. Captions
            matter here, so use the labeling grid to fill in anything missing
            before you train.
          </>
        ),
      },
      {
        id: "train-start",
        target: "images-train-start",
        title: "Start training",
        body: (
          <>
            Runs a LoRA on the base model above. Progress and sample images
            appear on the right, and finished LoRAs load straight back into
            Create.
          </>
        ),
      },
    ];
  }

  return [
    modeStep,
    {
      id: "model",
      target: "images-model",
      title: "Pick a model",
      body: (
        <>
          Image models you have downloaded, plus anything on Hugging Face. The
          model decides which controls appear below, such as LoRAs and
          ControlNet.
        </>
      ),
    },
    {
      id: "prompt",
      target: "images-prompt",
      title: "Prompt",
      body: (
        <>
          Describe what you want. In the edit and inpaint workflows this becomes
          an instruction about the source image instead.
        </>
      ),
    },
    {
      id: "settings",
      target: "images-settings",
      title: "Settings",
      body: (
        <>
          Aspect ratio, steps, guidance, batch size and seed. Save a setup as a
          preset to reuse it. Reuse a seed to repeat a result exactly.
        </>
      ),
    },
    {
      id: "generate",
      target: "images-generate",
      title: "Generate",
      body: (
        <>
          Starts the run and turns into Stop while it works. The model loads on
          first use, so the first image takes longer than the rest.
        </>
      ),
    },
    {
      id: "preview",
      target: "images-preview",
      title: "Results",
      body: (
        <>
          The selected image fills this pane, and everything you make lines up
          in the strip below. From here you can save it, or send it into another
          workflow like upscale.
        </>
      ),
    },
  ];
}
