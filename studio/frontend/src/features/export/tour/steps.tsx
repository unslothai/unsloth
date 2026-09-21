// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TourStep } from "@/features/tour";

export const exportTourSteps: TourStep[] = [
  {
    id: "training-run",
    target: "export-training-run",
    title: "Pick a run",
    body: (
      <>
        Start here. Each run holds the checkpoints from one fine-tuning job.
        Runs you started on the Train page show up automatically.
      </>
    ),
  },
  {
    id: "checkpoint",
    target: "export-checkpoint",
    title: "Pick a checkpoint",
    body: (
      <>
        The last checkpoint is not always the best one. Export one or two
        candidates and compare them in Chat before you commit.
      </>
    ),
  },
  {
    id: "method",
    target: "export-method",
    title: "Choose a format",
    body: (
      <>
        Merged is a full 16-bit model for vLLM or Transformers. LoRA Only ships
        the adapter alone and needs the base model at inference. GGUF is for
        llama.cpp and Ollama, with a quant level you pick below.
      </>
    ),
  },
  {
    id: "cta",
    target: "export-cta",
    title: "Export",
    body: (
      <>
        Save to this device or push straight to a Hugging Face repo. Anything
        exported locally shows up in Chat under On Device, so you can test it
        against the base model right away.
      </>
    ),
  },
];
