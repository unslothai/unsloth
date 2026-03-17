// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TourStep } from "@/features/tour";

export const exportTourSteps: TourStep[] = [
  {
    id: "training-run",
    target: "export-training-run",
    title: "Pick training run",
    body: (
      <>
        Start by selecting the training run. Each run groups the checkpoints
        produced by that specific fine-tuning job.
      </>
    ),
  },
  {
    id: "checkpoint",
    target: "export-checkpoint",
    title: "Pick checkpoint",
    body: (
      <>
        Pick which checkpoint to export. If you trained multiple checkpoints,
        it’s worth exporting 1-2 candidates and testing in Chat.
      </>
    ),
  },
  {
    id: "method",
    target: "export-method",
    title: "Export method",
    body: (
      <>
        Choose the packaging. GGUF is for llama.cpp-style runtimes (pick a
        quant). Safetensors is for HF/Transformers-style usage. If you’re unsure,
        start with safetensors.
      </>
    ),
  },
  {
    id: "cta",
    target: "export-cta",
    title: "Export",
    body: (
      <>
        Export to local or push to HF Hub. After export, test in Chat and compare
        against base to confirm behavior is what you expect.
      </>
    ),
  },
];
