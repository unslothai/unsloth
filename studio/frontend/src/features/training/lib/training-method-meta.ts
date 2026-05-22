// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TrainingMethod } from "@/types/training";

export const TRAINING_METHOD_LABELS: Record<TrainingMethod, string> = {
  qlora: "QLoRA",
  lora: "LoRA",
  full: "Full fine-tune",
  cpt: "Continued pretraining",
};

export const TRAINING_METHOD_DESCRIPTIONS: Record<TrainingMethod, string> = {
  qlora: "QLoRA (4-bit)",
  lora: "LoRA (16-bit)",
  full: "Full fine-tune",
  cpt: "Continued pretraining",
};

export const TRAINING_METHOD_HINTS: Record<TrainingMethod, string> = {
  qlora: "4-bit quantization. Lowest VRAM, fastest to start.",
  lora: "16-bit adapters. Balanced quality and memory.",
  full: "Trains all weights. Highest quality, needs the most VRAM.",
  cpt: "Continued pretraining for new domains or languages.",
};

export const TRAINING_METHOD_DOTS: Record<TrainingMethod, string> = {
  qlora: "bg-emerald-500/70",
  lora: "bg-sky-500/70",
  full: "bg-amber-500/70",
  cpt: "bg-violet-500/70",
};

export const TRAINING_METHOD_NOTES: Record<TrainingMethod, string> = {
  qlora: "4-bit",
  lora: "16-bit",
  full: "fp16",
  cpt: "continued",
};
