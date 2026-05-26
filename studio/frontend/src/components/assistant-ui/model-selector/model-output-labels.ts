// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { LoraModelOption } from "./types";

export function isStudioTrainedModel(model: LoraModelOption): boolean {
  return (
    !!model.exportType &&
    (model.source === "training" || model.source === "exported")
  );
}

export function trainingMethodLabel(method?: string | null): string | null {
  const normalized = method?.toLowerCase().trim();
  if (!normalized) return null;
  if (normalized === "qlora") return "QLoRA";
  if (normalized === "lora") return "LoRA";
  if (
    normalized === "cpt" ||
    normalized === "continued pretraining" ||
    normalized === "continued_pretraining"
  ) {
    return "CPT";
  }
  if (
    normalized === "full" ||
    normalized === "ft" ||
    normalized === "full finetuning" ||
    normalized === "full fine-tuning" ||
    normalized === "full_finetuning"
  ) {
    return "Full";
  }
  if (normalized === "lora/qlora") return "LoRA/QLoRA";
  return method?.trim() ?? null;
}

export function trainedOutputMeta(model: LoraModelOption): string {
  const method = trainingMethodLabel(model.trainingMethod);
  const format = model.exportType
    ? model.exportType === "gguf"
      ? "GGUF"
      : model.exportType === "lora"
        ? "Adapter"
        : "Safetensors"
    : null;
  if (!format) return method ?? "LoRA";
  return method ? `${method} · ${format}` : format;
}

export function trainedModelDescription(model: LoraModelOption): string {
  const label = trainedOutputMeta(model);
  return model.source === "exported" ? `${label} · Exported` : label;
}
