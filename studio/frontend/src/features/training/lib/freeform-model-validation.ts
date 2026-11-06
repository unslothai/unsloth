// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  ModelInventoryCapabilities,
  ModelInventoryFormat,
} from "@/features/hub";
import { classifyUnslothSupport, isGgufLike } from "@/features/hub";
import type { TranslationKey } from "@/i18n";

// A single adapter file is not loadable on its own; its folder is.
const ADAPTER_ARTIFACT_PATTERN =
  /(?:^|[/\\])adapter_(?:config\.json|model\.(?:safetensors|bin))$/i;

export type TrainingModelValidationCandidate = {
  id: string;
  modelFormat?: ModelInventoryFormat | null;
  capabilities?: ModelInventoryCapabilities | null;
  pipelineTag?: string | null;
  tags?: readonly string[] | null;
  libraryName?: string | null;
  quantMethod?: string | null;
};

export type TrainingModelValidationResult =
  | { ok: true }
  | { ok: false; reasonKey: TranslationKey; reasonText?: string };

export function validateTrainingModelCandidate(
  candidate: TrainingModelValidationCandidate,
  options: { deviceType?: string | null } = {},
): TrainingModelValidationResult {
  const id = candidate.id.trim();
  if (!id) {
    return { ok: false, reasonKey: "studio.modelPicker.reasonEmptyId" };
  }
  if (candidate.modelFormat === "gguf" || isGgufLike(id)) {
    return { ok: false, reasonKey: "studio.modelPicker.reasonGguf" };
  }
  if (ADAPTER_ARTIFACT_PATTERN.test(id)) {
    return { ok: false, reasonKey: "studio.modelPicker.reasonAdapter" };
  }
  // Adapter / PEFT dirs are allowed here: LoRA and CPT can continue from a saved
  // adapter (#13140). Full finetune is refused later in validateTrainingConfig.
  // The inventory marks adapters canTrain=false (not a base), so skip that check for them.
  if (
    candidate.modelFormat !== "adapter" &&
    candidate.capabilities &&
    !candidate.capabilities.canTrain
  ) {
    return { ok: false, reasonKey: "studio.modelPicker.reasonNotTrainable" };
  }
  const support = classifyUnslothSupport({
    modelId: id,
    pipelineTag: candidate.pipelineTag,
    tags: candidate.tags,
    libraryName: candidate.libraryName,
    quantMethod: candidate.quantMethod,
    deviceType: options.deviceType,
  });
  if (support.status === "unsupported") {
    return {
      ok: false,
      reasonKey: "studio.modelPicker.reasonUnsupportedFormat",
      reasonText: support.reason ?? undefined,
    };
  }
  return { ok: true };
}
