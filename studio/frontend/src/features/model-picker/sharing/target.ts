// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type ChatLoraSummary,
  type ChatModelSummary,
  isExternalModelId,
} from "@/features/chat";
import { isValidRepoId as isShareableModelId } from "@/features/deep-links";
import { looksLikeLocalPath } from "@/lib/local-path";
import type { ModelConfigHandoffRequest } from "../model-config/model-config-handoff";
import {
  ggufVariantsMatch,
  isOllamaModelId,
  isStandaloneGgufPath,
  residentModelIdMatches,
} from "../model-config/model-identity";
import type { SharedRunConfig } from "./links";

const invalidCharacters = /[\p{Cc}\p{Cs}]/u;

export function isRunConfigModelInput(model: string): boolean {
  return (
    model.length > 0 &&
    model === model.trim() &&
    !invalidCharacters.test(model) &&
    !model.includes("://") &&
    (!model.includes(":") ||
      looksLikeLocalPath(model) ||
      isStandaloneGgufPath(model) ||
      isOllamaModelId(model))
  );
}

type ModelSelection = {
  params: { checkpoint: string };
  activeGgufVariant: string | null;
  loadedIsGguf: boolean | null;
  activeNativePathToken: string | null;
  activeLoadId: string | null;
  models: readonly Pick<ChatModelSummary, "id" | "isGguf" | "isLora">[];
  loras: readonly Pick<ChatLoraSummary, "id" | "exportType">[];
};

function knownModel(id: string, selection: ModelSelection) {
  const sameModel = residentModelIdMatches(id, selection.params.checkpoint);
  const model = selection.models.find((entry) =>
    residentModelIdMatches(id, entry.id),
  );
  const lora = selection.loras.find((entry) =>
    residentModelIdMatches(id, entry.id),
  );
  return {
    sameModel,
    model,
    lora,
    isGguf:
      (sameModel ? selection.loadedIsGguf : null) ??
      model?.isGguf ??
      (lora?.exportType ? lora.exportType === "gguf" : undefined),
  };
}

export function isKnownNonGgufModel(
  id: string,
  selection: ModelSelection,
): boolean {
  return (
    id !== "" &&
    !isStandaloneGgufPath(id) &&
    knownModel(id, selection).isGguf === false
  );
}

export function resolveRunConfigTarget(
  value: SharedRunConfig,
  selection: ModelSelection,
  selectedModel?: string,
): Pick<ModelConfigHandoffRequest, "id" | "meta"> | null {
  const id = selectedModel ?? value.model ?? selection.params.checkpoint;
  if (!id || isExternalModelId(id)) {
    return null;
  }
  const { sameModel, model, lora, isGguf: knownFormat } = knownModel(
    id,
    selection,
  );
  const singleFile =
    isStandaloneGgufPath(id) ||
    (Boolean(selectedModel) && isOllamaModelId(id));
  const isGguf = singleFile || Boolean(value.model) || knownFormat !== false;
  const ggufVariant =
    isGguf && !singleFile
      ? (value.ggufVariant ??
        (sameModel ? selection.activeGgufVariant : null) ??
        undefined)
      : undefined;
  const sameArtifact =
    sameModel &&
    selection.loadedIsGguf === isGguf &&
    (isStandaloneGgufPath(id) ||
      ggufVariantsMatch(ggufVariant, selection.activeGgufVariant));
  return {
    id,
    meta: {
      source: isShareableModelId(id) ? "hub" : "local",
      isLora: model?.isLora ?? lora?.exportType === "lora",
      isGguf,
      ggufVariant,
      ...(sameArtifact
        ? {
            loadId: selection.activeLoadId ?? selection.params.checkpoint,
            isDownloaded: true,
          }
        : {}),
      ...(sameArtifact && isGguf && selection.activeNativePathToken
        ? { nativePathToken: selection.activeNativePathToken }
        : {}),
    },
  };
}
