// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type ChatLoraSummary,
  type ChatModelSummary,
  isExternalModelId,
} from "@/features/chat";
import { looksLikeLocalPath } from "@/lib/local-path";
import type { ModelConfigHandoffRequest } from "../model-config/model-config-handoff";
import {
  ggufVariantsMatch,
  isOllamaModelId,
  isStandaloneGgufPath,
  residentModelIdMatches,
} from "../model-config/model-identity";
import { type SharedRunConfig, isShareableModelId } from "./links";

const ggufName = /(?:-gguf|\.gguf)$/i;
const invalidCharacters = /[\p{Cc}\p{Cs}]/u;

export function isRunConfigModelInput(model: string): boolean {
  return (
    model.length > 0 &&
    model === model.trim() &&
    !invalidCharacters.test(model) &&
    !model.includes("://") &&
    (isShareableModelId(model) ||
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

function resolveFormat(
  value: SharedRunConfig,
  id: string,
  knownFormat: boolean | null | undefined,
  selectedVariant: string | null,
) {
  const format = value.model ? value.isGguf : (knownFormat ?? value.isGguf);
  const ggufVariant =
    format === false || isStandaloneGgufPath(id)
      ? undefined
      : (value.ggufVariant ?? selectedVariant ?? undefined);
  return {
    ggufVariant,
    isGguf:
      format ?? Boolean(ggufVariant || (knownFormat ?? ggufName.test(id))),
  };
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
  const sameModel = residentModelIdMatches(id, selection.params.checkpoint);
  const model = selection.models.find((entry) =>
    residentModelIdMatches(id, entry.id),
  );
  const lora = selection.loras.find((entry) =>
    residentModelIdMatches(id, entry.id),
  );
  const loraFormat = lora?.exportType && lora.exportType === "gguf";
  const knownFormat =
    (sameModel ? selection.loadedIsGguf : null) ?? model?.isGguf ?? loraFormat;
  const { isGguf, ggufVariant } = resolveFormat(
    selectedModel && (isStandaloneGgufPath(id) || isOllamaModelId(id))
      ? { ...value, isGguf: true, ggufVariant: undefined }
      : value,
    id,
    knownFormat,
    sameModel ? selection.activeGgufVariant : null,
  );
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
