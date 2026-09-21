// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ModelConfigHandoffRequest } from "@/features/model-picker";
import { isExternalModelId } from "../chat/external-providers";
import type { ChatLoraSummary, ChatModelSummary } from "../chat/types/runtime";
import {
  ggufVariantsMatch,
  isStandaloneGgufPath,
  residentModelIdMatches,
} from "../model-picker/model-config/model-identity";
import { type SharedRunConfig, isShareableModelId } from "./links";

const ggufName = /(?:-gguf|\.gguf)$/i;

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
  const ggufVariant =
    value.isGguf === false || isStandaloneGgufPath(id)
      ? undefined
      : (value.ggufVariant ?? selectedVariant ?? undefined);
  return {
    ggufVariant,
    isGguf:
      value.isGguf ??
      Boolean(ggufVariant || (knownFormat ?? ggufName.test(id))),
  };
}

export function resolveRunConfigTarget(
  value: SharedRunConfig,
  selection: ModelSelection,
): Pick<ModelConfigHandoffRequest, "id" | "meta"> | null {
  const id = value.model ?? selection.params.checkpoint;
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
    value,
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
