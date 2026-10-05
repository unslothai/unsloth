// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ModelConfigHandoffRequest } from "@/features/model-picker";
import type { SelectedModelView } from "../types";
import { EMBEDDING_TAGS } from "./hf-model-meta.ts";
import { routableToMediaPage } from "./local-path.ts";

export interface HubModelRunSelection {
  ggufVariant?: string;
  ggufFilename?: string;
  expectedBytes?: number;
}

function isPresent(value: string | null | undefined): value is string {
  return Boolean(value?.trim());
}

function hasSupportedFormat(model: SelectedModelView): boolean {
  if (model.modelFormat === "gguf") {
    return model.isGguf;
  }
  return (
    (model.modelFormat === "safetensors" ||
      model.modelFormat === "checkpoint" ||
      model.modelFormat === "adapter") &&
    !model.isGguf
  );
}

const GENERATIVE_CAPABILITIES = new Set([
  "conversational",
  "tools",
  "reasoning",
  "code",
  "vision",
  "audio",
  "diffusion",
]);

function isEmbeddingOnly(model: SelectedModelView): boolean {
  if (
    model.isGguf ||
    !model.capabilities.some((capability) => capability.key === "embedding")
  ) {
    return false;
  }
  if (EMBEDDING_TAGS.has(model.pipelineTag?.trim().toLowerCase() ?? "")) {
    return true;
  }
  return !model.capabilities.some((capability) =>
    GENERATIVE_CAPABILITIES.has(capability.key),
  );
}

function hasCompleteInventoryModel(model: SelectedModelView): boolean {
  return model.isDownloaded && !model.isPartial && isPresent(model.loadId);
}

function hasRunnableChatModel(model: SelectedModelView): boolean {
  return (
    model.runtimeCanChat &&
    hasCompleteInventoryModel(model) &&
    !isEmbeddingOnly(model) &&
    hasSupportedFormat(model)
  );
}

function hasValidSelection(
  model: SelectedModelView,
  selection: HubModelRunSelection,
): boolean {
  const hasGgufVariant = isPresent(selection.ggufVariant);
  const hasGgufFilename = isPresent(selection.ggufFilename);
  if (model.modelFormat !== "gguf") {
    return !(hasGgufVariant || hasGgufFilename);
  }
  return !model.requiresVariant || hasGgufVariant;
}

function createHandoffMeta(
  model: SelectedModelView,
  selection: HubModelRunSelection,
  usesLocalIdentity: boolean,
): ModelConfigHandoffRequest["meta"] {
  const meta: ModelConfigHandoffRequest["meta"] = {
    source: usesLocalIdentity ? "local" : "hub",
    isLora: model.modelFormat === "adapter",
    loadId: model.loadId,
    isDownloaded: true,
    isGguf: model.modelFormat === "gguf",
    pipelineTag: model.task ?? model.pipelineTag ?? null,
  };
  if (isPresent(selection.ggufVariant)) {
    meta.ggufVariant = selection.ggufVariant;
  }
  if (isPresent(selection.ggufFilename)) {
    meta.ggufFilename = selection.ggufFilename;
  }
  if (
    selection.expectedBytes != null &&
    Number.isFinite(selection.expectedBytes) &&
    selection.expectedBytes > 0
  ) {
    meta.expectedBytes = selection.expectedBytes;
  }
  return meta;
}

export function isHubModelRunEligible({
  model,
  isDataset,
  mediaRuntime,
  nonGgufRuntimeAvailable,
}: {
  model: SelectedModelView | null;
  isDataset: boolean;
  mediaRuntime: boolean;
  nonGgufRuntimeAvailable: boolean;
}): boolean {
  if (!model || isDataset) {
    return false;
  }

  if (mediaRuntime) {
    return (
      hasCompleteInventoryModel(model) &&
      isPresent(model.hubRepoId) &&
      routableToMediaPage(model.kind, model.localSource)
    );
  }

  if (!hasRunnableChatModel(model)) {
    return false;
  }

  return model.modelFormat === "gguf" || nonGgufRuntimeAvailable;
}

export function createHubModelConfigHandoff({
  requestId,
  model,
  selection,
}: {
  requestId: string;
  model: SelectedModelView;
  selection: HubModelRunSelection;
}): ModelConfigHandoffRequest | null {
  if (!isPresent(requestId)) {
    return null;
  }
  if (!hasRunnableChatModel(model)) {
    return null;
  }

  if (!hasValidSelection(model, selection)) {
    return null;
  }

  const usesLocalIdentity = model.isLocal && model.localSource !== "hf_cache";
  const id = usesLocalIdentity ? model.id : (model.hubRepoId ?? model.id);
  if (!isPresent(id)) {
    return null;
  }

  return {
    requestId,
    id,
    displayName: model.title,
    meta: createHandoffMeta(model, selection, usesLocalIdentity),
  };
}
