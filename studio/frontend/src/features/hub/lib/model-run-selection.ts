// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ModelConfigHandoffRequest } from "@/features/model-picker";
import type { SelectedModelView } from "../types";
import { detectResultFormat } from "./format-filters.ts";

export interface HubModelRunSelection {
  ggufVariant?: string;
  ggufFilename?: string;
  expectedBytes?: number;
}

function isPresent(value: string | null | undefined): value is string {
  return Boolean(value?.trim());
}

function hasSupportedFormat(model: SelectedModelView): boolean {
  const detectedFormat = detectResultFormat({
    id: model.hubRepoId ?? model.id,
    isGguf: model.isGguf,
    libraryName: model.libraryName,
    tags: model.tags,
  });
  return (
    (model.modelFormat === "gguf" && detectedFormat === "gguf") ||
    (model.modelFormat === "safetensors" && detectedFormat === "safetensors")
  );
}

function hasRunnableInventoryModel(model: SelectedModelView): boolean {
  return (
    model.runtimeCanChat &&
    model.isDownloaded &&
    !model.isPartial &&
    isPresent(model.loadId) &&
    hasSupportedFormat(model)
  );
}

function hasValidSelection(
  model: SelectedModelView,
  selection: HubModelRunSelection,
): boolean {
  const hasGgufVariant = isPresent(selection.ggufVariant);
  const hasGgufFilename = isPresent(selection.ggufFilename);
  if (model.modelFormat === "safetensors") {
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
    isLora: false,
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
  safetensorsRuntimeAvailable,
}: {
  model: SelectedModelView | null;
  isDataset: boolean;
  mediaRuntime: boolean;
  safetensorsRuntimeAvailable: boolean;
}): boolean {
  if (
    !model ||
    isDataset ||
    mediaRuntime ||
    !hasRunnableInventoryModel(model)
  ) {
    return false;
  }

  return (
    model.modelFormat === "gguf" ||
    (model.modelFormat === "safetensors" && safetensorsRuntimeAvailable)
  );
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
  if (!hasRunnableInventoryModel(model)) {
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
