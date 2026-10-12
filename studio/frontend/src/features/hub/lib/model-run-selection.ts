// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { AUDIO_CPP_REPO } from "@/features/audio/audio-cpp-catalog";
import { audioWorkflowForPick } from "@/features/audio/route-search";
import type { ModelConfigHandoffRequest } from "@/features/model-picker";
import {
  audioPickIsRoutable,
  curatedAudioInventoryTask,
} from "@/features/model-picker/components/model-selector/audio-picker-policy";
import {
  AUDIO_CATALOG,
  artifactForRepoId,
} from "@/features/model-picker/components/model-selector/model-catalog";
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

/** The task the Audio handoff runs on: offline, a cached curated audio GGUF carries only the
 *  backend's generic text-generation task, which the picker maps back from the catalog too. */
export function hubAudioTask(
  model: SelectedModelView,
  task: string | null,
): string | null {
  const artifact = model.hubRepoId
    ? artifactForRepoId(model.hubRepoId, AUDIO_CATALOG)
    : null;
  return curatedAudioInventoryTask({
    inventoryTask: task,
    isExactCatalogArtifact: artifact !== null,
    catalogScope: artifact?.group.scope,
    catalogTask: artifact?.group.task,
  });
}

/** Whether Run opens this model on the Audio page rather than in chat, which refuses audio models.
 *  Judged on the gate the chat picker routes by, so both surfaces send the same models there. */
export function hubModelRunsOnAudioPage(
  model: SelectedModelView,
  task: string | null,
): boolean {
  // A filesystem row has no Hub id for the Audio handoff, so it keeps its chat Run.
  const id = model.hubRepoId;
  if (!id || !routableToMediaPage(model.kind, model.localSource)) return false;
  // The shared GGUF audio repo holds many models; the Audio page lists them one by one.
  if (id.toLowerCase() === AUDIO_CPP_REPO.toLowerCase()) return false;
  task = hubAudioTask(model, task);
  const audioType = model.audioType ?? null;
  if (audioWorkflowForPick({ id, task, audioType }) === null) return false;
  return audioPickIsRoutable({
    id,
    task,
    isGguf: model.isGguf,
    isCurated: artifactForRepoId(id, AUDIO_CATALOG) !== null,
    // An on-device GGUF is tagged text-to-audio only by the audio runtime's header classifier.
    taskFromGgufArch: model.isGguf && model.task === "text-to-audio",
    baseModel: model.baseModel,
    tags: model.tags,
    libraryName: model.libraryName,
    audioType,
  });
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
