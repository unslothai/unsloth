// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// eslint-disable-next-line no-restricted-imports -- Avoid the hub barrel's React and download-manager exports.
import {
  isHfCacheSnapshotPath,
  isOllamaModelId,
  modelIdsMatch,
  publicModelId,
} from "@/features/hub/lib/model-identity";
import type {
  LoraModelOption,
  ModelSelectorChangeMeta,
} from "@/features/model-picker/components/model-selector/types";
import {
  ggufQuantLabel,
  ggufVariantsMatch,
  normalizeGgufVariantIdentity,
  residentModelIdMatches,
} from "../../model-picker/model-config/model-identity";
import { resolveOnlyRememberedGgufVariant } from "../../model-picker/model-config/per-model-config";
import { isExternalModelId } from "../external-providers";

export type ChatModelSwitchTarget = {
  modelId: string;
  ggufVariant?: string | null;
};

// External and Ollama ids are opaque and case-sensitive: folding them would merge distinct models.
function isExactOnlyIdentity(id: string): boolean {
  return isExternalModelId(id) || isOllamaModelId(id);
}

/** Same HF cache repo: snapshot path vs repo id, either direction, or two snapshots of one repo. */
function sameHfCacheIdentity(left: string, right: string): boolean {
  if (left === right) {
    return true;
  }
  if (isExactOnlyIdentity(left) || isExactOnlyIdentity(right)) {
    return false;
  }
  if (
    residentModelIdMatches(left, right) ||
    residentModelIdMatches(right, left)
  ) {
    return true;
  }
  if (!(isHfCacheSnapshotPath(left) && isHfCacheSnapshotPath(right))) {
    return false;
  }
  const leftRepo = publicModelId(left);
  return (
    leftRepo.includes("/") && modelIdsMatch(leftRepo, publicModelId(right))
  );
}

export function chatModelIsResident(
  createdModel: ChatModelSwitchTarget,
  checkpoint: string,
  activeGgufVariant: string | null,
): boolean {
  if (!sameHfCacheIdentity(checkpoint, createdModel.modelId)) {
    return false;
  }
  return (
    createdModel.ggufVariant == null ||
    ggufVariantsMatch(createdModel.ggufVariant, activeGgufVariant)
  );
}

/** The live picker id Switch Back loads: the exact id, else a row of the same HF cache repo. */
export function chatModelSelectableId(
  modelId: string,
  selectableModelIds: ReadonlySet<string>,
): string | null {
  if (selectableModelIds.has(modelId)) {
    return modelId;
  }
  for (const id of selectableModelIds) {
    if (sameHfCacheIdentity(id, modelId)) {
      return id;
    }
  }
  return null;
}

type ChatModelThreadSnapshot = {
  id: string;
  modelId?: string | null;
  modelGgufVariant?: string | null;
};

export function resolveChatModelSwitchTarget(
  target: ChatModelSwitchTarget,
): ChatModelSwitchTarget {
  if (target.ggufVariant) {
    return target;
  }
  const remembered = resolveOnlyRememberedGgufVariant(target.modelId);
  const basename = target.modelId.replace(/\\/g, "/").split("/").pop() ?? "";
  if (
    !remembered ||
    normalizeGgufVariantIdentity(ggufQuantLabel(`${basename}.gguf`)) !==
      normalizeGgufVariantIdentity(remembered.ggufVariant)
  ) {
    return target;
  }
  return { ...target, ggufVariant: remembered.ggufVariant };
}

function chatModelSwitchTargetFromThread(
  thread: ChatModelThreadSnapshot | null | undefined,
): ChatModelSwitchTarget | null {
  return thread?.modelId
    ? resolveChatModelSwitchTarget({
        modelId: thread.modelId,
        ggufVariant: thread.modelGgufVariant ?? null,
      })
    : null;
}

function sameSwitchTarget(
  a: ChatModelSwitchTarget | null,
  b: ChatModelSwitchTarget | null,
): boolean {
  return (
    (a?.modelId ?? null) === (b?.modelId ?? null) &&
    (a?.ggufVariant ?? null) === (b?.ggufVariant ?? null)
  );
}

export function createChatModelHistoryReader(
  threadId: string,
  onModel: (model: ChatModelSwitchTarget | null) => void,
) {
  let disposed = false;
  let updateSeen = false;
  // undefined until the first emit, so an opening read of null still reaches the caller.
  let emitted: ChatModelSwitchTarget | null | undefined;
  const emit = (model: ChatModelSwitchTarget | null): void => {
    if (emitted !== undefined && sameSwitchTarget(emitted, model)) {
      return;
    }
    emitted = model;
    onModel(model);
  };
  return {
    applyInitial(thread: ChatModelThreadSnapshot | null | undefined): void {
      if (disposed || updateSeen) {
        return;
      }
      emit(chatModelSwitchTargetFromThread(thread));
    },
    // Renames and archives land here too, so compare the model before emitting.
    applyUpdate(thread: ChatModelThreadSnapshot): void {
      if (disposed || thread.id !== threadId) {
        return;
      }
      updateSeen = true;
      emit(chatModelSwitchTargetFromThread(thread));
    },
    dispose(): void {
      disposed = true;
    },
  };
}

/** Picker metadata a Switch back must carry, or undefined. Local/fine-tuned rows are in no list,
 *  so without it isGguf resolves false and /load drops the llama.cpp flags. Hub repos need only
 *  the variant. */
export function chatModelSwitchMeta(
  target: ChatModelSwitchTarget,
  loraModels: readonly LoraModelOption[],
): Partial<ModelSelectorChangeMeta> | undefined {
  const resolvedTarget = resolveChatModelSwitchTarget(target);
  const row = loraModels.find((model) => model.id === resolvedTarget.modelId);
  const ggufVariant = resolvedTarget.ggufVariant || undefined;
  if (!row) {
    return ggufVariant ? { ggufVariant } : undefined;
  }
  const isLocal = row.source === "local";
  return {
    source: isLocal ? "local" : row.source === "exported" ? "exported" : "lora",
    isLora:
      !isLocal && row.exportType !== "merged" && row.exportType !== "gguf",
    isDownloaded: true,
    isGguf: row.isDirectGguf === true || Boolean(ggufVariant),
    ...(ggufVariant ? { ggufVariant } : {}),
  };
}
