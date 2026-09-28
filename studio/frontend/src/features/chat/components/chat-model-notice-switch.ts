// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  LoraModelOption,
  ModelSelectorChangeMeta,
} from "@/features/model-picker/components/model-selector/types";
import {
  ggufQuantLabel,
  normalizeGgufVariantIdentity,
} from "../../model-picker/model-config/model-identity";
import { resolveOnlyRememberedGgufVariant } from "../../model-picker/model-config/per-model-config";

export type ChatModelSwitchTarget = {
  modelId: string;
  ggufVariant?: string | null;
};

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
    // Every history event for this thread lands here, renames and archives included, so the model is
    // compared before a new object is handed to the caller.
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

/** The picker metadata a "Switch back" has to carry, or undefined when the id needs none. A local
 *  or fine-tuned row is in neither `/api/models/list` nor the external ids, so with no metadata
 *  `isGguf` resolves false and the /load request drops the llama.cpp flags. A Hub repo resolves
 *  itself, and a synthesized `source: "hub"` would only route an on-disk model into the download
 *  manager, so just the variant travels. The remembered config stays off both branches:
 *  `stageOrLoad` already falls back to `rememberedConfigFor`. */
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
    // local folders and merged or gguf exports are not adapters.
    isLora:
      !isLocal && row.exportType !== "merged" && row.exportType !== "gguf",
    // every row here is already on disk.
    isDownloaded: true,
    isGguf: row.isDirectGguf === true || Boolean(ggufVariant),
    ...(ggufVariant ? { ggufVariant } : {}),
  };
}
