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
import {
  resolveInitialConfig,
  resolveOnlyRememberedGgufVariant,
} from "../../model-picker/model-config/per-model-config";

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

export function createChatModelHistoryReader(
  threadId: string,
  onModel: (model: ChatModelSwitchTarget | null) => void,
) {
  let disposed = false;
  let updateSeen = false;
  return {
    applyInitial(thread: ChatModelThreadSnapshot | null | undefined): void {
      if (disposed || updateSeen) {
        return;
      }
      onModel(chatModelSwitchTargetFromThread(thread));
    },
    applyUpdate(thread: ChatModelThreadSnapshot): void {
      if (disposed || thread.id !== threadId) {
        return;
      }
      updateSeen = true;
      onModel(chatModelSwitchTargetFromThread(thread));
    },
    dispose(): void {
      disposed = true;
    },
  };
}

export function chatModelSwitchMeta(
  target: ChatModelSwitchTarget,
  loraModels: readonly LoraModelOption[],
): ModelSelectorChangeMeta | undefined {
  const resolvedTarget = resolveChatModelSwitchTarget(target);
  const row = loraModels.find((model) => model.id === resolvedTarget.modelId);
  const ggufVariant = resolvedTarget.ggufVariant || undefined;
  const resolvedConfig = resolveInitialConfig(
    resolvedTarget.modelId,
    ggufVariant,
  );
  const config = resolvedConfig.remembered ? resolvedConfig.config : null;
  if (!row) {
    if (!ggufVariant) return undefined;
    return {
      source: "hub",
      isLora: false,
      ggufVariant,
      isGguf: true,
      ...(config ? { config } : {}),
    };
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
    ...(config ? { config } : {}),
  };
}
