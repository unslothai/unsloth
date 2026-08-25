// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type PromptEntry, savePromptEntry } from "../api/prompts-api";
import {
  type ChatPresetSource,
  type Preset,
  getOrderedPresets,
  getPresetSource,
  isSamePresetConfig,
} from "../presets/preset-policy";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import type { InferenceParams } from "../types/runtime";
import {
  applySystemPromptToParams,
  savedPromptNameFromText,
} from "./saved-system-prompt";

export function resolvePresetSourceAfterParamsChange(
  activePreset: string,
  customPresets: Preset[],
  nextParams: InferenceParams,
): ChatPresetSource {
  const preset = getOrderedPresets(customPresets).find(
    (item) => item.name === activePreset,
  );
  if (preset && isSamePresetConfig(preset.params, nextParams)) {
    return getPresetSource(activePreset);
  }
  return "modified";
}

export function applySavedPromptAsSystemPrompt(text: string): void {
  const {
    params,
    setParams,
    activePreset,
    customPresets,
    setActivePresetSource,
  } = useChatRuntimeStore.getState();
  const nextParams = applySystemPromptToParams(params, text);
  setActivePresetSource(
    resolvePresetSourceAfterParamsChange(
      activePreset,
      customPresets,
      nextParams,
    ),
  );
  setParams(nextParams);
}

function newSavedPromptId(): string {
  return crypto.randomUUID().replace(/-/g, "").slice(0, 12);
}

export async function saveTextAsPromptEntry(
  text: string,
  name?: string,
): Promise<PromptEntry | null> {
  const trimText = text.trim();
  if (!trimText) {
    return null;
  }
  const ts = Date.now();
  return await savePromptEntry({
    id: newSavedPromptId(),
    name: name?.trim() || savedPromptNameFromText(trimText),
    text: trimText,
    createdAt: ts,
    updatedAt: ts,
  });
}
