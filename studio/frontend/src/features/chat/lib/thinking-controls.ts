// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { parseExternalModelId } from "../external-providers";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import { useExternalProvidersStore } from "../stores/external-providers-store";
import {
  clampReasoningEffortToLevels,
  getExternalReasoningCapabilities,
  resolveExternalReasoningEffort,
} from "../provider-capabilities";
// eslint-disable-next-line no-restricted-imports -- The store-only import avoids the model-picker React barrel cycle.
import { useModelReasoningEffortStore } from "@/features/model-picker/components/model-selector/model-reasoning-effort";
import { applyQwenThinkingParams } from "../utils/qwen-params";
import { thinkingPresentation } from "./thinking-presentation";
import type { ReasoningEffortLevel } from "../model-catalog";

export function currentThinking() {
  const state = useChatRuntimeStore.getState();
  const selection = parseExternalModelId(state.params.checkpoint);
  const connections = useExternalProvidersStore.getState();
  const provider = connections.connectionsEnabled
    ? connections.providers.find((p) => p.id === selection?.providerId)
    : undefined;
  const caps = selection
    ? getExternalReasoningCapabilities(
        provider?.providerType,
        selection.modelId,
        {
          isReasoningProvider: provider?.isReasoningModel === true,
          baseUrl: provider?.baseUrl,
          apiType: provider?.apiType,
        },
      )
    : {
        supportsReasoning: state.supportsReasoning,
        reasoningStyle: state.reasoningStyle,
        reasoningAlwaysOn: state.reasoningAlwaysOn,
        supportsReasoningOff: state.supportsReasoningOff,
        reasoningEffortLevels: state.reasoningEffortLevels,
      };
  const view = thinkingPresentation(caps);
  const effort = caps.reasoningEffortLevels.length
    ? clampReasoningEffortToLevels(
        state.reasoningEffort,
        caps.reasoningEffortLevels,
      )
    : state.reasoningEffort;
  return { state, selection, provider, caps, view, effort };
}

export function changeThinking(
  enabled: boolean,
  effort?: ReasoningEffortLevel,
) {
  const { state, provider, caps, view } = currentThinking();
  if (!enabled && !view.canDisable) return;
  if (effort && !view.levels.includes(effort)) return;
  if (effort) state.setReasoningEffort(effort);
  else if (enabled && state.reasoningEffort === "none" && view.levels.length)
    state.setReasoningEffort(view.levels[0]);
  // Effort-only local templates represent Off with the actual none level.
  if (
    !enabled &&
    caps.reasoningStyle === "reasoning_effort" &&
    caps.reasoningEffortLevels.includes("none")
  )
    state.setReasoningEffort("none");
  state.setReasoningEnabled(enabled);
  applyQwenThinkingParams(enabled);
  if (!enabled) state.setPreserveThinking(false);
  if (enabled && provider?.providerType === "kimi" && state.toolsEnabled)
    state.setToolsEnabled(false, { persist: false });
}

export function resetThinking() {
  const { state, selection, provider, caps } = currentThinking();
  if (state.params.checkpoint)
    useModelReasoningEffortStore
      .getState()
      .setModelReasoningEffort(state.params.checkpoint, null);
  const effort = selection
    ? resolveExternalReasoningEffort({
        caps,
        providerType: provider?.providerType,
        apiType: provider?.apiType,
        current: "medium",
        pinned: null,
      })
    : clampReasoningEffortToLevels("medium", caps.reasoningEffortLevels);
  state.setReasoningEffort(effort);
  changeThinking(effort !== "none");
}
