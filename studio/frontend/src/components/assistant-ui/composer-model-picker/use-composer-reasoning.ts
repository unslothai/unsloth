// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { parseExternalModelId } from "@/features/chat/external-providers";
import {
  clampReasoningEffortToLevels,
  getExternalReasoningCapabilities,
  kimiSearchExcludesThinking,
  modelCatalogVersion,
  providerSupportsFastMode,
  subscribeModelCatalog,
} from "@/features/chat/provider-capabilities";
import {
  type ReasoningEffort,
  useChatRuntimeStore,
} from "@/features/chat/stores/chat-runtime-store";
import { useExternalProvidersStore } from "@/features/chat/stores/external-providers-store";
import { applyQwenThinkingParams } from "@/features/chat/utils/qwen-params";
import { useSyncExternalStore } from "react";
import {
  OFF_LEVEL,
  type SliderLevel,
  effortLabel,
  sliderLevels,
  sliderValue,
} from "./picker-model";

export interface ComposerReasoning {
  disabled: boolean;
  supportsReasoning: boolean;
  isEffort: boolean;
  lockedOn: boolean;
  levels: SliderLevel[];
  value: string;
  setValue: (value: string) => void;
  enabled: boolean;
  setEnabled: (next: boolean) => void;
  supportsPreserve: boolean;
  supportsFast: boolean;
  fast: boolean;
  setFast: (next: boolean) => void;
  preserve: boolean;
  setPreserve: (next: boolean) => void;
  chipLabel: string | null;
}

export function useComposerReasoning(): ComposerReasoning {
  const modelLoaded = useChatRuntimeStore(
    (s) => !!s.params.checkpoint && !s.modelLoading,
  );
  const checkpoint = useChatRuntimeStore((s) => s.params.checkpoint);
  const supportsReasoning = useChatRuntimeStore((s) => s.supportsReasoning);
  const reasoningAlwaysOn = useChatRuntimeStore((s) => s.reasoningAlwaysOn);
  const reasoningEnabled = useChatRuntimeStore((s) => s.reasoningEnabled);
  const setReasoningEnabled = useChatRuntimeStore((s) => s.setReasoningEnabled);
  const reasoningStyle = useChatRuntimeStore((s) => s.reasoningStyle);
  const reasoningEffort = useChatRuntimeStore((s) => s.reasoningEffort);
  const supportsReasoningOff = useChatRuntimeStore(
    (s) => s.supportsReasoningOff,
  );
  const reasoningEffortLevels = useChatRuntimeStore(
    (s) => s.reasoningEffortLevels,
  );
  const setReasoningEffort = useChatRuntimeStore((s) => s.setReasoningEffort);
  const toolsEnabled = useChatRuntimeStore((s) => s.toolsEnabled);
  const setToolsEnabled = useChatRuntimeStore((s) => s.setToolsEnabled);
  const supportsPreserve = useChatRuntimeStore(
    (s) => s.supportsPreserveThinking,
  );
  const preserve = useChatRuntimeStore((s) => s.preserveThinking);
  const setPreserveThinking = useChatRuntimeStore((s) => s.setPreserveThinking);
  const connectionsEnabled = useExternalProvidersStore(
    (s) => s.connectionsEnabled,
  );
  const providers = useExternalProvidersStore((s) => s.providers);
  const fast = useChatRuntimeStore((s) => s.params.fastMode === true);
  useSyncExternalStore(subscribeModelCatalog, modelCatalogVersion);

  const externalSelection = parseExternalModelId(checkpoint);
  const provider =
    externalSelection != null && connectionsEnabled
      ? providers.find((p) => p.id === externalSelection.providerId)
      : undefined;
  const isKimiExternal = kimiSearchExcludesThinking(
    provider?.providerType,
    externalSelection?.modelId,
  );
  const caps =
    externalSelection != null
      ? getExternalReasoningCapabilities(
          provider?.providerType,
          externalSelection.modelId,
          {
            isReasoningProvider: provider?.isReasoningModel === true,
            baseUrl: provider?.baseUrl ?? null,
            apiType: provider?.apiType,
            reasoningConfig: provider?.reasoningConfig,
          },
        )
      : null;
  const style = caps?.reasoningStyle ?? reasoningStyle;
  const alwaysOn = caps?.reasoningAlwaysOn ?? reasoningAlwaysOn;
  const supportsOff = caps?.supportsReasoningOff ?? supportsReasoningOff;
  const ladder = caps?.reasoningEffortLevels ?? reasoningEffortLevels;
  const supports = caps?.supportsReasoning ?? supportsReasoning;
  const lockedOn = supports && (alwaysOn || !supportsOff);
  const enabled = lockedOn ? true : reasoningEnabled;
  // The stored effort clamped to this ladder, without overwriting the choice.
  const effort =
    ladder.length > 0
      ? clampReasoningEffortToLevels(reasoningEffort, ladder)
      : reasoningEffort;
  const isEffort =
    supports &&
    (style === "reasoning_effort" || style === "enable_thinking_effort");
  const offStop = isEffort && supportsOff && !lockedOn;
  const levels = isEffort
    ? sliderLevels(ladder, offStop, externalSelection?.modelId)
    : [];
  const value = sliderValue(enabled, effort, offStop);

  const turnOn = () => {
    setReasoningEnabled(true);
    applyQwenThinkingParams(true);
    // Kimi's $web_search forbids thinking, so thinking turns Search off.
    if (isKimiExternal && toolsEnabled)
      setToolsEnabled(false, { persist: false });
  };
  const turnOff = () => {
    setReasoningEnabled(false);
    applyQwenThinkingParams(false);
    setPreserveThinking(false);
  };

  const setValue = (next: string) => {
    if (next === OFF_LEVEL) {
      turnOff();
      return;
    }
    setReasoningEffort(next as ReasoningEffort);
    turnOn();
  };
  const setEnabled = (next: boolean) => {
    if (lockedOn) return;
    if (next) turnOn();
    else turnOff();
  };
  const supportsFast =
    provider != null &&
    providerSupportsFastMode(provider.providerType, externalSelection?.modelId);
  const setFast = (next: boolean) => {
    const store = useChatRuntimeStore.getState();
    store.setParams({ ...store.params, fastMode: next });
  };
  const setPreserve = (next: boolean) => {
    setPreserveThinking(next);
    // Only local models couple this to generation controls.
    if (next && externalSelection === null) {
      setReasoningEnabled(true);
      applyQwenThinkingParams(true);
    }
  };

  // Not while loading: the store still holds the previous model's reasoning until the new one lands.
  const chipLabel =
    supports && modelLoaded
      ? isEffort
        ? value === OFF_LEVEL
          ? null
          : effortLabel(value, externalSelection?.modelId)
        : enabled
          ? "Thinking"
          : null
      : null;

  return {
    disabled: !modelLoaded,
    supportsReasoning: supports,
    isEffort,
    lockedOn,
    levels,
    value,
    setValue,
    enabled,
    setEnabled,
    supportsPreserve,
    supportsFast,
    fast,
    setFast,
    preserve,
    setPreserve,
    chipLabel,
  };
}
