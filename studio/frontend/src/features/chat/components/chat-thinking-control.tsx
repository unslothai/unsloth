// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useSyncExternalStore } from "react";
import { useShallow } from "zustand/react/shallow";
import { ThinkingControl } from "./thinking-control";
import {
  currentThinking,
  changeThinking,
  resetThinking,
} from "../lib/thinking-controls";
import { modelCatalogVersion, subscribeModelCatalog } from "../model-catalog";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import { useExternalProvidersStore } from "../stores/external-providers-store";

export function ChatThinkingControl({
  side = "top",
}: { side?: "top" | "bottom" }) {
  useChatRuntimeStore(
    useShallow((s) => [
      s.params.checkpoint,
      s.modelLoading,
      s.reasoningStyle,
      s.reasoningEffortLevels,
      s.reasoningAlwaysOn,
      s.supportsReasoning,
      s.supportsReasoningOff,
      s.reasoningEffort,
      s.reasoningEnabled,
      s.supportsPreserveThinking,
      s.preserveThinking,
    ]),
  );
  useExternalProvidersStore(
    useShallow((s) => [s.providers, s.connectionsEnabled]),
  );
  useSyncExternalStore(subscribeModelCatalog, modelCatalogVersion);
  const { state, caps, effort, selection } = currentThinking();
  return (
    <ThinkingControl
      caps={caps}
      effort={effort}
      enabled={state.reasoningEnabled}
      disabled={!state.params.checkpoint || state.modelLoading}
      side={side}
      onEnabledChange={changeThinking}
      onEffortChange={(level) => changeThinking(true, level)}
      onReset={resetThinking}
      preserve={state.preserveThinking}
      onPreserveChange={
        state.supportsPreserveThinking
          ? (enabled) => {
              state.setPreserveThinking(enabled);
              if (enabled && !selection) changeThinking(true);
            }
          : undefined
      }
    />
  );
}
