// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { buildExternalModelId } from "../external-providers";
import { providerSupportsFastMode } from "../provider-capabilities";
import { useExternalProvidersStore } from "../stores/external-providers-store";
import { parseExternalModelId } from "../external-providers";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import {
  getExternalReasoningCapabilities,
  clampReasoningEffortToLevels,
} from "../provider-capabilities";
import type { ReasoningEffortLevel } from "../model-catalog";

function currentThinking() {
  const state = useChatRuntimeStore.getState();
  const selection = parseExternalModelId(state.params.checkpoint);
  const provider = useExternalProvidersStore
    .getState()
    .providers.find((p) => p.id === selection?.providerId);
  const caps = getExternalReasoningCapabilities(
    provider?.providerType,
    selection?.modelId,
    {
      isReasoningProvider: provider?.isReasoningModel === true,
      baseUrl: provider?.baseUrl,
      apiType: provider?.apiType,
    },
  );
  const effort = caps.reasoningEffortLevels.length
    ? clampReasoningEffortToLevels(
        state.reasoningEffort,
        caps.reasoningEffortLevels,
      )
    : state.reasoningEffort;
  return {
    state,
    selection,
    provider,
    effort,
    view: {
      levels: caps.reasoningEffortLevels.filter(
        (level) => level !== "none",
      ) as ReasoningEffortLevel[],
      canDisable:
        !caps.reasoningAlwaysOn &&
        (caps.supportsReasoningOff ||
          caps.reasoningEffortLevels.includes("none")),
    },
  };
}

function changeThinking(enabled: boolean, effort?: ReasoningEffortLevel) {
  const { state } = currentThinking();
  if (effort) state.setReasoningEffort(effort);
  state.setReasoningEnabled(enabled);
}
import { fastCandidateModels } from "../model-catalog";
import { openRouterFastTier } from "./openrouter-fast-tier";
import { resolveFastPairs, verifiedFastVariant } from "./fast-variants";
import { updateProviderConfig } from "../api/providers-api";
import { toast } from "sonner";

let enablingCompanion = false;

export function currentFast() {
  const { state, selection, provider } = currentThinking();
  const variant = verifiedFastVariant(
    provider?.providerType,
    selection?.modelId,
    provider?.models ?? [],
    provider?.availableModels,
    resolveFastPairs(
      fastCandidateModels(),
      provider?.fastPairs,
      provider?.autoDetectFastVariants !== false,
    ),
  );
  const native = providerSupportsFastMode(
    provider?.providerType,
    selection?.modelId,
  );
  const tier =
    provider?.providerType === "openrouter" && selection
      ? openRouterFastTier(selection.modelId)
      : null;
  const busy =
    state.modelLoading || Object.values(state.runningByThreadId).some(Boolean);
  const connected = useExternalProvidersStore.getState().connectionsEnabled;
  return {
    variant: native ? null : variant,
    tier: tier?.supported ? tier : null,
    native,
    busy,
    connected,
    isFast: native ? state.params.fastMode : (variant?.isFast ?? false),
    provider,
    selection,
  };
}

/** Both the header and shortcut use the picker callback, including its capability/default resolution. */
export async function toggleFast(
  selectModel: (checkpoint: string) => void,
): Promise<boolean> {
  const fast = currentFast();
  if (fast.busy || !fast.connected || enablingCompanion) return false;
  const before = currentThinking();
  if (fast.variant && fast.provider) {
    if (fast.variant.reason) return false;
    if (!fast.variant.enabledCompanion) {
      enablingCompanion = true;
      try {
        const models = [...fast.provider.models, fast.variant.destination];
        await updateProviderConfig(fast.provider.id, { models });
        const store = useExternalProvidersStore.getState();
        store.setProviders(
          store.providers.map((provider) =>
            provider.id === fast.provider!.id
              ? {
                  ...provider,
                  models: [
                    ...new Set([...provider.models, fast.variant!.destination]),
                  ],
                  updatedAt: Date.now(),
                }
              : provider,
          ),
        );
      } catch {
        toast.error("Could not enable the Fast companion. Please try again.");
        return false;
      } finally {
        enablingCompanion = false;
      }
      const current = currentFast();
      if (
        current.busy ||
        !current.connected ||
        current.provider?.id !== fast.provider.id ||
        current.selection?.modelId !== fast.selection?.modelId ||
        current.variant?.reason
      )
        return false;
    }
    selectModel(
      buildExternalModelId(fast.provider.id, fast.variant.destination),
    );
    const after = currentThinking();
    // Only preserve a value the actual companion supports. Otherwise keep the selection path's result.
    if (after.selection?.modelId === fast.variant.destination) {
      if (
        (!before.state.reasoningEnabled || before.effort === "none") &&
        after.view.canDisable
      )
        changeThinking(false);
      else if (after.view.levels.includes(before.effort))
        changeThinking(true, before.effort);
    }
    return true;
  }
  if (fast.native) {
    if (fast.tier && !fast.tier.available && !before.state.params.fastMode)
      return false;
    before.state.setParams({
      ...before.state.params,
      fastMode: !before.state.params.fastMode,
    });
    return true;
  }
  return false;
}
