// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useState } from "react";

import { Spinner } from "@/components/ui/spinner";
import { BulbIcon } from "@/lib/bulb-icon";
import { toast } from "@/lib/toast";

import { fetchProviderModelReasoning } from "../api/providers-api";
import {
  getProviderModelReasoningCapabilities,
  setProviderModelReasoningCapabilities,
} from "../external-providers";
import {
  clampReasoningEffortToLevels,
  getExternalReasoningCapabilities,
} from "../provider-capabilities";
import {
  type ReasoningEffort,
  useChatRuntimeStore,
} from "../stores/chat-runtime-store";

/** Publish a cached probe result to the runtime store and return whether the
 * model actually supports reasoning. */
function applyReasoningCapsToStore(
  providerType: string,
  modelId: string,
  isReasoningProvider?: boolean,
  baseUrl?: string | null,
): boolean {
  const caps = getExternalReasoningCapabilities(providerType, modelId, {
    isReasoningProvider,
    baseUrl: baseUrl ?? null,
  });
  const state = useChatRuntimeStore.getState();
  const levels = caps.reasoningEffortLevels;
  const clamped = clampReasoningEffortToLevels(state.reasoningEffort, levels);
  const nextEffort: ReasoningEffort = caps.supportsReasoning
    ? levels.includes("medium")
      ? "medium"
      : clamped
    : state.reasoningEffort;
  useChatRuntimeStore.setState({
    supportsReasoning: caps.supportsReasoning,
    reasoningAlwaysOn: caps.reasoningAlwaysOn,
    reasoningStyle: caps.reasoningStyle,
    supportsReasoningOff: caps.supportsReasoningOff,
    reasoningEffortLevels: levels,
    reasoningEffort: nextEffort,
    reasoningEnabled: caps.supportsReasoning ? true : state.reasoningEnabled,
  });
  return caps.supportsReasoning;
}

/**
 * One-shot "Detect reasoning" affordance for a connected llama.cpp model whose
 * reasoning controls haven't been probed yet.
 *
 * llama.cpp reports a model's Jinja chat template only once it is (lazily)
 * loaded, so probing the whole catalog at list time would force-load every
 * model. This probes exactly the selected model on an explicit click, caches
 * the result, and publishes the reasoning caps to the runtime store so the
 * composer's Think control appears in place of this button.
 */
export function DetectReasoningButton({
  providerType,
  providerId,
  modelId,
  baseUrl,
  isReasoningProvider,
}: {
  providerType: string;
  providerId?: string;
  modelId: string;
  baseUrl?: string | null;
  isReasoningProvider?: boolean;
}) {
  const [detecting, setDetecting] = useState(false);

  // Only llama.cpp exposes a probeable Jinja template, and only an uncached
  // model needs detection (a probed negative is cached as
  // `supports_reasoning: false`, so it never shows the button again).
  if (providerType !== "llama_cpp") {
    return null;
  }
  if (
    getProviderModelReasoningCapabilities(providerType, modelId) !== undefined
  ) {
    return null;
  }

  async function detect() {
    setDetecting(true);
    try {
      const reasoning = await fetchProviderModelReasoning({
        providerType,
        providerId: providerId ?? null,
        apiKey: "",
        baseUrl: baseUrl ?? null,
        modelId,
      });
      setProviderModelReasoningCapabilities(providerType, modelId, reasoning);
      if (reasoning) {
        const supports = applyReasoningCapsToStore(
          providerType,
          modelId,
          isReasoningProvider,
          baseUrl,
        );
        if (!supports) {
          toast.info("This model doesn't expose reasoning controls.");
        }
      } else {
        toast.error("Couldn't read this model's chat template.");
      }
    } catch {
      toast.error("Couldn't detect reasoning controls.");
    } finally {
      setDetecting(false);
    }
  }

  return (
    <button
      type="button"
      onClick={detect}
      disabled={detecting}
      className="unsloth-thinking-pill"
      data-pill-label="Detect reasoning"
      aria-label="Detect reasoning capabilities"
    >
      {detecting ? (
        <Spinner className="size-[15.5px]" />
      ) : (
        <BulbIcon className="size-[15.5px]" />
      )}
      <span className="unsloth-thinking-label">Detect reasoning</span>
    </button>
  );
}
