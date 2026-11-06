// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Marks come only from what the request path resolves, never from the model's name.

import type { ProviderApiType } from "@/features/chat/api/providers-api";
// eslint-disable-next-line no-restricted-imports -- Avoid the chat barrel's React exports.
import { providerModelSupportsVision } from "@/features/chat/external-providers";
// eslint-disable-next-line no-restricted-imports -- Avoid the chat barrel's React exports.
import { providerSupportsBuiltinImageGeneration } from "@/features/chat/provider-capabilities";
import type { ModelCapabilities } from "./model-capabilities";

export interface ConnectedModelMarks {
  capabilities: ModelCapabilities;
  vision: boolean;
}

/** `modelId` is the provider's own id, not the `external::` id. */
export function connectedModelMarks(opts: {
  providerType: string | null | undefined;
  modelId: string | null | undefined;
  baseUrl?: string | null;
  apiType?: ProviderApiType;
}): ConnectedModelMarks {
  const { providerType, modelId, baseUrl, apiType } = opts;
  // null is unknown, drawn as no badge.
  const vision = providerModelSupportsVision(providerType, modelId) === true;
  return {
    capabilities: {
      vision,
      reasoning: false,
      // The audio adapter only resolves loaded local models, so audio is rejected for connected picks.
      audio: false,
      imageGen: providerSupportsBuiltinImageGeneration(
        providerType,
        modelId,
        baseUrl,
        apiType,
      ),
      // No connected provider serves video generation through the chat route.
      videoGen: false,
    },
    vision,
  };
}
