// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Every mark comes from what the request path already resolves (capability registry, provider
// gates), never from the model's name: a mark the request cannot honour is a promise the picker
// breaks. An On Device badge may guess, since it describes a repo you have not fetched yet.

import type { ProviderApiType } from "@/features/chat/api/providers-api";
// eslint-disable-next-line no-restricted-imports -- Avoid the chat barrel's React exports.
import { providerModelSupportsVision } from "@/features/chat/external-providers";
// eslint-disable-next-line no-restricted-imports -- Avoid the chat barrel's React exports.
import { providerSupportsBuiltinImageGeneration } from "@/features/chat/provider-capabilities";
import type { ModelCapabilities } from "./model-capabilities";

export interface ConnectedModelMarks {
  /** Capability glyphs (audio / generates images / generates video). */
  capabilities: ModelCapabilities;
  /** Reads images: drawn as the row's Vision badge rather than a glyph. */
  vision: boolean;
}

/** The badges one Connected row should draw.
 *
 *  `modelId` is the provider's own id (`gemini-2.5-flash-image`), not the `external::` id the
 *  picker selects by. `baseUrl` and `apiType` are the connection's and only image generation
 *  reads them. */
export function connectedModelMarks(opts: {
  providerType: string | null | undefined;
  modelId: string | null | undefined;
  baseUrl?: string | null;
  apiType?: ProviderApiType;
}): ConnectedModelMarks {
  const { providerType, modelId, baseUrl, apiType } = opts;
  // null is unknown, drawn as no badge rather than as a promise.
  const vision = providerModelSupportsVision(providerType, modelId) === true;
  return {
    capabilities: {
      vision,
      // CAPABILITY_BADGES draws no reasoning glyph, so it is not worth a second catalogue lookup.
      reasoning: false,
      // The audio adapter resolves the active model out of `models`, which holds loaded local
      // models only, so `add` rejects every audio file under a connected selection.
      audio: false,
      // The helper alone, no name fallback: the same gate the adapter and the Images pill use.
      imageGen: providerSupportsBuiltinImageGeneration(
        providerType,
        modelId,
        baseUrl,
        apiType,
      ),
      // Nothing we connect to serves video generation through the chat route, so the name would
      // be the only evidence and a glyph resting on it promises what selecting the row cannot do.
      videoGen: false,
    },
    vision,
  };
}
