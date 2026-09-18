// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// What a connected (API-served) model can do, in the shape the picker's row badges take. A
// connected model carries no HF tags and no local metadata, so every mark comes from what the app
// already resolves for the request itself: the backend capability registry and the provider gates
// the adapter uses. Not from the name, unlike an On Device row: that badge describes a repo you
// may be about to fetch, while this one describes what happens when you pick the row and send a
// message, so a mark the request path cannot honour is a promise the picker breaks. That rules
// out a published modality as well as a guess, where nothing carries it to the provider yet.

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
 *  picker selects by: every helper below keys on what the provider calls the model.
 *  `baseUrl` is the connection's, and only image generation reads it -- a Gemini connection
 *  pointed at an OpenAI-compatible proxy never returns inline image data. */
export function connectedModelMarks(opts: {
  providerType: string | null | undefined;
  modelId: string | null | undefined;
  baseUrl?: string | null;
}): ConnectedModelMarks {
  const { providerType, modelId, baseUrl } = opts;
  // Already layered: the backend's per-model registry entry, the image-stripping provider list,
  // the catalogue's modalities, then the provider-type default. Null there means unknown, which
  // the picker draws as no badge rather than as a promise.
  const vision = providerModelSupportsVision(providerType, modelId) === true;
  return {
    capabilities: {
      vision,
      // No glyph reads this (CAPABILITY_BADGES draws none for reasoning), so it is not worth a
      // second catalogue lookup here.
      reasoning: false,
      // Never claimed either, though catalogues do publish it: the audio adapter resolves the
      // active model out of `models`, which holds loaded local models only, so `add` rejects
      // every audio file under a connected selection. The glyph would offer a picker entry that
      // cannot be used, so it waits on a request path that can carry audio to a provider.
      audio: false,
      // The helper alone, with no fallback to the model's name. It is the same gate the adapter
      // enables image generation through, and the composer's Images pill with it, so a name that
      // reads like a diffusion model but resolves to false here can never be asked for an image.
      imageGen: providerSupportsBuiltinImageGeneration(
        providerType,
        modelId,
        baseUrl,
      ),
      // Never claimed. Nothing we connect to serves video generation through the chat route and
      // no provider publishes a capability for it, so the name is the only evidence there could
      // be, and a glyph resting on that promises a row something selecting it cannot do. An On
      // Device row can say it because that badge describes a repo you may be about to fetch.
      videoGen: false,
    },
    vision,
  };
}
