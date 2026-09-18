// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// What a connected (API-served) model can do, in the shape the picker's row badges take. A
// connected model carries no HF tags and no local metadata, so every mark comes from what the app
// already resolves: the backend capability registry, the connection's cached catalogue, and the
// name fallback the On Device rows use.

// eslint-disable-next-line no-restricted-imports -- Avoid the chat barrel's React exports.
import { providerModelSupportsVision } from "@/features/chat/external-providers";
// eslint-disable-next-line no-restricted-imports -- Avoid the chat barrel's React exports.
import { resolveModelCatalogEntry } from "@/features/chat/model-catalog";
// eslint-disable-next-line no-restricted-imports -- Avoid the chat barrel's React exports.
import { providerSupportsBuiltinImageGeneration } from "@/features/chat/provider-capabilities";
import {
  type ModelCapabilities,
  detectCapabilities,
} from "./model-capabilities";

export interface ConnectedModelMarks {
  /** Capability glyphs (audio / generates images / generates video). */
  capabilities: ModelCapabilities;
  /** Reads images: drawn as the row's Vision badge rather than a glyph. */
  vision: boolean;
  /** Published context window in tokens, or null when the catalogue does not say. */
  contextLength: number | null;
}

/** A context window as a chip: "128K", "1.1M". The exact count stays in Model info. */
export function formatContextLength(tokens: number): string {
  if (tokens >= 1_000_000) {
    const millions = tokens / 1_000_000;
    return `${millions >= 10 ? Math.round(millions) : Number(millions.toFixed(1))}M`;
  }
  if (tokens >= 1_000) return `${Math.round(tokens / 1_000)}K`;
  return String(tokens);
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
  // The fallback for ids no catalogue covers: a self-hosted "whisper-large-v3" or an
  // OpenRouter-served diffusion model is only ever described by what it is called.
  const byName = detectCapabilities({ id: modelId ?? "" });
  const entry = resolveModelCatalogEntry(providerType, modelId);
  const modalities = entry?.inputModalities ?? null;
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
      audio: modalities?.includes("audio") === true || byName.audio,
      imageGen:
        providerSupportsBuiltinImageGeneration(
          providerType,
          modelId,
          baseUrl,
        ) || byName.imageGen,
      // Never claimed. Nothing we connect to serves video generation through the chat route and
      // no provider publishes a capability for it, so the name is the only evidence there could
      // be, and a glyph resting on that promises a row something selecting it cannot do. An On
      // Device row can say it because that badge describes a repo you may be about to fetch.
      videoGen: false,
    },
    vision,
    contextLength:
      entry?.contextLength && entry.contextLength > 0
        ? entry.contextLength
        : null,
  };
}
