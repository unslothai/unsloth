// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The single Qwen3 sampling table. Kept free of store imports: qwen-params.ts
 * imports the store, so the table cannot live there without a cycle. Load, the
 * Think toggle and the migration then agree by construction.
 */

import { parseExternalModelId } from "../external-providers";

export type QwenThinkingParams = {
  temperature: number;
  topP: number;
  topK: number;
  minP: number;
  presencePenalty?: number;
};

// Boundary-anchored: "Qwen3.80" and "Qwen3.8B" are a future family and a
// parameter count, and a substring test would bump both. Any non-alphanumeric
// ends the family, since a path can separate it with a space as readily as "-".
const PRESENCE_BUMP_QWEN = /(?:^|[^a-z0-9])qwen3\.(5|6|8)(?:$|[^a-z0-9])/;

const OLLAMA_MANIFEST_REF_PREFIX = "ollama-manifest:";

/**
 * normalizeModelIdentity sees this wrapper rather than the percent-encoded path
 * inside it and folds the path's case with it, so callers compare these exactly:
 * two manifests differing only by case are two files.
 */
export function isOllamaManifestRef(modelId: string): boolean {
  return modelId.startsWith(OLLAMA_MANIFEST_REF_PREFIX);
}

/**
 * The bare model id, with any wrapper removed. Both wrappers percent-encode, so
 * "Qwen/Qwen3.8-27B" arrives with its slash as `%2F`, leaving an alphanumeric
 * "f" against the family segment that the boundary match would reject.
 */
function bareModelId(checkpoint: string): string {
  const external = parseExternalModelId(checkpoint)?.modelId;
  if (external !== undefined) {
    return external;
  }
  // An Ollama row carries its inventory reference into inference status, quoted
  // with safe='', so every separator arrives as %2F.
  if (checkpoint.startsWith(OLLAMA_MANIFEST_REF_PREFIX)) {
    const ref = checkpoint.slice(OLLAMA_MANIFEST_REF_PREFIX.length);
    try {
      return decodeURIComponent(ref);
    } catch {
      // A malformed escape is not a reason to lose the whole checkpoint.
      return ref;
    }
  }
  return checkpoint;
}

/** Resolve the sampling table shared by model load and the Think toggle. */
export function resolveQwenThinkingParams(
  checkpoint: string,
  thinkingOn: boolean,
): QwenThinkingParams | null {
  const normalized = bareModelId(checkpoint).toLowerCase();
  if (!normalized.includes("qwen3")) {
    return null;
  }

  const presenceBumpFamily = normalized.match(PRESENCE_BUMP_QWEN);
  const qwen38Thinking = thinkingOn && presenceBumpFamily?.[1] === "8";
  const base = thinkingOn
    ? {
        temperature: qwen38Thinking ? 1.0 : 0.6,
        topP: 0.95,
        topK: 20,
        minP: 0.0,
      }
    : { temperature: 0.7, topP: 0.8, topK: 20, minP: 0.0 };
  return presenceBumpFamily
    ? { ...base, presencePenalty: qwen38Thinking ? 0.0 : 1.5 }
    : base;
}
