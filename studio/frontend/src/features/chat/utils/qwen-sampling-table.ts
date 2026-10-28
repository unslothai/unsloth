// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Kept free of store imports to avoid a cycle with qwen-params.ts. */

import { parseExternalModelId } from "../external-providers";

export type QwenThinkingParams = {
  temperature: number;
  topP: number;
  topK: number;
  minP: number;
  presencePenalty?: number;
};

// Boundary-anchored so "Qwen3.80" or "Qwen3.8B" do not match.
const PRESENCE_BUMP_QWEN = /(?:^|[^a-z0-9])qwen3\.(?:5|6|8)(?:$|[^a-z0-9])/;

const OLLAMA_MANIFEST_REF_PREFIX = "ollama-manifest:";

/** Compare exactly: two manifests differing only by case are two files. */
export function isOllamaManifestRef(modelId: string): boolean {
  return modelId.startsWith(OLLAMA_MANIFEST_REF_PREFIX);
}

/** Wrappers percent-encode, so decode before the boundary match. */
function bareModelId(checkpoint: string): string {
  const external = parseExternalModelId(checkpoint)?.modelId;
  if (external !== undefined) {
    return external;
  }
  if (checkpoint.startsWith(OLLAMA_MANIFEST_REF_PREFIX)) {
    const ref = checkpoint.slice(OLLAMA_MANIFEST_REF_PREFIX.length);
    try {
      return decodeURIComponent(ref);
    } catch {
      return ref;
    }
  }
  return checkpoint;
}

export function resolveQwenThinkingParams(
  checkpoint: string,
  thinkingOn: boolean,
): QwenThinkingParams | null {
  const normalized = bareModelId(checkpoint).toLowerCase();
  if (!normalized.includes("qwen3")) {
    return null;
  }

  const needsPresencePenalty = PRESENCE_BUMP_QWEN.test(normalized);
  const base = thinkingOn
    ? { temperature: 0.6, topP: 0.95, topK: 20, minP: 0.0 }
    : { temperature: 0.7, topP: 0.8, topK: 20, minP: 0.0 };
  return needsPresencePenalty ? { ...base, presencePenalty: 1.5 } : base;
}
