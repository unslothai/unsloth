// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The single Qwen3 sampling table, kept free of store imports.
 *
 * qwen-params.ts imports the store, which imports this migration, so the table
 * cannot live there without a cycle. Keeping it pure lets model load, the Think
 * toggle and the migration agree by construction: change a value once and every
 * consumer follows, including rows already on disk.
 */

import { parseExternalModelId } from "../external-providers";

export type QwenThinkingParams = {
  temperature: number;
  topP: number;
  topK: number;
  minP: number;
  presencePenalty?: number;
};

// Boundary-anchored, not a substring: "Qwen3.80" and "Qwen3.8B" are a future
// family and a parameter count, and a substring test would give both the
// presence bump. Real ids still match, since a family segment always ends the
// string or runs into -, _, / or \.
const PRESENCE_BUMP_QWEN = /(?:^|[^a-z0-9])qwen3\.(?:5|6|8)(?:$|[-_/\\])/;

const OLLAMA_MANIFEST_REF_PREFIX = "ollama-manifest:";

/**
 * The bare model id, with any external wrapper removed.
 *
 * An external checkpoint is `external::<provider>::<percent-encoded id>`, so
 * "Qwen/Qwen3.8-27B" arrives with its slash as `%2F`. That leaves an
 * alphanumeric `f` against the family segment, which the boundary match would
 * reject, dropping the presence bump for every external Qwen.
 */
function bareModelId(checkpoint: string): string {
  const external = parseExternalModelId(checkpoint)?.modelId;
  if (external !== undefined) {
    return external;
  }
  // An Ollama row keeps its opaque inventory reference all the way into
  // inference status (_public_model_identifier), and the manifest path is
  // quoted with safe='', so every separator arrives as %2F. Undecoded, the
  // family segment sits between an alphanumeric "f" and a "%", and the
  // boundary match below drops the presence bump for every Ollama Qwen.
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

  const needsPresencePenalty = PRESENCE_BUMP_QWEN.test(normalized);
  const base = thinkingOn
    ? { temperature: 0.6, topP: 0.95, topK: 20, minP: 0.0 }
    : { temperature: 0.7, topP: 0.8, topK: 20, minP: 0.0 };
  return needsPresencePenalty ? { ...base, presencePenalty: 1.5 } : base;
}
