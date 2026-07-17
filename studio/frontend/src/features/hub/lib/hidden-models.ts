// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";

// Infra models hidden from every browse/preview list (Hub discover and the chat
// model selector). Mirrors the backend `_is_hidden_model`: the RAG embedding
// model and the llama.cpp validation probe are not usable chat models. Per-repo
// file/download views are NOT filtered, so a reinstall still shows the model as
// already downloaded.
const HIDDEN_NEEDLES = [
  "bge-small-en-v1.5", // RAG embedder: unsloth/bge-small-en-v1.5[-GGUF]
  "ggml-org/models", // llama.cpp validation probe repo
  "stories260k.gguf", // probe filename (carries .gguf so it stays specific)
];

let dynamicNeedles: readonly string[] = [];
let dynamicExactPaths: readonly string[] = [];
let matchersFetch: Promise<void> | null = null;

function toLowerStrings(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .filter((v): v is string => typeof v === "string" && v.length > 0)
    .map((v) => v.toLowerCase());
}

export function ensureHiddenModelMatchers(): Promise<void> {
  matchersFetch ??= (async () => {
    try {
      const response = await authFetch("/api/hub/hidden-models");
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const data = (await response.json()) as {
        needles?: unknown;
        exact_paths?: unknown;
      };
      dynamicNeedles = toLowerStrings(data.needles);
      dynamicExactPaths = toLowerStrings(data.exact_paths);
    } catch {
      matchersFetch = null;
    }
  })();
  return matchersFetch;
}

/** True if any id/path is a hidden infra model. */
export function isHiddenModelId(
  ...values: (string | null | undefined)[]
): boolean {
  return values.some((v) => {
    if (!v) return false;
    const lower = v.toLowerCase();
    return (
      HIDDEN_NEEDLES.some((needle) => lower.includes(needle)) ||
      dynamicNeedles.some((needle) => lower.includes(needle)) ||
      dynamicExactPaths.includes(lower)
    );
  });
}
