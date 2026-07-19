// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Infra models hidden from browse/preview lists (Hub Discover, the chat model
// selector, and local on-device rows). Mirrors the backend
// `utils.hidden_models`: the RAG embedding model and the llama.cpp validation
// probe are not usable chat models. Server-confirmed cache rows are trusted
// because the backend applies variant-aware filtering. Optimistic cache rows
// still use these needles until the server confirms them. Per-repo views are
// not filtered, so reinstall flows still show downloaded files.
const HIDDEN_NEEDLES = [
  "bge-small-en-v1.5", // RAG embedder: unsloth/bge-small-en-v1.5[-GGUF]
  "ggml-org/models", // llama.cpp validation probe repo
  "stories260k.gguf", // probe filename (carries .gguf so it stays specific)
];

/** True if any id/path is a hidden infra model. */
export function isHiddenModelId(
  ...values: (string | null | undefined)[]
): boolean {
  return values.some((v) => {
    if (!v) {
      return false;
    }
    const lower = v.toLowerCase();
    return HIDDEN_NEEDLES.some((needle) => lower.includes(needle));
  });
}

/** Exact-match configured infra repos without hiding similarly named models. */
export function isConfiguredHiddenModelId(
  configuredIds: ReadonlySet<string>,
  ...values: (string | null | undefined)[]
): boolean {
  return values.some(
    (value) => value != null && configuredIds.has(value.trim().toLowerCase()),
  );
}
