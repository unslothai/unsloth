// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Infra models hidden from browse/preview lists (Hub Discover, the chat model
// selector, and local on-device rows). Mirrors the backend
// `utils.hidden_models`: the RAG embedding model and the llama.cpp validation
// probe are not usable chat models. The cached on-device inventory is filtered
// server-side (hub cache_inventory) and trusted as-is: it already un-hides a
// GGUF infra repo once the user downloads a variant through the Hub, so these
// needles are NOT re-applied to cached rows (that would drop the variant from
// the list and the count). Per-repo file/download views are NOT filtered
// either, so a reinstall still shows the model as already downloaded.
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
    if (!v) return false;
    const lower = v.toLowerCase();
    return HIDDEN_NEEDLES.some((needle) => lower.includes(needle));
  });
}
