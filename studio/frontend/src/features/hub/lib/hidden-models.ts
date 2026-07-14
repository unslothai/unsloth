// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Infra models hidden from every browse/preview list (Hub discover and the chat
// model selector). Mirrors the backend `_is_hidden_model`: the RAG embedding
// model, STT models, and llama.cpp validation probe are not chat models. Per-repo
// file/download views are NOT filtered, so a reinstall still shows the model as
// already downloaded.
const HIDDEN_NEEDLES = [
  "bge-small-en-v1.5", // RAG embedder: unsloth/bge-small-en-v1.5[-GGUF]
  "ggml-org/models", // llama.cpp validation probe repo
  "stories260k.gguf", // probe filename (carries .gguf so it stays specific)
];
const HIDDEN_STT_REPOS = new Set([
  "unsloth/whisper-small",
  "unsloth/whisper-large-v3-turbo",
  "unsloth/whisper-large-v3",
]);
const HIDDEN_STT_CACHE_NAMES = [...HIDDEN_STT_REPOS].map((repo) =>
  repo.replace("/", "--"),
);

/** True if any id/path is a hidden infra model. */
export function isHiddenModelId(
  ...values: (string | null | undefined)[]
): boolean {
  return values.some((v) => {
    if (!v) return false;
    const lower = v.toLowerCase();
    const normalized = lower.trim().replace(/^\/+|\/+$/g, "");
    const pathParts = lower.split(/[\\/]/);
    return (
      HIDDEN_STT_REPOS.has(normalized) ||
      HIDDEN_STT_CACHE_NAMES.some((name) =>
        pathParts.includes(`models--${name}`),
      ) ||
      HIDDEN_NEEDLES.some((needle) => lower.includes(needle))
    );
  });
}
