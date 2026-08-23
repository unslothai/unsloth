// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { getInventoryVersion } from "../stores/inventory-events";
import {
  getHiddenModelMatchersSnapshot,
  type HiddenModelMatchersSnapshot,
  replaceHiddenModelMatchers,
} from "./hidden-model-matcher-store";

// Infra models hidden from browse/preview lists (Hub Discover, the chat model
// selector, and local on-device rows). Mirrors the backend
// `utils.hidden_models`: the RAG embedding model, STT dictation models, and the
// llama.cpp validation probe are not usable chat models. Server-confirmed cache
// rows are trusted because the backend applies variant-aware filtering.
// Optimistic cache rows still use these needles until the server confirms them.
// The dynamic matchers fetched from `/api/hub/hidden-models` add default and
// configured embedders as exact repo ids and exact resolved paths, never
// substring needles. Per-repo views are not filtered, so reinstall flows still
// show downloaded files.
const HIDDEN_NEEDLES = [
  "bge-small-en-v1.5", // RAG embedder: unsloth/bge-small-en-v1.5[-GGUF]
  "ggml-org/models", // llama.cpp validation probe repo
  "stories260k.gguf", // probe filename (carries .gguf so it stays specific)
];
const HIDDEN_STT_REPOS = new Set([
  // Transformers safetensors repos and their whisper.cpp GGUF companions
  // (unslothai/whisper-*-GGUF): STT-only, never chat models. The Qwen3-ASR
  // GGUFs are here for the same reason: llama.cpp will load one as a chat
  // model, where it only answers with transcripts.
  "unsloth/whisper-tiny",
  "unsloth/whisper-base",
  "unsloth/whisper-small",
  "unsloth/whisper-large-v3-turbo",
  "unsloth/whisper-large-v3",
  "unslothai/whisper-tiny-gguf",
  "unslothai/whisper-base-gguf",
  "unslothai/whisper-small-gguf",
  "unslothai/whisper-large-v3-turbo-gguf",
  "unslothai/whisper-large-v3-gguf",
  "unslothai/qwen3-asr-0.6b-gguf",
  "unslothai/qwen3-asr-1.7b-gguf",
]);
const HIDDEN_STT_CACHE_NAMES = [...HIDDEN_STT_REPOS].map((repo) =>
  repo.replace("/", "--"),
);

let matchersFetch: Promise<void> | null = null;
let matchersFetchVersion = -1;

function toLowerStrings(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .filter((v): v is string => typeof v === "string" && v.length > 0)
    .map((v) => v.toLowerCase());
}

export function ensureHiddenModelMatchers(): Promise<void> {
  const version = getInventoryVersion();
  if (matchersFetch && matchersFetchVersion === version) {
    return matchersFetch;
  }
  matchersFetchVersion = version;
  matchersFetch = (async () => {
    try {
      const response = await authFetch("/api/hub/hidden-models");
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const data = (await response.json()) as {
        needles?: unknown;
        exact_ids?: unknown;
        exact_paths?: unknown;
      };
      if (
        getInventoryVersion() !== version ||
        matchersFetchVersion !== version
      ) {
        return;
      }
      replaceHiddenModelMatchers({
        needles: toLowerStrings(data.needles),
        exactIds: toLowerStrings(data.exact_ids),
        exactPaths: toLowerStrings(data.exact_paths),
      });
    } catch {
      if (
        getInventoryVersion() === version &&
        matchersFetchVersion === version
      ) {
        matchersFetch = null;
      }
    }
  })();
  return matchersFetch;
}

/** True if any id/path is a hidden infra model. */
export function isHiddenModelIdWithMatchers(
  matchers: HiddenModelMatchersSnapshot,
  ...values: (string | null | undefined)[]
): boolean {
  return values.some((v) => {
    if (!v) {
      return false;
    }
    const lower = v.toLowerCase();
    const normalized = lower.trim().replace(/^\/+|\/+$/g, "");
    const pathParts = lower.split(/[\\/]/);
    return (
      HIDDEN_STT_REPOS.has(normalized) ||
      HIDDEN_STT_CACHE_NAMES.some((name) =>
        pathParts.includes(`models--${name}`),
      ) ||
      HIDDEN_NEEDLES.some((needle) => lower.includes(needle)) ||
      matchers.needles.some((needle) => lower.includes(needle)) ||
      matchers.exactIds.includes(lower) ||
      matchers.exactPaths.includes(lower)
    );
  });
}

export function isHiddenModelId(
  ...values: (string | null | undefined)[]
): boolean {
  return isHiddenModelIdWithMatchers(
    getHiddenModelMatchersSnapshot(),
    ...values,
  );
}
