// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { looksLikeLocalPath } from "@/lib/local-path";
import type { ModelPickTarget } from "../components/model-selector/types";
import { isOllamaModelId } from "./model-identity";

export type ModelDiskLocation =
  | { kind: "path"; path: string }
  /** The id names a managed-cache repo: the server resolves id (+ quant) to a path. */
  | { kind: "cachedRepo"; repoId: string; variant?: string };

/** On-disk location for a load-settings target; null when nothing on disk backs it. */
export function resolveModelLocation(
  target: ModelPickTarget,
): ModelDiskLocation | null {
  const id = target.id.trim();

  // Before the path test: a materialized Ollama link is itself a path, but it points into
  // Ollama's blob store rather than naming files this app loads.
  if (isOllamaModelId(id)) {
    return null;
  }

  const pickedPath = target.meta.localPath?.trim();
  if (pickedPath) {
    return { kind: "path", path: pickedPath };
  }
  if (looksLikeLocalPath(id)) {
    return { kind: "path", path: id };
  }
  // A snapshot pin (a repo outside the active cache) is what the load reads; the cache lookup
  // would answer for whichever cached copy is largest.
  const loadId = target.meta.loadId?.trim();
  if (loadId && looksLikeLocalPath(loadId)) {
    return { kind: "path", path: loadId };
  }

  if (target.meta.source === "external") {
    return null;
  }
  // A repo that is not on disk yet has no path to show; the request would only 404.
  if (target.meta.source === "hub" && target.meta.isDownloaded === false) {
    return null;
  }

  const repoId = (target.configId ?? id).trim();
  if (!repoId || looksLikeLocalPath(repoId) || isOllamaModelId(repoId)) {
    return null;
  }
  // The endpoint's variant lookup matches GGUF files only; an MLX precision folder resolves to its snapshot.
  const variant = target.isGguf ? target.ggufVariant?.trim() : undefined;
  return { kind: "cachedRepo", repoId, ...(variant ? { variant } : {}) };
}

const SNAPSHOT_HASH_RE = /^[0-9a-f]{40}$/i;

/** Collapsed label: the last path segment; an HF snapshot dir reads `repo@hash8`. */
export function shortModelLocation(path: string): string {
  const parts = path.split(/[\\/]+/).filter(Boolean);
  const last = parts.at(-1);
  if (!last) return path;
  let name = last;
  if (SNAPSHOT_HASH_RE.test(last)) {
    const repoDir = parts.at(-2) === "snapshots" ? parts.at(-3) : undefined;
    const repo = repoDir?.startsWith("models--") ? repoDir.split("--").slice(2).join("--") : "";
    name = repo ? `${repo}@${last.slice(0, 8)}` : last.slice(0, 8);
  }
  return parts.length > 1 ? `…/${name}` : name;
}
