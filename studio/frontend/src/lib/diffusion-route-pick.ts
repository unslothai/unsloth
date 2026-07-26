// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Load kind + repo for a diffusion pick that arrived through the URL. */
export interface DiffusionRoutePick {
  repoId: string;
  opts: { kind: "gguf" | "single_file" | "pipeline"; filename?: string };
}

/**
 * What `/images?model=...&quant=...` (or the Video twin) should load.
 *
 * A pick routed from the chat picker carries no picker metadata, only the two search params, so
 * a bare local single file has to be recognised here. Loading one as a pipeline evicts the
 * resident model and then fails on the missing `model_index.json`, since an explicit
 * `model_kind` wins over the backend's filename sniffing. Split into (parent dir, basename) the
 * same way the pages' own picker handlers do.
 */
export function diffusionRoutePick(
  model: string,
  quant?: string | null,
): DiffusionRoutePick {
  if (quant) return { repoId: model, opts: { kind: "gguf", filename: quant } };
  const norm = model.replace(/\\/g, "/");
  const slash = norm.lastIndexOf("/");
  const filename = slash >= 0 ? norm.slice(slash + 1) : norm;
  const dir = slash >= 0 ? norm.slice(0, slash) : ".";
  const lower = filename.toLowerCase();
  if (lower.endsWith(".gguf")) {
    return { repoId: dir, opts: { kind: "gguf", filename } };
  }
  if (lower.endsWith(".safetensors")) {
    return { repoId: dir, opts: { kind: "single_file", filename } };
  }
  // A repo id: the curated/pipeline case, loaded as before.
  return { repoId: model, opts: { kind: "pipeline" } };
}
