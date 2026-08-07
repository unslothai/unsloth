// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Load kind + repo for a diffusion pick that arrived through the URL. */
export interface DiffusionRoutePick {
  repoId: string;
  opts: { kind: "gguf" | "single_file" | "pipeline"; filename?: string };
}

/** What `/images?model=...&quant=...` (or the Video twin) should load. A pick routed from the chat picker carries no picker metadata, only the two search params, so a bare local single file has to be recognised here: loading one as a pipeline evicts the resident model and then fails on the missing `model_index.json`. `spec` is the target page's catalog entry (`loadSpecFor`), which the chat picker cannot put in the URL, so a curated single-file artifact (LTX-2.3, an FP8 checkpoint) would otherwise arrive with no `quant` and read as a pipeline. */
export function diffusionRoutePick(
  model: string,
  quant?: string | null,
  spec?: { kind: "gguf" | "single_file" | "pipeline"; filename?: string } | null,
): DiffusionRoutePick {
  if (quant) return { repoId: model, opts: { kind: "gguf", filename: quant } };
  // A spec exists only for a catalog repo id (loadSpecFor matches on that), never a local path, so it beats the extension sniffing below.
  if (spec) return { repoId: model, opts: { kind: spec.kind, filename: spec.filename } };
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
