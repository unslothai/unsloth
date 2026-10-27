// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export interface DiffusionRoutePick {
  repoId: string;
  opts: { kind: "gguf" | "single_file" | "pipeline"; filename?: string };
}

/** Chat-routed picks carry only search params, so recognise bare local single files here (a pipeline
 * load would evict the resident model and fail). `spec` covers curated single-file artifacts. */
export function diffusionRoutePick(
  model: string,
  quant?: string | null,
  spec?: { kind: "gguf" | "single_file" | "pipeline"; filename?: string } | null,
): DiffusionRoutePick {
  if (quant) return { repoId: model, opts: { kind: "gguf", filename: quant } };
  // A spec exists only for catalog repo ids, so it beats extension sniffing.
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
  return { repoId: model, opts: { kind: "pipeline" } };
}
