// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** What the chat picker puts in the URL when a diffusion pick routes to /images or /video. */
export interface DiffusionRouteSearch {
  model: string;
  /** An exact repo filename, never a label: the target page uses it verbatim as the gguf filename. */
  quant?: string;
  /** A quant label (`Q4_K_S`) for a pick that has no filename, e.g. a pinned row. The page resolves it against the listing. */
  ggufQuant?: string;
}

const trimmed = (value: string | null | undefined): string | null =>
  typeof value === "string" && value.trim().length > 0 ? value.trim() : null;

/** The search params for a diffusion pick routed out of the chat picker. A pinned row carries only a label, so forwarding the
 *  filename alone dropped it at the URL boundary and the page saw a bare repo id: curated ones still resolved through the
 *  catalog, every other on-device GGUF repo read as a pipeline, which the backend rejects. The label rides its own param
 *  because `quant` is consumed as a filename. */
export function diffusionRouteSearch(
  model: string,
  meta: { ggufFilename?: string | null; ggufVariant?: string | null },
): DiffusionRouteSearch {
  const filename = trimmed(meta.ggufFilename);
  if (filename) return { model, quant: filename };
  const label = trimmed(meta.ggufVariant);
  return label ? { model, ggufQuant: label } : { model };
}
