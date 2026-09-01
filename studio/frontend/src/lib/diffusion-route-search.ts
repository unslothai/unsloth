// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isGgufName } from "./gguf-filename-pick.ts";

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

type DiffusionPickSource = "hub" | "lora" | "exported" | "local" | "external";

/** Resolve a complete cached pipeline row to the exact snapshot that established its manifest.
 *
 * Active-cache rows repeat the Hub id as loadId and must keep the Hub preflight/staging path.
 * A distinct loadId is a pinned snapshot/path; treating only that identity as local avoids resolving
 * the bare Hub id through a different cache (or the network) after admission from on-device evidence.
 */
export function diffusionPipelineLoadTarget(
  model: string,
  meta: { loadId?: string | null; source: DiffusionPickSource },
): { repoId: string; source: DiffusionPickSource } {
  const loadId = trimmed(meta.loadId);
  return loadId && loadId !== model.trim()
    ? { repoId: loadId, source: "local" }
    : { repoId: model, source: meta.source };
}

/** Search params for a diffusion pick routed out of the chat picker. The label rides its own param because `quant` is
 *  consumed as a filename; forwarding the filename alone left a pinned row as a bare repo id that read as a pipeline. */
export function diffusionRouteSearch(
  model: string,
  meta: { ggufFilename?: string | null; ggufVariant?: string | null },
): DiffusionRouteSearch {
  const filename = trimmed(meta.ggufFilename);
  if (filename) return { model, quant: filename };
  const label = trimmed(meta.ggufVariant);
  return label ? { model, ggufQuant: label } : { model };
}

/** The exact .gguf a routed pick names, if it names one. */
export function routedGgufFilename(
  search: Pick<DiffusionRouteSearch, "quant">,
): string | null {
  const quant = trimmed(search.quant);
  return quant && isGgufName(quant) ? quant : null;
}

/** The quant label a routed pick carries. A non-filename in `quant` (hand-built link, older producer) is a label too, so it
 *  joins ggufQuant to be resolved rather than posted verbatim. */
export function routedGgufLabel(
  search: Pick<DiffusionRouteSearch, "quant" | "ggufQuant">,
): string | null {
  const quant = trimmed(search.quant);
  if (quant && isGgufName(quant)) return null;
  return trimmed(search.ggufQuant) ?? quant;
}
