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

/** The search params for a diffusion pick routed out of the chat picker. A pinned row carries only a label, so forwarding the
 *  filename alone left the page a bare repo id: curated repos still resolved through the catalog, every other on-device GGUF
 *  repo read as a pipeline. The label rides its own param because `quant` is consumed as a filename. */
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

/** The quant label a routed pick carries. `quant` is used verbatim as a filename, so a value there that is not one is a
 *  label (a hand-built link, or a producer that predates the split) and joins ggufQuant to be resolved rather than posted. */
export function routedGgufLabel(
  search: Pick<DiffusionRouteSearch, "quant" | "ggufQuant">,
): string | null {
  const quant = trimmed(search.quant);
  if (quant && isGgufName(quant)) return null;
  return trimmed(search.ggufQuant) ?? quant;
}
