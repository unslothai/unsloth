// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isGgufName } from "./gguf-filename-pick.ts";

export interface DiffusionRouteSearch {
  model: string;
  /** An exact repo filename, never a label. */
  quant?: string;
  /** A quant label (`Q4_K_S`), resolved against the listing. */
  ggufQuant?: string;
}

const trimmed = (value: string | null | undefined): string | null =>
  typeof value === "string" && value.trim().length > 0 ? value.trim() : null;

/** The label rides its own param because `quant` is consumed as a filename. */
export function diffusionRouteSearch(
  model: string,
  meta: { ggufFilename?: string | null; ggufVariant?: string | null },
): DiffusionRouteSearch {
  const filename = trimmed(meta.ggufFilename);
  if (filename) return { model, quant: filename };
  const label = trimmed(meta.ggufVariant);
  return label ? { model, ggufQuant: label } : { model };
}

export function routedGgufFilename(
  search: Pick<DiffusionRouteSearch, "quant">,
): string | null {
  const quant = trimmed(search.quant);
  return quant && isGgufName(quant) ? quant : null;
}

/** A non-filename in `quant` (hand-built link, older producer) is a label too. */
export function routedGgufLabel(
  search: Pick<DiffusionRouteSearch, "quant" | "ggufQuant">,
): string | null {
  const quant = trimmed(search.quant);
  if (quant && isGgufName(quant)) return null;
  return trimmed(search.ggufQuant) ?? quant;
}
