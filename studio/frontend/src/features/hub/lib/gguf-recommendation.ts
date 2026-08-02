// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

type RecommendableGgufVariant = {
  quant: string;
  size_bytes: number;
};

type DownloadableGgufVariant = {
  quant: string;
  downloaded?: boolean;
  partial?: boolean;
};

/**
 * Shared recommendation rule for both GGUF pickers: prefer the default variant
 * when it fits, otherwise the largest fitting one, otherwise the smallest.
 *
 * `fitsInMemory` is injected so each surface keeps its own fit thresholds and
 * stays internally consistent with the fit indicator it renders.
 */
export function recommendedGgufVariant<
  Variant extends RecommendableGgufVariant,
>(
  variants: readonly Variant[],
  defaultVariant: string | null,
  fitsInMemory: (sizeBytes: number) => boolean,
): Variant | null {
  if (variants.length === 0) {
    return null;
  }

  const defaultCandidate = defaultVariant
    ? (variants.find((variant) => variant.quant === defaultVariant) ?? null)
    : null;
  if (defaultCandidate && fitsInMemory(defaultCandidate.size_bytes)) {
    return defaultCandidate;
  }

  const fitting = variants.filter((variant) =>
    fitsInMemory(variant.size_bytes),
  );
  if (fitting.length > 0) {
    return fitting.reduce((largest, variant) =>
      variant.size_bytes > largest.size_bytes ? variant : largest,
    );
  }

  return variants.reduce((smallest, variant) =>
    variant.size_bytes < smallest.size_bytes ? variant : smallest,
  );
}

export function shouldShowGgufRecommendation(
  variant: DownloadableGgufVariant | null,
  recommendedVariant: Pick<DownloadableGgufVariant, "quant"> | null,
): boolean {
  return Boolean(
    variant &&
      recommendedVariant &&
      variant.quant === recommendedVariant.quant &&
      !variant.downloaded &&
      !variant.partial,
  );
}
