// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { GgufVariantDetail } from "@/features/hub/inventory";
import { classifyGgufFit } from "@/features/hub/lib/gguf-fit";
import { ggufVariantsMatch } from "@/features/hub/lib/model-identity";

type GgufVariantResources = {
  gpuGb?: number;
  systemRamGb?: number;
};

export function ggufVariantDisplayLabel(
  variant: Pick<GgufVariantDetail, "display_label" | "quant">,
): string {
  return variant.display_label?.trim() || variant.quant;
}

export function ggufVariantDownloadSizeBytes(
  variant: Pick<GgufVariantDetail, "download_size_bytes" | "size_bytes">,
): number {
  return variant.download_size_bytes ?? variant.size_bytes;
}

export function ggufVariantFitRank(
  variant: GgufVariantDetail,
  resources: GgufVariantResources,
): number {
  switch (classifyGgufFit(variant.size_bytes, resources)) {
    case "fits":
      return 0;
    case "marginal":
      return 1;
    case "partial":
    case "ram":
      return 2;
    default:
      return 3;
  }
}

export function compareGgufVariantFitAndSize(
  a: GgufVariantDetail,
  b: GgufVariantDetail,
  resources: GgufVariantResources,
): number {
  const aFit = ggufVariantFitRank(a, resources);
  const bFit = ggufVariantFitRank(b, resources);
  if (aFit !== bFit) return aFit - bFit;
  return aFit === 3 ? a.size_bytes - b.size_bytes : b.size_bytes - a.size_bytes;
}

export function ggufVariantDownloadStatusRank(
  variant: GgufVariantDetail,
): number {
  if (variant.downloaded) return 0;
  if (variant.partial) return 1;
  return 2;
}

export function sortDownloadableGgufVariants(
  variants: readonly GgufVariantDetail[],
  resources: GgufVariantResources,
): GgufVariantDetail[] {
  return [...variants].sort((a, b) => {
    const statusDelta =
      ggufVariantDownloadStatusRank(a) - ggufVariantDownloadStatusRank(b);
    if (statusDelta !== 0) return statusDelta;
    return compareGgufVariantFitAndSize(a, b, resources);
  });
}

export function sortLocalGgufVariants(
  variants: readonly GgufVariantDetail[],
  options: GgufVariantResources & {
    activeGgufVariant?: string | null;
    defaultVariant?: string | null;
  },
): GgufVariantDetail[] {
  return [...variants].sort((a, b) => {
    const aActive = ggufVariantsMatch(a.quant, options.activeGgufVariant);
    const bActive = ggufVariantsMatch(b.quant, options.activeGgufVariant);
    if (aActive !== bActive) return aActive ? -1 : 1;
    const aDefault = ggufVariantsMatch(a.quant, options.defaultVariant);
    const bDefault = ggufVariantsMatch(b.quant, options.defaultVariant);
    if (aDefault !== bDefault) return aDefault ? -1 : 1;
    return compareGgufVariantFitAndSize(a, b, options);
  });
}
