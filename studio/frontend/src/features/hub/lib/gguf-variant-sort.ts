// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { GgufVariantDetail } from "@/features/hub/inventory";
import { formatBytes } from "@/features/hub/lib/format";
import { ggufVariantsMatch } from "@/features/hub/lib/model-identity";
import { classifyGgufFit } from "@/lib/gguf-fit";

export type GgufVariantResources = {
  gpuGb?: number;
  systemRamGb?: number;
  /** The saved VRAM Budget, so the sort ranks against the line the loader will
   *  actually admit at rather than against the default. */
  budgetFraction?: number;
  /** GPUs gpuGb sums, so the sort charges the loader's per-card VRAM reserve as
   *  many times as the loader does. */
  gpuCount?: number;
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

/** Conservative byte basis for a pre-download fit verdict.
 *
 * Before a quant is on disk the load planner cannot inspect its GGUF headers,
 * so the Hub's fit badge has to work from the variant listing. `size_bytes`
 * names only the main weights, while `download_size_bytes` also includes the
 * projector and drafter GGUFs a default launch can open beside them. It can
 * also contain small support files that are not resident, so it is deliberately
 * a conservative proxy rather than a replacement for the on-disk planner.
 * Scoring only the main weights called a Muse Glimmer Q3 load a full 16 GiB GPU
 * fit even though its required projector pushed the launch well over the card.
 *
 * Take the larger figure rather than trusting `download_size_bytes` blindly:
 * old or incomplete listings may report it as zero, and a fit estimate must
 * never become smaller than the weights themselves. */
export function ggufVariantFitSizeBytes(
  variant: Pick<GgufVariantDetail, "download_size_bytes" | "size_bytes">,
): number {
  return Math.max(variant.size_bytes, variant.download_size_bytes ?? 0);
}

/** The Hub-wide fit verdict for a concrete GGUF variant. */
export function classifyGgufVariantFit(
  variant: Pick<GgufVariantDetail, "download_size_bytes" | "size_bytes">,
  resources: GgufVariantResources,
) {
  return classifyGgufFit(ggufVariantFitSizeBytes(variant), resources);
}

type GgufVariantTransfer = Pick<
  GgufVariantDetail,
  "download_size_bytes" | "size_bytes" | "download_remaining_bytes" | "partial"
>;

/** What starting this variant now would transfer. On a partial that is the
 * remainder the backend measured; everywhere else it is the full size. An
 * unmeasured partial falls back to the total, the costlier of the two. */
export function ggufVariantTransferBytes(variant: GgufVariantTransfer): number {
  const total = ggufVariantDownloadSizeBytes(variant);
  if (!variant.partial) return total;
  const remaining = variant.download_remaining_bytes;
  return typeof remaining === "number" && remaining >= 0 ? remaining : total;
}

/** Labelled form of the above. A partial says what is LEFT: the full size there
 * reads as "this downloads all over again", which is only true for a one-file
 * quant. */
export function ggufVariantTransferLabel(variant: GgufVariantTransfer): string {
  const label = formatBytes(ggufVariantTransferBytes(variant));
  return variant.partial ? `${label} left` : label;
}

export function ggufVariantFitRank(
  variant: GgufVariantDetail,
  resources: GgufVariantResources,
): number {
  switch (classifyGgufVariantFit(variant, resources)) {
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
