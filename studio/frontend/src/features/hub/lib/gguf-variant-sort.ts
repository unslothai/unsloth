


import type { GgufVariantDetail } from "@/features/hub/inventory";
import { formatBytes } from "@/features/hub/lib/format";
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
