// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type GgufFitClass,
  type GgufVariantSizes,
  classifyVariantFitOnCard,
} from "../../../../lib/gguf-fit.ts";
import { recommendedQuantForDevice } from "./model-catalog.ts";
import {
  type GgufVariantPresentationGroup,
  groupGgufVariantsForPicker,
  preferredGgufVariantByGroup,
} from "./variant-presentation.ts";

type OrderableVariant = GgufVariantSizes & {
  filename: string;
  quant: string;
  display_label?: string | null;
  downloaded?: boolean;
};

export function recommendedQuantByGroup<T extends OrderableVariant>(
  groups: readonly GgufVariantPresentationGroup<T>[],
  defaultVariant: string | null,
  {
    fitOf,
    anyBudget,
    budgetKnown,
    cardGb,
  }: {
    fitOf: (sizeBytes: number) => GgufFitClass;
    anyBudget: boolean;
    budgetKnown: boolean;
    cardGb: number;
  },
): Map<string, string> {
  const preferredByGroup = preferredGgufVariantByGroup(groups, defaultVariant);
  const recommended = new Map<string, string>();
  for (const group of groups) {
    const preferred = preferredByGroup.get(group.key) ?? null;
    const pick =
      !anyBudget && !budgetKnown
        ? preferred
        : (recommendedQuantForDevice(
            group.variants,
            fitOf,
            preferred,
            cardGb,
          ) ?? preferred);
    if (pick) {
      recommended.set(group.key, pick.quant);
    }
  }
  return recommended;
}

/** Downloaded first, then fits / tight / OOM; recommended leads its tier; fits largest first, else smallest. */
export function orderGgufVariantsForPicker<T extends OrderableVariant>(
  groups: readonly GgufVariantPresentationGroup<T>[],
  fitOfVariant: (variant: T) => GgufFitClass,
  recommendedByGroup: ReadonlyMap<string, string>,
): T[] {
  const tierOf = (variant: T) => {
    const fit = fitOfVariant(variant);
    if (fit === "oom") {
      return 4;
    }
    const base = fit === "fits" ? 0 : 1;
    return variant.downloaded ? base : base + 2;
  };
  return groups.flatMap((group) => {
    const recommended = recommendedByGroup.get(group.key);
    return [...group.variants].sort((a, b) => {
      const aTier = tierOf(a);
      const bTier = tierOf(b);
      if (aTier !== bTier) {
        return aTier - bTier;
      }
      const aIsRec = a.quant === recommended;
      const bIsRec = b.quant === recommended;
      if (aIsRec !== bIsRec) {
        return aIsRec ? -1 : 1;
      }
      const fitsInGpu = aTier === 0 || aTier === 2;
      return fitsInGpu
        ? b.size_bytes - a.size_bytes
        : a.size_bytes - b.size_bytes;
    });
  });
}

export function pickerOrderedGgufVariants<T extends OrderableVariant>(
  variants: readonly T[],
  defaultVariant: string | null,
  budget: {
    fitOf: (sizeBytes: number) => GgufFitClass;
    anyBudget: boolean;
    budgetKnown: boolean;
    cardGb: number;
  },
): { ordered: T[]; recommended: ReadonlySet<string> } {
  const groups = groupGgufVariantsForPicker(variants);
  const recommendedByGroup = recommendedQuantByGroup(
    groups,
    defaultVariant,
    budget,
  );
  const ordered = orderGgufVariantsForPicker(
    groups,
    (variant) => classifyVariantFitOnCard(budget.fitOf, variant, budget.cardGb),
    recommendedByGroup,
  );
  return { ordered, recommended: new Set(recommendedByGroup.values()) };
}
