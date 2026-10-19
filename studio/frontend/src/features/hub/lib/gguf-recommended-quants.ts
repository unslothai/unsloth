// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { recommendedQuantForDevice } from "@/features/model-picker/components/model-selector/model-catalog";
import {
  groupGgufVariantsForPicker,
  preferredGgufVariantByGroup,
} from "@/features/model-picker/components/model-selector/variant-presentation";
import type { GgufFitClass } from "@/lib/gguf-fit";
import type { GgufVariantDetail } from "../inventory";

/** Per group, the quant the chat picker stars (`effectiveRecommendedByGroup` in pickers.tsx).
 *  A null `fitOf` (no measured budget) falls back to the repo default. */
export function recommendedGgufQuantsByGroup(
  variants: readonly GgufVariantDetail[],
  defaultVariant: string | null,
  fitOf: ((sizeBytes: number) => GgufFitClass) | null,
): string[] {
  const groups = groupGgufVariantsForPicker(variants);
  const preferredByGroup = preferredGgufVariantByGroup(groups, defaultVariant);
  const quants: string[] = [];
  for (const group of groups) {
    const preferred = preferredByGroup.get(group.key) ?? null;
    const pick = fitOf
      ? (recommendedQuantForDevice(group.variants, fitOf, preferred) ??
        preferred)
      : preferred;
    if (pick) quants.push(pick.quant);
  }
  return quants;
}
