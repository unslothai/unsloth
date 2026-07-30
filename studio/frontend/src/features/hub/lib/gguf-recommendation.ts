// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

type DownloadableVariant = {
  downloaded?: boolean;
  partial?: boolean;
};

export function recommendedDownloadableGgufVariant<
  Variant extends DownloadableVariant,
>(sortedVariants: readonly Variant[]): Variant | null {
  return (
    sortedVariants.find(
      (variant) => !(variant.downloaded || variant.partial),
    ) ?? null
  );
}
