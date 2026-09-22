// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  UI_FONT_SIZE_RANGE,
  useAppearanceCustomStore,
} from "@/features/settings/stores/appearance-custom-store";

/**
 * The JS twin of --ui-space-scale, for geometry that only exists in JS:
 * virtualizer slot heights and anything measured against them. Everything
 * else scales through the CSS variable.
 */
export function useUiSpaceScale(): number {
  const uiFontSize = useAppearanceCustomStore((s) => s.customization.uiFontSize);
  return (
    (uiFontSize ?? UI_FONT_SIZE_RANGE.default) / UI_FONT_SIZE_RANGE.default
  );
}
