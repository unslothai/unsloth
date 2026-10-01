// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  UI_FONT_SIZE_RANGE,
  useAppearanceCustomStore,
} from "@/features/settings/stores/appearance-custom-store";
import {
  useInterfaceScaleStore,
  webInterfaceScaleFactor,
} from "@/features/settings/stores/interface-scale-store";

/**
 * The JS twin of --ui-space-scale, for geometry that only exists in JS:
 * virtualizer slot heights and anything measured against them. Everything
 * else scales through the CSS variable. Includes the browser interface scale.
 */
export function useUiSpaceScale(): number {
  const uiFontSize = useAppearanceCustomStore((s) => s.customization.uiFontSize);
  const interfaceScale = useInterfaceScaleStore((s) => s.scale);
  return (
    ((uiFontSize ?? UI_FONT_SIZE_RANGE.default) / UI_FONT_SIZE_RANGE.default) *
    webInterfaceScaleFactor(interfaceScale)
  );
}
