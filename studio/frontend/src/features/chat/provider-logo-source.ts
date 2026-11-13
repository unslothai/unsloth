// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  isCustomProviderType,
  isProviderFaviconDataUrl,
} from "./external-providers";
import { providerLogoPath } from "./provider-logo-path";

/** Resolve the rendered logo source: an opted-in custom-provider site icon wins, else the static mark. */
export function resolveApiProviderLogoSrc(
  providerType: string | undefined | null,
  faviconDataUrl?: string | null | undefined,
  baseUrl = "",
): string | undefined {
  if (
    isCustomProviderType(providerType) &&
    isProviderFaviconDataUrl(faviconDataUrl)
  ) {
    return faviconDataUrl;
  }
  const path = providerLogoPath(providerType);
  return path ? `${baseUrl}${path.slice(1)}` : undefined;
}

/**
 * True for a dark, colourless glyph (e.g. a black mark on transparency) that
 * vanishes on dark surfaces and should get the same `dark:invert` as preset marks.
 * Input is canvas RGBA. Pixels under alpha 32 are background; the rest must be
 * at least 16 pixels, 90% of them near-grey (channel spread <= 40), and their
 * alpha-weighted mean luminance under 0.35. Weighting keeps faint anti-aliased
 * grey rims from outvoting the solid body of the glyph.
 */
export function isDarkMonochromeIcon(rgba: ArrayLike<number>): boolean {
  let opaque = 0;
  let grey = 0;
  let weight = 0;
  let luminance = 0;
  for (let i = 0; i + 3 < rgba.length; i += 4) {
    const a = rgba[i + 3];
    if (a < 32) continue;
    const r = rgba[i], g = rgba[i + 1], b = rgba[i + 2];
    opaque++;
    if (Math.max(r, g, b) - Math.min(r, g, b) <= 40) grey++;
    weight += a;
    luminance += (a * (0.2126 * r + 0.7152 * g + 0.0722 * b)) / 255;
  }
  return opaque >= 16 && grey >= opaque * 0.9 && luminance / weight < 0.35;
}
