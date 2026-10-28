// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A1111 inpaint mask options (modules/processing.py:1563-1569); the defaults send nothing.
export const MASK_BLUR_DEFAULT = 0;
export const MASK_BLUR_MAX = 64;
export const INPAINT_PADDING_DEFAULT = 32;
export const INPAINT_PADDING_MAX = 256;

export interface InpaintMaskOptions {
  maskBlur: number;
  onlyMasked: boolean;
  padding: number;
}

export interface InpaintMaskRequestFields {
  mask_blur?: number;
  inpaint_only_masked?: boolean;
  inpaint_padding?: number;
}

/** Only non-default fields; "Only masked" only when the pipeline supports it. */
export function inpaintMaskRequestFields(
  options: InpaintMaskOptions,
  supportsOnlyMasked: boolean,
): InpaintMaskRequestFields {
  const out: InpaintMaskRequestFields = {};
  const blur = Math.round(
    Math.min(MASK_BLUR_MAX, Math.max(0, options.maskBlur)),
  );
  if (Number.isFinite(blur) && blur > 0) out.mask_blur = blur;
  if (options.onlyMasked && supportsOnlyMasked) {
    out.inpaint_only_masked = true;
    const padding = Math.round(
      Math.min(INPAINT_PADDING_MAX, Math.max(0, options.padding)),
    );
    out.inpaint_padding = Number.isFinite(padding)
      ? padding
      : INPAINT_PADDING_DEFAULT;
  }
  return out;
}

interface InpaintMaskRecord {
  workflow?: string | null;
  mask_blur?: number | null;
  inpaint_only_masked?: boolean | null;
  inpaint_padding?: number | null;
}

/** The mask options a restore applies. An inpaint record omits its defaults, so an absent field resets to it. */
export function restoredInpaintMaskOptions(
  image: InpaintMaskRecord,
): Partial<InpaintMaskOptions> {
  const inpaint = image.workflow === "inpaint";
  const out: Partial<InpaintMaskOptions> = {};
  if (typeof image.mask_blur === "number") out.maskBlur = image.mask_blur;
  else if (inpaint) out.maskBlur = MASK_BLUR_DEFAULT;
  if (typeof image.inpaint_only_masked === "boolean")
    out.onlyMasked = image.inpaint_only_masked;
  else if (inpaint) out.onlyMasked = false;
  if (typeof image.inpaint_padding === "number")
    out.padding = image.inpaint_padding;
  else if (inpaint) out.padding = INPAINT_PADDING_DEFAULT;
  return out;
}

/** A restored "Only masked" a pipeline that cannot crop will silently drop, for the not-applied toast. */
export function inpaintOptionsNotApplied(
  settings: { inpaint_only_masked?: boolean | null; inpaint_padding?: number | null },
  supportsOnlyMasked: boolean,
): string[] {
  if (settings.inpaint_only_masked !== true || supportsOnlyMasked) return [];
  const padding =
    typeof settings.inpaint_padding === "number"
      ? settings.inpaint_padding
      : INPAINT_PADDING_DEFAULT;
  return [`inpaint area only masked (padding ${padding})`];
}
