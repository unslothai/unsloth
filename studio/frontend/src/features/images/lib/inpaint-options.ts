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
