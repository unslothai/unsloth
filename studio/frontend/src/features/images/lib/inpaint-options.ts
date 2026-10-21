// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// AUTOMATIC1111's inpaint "Mask blur" and "Inpaint area: Only masked" + "Only masked padding"
// (modules/processing.py:1563-1569), sent as the backend's mask_blur / inpaint_only_masked /
// inpaint_padding. The defaults send nothing, so an untouched form makes today's exact request.
export const MASK_BLUR_DEFAULT = 0;
export const MASK_BLUR_MAX = 64;
// A1111's default "Only masked padding, pixels".
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

/** Request fields for an inpaint run: only what differs from the default, and "Only masked" only
 *  when the loaded model's inpaint pipeline can crop to the mask. */
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
