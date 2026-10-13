// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// decode_b64_image (studio/backend/core/inference/diffusion.py) refuses a source image above this per side.
export const MAX_UPLOAD_IMAGE_SIDE = 4096;

/** The size that fits `maxSide` at the same aspect, or null when the image already fits. */
export function fitWithinMaxSide(
  width: number,
  height: number,
  maxSide: number,
): { width: number; height: number } | null {
  const longest = Math.max(width, height);
  if (!(width > 0 && height > 0 && longest > maxSide)) return null;
  const scale = maxSide / longest;
  return {
    width: Math.min(maxSide, Math.max(1, Math.round(width * scale))),
    height: Math.min(maxSide, Math.max(1, Math.round(height * scale))),
  };
}

/** JPEG only for a JPEG source: a JPEG export paints transparent pixels black. */
export function fitOutputType(dataUrl: string): "image/jpeg" | "image/png" {
  return /^data:image\/(jpeg|jpg|pjpeg)[;,]/i.test(dataUrl) ? "image/jpeg" : "image/png";
}

/** Downscale an image data URL so neither side exceeds `maxSide`; anything that fits, or that this
 *  browser cannot decode, is returned unchanged so the server still reports it. */
export async function fitImageDataUrl(dataUrl: string, maxSide: number): Promise<string> {
  let image: HTMLImageElement;
  try {
    image = await new Promise<HTMLImageElement>((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = dataUrl;
    });
  } catch {
    return dataUrl;
  }
  // naturalWidth/Height and drawImage both follow EXIF orientation, so the result is upright.
  const size = fitWithinMaxSide(image.naturalWidth, image.naturalHeight, maxSide);
  if (!size) return dataUrl;
  const canvas = document.createElement("canvas");
  canvas.width = size.width;
  canvas.height = size.height;
  const context = canvas.getContext("2d");
  if (!context) return dataUrl;
  context.imageSmoothingQuality = "high";
  context.drawImage(image, 0, 0, size.width, size.height);
  const fitted = canvas.toDataURL(fitOutputType(dataUrl), 0.92);
  return fitted.startsWith("data:image/") ? fitted : dataUrl;
}
