// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Draw a "Mask from text" result into a painter's native-resolution white-on-black layer and rebuild its overlay, so
// the mask stays editable with the same brush. Shared by the inpaint MaskCanvas and LocalizedEditCanvas.

import { MASK_OVERLAY_RGBA, tintMaskPixels } from "./lib/mask-from-text.ts";

function loadMaskImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("Could not read the mask image."));
    img.src = src;
  });
}

/** Replace (or, with ``additive``, union) ``layer`` with the mask at ``dataUrl``, scaled to the layer's size, then
 *  repaint ``display`` from the layer in the ``overlay`` colour. Resolves to the layer as a PNG data URL, or null if
 *  either canvas has no 2D context or ``isCurrent`` says the request went stale (Clear, a new source or a newer mask)
 *  while the image decoded: the canvases are reused, so a stale result never touches them. */
export async function applyMaskToLayer({
  layer,
  display,
  dataUrl,
  additive,
  overlay,
  isCurrent,
}: {
  layer: HTMLCanvasElement;
  display: HTMLCanvasElement;
  dataUrl: string;
  additive: boolean;
  overlay: "mask" | "paint";
  isCurrent?: () => boolean;
}): Promise<string | null> {
  const img = await loadMaskImage(dataUrl);
  if (isCurrent && !isCurrent()) return null;
  const w = layer.width;
  const h = layer.height;
  const lctx = layer.getContext("2d");
  const dctx = display.getContext("2d");
  if (!lctx || !dctx || w === 0 || h === 0) return null;
  lctx.save();
  if (!additive) {
    lctx.globalCompositeOperation = "source-over";
    lctx.fillStyle = "#000";
    lctx.fillRect(0, 0, w, h);
  }
  // White on black either way: "lighten" keeps whatever is already white, so additive is a union.
  lctx.globalCompositeOperation = "lighten";
  lctx.imageSmoothingEnabled = true;
  lctx.drawImage(img, 0, 0, w, h);
  lctx.restore();
  const pixels = lctx.getImageData(0, 0, w, h);
  const tinted = tintMaskPixels(pixels.data, MASK_OVERLAY_RGBA[overlay]);
  dctx.clearRect(0, 0, w, h);
  dctx.putImageData(new ImageData(tinted, w, h), 0, 0);
  return layer.toDataURL("image/png");
}
