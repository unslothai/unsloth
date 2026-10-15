// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export interface CropPoint {
  x: number;
  y: number;
}

export interface CropRect extends CropPoint {
  width: number;
  height: number;
}

export interface StagedReferenceImage {
  originalDataUrl: string;
  dataUrl: string;
  crop: CropRect | null;
}

export interface ImageSize {
  width: number;
  height: number;
}

export interface CropRasterCanvas {
  width: number;
  height: number;
  getContext(kind: "2d"): {
    drawImage(
      source: CanvasImageSource,
      sourceX: number,
      sourceY: number,
      sourceWidth: number,
      sourceHeight: number,
      destinationX: number,
      destinationY: number,
      destinationWidth: number,
      destinationHeight: number,
    ): void;
  } | null;
  toDataURL(type: string): string;
}

// Keep aligned with VideoGenerateRequest.reference_images in models/inference.py.
export const MAX_REFERENCE_IMAGE_DATA_URL_LENGTH = 32 * 1024 * 1024;

// fit_h3_reference_image keeps at most a 2048px short edge, so a larger export only wastes the cap;
// the long edge is bounded too.
export const MAX_REFERENCE_CROP_SHORT_EDGE = 2048;
export const MAX_REFERENCE_CROP_LONG_EDGE = 4096;

function finite(value: number): number {
  return Number.isFinite(value) ? value : 0;
}

function between(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}

function boundedSize(size: ImageSize): ImageSize {
  return {
    width: Math.max(0, Math.floor(finite(size.width))),
    height: Math.max(0, Math.floor(finite(size.height))),
  };
}

export function clampCropRect(rect: CropRect, image: ImageSize): CropRect {
  const bounds = boundedSize(image);
  const x = between(Math.floor(finite(rect.x)), 0, bounds.width);
  const y = between(Math.floor(finite(rect.y)), 0, bounds.height);
  const right = between(
    Math.ceil(finite(rect.x) + Math.max(0, finite(rect.width))),
    x,
    bounds.width,
  );
  const bottom = between(
    Math.ceil(finite(rect.y) + Math.max(0, finite(rect.height))),
    y,
    bounds.height,
  );
  return { x, y, width: right - x, height: bottom - y };
}

export function cropRectFromPoints(
  start: CropPoint,
  end: CropPoint,
  image: ImageSize,
): CropRect {
  const bounds = boundedSize(image);
  const startX = between(finite(start.x), 0, bounds.width);
  const startY = between(finite(start.y), 0, bounds.height);
  const endX = between(finite(end.x), 0, bounds.width);
  const endY = between(finite(end.y), 0, bounds.height);
  const left = Math.floor(Math.min(startX, endX));
  const top = Math.floor(Math.min(startY, endY));
  const right = Math.ceil(Math.max(startX, endX));
  const bottom = Math.ceil(Math.max(startY, endY));
  return clampCropRect(
    { x: left, y: top, width: right - left, height: bottom - top },
    bounds,
  );
}

export function moveCropRect(
  rect: CropRect,
  delta: CropPoint,
  image: ImageSize,
): CropRect {
  const crop = clampCropRect(rect, image);
  const bounds = boundedSize(image);
  return {
    ...crop,
    x: between(
      Math.round(crop.x + finite(delta.x)),
      0,
      bounds.width - crop.width,
    ),
    y: between(
      Math.round(crop.y + finite(delta.y)),
      0,
      bounds.height - crop.height,
    ),
  };
}

export function displayPointToSource(
  point: CropPoint,
  display: ImageSize,
  source: ImageSize,
): CropPoint {
  if (display.width <= 0 || display.height <= 0) {
    return { x: 0, y: 0 };
  }
  return {
    x: (point.x / display.width) * source.width,
    y: (point.y / display.height) * source.height,
  };
}

export interface CropImageLoadClaim {
  dataUrl: string;
  cancel(): void;
  isCurrent(): boolean;
}

export function createCropImageLoadGate(): {
  begin(dataUrl: string): CropImageLoadClaim;
} {
  let revision = 0;
  return {
    begin(dataUrl) {
      revision += 1;
      const claimedRevision = revision;
      let cancelled = false;
      return {
        dataUrl,
        cancel() {
          cancelled = true;
        },
        isCurrent() {
          return !cancelled && claimedRevision === revision;
        },
      };
    },
  };
}

export function createReferenceImageEditorActions(callbacks: {
  onApply(dataUrl: string, crop: CropRect | null): void;
  onOpenChange(open: boolean): void;
}): {
  apply(dataUrl: string, crop: CropRect | null): void;
  cancel(): void;
} {
  return {
    apply(dataUrl, crop) {
      callbacks.onApply(dataUrl, crop);
      callbacks.onOpenChange(false);
    },
    cancel() {
      callbacks.onOpenChange(false);
    },
  };
}

export function referenceCropExportSize(crop: ImageSize): ImageSize {
  const longest = Math.max(crop.width, crop.height);
  const shortest = Math.min(crop.width, crop.height);
  const scale = Math.min(
    1,
    MAX_REFERENCE_CROP_SHORT_EDGE / Math.max(1, shortest),
    MAX_REFERENCE_CROP_LONG_EDGE / Math.max(1, longest),
  );
  if (scale >= 1) return { width: crop.width, height: crop.height };
  return {
    width: Math.max(1, Math.floor(crop.width * scale)),
    height: Math.max(1, Math.floor(crop.height * scale)),
  };
}

export function rasterizeReferenceImageCrop(
  source: CanvasImageSource,
  selection: CropRect,
  sourceSize: ImageSize,
  createCanvas: () => CropRasterCanvas = () => document.createElement("canvas"),
): string {
  const crop = clampCropRect(selection, sourceSize);
  if (crop.width < 1 || crop.height < 1) {
    throw new Error("Select an area at least one pixel wide and high.");
  }
  const output = referenceCropExportSize(crop);
  const canvas = createCanvas();
  canvas.width = output.width;
  canvas.height = output.height;
  const context = canvas.getContext("2d");
  if (!context)
    throw new Error("This browser could not start the crop canvas.");
  context.drawImage(
    source,
    crop.x,
    crop.y,
    crop.width,
    crop.height,
    0,
    0,
    output.width,
    output.height,
  );
  const dataUrl = canvas.toDataURL("image/png");
  if (!dataUrl.startsWith("data:image/png;base64,")) {
    throw new Error("This browser could not export the cropped picture.");
  }
  return dataUrl;
}

export function stageReferenceImage(dataUrl: string): StagedReferenceImage {
  return { originalDataUrl: dataUrl, dataUrl, crop: null };
}

export function applyReferenceImageCrop(
  images: StagedReferenceImage[],
  index: number,
  dataUrl: string,
  crop: CropRect | null,
): StagedReferenceImage[] {
  return images.map((image, current) =>
    current === index ? { ...image, dataUrl, crop } : image,
  );
}

export function referenceImageDataUrls(
  images: StagedReferenceImage[],
): string[] {
  return images.map((image) => image.dataUrl);
}

export function referenceImageDataUrlError(dataUrl: string): string | null {
  return dataUrl.length <= MAX_REFERENCE_IMAGE_DATA_URL_LENGTH
    ? null
    : "The cropped picture is too large. Select a smaller area and try again.";
}
