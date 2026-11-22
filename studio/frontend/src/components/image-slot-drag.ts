// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const IMAGE_SLOT_DRAG_TYPE = "application/x-unsloth-image-slot";

const NATIVE_IMAGE_EXTS = ["jpg", "jpeg", "png", "webp", "gif"];

// The desktop shell reports the drop natively, after dragend.
const NATIVE_DROP_GRACE_MS = 1000;

let draggedSlotImage: string | null = null;
let endedAt: number | null = null;

// Capture runs before the slot's own onDragStart re-arms it.
if (typeof document !== "undefined") {
  document.addEventListener(
    "dragstart",
    () => {
      draggedSlotImage = null;
      endedAt = null;
    },
    true,
  );
}

export function startImageSlotDrag(dataUrl: string): void {
  draggedSlotImage = dataUrl;
  endedAt = null;
}

export function endImageSlotDrag(now = Date.now()): void {
  endedAt = now;
}

export function draggedImageSlot(now = Date.now()): string | null {
  if (endedAt !== null && now - endedAt > NATIVE_DROP_GRACE_MS) {
    draggedSlotImage = null;
    endedAt = null;
  }
  return draggedSlotImage;
}

export type NativeImageDrop = "none" | "image" | "in-app" | "unsupported";

export function classifyNativeImageDrop(
  path: string | undefined,
): NativeImageDrop {
  if (!path) return "none";
  // An image dragged out of the page reaches the native handler as its URL, not a file.
  if (/^(data|blob|https?):/i.test(path)) return "in-app";
  const ext = path.split(".").pop()?.toLowerCase() ?? "";
  return NATIVE_IMAGE_EXTS.includes(ext) ? "image" : "unsupported";
}
