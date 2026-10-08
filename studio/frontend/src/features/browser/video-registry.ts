// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isTauri } from "@/lib/api-base";

// The player of each video tab, so the toolbar can act on the frame on screen.
const videos = new Map<string, HTMLVideoElement>();

export function registerTabVideo(tabId: string, video: HTMLVideoElement): () => void {
  videos.set(tabId, video);
  return () => {
    if (videos.get(tabId) === video) videos.delete(tabId);
  };
}

export function tabVideo(tabId: string): HTMLVideoElement | null {
  return videos.get(tabId) ?? null;
}

/** The desktop app can't write images to the clipboard. */
export const canCopyVideoFrame = (): boolean =>
  !isTauri && typeof ClipboardItem !== "undefined" && typeof navigator.clipboard?.write === "function";

/** Copies the frame on screen as a PNG; false where the clipboard takes no images. */
export async function copyVideoFrame(video: HTMLVideoElement): Promise<boolean> {
  if (!canCopyVideoFrame()) return false;
  if (!video.videoWidth || !video.videoHeight) return false;
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d")?.drawImage(video, 0, 0);
  const png = new Promise<Blob>((resolve, reject) =>
    canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error("no frame"))), "image/png"),
  );
  try {
    await navigator.clipboard.write([new ClipboardItem({ "image/png": png })]);
    return true;
  } catch {
    return false;
  }
}
