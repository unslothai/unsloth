// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { canvasBlob, decodeImage } from "./image";
import { DEFAULT_ADJUSTMENTS, normalizedSize, validUpload } from "./model";
import {
  beginBackgroundUpload,
  failBackgroundUpload,
  finishBackgroundUpload,
} from "./storage";

/** Errors land in the background store, where the settings and onboarding pickers read them. */
export async function uploadBackgroundFile(file: File): Promise<void> {
  const ticket = beginBackgroundUpload();
  if (!validUpload(file)) {
    failBackgroundUpload(ticket, "file");
    return;
  }
  try {
    const decoded = await decodeImage(file);
    try {
      const [width, height] = normalizedSize(decoded.width, decoded.height);
      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext("2d");
      if (!ctx) throw new Error("decode");
      ctx.drawImage(decoded.source, 0, 0, width, height);
      await finishBackgroundUpload(ticket, {
        ...DEFAULT_ADJUSTMENTS,
        blob: await canvasBlob(canvas),
      });
    } finally {
      decoded.close();
    }
  } catch (error) {
    failBackgroundUpload(
      ticket,
      error instanceof Error && error.message === "dimensions"
        ? "dimensions"
        : "decode",
    );
  }
}

export async function setBackgroundCanvas(
  canvas: HTMLCanvasElement,
): Promise<void> {
  const ticket = beginBackgroundUpload();
  try {
    await finishBackgroundUpload(ticket, {
      ...DEFAULT_ADJUSTMENTS,
      blob: await canvasBlob(canvas),
    });
  } catch {
    failBackgroundUpload(ticket, "storage");
  }
}
