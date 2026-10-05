// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Which tool calls stay visible whatever the fold and collapse preferences say. One place for the
// rule, so the tool group that shows the run and the Thinking header that counts it agree.

import { hasCreatedFiles } from "./sandbox-files.ts";

export interface ToolPartLike {
  readonly type: string;
  readonly toolName?: string;
  readonly toolCallId?: string;
  readonly result?: unknown;
}

/** An MCP-style result with images in it, the shape the fallback card shows previews for. */
export function carriesImages(result: unknown): boolean {
  if (typeof result !== "object" || result === null) {
    return false;
  }
  const v = result as { text?: unknown; images?: unknown };
  return (
    typeof v.text === "string" &&
    Array.isArray(v.images) &&
    v.images.length > 0 &&
    v.images.every(
      (img: unknown) =>
        typeof img === "object" &&
        img !== null &&
        typeof (img as { data?: unknown }).data === "string" &&
        typeof (img as { mimeType?: unknown }).mimeType === "string",
    )
  );
}

/** Canvases, Python, generated images, results that carry images and anything that wrote a
 *  file: their output lives only in the card, so it is never tucked away. */
export function holdsOwnOutput(part: ToolPartLike): boolean {
  return (
    part.type === "tool-call" &&
    (part.toolName === "render_html" ||
      part.toolName === "python" ||
      part.toolName === "image_generation" ||
      carriesImages(part.result) ||
      hasCreatedFiles(part.toolName, part.result))
  );
}

/** A blocking allow or deny prompt must never sit behind a collapsed header. */
export function awaitsConfirmation(
  part: ToolPartLike,
  toolConfirmations: Record<string, unknown>,
): boolean {
  return (
    part.type === "tool-call" &&
    part.toolCallId !== undefined &&
    Object.prototype.hasOwnProperty.call(toolConfirmations, part.toolCallId)
  );
}

/** Whether the run of calls at [start, end] stays visible instead of folding. */
export function toolRunIsExempt(
  parts: readonly ToolPartLike[],
  start: number,
  end: number,
  toolConfirmations: Record<string, unknown>,
): boolean {
  for (let i = start; i <= end && i < parts.length; i += 1) {
    const part = parts[i];
    if (!part) {
      continue;
    }
    if (holdsOwnOutput(part) || awaitsConfirmation(part, toolConfirmations)) {
      return true;
    }
  }
  return false;
}
