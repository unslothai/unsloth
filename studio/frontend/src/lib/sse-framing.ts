// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type SseFrame = {
  event: string;
  remainder: string;
};

const SSE_FRAME_SEPARATOR = /\r?\n\r?\n/;

export function takeSseFrame(buffer: string): SseFrame | null {
  const separator = buffer.match(SSE_FRAME_SEPARATOR);
  const separatorIndex = separator?.index;
  if (separatorIndex === undefined || !separator) {
    return null;
  }
  return {
    event: buffer.slice(0, separatorIndex),
    remainder: buffer.slice(separatorIndex + separator[0].length),
  };
}
