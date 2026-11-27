// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Records start with a four-byte big-endian length. Zero length is a heartbeat. */
export async function readFrames(
  stream: ReadableStream<Uint8Array>,
  onFrame: (frame: Uint8Array<ArrayBuffer>) => void,
): Promise<void> {
  const reader = stream.getReader();
  const header = new DataView(new ArrayBuffer(4));
  let headerSize = 0;
  let frame: Uint8Array<ArrayBuffer> | undefined;
  let position = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      for (let offset = 0; offset < value.length;) {
        if (!frame) {
          header.setUint8(headerSize++, value[offset++]);
          if (headerSize !== 4) continue;
          headerSize = 0;
          const size = header.getUint32(0);
          if (size > 4 * 1024 * 1024) throw new Error("Viewport frame is too large.");
          if (size) frame = new Uint8Array(size);
          continue;
        }
        const count = Math.min(value.length - offset, frame.length - position);
        frame.set(value.subarray(offset, offset + count), position);
        position += count;
        offset += count;
        if (position === frame.length) {
          onFrame(frame);
          frame = undefined;
          position = 0;
        }
      }
    }
  } finally {
    await reader.cancel().catch(() => undefined);
    reader.releaseLock();
  }
}
