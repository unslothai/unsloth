// SPDX-License-Identifier: AGPL-3.0-only

import assert from "node:assert/strict";
import test from "node:test";

import { copyPngTextChunks } from "../src/features/images/lib/info-stamp.ts";

const SIGNATURE = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
const bytes = (text: string) => new TextEncoder().encode(text);

function chunk(type: string, data: Uint8Array = new Uint8Array()) {
  const result = new Uint8Array(data.length + 12);
  new DataView(result.buffer).setUint32(0, data.length);
  result.set(bytes(type), 4);
  result.set(data, 8);
  // Structural fixtures only: sentinel CRC bytes exercise verbatim copying.
  result.set([1, 2, 3, 4], 8 + data.length);
  return result;
}

function png(...chunks: Uint8Array[]) {
  return new Uint8Array([...SIGNATURE, ...chunks.flatMap((part) => [...part])]);
}

function viewWithOffset(input: Uint8Array, offset: number) {
  const storage = new Uint8Array(offset + input.length + 9).fill(0xa5);
  storage.set(input, offset);
  return storage.subarray(offset, offset + input.length);
}

for (const sourceOffset of [0, 7]) {
  for (const targetOffset of [0, 11]) {
    test(`text chunks survive PNG views at offsets ${sourceOffset}/${targetOffset}`, () => {
      const metadata = [
        chunk("tEXt", bytes("unsloth\0{}")),
        chunk("iTXt", bytes("recipe\0\0\0\0\0source")),
        chunk("zTXt", new Uint8Array([114, 0, 0, 120, 156])),
      ];
      const source = viewWithOffset(
        png(chunk("IHDR", new Uint8Array([1])), ...metadata,
          chunk("IDAT", new Uint8Array([10])), chunk("IEND")),
        sourceOffset,
      );
      const targetHeader = chunk("IHDR", new Uint8Array([2]));
      const nonText = chunk("pHYs", new Uint8Array([8]));
      const targetPixels = chunk("IDAT", new Uint8Array([20]));
      const target = viewWithOffset(
        png(targetHeader, nonText, chunk("tEXt", bytes("stale\0text")),
          chunk("iTXt", bytes("stale\0international")),
          chunk("zTXt", bytes("stale\0compressed")), targetPixels, chunk("IEND")),
        targetOffset,
      );
      const sourceBefore = new Uint8Array(source.buffer).slice();
      const targetBefore = new Uint8Array(target.buffer).slice();
      const result = copyPngTextChunks(source, target);
      const expected = png(targetHeader, ...metadata, nonText, targetPixels, chunk("IEND"));

      assert.deepEqual(result, expected);
      assert.deepEqual(new Uint8Array(source.buffer), sourceBefore);
      assert.deepEqual(new Uint8Array(target.buffer), targetBefore);
      assert.deepEqual(copyPngTextChunks(source, result), result);
    });
  }
}
