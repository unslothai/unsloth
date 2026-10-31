// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const PLAIN_FENCE_CHUNK_LINES = 64;

export type PlainFenceChunk = readonly string[];

export type PlainFenceChunks = {
  readonly source: string;
  readonly chunks: readonly PlainFenceChunk[];
  readonly trimmedEnd: number;
  readonly tailStart: number;
};

const sameNodes = (a: PlainFenceChunk, b: PlainFenceChunk): boolean =>
  a.length === b.length && a.every((node, index) => node === b[index]);

function readChunks(
  previous: PlainFenceChunks | null | undefined,
  source: string,
  trimmedEnd: number,
  appended: boolean,
): Pick<PlainFenceChunks, "chunks" | "tailStart"> {
  const chunks: PlainFenceChunk[] =
    appended && previous ? previous.chunks.slice(0, -1) : [];
  let offset = appended && previous ? previous.tailStart : 0;
  let chunkStart = offset;
  let tailStart = offset;
  let lineIndex = chunks.length * PLAIN_FENCE_CHUNK_LINES;
  let nodes: string[] = [];

  while (offset <= trimmedEnd) {
    const newline = source.indexOf("\n", offset);
    const end = newline < 0 || newline >= trimmedEnd ? trimmedEnd : newline;
    if (lineIndex > 0) {
      nodes.push("\n");
    }
    if (end > offset) {
      nodes.push(source.slice(offset, end));
    }
    lineIndex += 1;

    if (lineIndex % PLAIN_FENCE_CHUNK_LINES === 0 || end === trimmedEnd) {
      const oldNodes = previous?.chunks[chunks.length];
      chunks.push(oldNodes && sameNodes(oldNodes, nodes) ? oldNodes : nodes);
      tailStart = chunkStart;
      chunkStart = end + 1;
      nodes = [];
    }
    if (end === trimmedEnd) {
      break;
    }
    offset = end + 1;
  }

  return { chunks, tailStart };
}

export function updatePlainFenceChunks(
  previous: PlainFenceChunks | null | undefined,
  source: string,
): PlainFenceChunks {
  if (previous?.source === source) {
    return previous;
  }

  let trimmedEnd = source.length;
  while (trimmedEnd > 0 && source[trimmedEnd - 1] === "\n") {
    trimmedEnd -= 1;
  }

  const appended =
    previous !== null &&
    previous !== undefined &&
    source.slice(0, previous.source.length) === previous.source;
  if (appended && trimmedEnd === previous.trimmedEnd) {
    return { ...previous, source };
  }
  if (trimmedEnd === 0) {
    return { source, chunks: [["\n"]], trimmedEnd, tailStart: 0 };
  }

  return {
    source,
    trimmedEnd,
    ...readChunks(
      previous,
      source,
      trimmedEnd,
      appended && previous.trimmedEnd > 0,
    ),
  };
}
