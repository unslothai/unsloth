// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isWithinCodeBlock, isWithinMathBlock } from "remend";

const AMBIGUOUS_BOLD_ASTERISK_ITEM_RE = /^ {0,3}\* {1,4}\*\*[ \t]*$/;
const TILDE_FENCE_RE = /^ {0,3}(~{3,})(.*)$/;

function isWithinTildeFence(text: string, lineStart: number): boolean {
  let fenceLength = 0;
  for (const line of text.slice(0, lineStart).split("\n")) {
    const fence = line.match(TILDE_FENCE_RE);
    if (!fence) {
      continue;
    }

    if (fenceLength === 0) {
      fenceLength = fence[1].length;
      continue;
    }

    if (fence[1].length >= fenceLength && fence[2].trim().length === 0) {
      fenceLength = 0;
    }
  }
  return fenceLength > 0;
}

export function stabilizeStreamingMarkdown(
  text: string,
  isStreaming: boolean,
): string {
  if (!isStreaming) {
    return text;
  }

  const lineStart = text.lastIndexOf("\n") + 1;
  const line = text.slice(lineStart);
  if (!AMBIGUOUS_BOLD_ASTERISK_ITEM_RE.test(line)) {
    return text;
  }

  const markerIndex = lineStart + line.indexOf("*");
  if (
    isWithinCodeBlock(text, markerIndex) ||
    isWithinTildeFence(text, lineStart) ||
    isWithinMathBlock(text, markerIndex)
  ) {
    return text;
  }

  // `* **` is both a valid thematic break and the streaming prefix of an
  // asterisk list item whose content starts bold. Buffer the ambiguous line
  // until text arrives, rather than briefly rendering the wrong block type.
  return text.slice(0, lineStart);
}
