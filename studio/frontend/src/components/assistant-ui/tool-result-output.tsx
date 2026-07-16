// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { useMemo, useState } from "react";

/** Tail-line cap so a huge output never mounts a megabyte <pre> block. */
const TAIL_LINES = 2000;
/** Char backstop for pathological single-line outputs. */
const TAIL_CHARS = 200_000;

interface Tail {
  visible: string;
  hiddenLines: number;
  hiddenChars: number;
}

export function tailText(text: string): Tail {
  let visible = text;
  let hiddenLines = 0;
  let hiddenChars = 0;
  const lines = visible.split("\n");
  if (lines.length > TAIL_LINES) {
    hiddenLines = lines.length - TAIL_LINES;
    visible = lines.slice(hiddenLines).join("\n");
  }
  if (visible.length > TAIL_CHARS) {
    hiddenChars = visible.length - TAIL_CHARS;
    visible = visible.slice(hiddenChars);
  }
  return { visible, hiddenLines, hiddenChars };
}

/**
 * Finished-tool output pane: renders the tail (~2000 lines) with a "Show all"
 * toggle so a large output stays scrollable without janking the DOM. Copy
 * buttons still copy the FULL text (owned by the caller), not the tail.
 */
export function ToolResultOutput({ text }: { text: string }) {
  const [showAll, setShowAll] = useState(false);
  const tail = useMemo(() => tailText(text), [text]);
  const truncated = !showAll && (tail.hiddenLines > 0 || tail.hiddenChars > 0);

  return (
    <>
      {truncated && (
        <button
          type="button"
          onClick={() => setShowAll(true)}
          className="mt-1 rounded px-1.5 py-0.5 text-xs text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
        >
          {tail.hiddenLines > 0
            ? `Show all (${tail.hiddenLines.toLocaleString()} earlier lines hidden)`
            : `Show all (${tail.hiddenChars.toLocaleString()} earlier chars hidden)`}
        </button>
      )}
      <pre className="mt-1 max-h-60 overflow-auto whitespace-pre-wrap break-words font-mono text-xs">
        {showAll ? text : tail.visible}
      </pre>
    </>
  );
}
