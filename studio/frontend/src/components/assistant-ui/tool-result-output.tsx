// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { stripAnsi, tailToolOutput } from "@/lib/strip-ansi";
import { useMemo, useState } from "react";


/**
 * Finished-tool output pane: renders the tail (~2000 lines) with a "Show all"
 * toggle so a large output stays scrollable without janking the DOM. Copy
 * buttons still copy the FULL text (owned by the caller), not the tail.
 */
export function ToolResultOutput({ text }: { text: string }) {
  const [showAll, setShowAll] = useState(false);
  // Strip SGR before tailing so colour codes neither inflate the char budget
  // nor leak into the DOM as literal escape text (#7962).
  const cleaned = useMemo(() => stripAnsi(text), [text]);
  const tail = useMemo(() => tailToolOutput(cleaned), [cleaned]);
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
        {showAll ? cleaned : tail.visible}
      </pre>
    </>
  );
}
