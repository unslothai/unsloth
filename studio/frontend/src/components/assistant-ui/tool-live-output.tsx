// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { toolOutputKey, useToolPaneScope } from "@/features/chat";
import { useEffect, useMemo, useRef } from "react";
import { tailText } from "./tool-result-output";

/**
 * Live-scrolling stdout/stderr pane for a running server-side tool.
 *
 * Backed by the transient `toolLiveOutput` store map fed by `tool_output`
 * SSE events (mirrors the thinking block's streaming presentation): renders
 * nothing until the first output chunk arrives, then follows the tail as new
 * chunks stream in. The finished tool card shows the persisted result
 * instead, so this is only mounted while the tool is running.
 */
export function ToolLiveOutput({ toolCallId }: { toolCallId: string }) {
  const paneScope = useToolPaneScope();
  const output = useChatRuntimeStore(
    (s) => s.toolLiveOutput[toolOutputKey(paneScope, toolCallId)] ?? "",
  );
  const scrollRef = useRef<HTMLPreElement>(null);
  // Whether the pane is pinned to the bottom. Starts true so the pane follows
  // the tail by default; flipped by the scroll handler when the user scrolls
  // up to read earlier output, so streaming chunks no longer yank them down.
  const pinnedToBottom = useRef(true);

  // The stream can grow to hundreds of KB; render only the tail while it is
  // live (the pane follows the tail anyway). The finished card offers the
  // full text with a "Show all" control.
  const visible = useMemo(() => tailText(output).visible, [output]);

  const handleScroll = () => {
    const el = scrollRef.current;
    if (!el) {
      return;
    }
    // Treat "within 40px of the bottom" as pinned, so a small manual nudge or
    // sub-pixel rounding still counts as following the tail.
    pinnedToBottom.current =
      el.scrollHeight - el.scrollTop - el.clientHeight < 40;
  };

  useEffect(() => {
    const el = scrollRef.current;
    if (el && pinnedToBottom.current) {
      el.scrollTop = el.scrollHeight;
    }
  }, [visible]);

  if (!output) {
    return null;
  }

  return (
    <div className="aui-tool-live-output mt-2 border-t border-dashed pt-2">
      <span className="text-xs font-medium text-muted-foreground">output</span>
      <pre
        ref={scrollRef}
        onScroll={handleScroll}
        className="mt-1 max-h-60 overflow-auto whitespace-pre-wrap break-words font-mono text-xs"
      >
        {visible}
      </pre>
    </div>
  );
}
