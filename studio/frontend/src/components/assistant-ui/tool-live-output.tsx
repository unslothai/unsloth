// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { useToolOutputFor, useToolPaneScope } from "@/features/chat";
import { stripAnsi, tailToolOutput } from "@/lib/strip-ansi";
import { useEffect, useMemo, useRef } from "react";

/**
 * Live-scrolling stdout/stderr pane for a running server-side tool, backed by
 * the transient `toolLiveOutput` map fed by `tool_output` SSE events. Renders
 * nothing until the first chunk, then follows the tail. Mounted only while
 * running; the finished card shows the persisted result instead.
 */
export function ToolLiveOutput({ toolCallId }: { toolCallId: string }) {
  const paneScope = useToolPaneScope();
  const output = useToolOutputFor(
    useChatRuntimeStore((s) => s.toolLiveOutput),
    paneScope,
    toolCallId,
  );
  return output ? <ToolLiveOutputPane output={output} /> : null;
}

/** Presentational production pane, exported so browser smoke tests this exact path. */
export function ToolLiveOutputPane({ output }: { output: string }) {
  const scrollRef = useRef<HTMLPreElement>(null);
  // Pinned to the bottom until the user scrolls up (handler below), so
  // streaming chunks no longer yank them down.
  const pinnedToBottom = useRef(true);

  // The stream can reach hundreds of KB; render only the clean tail while live.
  const visible = useMemo(
    () => tailToolOutput(stripAnsi(output)).visible,
    [output],
  );

  const handleScroll = () => {
    const el = scrollRef.current;
    if (!el) {
      return;
    }
    // Within 40px of the bottom counts as pinned (tolerates small nudges).
    pinnedToBottom.current =
      el.scrollHeight - el.scrollTop - el.clientHeight < 40;
  };

  useEffect(() => {
    const el = scrollRef.current;
    if (el && pinnedToBottom.current) {
      el.scrollTop = el.scrollHeight;
    }
  }, [visible]);

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
