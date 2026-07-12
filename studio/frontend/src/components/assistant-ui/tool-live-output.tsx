// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { useEffect, useRef } from "react";

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
  const output = useChatRuntimeStore(
    (s) => s.toolLiveOutput[toolCallId] ?? "",
  );
  const scrollRef = useRef<HTMLPreElement>(null);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) {
      el.scrollTop = el.scrollHeight;
    }
  }, [output]);

  if (!output) {
    return null;
  }

  return (
    <div className="aui-tool-live-output mt-2 border-t border-dashed pt-2">
      <span className="text-xs font-medium text-muted-foreground">output</span>
      <pre
        ref={scrollRef}
        className="mt-1 max-h-60 overflow-auto whitespace-pre-wrap break-words font-mono text-xs"
      >
        {output}
      </pre>
    </div>
  );
}
