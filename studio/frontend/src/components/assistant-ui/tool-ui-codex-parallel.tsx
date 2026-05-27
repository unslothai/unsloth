// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

/**
 * Tool-call renderer for Codex parallel-calls fan-out.
 *
 * Driven by the chat-adapter pushing a tool-call part with
 * ``toolName === "codex_parallel"`` whose ``args.state`` is a
 * ``CodexParallelState`` value. We just unpack the state and hand it
 * to the existing ``CodexParallelTabs`` component. Mounted via the
 * ``tools.by_name`` map on ``MessagePrimitive.Parts`` in
 * ``thread.tsx`` so it renders inline above the assistant's prose.
 */

import type { ToolCallMessagePartComponent } from "@assistant-ui/react";
import { memo } from "react";
import {
  CodexParallelTabs,
  EMPTY_CODEX_PARALLEL_STATE,
  type CodexParallelState,
} from "@/features/chat/components/codex-parallel-tabs";

const CodexParallelToolUIImpl: ToolCallMessagePartComponent = ({ args }) => {
  const state = (args as { state?: CodexParallelState } | undefined)?.state;
  return <CodexParallelTabs state={state ?? EMPTY_CODEX_PARALLEL_STATE} />;
};

export const CodexParallelToolUI = memo(
  CodexParallelToolUIImpl,
) as ToolCallMessagePartComponent;
CodexParallelToolUI.displayName = "CodexParallelToolUI";
