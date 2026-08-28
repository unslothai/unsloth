// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createRoot } from "react-dom/client";
import { ToolResultOutput } from "@/components/assistant-ui/tool-result-output";
import { ToolFallbackResult } from "@/components/assistant-ui/tool-fallback";

import { ToolLiveOutputPane } from "@/components/assistant-ui/tool-live-output";
import { CodeExecutionResultOutput } from "@/components/assistant-ui/tool-ui-code-execution";

// Load-bearing import order. Several assistant-ui tool components reach the
// `@/features/chat` barrel, which re-exports chat-runtime-store and sits in a
// cycle with them; entering that graph from the barrel dies with "Cannot access
// 'CHAT_GPU_MEMORY_MODE_KEY' before initialization". Enter from the store
// module instead. Same pattern as smoke-tool-activity-main.tsx.
/* eslint-disable no-restricted-imports -- a harness entry point, not app code. */
import "@/features/chat/stores/chat-runtime-store";
import { preferSanitizedFullToolOutput } from "@/features/chat/tool-output-scope";

const ESC = "\u001b";
const coloured = `${ESC}[32mfile.txt${ESC}[0m\n${ESC}[01;31merror${ESC}[0m`;

const truncated = "file.txt\n\n... (truncated; full output available in UI)";

const root = document.getElementById("root");
if (!root) {
  throw new Error("missing #root");
}

createRoot(root).render(
  <>
    <section data-smoke="tool-result-output">
      <h1>ToolResultOutput</h1>
      <ToolResultOutput text={coloured} />
    </section>
    <section data-smoke="tool-fallback-result">
      <h1>ToolFallbackResult</h1>
      <ToolFallbackResult result={coloured} />
    </section>
    <section data-smoke="tool-live-output">
      <h1>ToolLiveOutput</h1>
      <ToolLiveOutputPane output={coloured} />
    </section>
    <section data-smoke="code-execution-result">
      <h1>CodeExecutionResultOutput</h1>
      <CodeExecutionResultOutput result={coloured} />
    </section>
    <section data-smoke="reconciled-terminal-result">
      <h1>Reconciled terminal result</h1>
      <pre>{preferSanitizedFullToolOutput(coloured, truncated)}</pre>
    </section>


  </>,
);
