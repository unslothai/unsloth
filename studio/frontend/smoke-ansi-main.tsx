// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createRoot } from "react-dom/client";
// The app mounts every one of these components inside the provider tree in app/provider.tsx.
// This page mounts them bare, so any context they depend on has to be supplied here too, and
// a missing one is not a small thing: `Tooltip` throws outside its provider, which takes the
// whole React root down and leaves every section below empty. #11425 gave the code cell an
// icon button with a tooltip, and CodeExecutionResultOutput renders it.
import { TooltipProvider } from "@/components/ui/tooltip";
import { ToolResultOutput } from "@/components/assistant-ui/tool-result-output";
import { ToolFallbackResult } from "@/components/assistant-ui/tool-fallback";

import { ToolLiveOutputPane } from "@/components/assistant-ui/tool-live-output";
import { CodeExecutionResultOutput } from "@/components/assistant-ui/tool-ui-code-execution";

import { preferSanitizedFullToolOutput } from "@/features/chat";

const ESC = "\u001b";
const coloured = `${ESC}[32mfile.txt${ESC}[0m\n${ESC}[01;31merror${ESC}[0m`;

const truncated = "file.txt\n\n... (truncated; full output available in UI)";

const root = document.getElementById("root");
if (!root) {
  throw new Error("missing #root");
}

createRoot(root).render(
  <TooltipProvider>
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


  </TooltipProvider>,
);
