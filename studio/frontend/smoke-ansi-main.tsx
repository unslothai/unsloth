// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createRoot } from "react-dom/client";
import { ToolResultOutput } from "@/components/assistant-ui/tool-result-output";
import { ToolFallbackResult } from "@/components/assistant-ui/tool-fallback";

const ESC = "\u001b";
const coloured = `${ESC}[32mfile.txt${ESC}[0m\n${ESC}[01;31merror${ESC}[0m`;

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
  </>,
);
