// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// #8483: the desktop app froze during a deep research run. Two of the costs are invisible to a
// rendering test and only show under a real stream, so they are pinned here at the source: what
// the run-carrying components subscribe to, and how much work the finished report's markdown
// commit signs up for. Same shape as drag-costs-no-render.test.ts - assert the cheap path, and
// assert the expensive one is gone.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { markdownPluginNeeds, MAX_HIGHLIGHT_CHARS } from "../src/lib/markdown-plugins.ts";

function source(path: string): string {
  return readFileSync(new URL(`../src/${path}`, import.meta.url), "utf8");
}

test("no research subscriber selects the whole run object", () => {
  // Each of these re-rendered its subtree on every streamed delta. thread.tsx's
  // useThreadResearchActive was already correct and is the shape the others now follow.
  for (const path of [
    "features/chat/chat-page.tsx",
    "components/assistant-ui/thread.tsx",
  ]) {
    const text = source(path);
    assert.doesNotMatch(
      text,
      /state\.sessions\[[^\]]+\]\?\.run;/,
      `${path} selects a run object out of the research store`,
    );
    assert.doesNotMatch(
      text,
      /return runId \? state\.sessions\[runId\]\?\.run : undefined;/,
      `${path} selects a run object out of the research store`,
    );
  }
});

test("chat-page derives the research pane from strings", () => {
  const page = source("features/chat/chat-page.tsx");
  assert.match(page, /state\.sessions\[openResearchRunId\]\?\.run\.threadId/);
  assert.match(page, /state\.sessions\[latestResearchRunId\]\?\.run\.status/);
});

test("Thread is memoized", () => {
  const thread = source("components/assistant-ui/thread.tsx");
  assert.match(thread, /export const Thread: FC<\{[^}]*\}> = memo\(/s);
  assert.match(thread, /Thread\.displayName = "Thread";/);
});

test("the report renderer is deferred and its plugins are conditional", () => {
  const preview = source("components/markdown/markdown-preview.tsx");
  assert.match(preview, /markdownPluginNeeds\(markdown\)/);
  assert.match(preview, /scheduleIdleTask\(\(\) => setReady\(true\)/);
  // The old path: all three plugins, always, in one synchronous commit.
  assert.doesNotMatch(preview, /const MARKDOWN_PLUGINS = \{ code, math, mermaid \}/);
  const message = source("features/chat/components/research-message.tsx");
  assert.match(message, /markdown=\{run\.report\}[\s\S]*?defer=\{true\}/);
});

test("plugin needs follow the document", () => {
  assert.deepEqual(markdownPluginNeeds("plain prose with `code` spans"), {
    math: false,
    mermaid: false,
    code: true,
  });
  assert.equal(markdownPluginNeeds("a $$x^2$$ b").math, true);
  assert.equal(markdownPluginNeeds("\\(x\\)").math, true);
  assert.equal(markdownPluginNeeds("\\[x\\]").math, true);
  // A lone $ is too common in prose (prices, shell prompts) to pull KaTeX in for.
  assert.equal(markdownPluginNeeds("costs $5 to run").math, false);
  assert.equal(markdownPluginNeeds("```mermaid\ngraph TD;\n```").mermaid, true);
  assert.equal(markdownPluginNeeds("```python\npass\n```").mermaid, false);
  // CommonMark opens a fenced block with three-or-more backticks *or* tildes, and allows a
  // longer fence than three of either. All of these reach the renderer as lang "mermaid".
  assert.equal(markdownPluginNeeds("~~~mermaid\ngraph TD;\n~~~").mermaid, true);
  assert.equal(markdownPluginNeeds("````mermaid\ngraph TD;\n````").mermaid, true);
  assert.equal(markdownPluginNeeds("~~~~mermaid\ngraph TD;\n~~~~").mermaid, true);
  assert.equal(markdownPluginNeeds("~~~ mermaid\ngraph TD;\n~~~").mermaid, true);
  assert.equal(markdownPluginNeeds("~~~python\npass\n~~~").mermaid, false);
  // Two tildes is strikethrough, not a fence.
  assert.equal(markdownPluginNeeds("~~mermaid~~ is a tool").mermaid, false);
});

test("highlighting is capped, and the cap is one constant", () => {
  assert.equal(markdownPluginNeeds("x".repeat(MAX_HIGHLIGHT_CHARS)).code, true);
  assert.equal(
    markdownPluginNeeds("x".repeat(MAX_HIGHLIGHT_CHARS + 1)).code,
    false,
  );
  const cell = source("components/assistant-ui/tool-code-cell.tsx");
  assert.match(cell, /import \{ MAX_HIGHLIGHT_CHARS \} from "@\/lib\/markdown-plugins";/);
  assert.doesNotMatch(cell, /const MAX_HIGHLIGHT_CHARS = /);
});
