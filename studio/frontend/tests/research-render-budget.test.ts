// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Two stream-only costs are pinned at the source: run subscriptions and markdown plugins.

import assert from "node:assert/strict";
import test from "node:test";

import { markdownPluginNeeds, MAX_HIGHLIGHT_CHARS } from "../src/lib/markdown-plugins.ts";

import { readSrc } from "./helpers/kit.ts";

test("no research subscriber selects the whole run object", () => {
  for (const path of [
    "features/chat/chat-page.tsx",
    "components/assistant-ui/thread.tsx",
  ]) {
    const text = readSrc(path);
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
  const page = readSrc("features/chat/chat-page.tsx");
  assert.match(page, /state\.sessions\[openResearchRunId\]\?\.run\.threadId/);
  assert.match(page, /state\.sessions\[latestResearchRunId\]\?\.run\.status/);
});

test("Thread is memoized", () => {
  const thread = readSrc("components/assistant-ui/thread.tsx");
  assert.match(thread, /export const Thread: FC<\{[^}]*\}> = memo\(/s);
  assert.match(thread, /Thread\.displayName = "Thread";/);
});

test("the report renderer is deferred and its plugins are conditional", () => {
  const preview = readSrc("components/markdown/markdown-preview.tsx");
  assert.match(preview, /markdownPluginNeeds\(markdown\)/);
  assert.match(preview, /scheduleIdleTask\(\(\) => setReadyMarkdown\(markdown\), 200\)/);
  assert.doesNotMatch(preview, /const MARKDOWN_PLUGINS = \{ code, math, mermaid \}/);
  const message = readSrc("features/chat/components/research-message.tsx");
  assert.match(message, /markdown=\{run\.report\}[\s\S]*?defer=\{true\}/);
});

test("deferred readiness belongs to a markdown value, not to the component", () => {
  // Blanking readiness in a passive effect lands a commit late and parses twice.
  const preview = readSrc("components/markdown/markdown-preview.tsx");
  assert.match(preview, /const ready = !defer \|\| readyMarkdown === markdown;/);
  assert.match(preview, /scheduleIdleTask\(\(\) => setReadyMarkdown\(markdown\), 200\)/);
  assert.doesNotMatch(preview, /useState\(!defer\)/);
  assert.doesNotMatch(preview, /setReady\(false\)/);
  assert.match(preview, /\{ready \? \(\s*<Streamdown/);
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
  // The bare @streamdown/math export disables singleDollarTextMath; keep NEEDS_MATH in sync.
  assert.equal(markdownPluginNeeds("the area is $x^2$ per unit").math, false);
  assert.match(
    readSrc("components/markdown/markdown-preview.tsx"),
    /import \{ math \} from "@streamdown\/math";/,
  );
  assert.equal(markdownPluginNeeds("```mermaid\ngraph TD;\n```").mermaid, true);
  assert.equal(markdownPluginNeeds("```python\npass\n```").mermaid, false);
  // CommonMark fences use 3+ backticks or tildes, all reaching the renderer as mermaid.
  assert.equal(markdownPluginNeeds("~~~mermaid\ngraph TD;\n~~~").mermaid, true);
  assert.equal(markdownPluginNeeds("````mermaid\ngraph TD;\n````").mermaid, true);
  assert.equal(markdownPluginNeeds("~~~~mermaid\ngraph TD;\n~~~~").mermaid, true);
  assert.equal(markdownPluginNeeds("~~~ mermaid\ngraph TD;\n~~~").mermaid, true);
  assert.equal(markdownPluginNeeds("~~~python\npass\n~~~").mermaid, false);
  assert.equal(markdownPluginNeeds("~~mermaid~~ is a tool").mermaid, false);
});

test("highlighting is capped, and the cap is one constant", () => {
  assert.equal(markdownPluginNeeds("x".repeat(MAX_HIGHLIGHT_CHARS)).code, true);
  assert.equal(
    markdownPluginNeeds("x".repeat(MAX_HIGHLIGHT_CHARS + 1)).code,
    false,
  );
  const imports =
    /import \{[^}]*\bMAX_HIGHLIGHT_CHARS\b[^}]*\} from "@\/lib\/markdown-plugins";/;
  for (const path of [
    "components/assistant-ui/tool-code-cell.tsx",
    "components/assistant-ui/attachment-preview.tsx",
  ]) {
    const cell = readSrc(path);
    assert.match(cell, imports, path);
    assert.doesNotMatch(cell, /const MAX_HIGHLIGHT_CHARS = /, path);
  }
});
