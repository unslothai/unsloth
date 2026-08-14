// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// #8483: the desktop app froze during a deep research run. Two costs only show under a real
// stream, so they are pinned at the source: what run-carrying components subscribe to, and what
// the finished report's markdown commit signs up for. Like drag-costs-no-render.test.ts: assert
// the cheap path, assert the expensive one is gone.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { markdownPluginNeeds, MAX_HIGHLIGHT_CHARS } from "../src/lib/markdown-plugins.ts";

function source(path: string): string {
  return readFileSync(new URL(`../src/${path}`, import.meta.url), "utf8");
}

test("no research subscriber selects the whole run object", () => {
  // Each re-rendered its subtree on every streamed delta; thread.tsx's useThreadResearchActive
  // was already correct and is the shape the others now follow.
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
  assert.match(preview, /scheduleIdleTask\(\(\) => setReadyMarkdown\(markdown\), 200\)/);
  // The old path: all three plugins, always, in one synchronous commit.
  assert.doesNotMatch(preview, /const MARKDOWN_PLUGINS = \{ code, math, mermaid \}/);
  const message = source("features/chat/components/research-message.tsx");
  assert.match(message, /markdown=\{run\.report\}[\s\S]*?defer=\{true\}/);
});

test("deferred readiness belongs to a markdown value, not to the component", () => {
  // Blanking readiness from a passive effect lands one commit late, so the parse is paid twice.
  // Measured on a 202KB report: a wasted 576ms parse, then a second one, ~1.11s blocked against
  // ~0.66s once readiness is derived during render.
  const preview = source("components/markdown/markdown-preview.tsx");
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
  // Nor a balanced pair: these callers install `@streamdown/math`'s bare export, which pins
  // singleDollarTextMath to false, so `$x^2$` renders literally with the plugin loaded too.
  // (`markdown-text.tsx` enables it, but does not use this detector.) If that changes, NEEDS_MATH
  // must change with it - and "costs $5 and $10" is why a balanced-pair regex is not the answer.
  assert.equal(markdownPluginNeeds("the area is $x^2$ per unit").math, false);
  assert.match(
    source("components/markdown/markdown-preview.tsx"),
    /import \{ math \} from "@streamdown\/math";/,
  );
  assert.equal(markdownPluginNeeds("```mermaid\ngraph TD;\n```").mermaid, true);
  assert.equal(markdownPluginNeeds("```python\npass\n```").mermaid, false);
  // CommonMark fences with three or more backticks *or* tildes, and allows longer fences. All of
  // these reach the renderer as lang "mermaid".
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
