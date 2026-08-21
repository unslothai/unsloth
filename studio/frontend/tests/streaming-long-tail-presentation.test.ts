// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import remend from "remend";
import { parseMarkdownIntoBlocks } from "streamdown";

import { OversizedStreamingCodeBlock } from "../src/components/assistant-ui/oversized-streaming-code-block.ts";
import {
  MAX_AUTO_HIGHLIGHT_SOURCE_CODE_UNITS,
  OVERSIZED_OPEN_CODE_CHARS,
  getCodeFenceFilename,
  getStreamingCodeFence,
  getTerminalStreamingCodeFence,
  isOversizedStreamingCode,
  normalizeCodeFenceLanguage,
  shouldAutoHighlightStreamingCode,
} from "../src/components/assistant-ui/streaming-code-policy.ts";
import { IncrementalMarkdownCache } from "../src/components/assistant-ui/streaming-render-schedule.ts";
import {
  LONG_STREAM_PRESENTATION_CHARS,
  LONG_STREAM_PRESENTATION_MS,
  createStreamingTextPresentationScheduler,
  scheduleAfterPaint,
} from "../src/components/assistant-ui/streaming-text-presentation.ts";
import { renderBlockContents } from "./streaming-render-plan.ts";

class FakeClock {
  now = 0;
  private nextHandle = 1;
  private tasks = new Map<
    number,
    { at: number; callback: (now: number) => void }
  >();

  requestFrame = (callback: FrameRequestCallback): number => {
    const handle = this.nextHandle++;
    const at = Math.ceil((this.now + 0.001) / 16) * 16;
    this.tasks.set(handle, { at, callback });
    return handle;
  };

  cancelFrame = (handle: number): void => {
    this.tasks.delete(handle);
  };

  setTimer = (callback: () => void, delay: number): number => {
    const handle = this.nextHandle++;
    // Even a zero-delay task runs after the current frame callback returns.
    this.tasks.set(handle, {
      at: this.now + Math.max(delay, 0.001),
      callback: () => callback(),
    });
    return handle;
  };

  clearTimer = (handle: number): void => {
    this.tasks.delete(handle);
  };

  advanceTo(target: number): void {
    for (;;) {
      const next = [...this.tasks.entries()]
        .filter(([, task]) => task.at <= target)
        .sort((left, right) => left[1].at - right[1].at)[0];
      if (!next) break;
      const [handle, task] = next;
      this.tasks.delete(handle);
      this.now = task.at;
      task.callback(this.now);
    }
    this.now = target;
  }
}

const codeSource = (characters: number): string => {
  const line = 'export const row = { id: 42, label: "value" };\n';
  return line.repeat(Math.ceil(characters / line.length)).slice(0, characters);
};

const prosePrefix = Array.from(
  { length: 24 },
  (_, index) => `Paragraph ${index} stays committed while the code grows.\n\n`,
).join("");




test("budget boundaries survive marker, line-ending, close, completion, and reload", () => {
  const prefix = "## Stable prefix\n\nParagraph with **bold** and a table.\n\n";
  for (const target of [
    MAX_AUTO_HIGHLIGHT_SOURCE_CODE_UNITS - 1,
    MAX_AUTO_HIGHLIGHT_SOURCE_CODE_UNITS,
    MAX_AUTO_HIGHLIGHT_SOURCE_CODE_UNITS + 1,
  ]) {
    for (const { lineEnding, marker } of [
      { lineEnding: "\n", marker: "```" },
      { lineEnding: "\r\n", marker: "~~~~" },
    ]) {
      const canonicalSource = codeSource(target);
      const rawSource = canonicalSource.replaceAll("\n", lineEnding);
      const rawPrefix = prefix.replaceAll("\n", lineEnding);
      const opening = `${marker}typescript title="boundary.ts"${lineEnding}`;
      const openMarkdown = rawPrefix + opening + rawSource;
      const closedMarkdown = `${openMarkdown}${lineEnding}${marker}`;
      const expected = target <= MAX_AUTO_HIGHLIGHT_SOURCE_CODE_UNITS;
      const cache = new IncrementalMarkdownCache();

      const open = cache.update(openMarkdown, true);
      assert.equal(open.terminalCodeTail?.source, canonicalSource);
      assert.equal(
        shouldAutoHighlightStreamingCode(open.terminalCodeTail?.source ?? ""),
        expected,
      );

      const closedWhileStreaming = cache.update(closedMarkdown, true);
      assert.equal(closedWhileStreaming.terminalCodeTail?.isClosed, true);
      assert.equal(closedWhileStreaming.terminalCodeTail?.source, canonicalSource);
      assert.equal(
        shouldAutoHighlightStreamingCode(
          closedWhileStreaming.terminalCodeTail?.source ?? "",
        ),
        expected,
      );

      const completed = cache.update(closedMarkdown, false);
      assert.equal(completed.terminalCodeTail?.source, canonicalSource);
      assert.equal(
        shouldAutoHighlightStreamingCode(completed.terminalCodeTail?.source ?? ""),
        expected,
      );

      const reloaded = new IncrementalMarkdownCache().update(
        closedMarkdown,
        false,
      );
      assert.equal(reloaded.terminalCodeTail, null);
      const reloadedFence = renderBlockContents(reloaded)
        .map((content) => getStreamingCodeFence(content))
        .find((fence) => fence !== null);
      assert.equal(reloadedFence?.source, canonicalSource);
      assert.equal(
        shouldAutoHighlightStreamingCode(reloadedFence?.source ?? ""),
        expected,
      );
    }
  }
});


test("open fence extraction preserves exact bytes across supported fence forms", () => {
  const backtickSource = "const first = 1;\r\n```\r\nconst last = 2;";
  assert.deepEqual(
    getStreamingCodeFence(
      `   \`\`\`\`typescript title="generated.ts"\r\n${backtickSource}`,
    ),
    { language: "typescript", source: backtickSource },
  );

  const tildeSource = "alpha\n~~~\nomega";
  assert.deepEqual(
    getStreamingCodeFence(`~~~~ ts linenums\n${tildeSource}`),
    { language: "ts", source: tildeSource },
  );
});


test("a mixed mutable shell plans only its oversized terminal fence as plain code", () => {
  const prefix = `## Live design\n\n${"Paragraph with **bold**, `inline code`, and a [link](https://example.com). ".repeat(
    40,
  )}The mutable paragraph still has *unfinished emphasis and a footnote[^live].\n\n`;
  const source = codeSource(OVERSIZED_OPEN_CODE_CHARS + 2_000);
  const opening = "````typescript title=generated.ts\n";
  const plan = new IncrementalMarkdownCache().update(
    `${prefix}${opening}${source}`,
  );

  assert.ok(
    plan.terminalCodeTail,
    "the mixed shell did not plan its terminal fence",
  );
  assert.equal(plan.terminalCodeTail.prefixMarkdown, prefix);
  assert.equal(plan.terminalCodeTail.fenceMarkdown, opening + source);
  assert.equal(plan.terminalCodeTail.source, source);
  assert.equal(plan.terminalCodeTail.openingOffset, prefix.length);
});

test("stopped and cold unclosed terminal fences retain the bounded tail", () => {
  const prefix =
    "## Persisted malformed answer\n\n" +
    "Surrounding **Markdown** remains rendered[^stable].\n\n" +
    "| state | result |\n| --- | --- |\n| stopped | plain code |\n\n" +
    "[^stable]: global provider definitions remain available.\n\n";
  const opening = "```typescript title=generated.ts\n";
  const source = codeSource(MAX_AUTO_HIGHLIGHT_SOURCE_CODE_UNITS + 1);
  const markdown = prefix + opening + source;
  const cache = new IncrementalMarkdownCache();

  const running = cache.update(markdown, true);
  const stopped = cache.update(markdown, false);
  const cold = new IncrementalMarkdownCache().update(markdown, false);

  assert.ok(running.terminalCodeTail);
  assert.ok(stopped.terminalCodeTail);
  assert.ok(cold.terminalCodeTail);
  assert.equal(stopped.terminalCodeTail.id, running.terminalCodeTail.id);
  for (const plan of [running, stopped, cold]) {
    assert.equal(plan.terminalCodeTail?.isClosed, false);
    assert.equal(plan.terminalCodeTail?.prefixMarkdown, prefix);
    assert.equal(plan.terminalCodeTail?.source, source);
    assert.equal(
      shouldAutoHighlightStreamingCode(plan.terminalCodeTail?.source ?? ""),
      false,
    );
  }
});

test("terminal fence extraction keeps exact offsets and rejects Markdown lookalikes", () => {
  const prefix =
    "> quoted prefix\r\n\r\nParagraph with **bold** and `inline code`.\r\n\r\n";
  const source = `first\r\n${codeSource(OVERSIZED_OPEN_CODE_CHARS + 20)}`;
  const opening = "  ~~~~typescript title=generated.ts\r\n";
  assert.deepEqual(
    getTerminalStreamingCodeFence(prefix + opening + source),
    {
      fenceMarkdown: opening + source,
      isClosed: false,
      language: "typescript",
      openingLine: opening,
      openingOffset: prefix.length,
      rawSource: source,
      source,
    },
  );

  const ordinary = "ordinary prose ".repeat(400);
  for (const markdown of [
    `${prefix}\`inline ~~~ not a fence\`\n${ordinary}`,
    `${prefix}    ~~~ts\n${ordinary}`,
    `<div>\n~~~ts\n${ordinary}`,
    `${prefix}~~~ts\n${ordinary}\n~~~\nAfter the closed fence.`,
    ordinary,
  ]) {
    assert.equal(getTerminalStreamingCodeFence(markdown), null);
  }
});

test("the mixed split keeps identity through growth and close, then invalidates narrowly", () => {
  const prefix =
    "## Rich prefix\n\nParagraph with **bold** and a footnote[^live].\n\n";
  const opening = "````typescript meta\n";
  const firstSource = codeSource(OVERSIZED_OPEN_CODE_CHARS + 200);
  const cache = new IncrementalMarkdownCache();
  const first = cache.update(prefix + opening + firstSource, true);
  const firstTail = first.terminalCodeTail;
  assert.ok(firstTail);

  const grownSource = firstSource + codeSource(700);
  const grown = cache.update(prefix + opening + grownSource, true);
  assert.equal(grown.terminalCodeTail?.id, firstTail.id);
  assert.equal(grown.terminalCodeTail?.prefixBlocks, firstTail.prefixBlocks);
  assert.equal(grown.terminalCodeTail?.source, grownSource);

  const rewoundSource = firstSource.slice(0, -100);
  const rewound = cache.update(prefix + opening + rewoundSource, true);
  assert.ok(rewound.terminalCodeTail);
  assert.notEqual(
    rewound.terminalCodeTail.id,
    firstTail.id,
    "a canonical source rewind retained the stale wrapper identity",
  );
  const regrown = cache.update(prefix + opening + grownSource, true);
  assert.equal(regrown.terminalCodeTail?.id, rewound.terminalCodeTail.id);
  assert.equal(regrown.terminalCodeTail?.source, grownSource);
  const closedMarkdown = `${prefix}${opening}${grownSource}\n\`\`\`\``;
  const closed = cache.update(closedMarkdown, false);
  assert.equal(closed.terminalCodeTail?.id, regrown.terminalCodeTail?.id);
  assert.equal(closed.terminalCodeTail?.isClosed, true);
  assert.equal(closed.terminalCodeTail?.source, grownSource);
  assert.deepEqual(
    renderBlockContents(closed),
    parseMarkdownIntoBlocks(remend(closedMarkdown)),
  );

  const historical = new IncrementalMarkdownCache().update(
    closedMarkdown,
    false,
  );
  assert.equal(
    historical.terminalCodeTail,
    null,
    "a closed historical fence was split",
  );

  const replaced = cache.update(
    `${prefix}\`\`not a fence\n${grownSource}`,
    true,
  );
  assert.equal(replaced.terminalCodeTail, null);
  assert.deepEqual(
    renderBlockContents(replaced),
    parseMarkdownIntoBlocks(remend(`${prefix}\`\`not a fence\n${grownSource}`)),
  );
});
test("completion preserves a canonical trailing line ending for source actions", () => {
  for (const { lineEnding, marker } of [
    { lineEnding: "\n", marker: "```" },
    { lineEnding: "\r\n", marker: "~~~~" },
  ]) {
    const prefix = `## Exact source${lineEnding}${lineEnding}`;
    const opening = `${marker}typescript title="generated.ts"${lineEnding}`;
    const source =
      codeSource(OVERSIZED_OPEN_CODE_CHARS + 2_900).replaceAll(
        "\n",
        lineEnding,
      ) + lineEnding;
    const canonicalSource = source.replaceAll("\r\n", "\n");
    const cache = new IncrementalMarkdownCache();
    const open = cache.update(prefix + opening + source, true);
    assert.equal(open.terminalCodeTail?.source, canonicalSource);

    const closed = cache.update(prefix + opening + source + marker, false);
    assert.equal(closed.terminalCodeTail?.source, canonicalSource);
  }
});




test("status completion before the final close keeps exact terminal source identity", () => {
  const prefix =
    "## Mixed terminal-fence fixture\n\n" +
    "- first stable item\n- second stable item\n\n" +
    "| phase | renderer |\n| --- | --- |\n| open | plain code |\n\n" +
    "> Stable quote.\n\n" +
    "$$x^2 + y^2 = z^2$$\n\n";
  const source = `${codeSource(MAX_AUTO_HIGHLIGHT_SOURCE_CODE_UNITS)}\n`;
  const opening = '```typescript title="boundary.ts"\n';
  const openMarkdown = prefix + opening + source;
  const completedMarkdown = `${openMarkdown}\`\`\`\n\n${"Settled after.\n\n".repeat(12)}`;
  const cache = new IncrementalMarkdownCache();

  for (let cursor = 512; cursor < openMarkdown.length; cursor += 512) {
    cache.update(openMarkdown.slice(0, cursor), true);
  }
  const open = cache.update(openMarkdown, true);
  assert.equal(open.terminalCodeTail?.source, source);

  const statusFirst = cache.update(openMarkdown, false);
  assert.equal(statusFirst.terminalCodeTail?.source, source);
  const completed = cache.update(completedMarkdown, false);
  const fence = [...completed.chunks.flatMap((chunk) => chunk.blocks), ...completed.tail]
    .flatMap((block) => block.codeFences)
    .find((candidate) => candidate.language === "typescript");
  assert.equal(fence?.source, source);
  assert.equal(shouldAutoHighlightStreamingCode(fence?.source ?? ""), false);
});








test("fence metastrings stay out of language labels and filenames", () => {
  assert.equal(
    normalizeCodeFenceLanguage('  TypeScript\ttitle="generated.ts" {1,3}  '),
    "TypeScript",
  );
  assert.equal(
    getCodeFenceFilename('TypeScript title="generated.ts" {1,3}'),
    "snippet.ts",
  );
  assert.equal(getCodeFenceFilename("custom lang metadata"), "snippet.custom");
  assert.equal(normalizeCodeFenceLanguage(" \t "), null);
  assert.equal(getCodeFenceFilename(" \t "), "snippet.txt");
});


test("the oversized open presentation keeps Streamdown chrome and escaped exact source", () => {
  const source = `<script>alert("unsafe")</script>\n${codeSource(
    OVERSIZED_OPEN_CODE_CHARS,
  )}`;
  const html = renderToStaticMarkup(
    createElement(OversizedStreamingCodeBlock, {
      isFenceOpen: true,
      language: "typescript",
      source,
    }),
  );

  assert.match(html, /data-streamdown="code-block"/);
  assert.match(html, /data-streamdown="code-block-header"/);
  assert.match(html, /data-streamdown="code-block-body"/);
  assert.match(html, /data-incomplete="true"/);
  assert.match(html, /typescript/);
  assert.match(
    html,
    /&lt;script&gt;alert\(&quot;unsafe&quot;\)&lt;\/script&gt;/,
  );
  assert.doesNotMatch(
    html,
    /<mark>/,
    "Shiki's subtree mounted during the bypass",
  );
});



test("deferred highlighting cannot run before a completed plain block paints", () => {
  const clock = new FakeClock();
  let highlighted = 0;
  const cancel = scheduleAfterPaint(
    () => {
      highlighted += 1;
    },
    {
      requestFrame: clock.requestFrame,
      cancelFrame: clock.cancelFrame,
      setTimer: clock.setTimer,
      clearTimer: clock.clearTimer,
    },
  );

  clock.advanceTo(16);
  assert.equal(
    highlighted,
    0,
    "the first frame is reserved for plain completion",
  );
  clock.advanceTo(32);
  assert.equal(
    highlighted,
    0,
    "the second frame has not painted until it returns",
  );
  clock.advanceTo(32.001);
  assert.equal(highlighted, 1);
  cancel();
});

test("short streams remain frame-gated while long streams stay at or below 15 Hz", () => {
  const clock = new FakeClock();
  const commits: number[] = [];
  const scheduler = createStreamingTextPresentationScheduler({
    publish: () => commits.push(clock.now),
    now: () => clock.now,
    requestFrame: clock.requestFrame,
    cancelFrame: clock.cancelFrame,
    setTimer: clock.setTimer,
    clearTimer: clock.clearTimer,
  });

  for (let at = 0; at <= 96; at += 4) {
    clock.advanceTo(at);
    scheduler.schedule(1000 + at, 1000 + at);
  }
  clock.advanceTo(128);
  assert.ok(
    commits.length >= 6,
    `short stream committed only ${commits.length} frames`,
  );

  commits.length = 0;
  for (let at = 200; at <= 2200; at += 8) {
    clock.advanceTo(at);
    scheduler.schedule(LONG_STREAM_PRESENTATION_CHARS + at, at);
  }
  clock.advanceTo(2400);
  for (let index = 1; index < commits.length; index += 1) {
    assert.ok(
      commits[index] - commits[index - 1] >= 1000 / 15,
      `${commits[index] - commits[index - 1]} ms between long-tail commits exceeds 15 Hz`,
    );
  }
  assert.ok(
    commits.length <= Math.ceil(2200 / LONG_STREAM_PRESENTATION_MS) + 1,
    `long stream published ${commits.length} times in about two seconds`,
  );
});

test("a completion flush cancels the cadence and publishes the exact pending value", () => {
  const clock = new FakeClock();
  let pending = "old";
  const published: string[] = [];
  const scheduler = createStreamingTextPresentationScheduler({
    publish: (value: string) => published.push(value),
    now: () => clock.now,
    requestFrame: clock.requestFrame,
    cancelFrame: clock.cancelFrame,
    setTimer: clock.setTimer,
    clearTimer: clock.clearTimer,
  });

  scheduler.schedule(LONG_STREAM_PRESENTATION_CHARS, pending);
  clock.advanceTo(16);
  pending = "complete exact value";
  scheduler.schedule(LONG_STREAM_PRESENTATION_CHARS + pending.length, pending);
  scheduler.flush(pending);
  assert.equal(published.at(-1), pending);
  const count = published.length;
  clock.advanceTo(500);
  assert.equal(
    published.length,
    count,
    "a stale cadence fired after completion",
  );
});

test("a 100K code stream bypasses Shiki, bounds commits, and stays plain on close", () => {
  const clock = new FakeClock();
  const cache = new IncrementalMarkdownCache();
  const body = codeSource(105_000);
  let pendingBody = body.slice(0, LONG_STREAM_PRESENTATION_CHARS);
  let pendingMarkdown = `${prosePrefix}\`\`\`typescript\n${pendingBody}`;
  let pendingIncomplete = true;
  let finalPlan = cache.update(pendingMarkdown);
  const firstChunks = new Map(
    finalPlan.chunks.map((chunk) => [chunk.id, chunk]),
  );
  let repeatedCommittedChunkRenders = 0;
  let openPresentations = 0;


  const scheduler = createStreamingTextPresentationScheduler({
    publish: ({
      markdown,
      source,
      incomplete,
    }: {
      markdown: string;
      source: string;
      incomplete: boolean;
    }) => {
      finalPlan = cache.update(markdown);
      for (const chunk of finalPlan.chunks) {
        const first = firstChunks.get(chunk.id);
        if (first && first !== chunk) repeatedCommittedChunkRenders += 1;
      }
      if (incomplete) {
        assert.equal(isOversizedStreamingCode(source.length), true);
        openPresentations += 1;
      }

      renderToStaticMarkup(
        createElement(OversizedStreamingCodeBlock, {
          isFenceOpen: incomplete,
          language: "typescript",
          source,
        }),
      );
    },
    now: () => clock.now,
    requestFrame: clock.requestFrame,
    cancelFrame: clock.cancelFrame,
    setTimer: clock.setTimer,
    clearTimer: clock.clearTimer,
  });

  let arrival = 0;
  for (
    let length = LONG_STREAM_PRESENTATION_CHARS;
    length <= body.length;
    length += 512
  ) {
    clock.advanceTo(arrival);
    pendingBody = body.slice(0, Math.min(length, body.length));

    pendingMarkdown = `${prosePrefix}\`\`\`typescript\n${pendingBody}`;
    scheduler.schedule(prosePrefix.length + pendingBody.length, {
      markdown: pendingMarkdown,
      source: pendingBody,
      incomplete: true,
    });
    arrival += 8;
  }
  pendingBody = body;

  pendingMarkdown = `${prosePrefix}\`\`\`typescript\n${pendingBody}`;

  scheduler.schedule(prosePrefix.length + pendingBody.length, {
    markdown: pendingMarkdown,
    source: pendingBody,
    incomplete: true,
  });
  clock.advanceTo(arrival + 200);
  assert.ok(openPresentations > 0);
  assert.ok(
    openPresentations <=
      Math.ceil((arrival + 200) / LONG_STREAM_PRESENTATION_MS) + 1,
  );
  assert.equal(repeatedCommittedChunkRenders, 0);

  const closedMarkdown = `${prosePrefix}\`\`\`typescript\n${body}\n\`\`\``;
  pendingMarkdown = closedMarkdown;
  pendingIncomplete = false;
  scheduler.flush({
    markdown: pendingMarkdown,
    source: pendingBody,
    incomplete: pendingIncomplete,
  });
  renderToStaticMarkup(
    createElement(OversizedStreamingCodeBlock, {
      isFenceOpen: false,
      language: "typescript",
      source: body,
    }),
  );

  assert.equal(shouldAutoHighlightStreamingCode(body), false);
  assert.deepEqual(
    renderBlockContents(finalPlan),
    parseMarkdownIntoBlocks(remend(closedMarkdown)),
  );
});
