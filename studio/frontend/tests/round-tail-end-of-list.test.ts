// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { join, sep } from "node:path";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { betweenToolRounds, runningReplyTail } = await import(
  "../src/features/chat/lib/generation-progress.ts"
);

const done = { type: "tool-call", result: "ok" };
const pending = { type: "tool-call" };
const source = { type: "source" };
const text = { type: "text" };
const reasoning = { type: "reasoning" };

test("the tail is the empty text part after any non-text end, plus the status once a call finished", () => {
  assert.equal(runningReplyTail(true, [reasoning, pending]), 1, "a call still running: empty part only");
  assert.equal(runningReplyTail(true, [reasoning, done]), 2, "a finished call: empty part and status");
  assert.equal(runningReplyTail(true, [done, source, source]), 2, "sources after the call");
  assert.equal(runningReplyTail(true, [pending, source]), 1);
  assert.equal(runningReplyTail(true, [source]), 1);
  assert.equal(runningReplyTail(true, [{ type: "image" }]), 1);
  assert.equal(runningReplyTail(true, [done, text]), 0, "the next pass streamed text");
  assert.equal(runningReplyTail(true, [done, reasoning]), 0, "the next pass streamed reasoning");
  assert.equal(runningReplyTail(true, []), 0, "GeneratingIndicator covers an empty reply");
  assert.equal(runningReplyTail(false, [done]), 0, "a finished reply has no tail");
  for (const parts of [[done], [pending], [done, source], [source], [done, text], [reasoning], []]) {
    for (const running of [true, false]) {
      assert.equal(
        runningReplyTail(running, parts) === 2,
        betweenToolRounds(running, parts),
        "the status shows exactly when betweenToolRounds says so",
      );
    }
  }
});

test("assistant-ui's empty part fallback still matches the replica the tail renders", () => {
  const require = createRequire(import.meta.url);
  const entry = require.resolve("@assistant-ui/core");
  const dist = entry.slice(0, entry.lastIndexOf(`${sep}dist${sep}`) + 6);
  const parts = readFileSync(join(dist, "react/primitives/message/MessageParts.js"), "utf8");
  assert.match(parts, /return lastPart\?\.type !== "text" && lastPart\?\.type !== "reasoning";/);
  assert.match(parts, /if \(status\.type !== "running"\)\s*return null;/);
  assert.match(parts, /_jsx\(TextMessagePartProvider, \{ text: "", isRunning: status\.type === "running", children: _jsx\(Component,/);
  assert.match(parts, /_jsx\(ConditionalEmpty, \{ components: components, enabled: unstable_showEmptyOnNonTextEnd \}\)/);
});

test("both tail nodes sit in one display:contents box that is a direct sibling of the parts", () => {
  const thread = readSrc("components/assistant-ui/thread.tsx");
  const tail = thread.slice(
    thread.indexOf("const PartsWithRoundTail"),
    thread.indexOf("// Mounted only under a running reply"),
  );
  // Only as a direct child of the component owning {children} is it removed before the next part lands.
  assert.match(
    tail,
    /return \(\s*<>\s*\{children\}\s*\{tail !== 0 && \(\s*<div data-slot="round-tail" className="contents">\s*<TextMessagePartProvider text="" isRunning=\{true\}>\s*<MarkdownText \/>\s*<\/TextMessagePartProvider>\s*\{tail === 2 && \(\s*<div data-slot="next-round-indicator" className="mt-2">/,
  );
  assert.match(
    thread,
    /<PartsWithRoundTail>\s*<MessagePrimitive\.Parts\s+components=\{ASSISTANT_PART_COMPONENTS\}\s+unstable_showEmptyOnNonTextEnd=\{false\}\s*\/>\s*<\/PartsWithRoundTail>/,
  );
  assert.equal(thread.match(/<MessagePrimitive\.Parts components=\{ASSISTANT_PART_COMPONENTS\}/g), null);
  assert.equal(thread.includes("NextRoundIndicator"), false);
});

test("the tail's children keep the reply's spacing rules through the contents box", () => {
  const css = readSrc("index.css");
  for (const rule of [
    ':where(.aui-assistant-message-content > [data-slot="round-tail"] > *) {\n\t\tmargin-top: calc(1rem * var(--ui-space-scale, 1));',
    ':where(.aui-assistant-message-content > .h-0 + [data-slot="round-tail"] > *) {\n\t\tmargin-top: 0;',
    '.aui-assistant-message-content > [data-slot="round-tail"]:last-child > *:last-child > *:last-child > *:last-child > *:last-child > *:last-child > *:last-child {\n\t\tmargin-bottom: 0 !important;',
  ]) {
    assert.ok(css.includes(rule), rule);
  }
  assert.ok(css.includes(":where(.aui-assistant-message-content > *):where(:not(:first-child)) {\n\t\tmargin-top: calc(1rem * var(--ui-space-scale, 1));"));
  assert.ok(css.includes(".aui-assistant-message-content > *:last-child > *:last-child > *:last-child > *:last-child > *:last-child > *:last-child {"));
});
