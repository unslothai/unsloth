// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import ts from "typescript";

const MARKDOWN_TEXT_PATH = new URL(
  "../src/components/assistant-ui/markdown-text.tsx",
  import.meta.url,
);
const source = ts.createSourceFile(
  MARKDOWN_TEXT_PATH.pathname,
  readFileSync(MARKDOWN_TEXT_PATH, "utf8"),
  ts.ScriptTarget.ESNext,
  true,
  ts.ScriptKind.TSX,
);

function findImmediateUpdateConfig(): ts.ObjectLiteralExpression | null {
  let config: ts.ObjectLiteralExpression | null = null;
  const visit = (node: ts.Node): void => {
    if (
      ts.isVariableDeclaration(node) &&
      node.name.getText(source) === "STREAMDOWN_IMMEDIATE_UPDATES" &&
      node.initializer &&
      ts.isSatisfiesExpression(node.initializer) &&
      ts.isObjectLiteralExpression(node.initializer.expression)
    ) {
      config = node.initializer.expression;
    }
    node.forEachChild(visit);
  };
  source.forEachChild(visit);
  return config;
}

function findChatStreamdown(): ts.JsxOpeningLikeElement | null {
  let streamdown: ts.JsxOpeningLikeElement | null = null;
  const visit = (node: ts.Node): void => {
    if (
      (ts.isJsxOpeningElement(node) || ts.isJsxSelfClosingElement(node)) &&
      node.tagName.getText(source) === "Streamdown"
    ) {
      streamdown = node;
    }
    node.forEachChild(visit);
  };
  source.forEachChild(visit);
  return streamdown;
}

function jsxAttribute(
  opening: ts.JsxOpeningLikeElement,
  name: string,
): ts.JsxAttribute | null {
  for (const attribute of opening.attributes.properties) {
    if (
      ts.isJsxAttribute(attribute) &&
      attribute.name.getText(source) === name
    ) {
      return attribute;
    }
  }
  return null;
}

test("chat streaming bypasses Streamdown's starvable transition without visible animation", () => {
  const config = findImmediateUpdateConfig();
  assert.ok(config, "STREAMDOWN_IMMEDIATE_UPDATES is missing");

  const values = new Map(
    config.properties.flatMap((property) => {
      if (
        !(
          ts.isPropertyAssignment(property) &&
          ts.isNumericLiteral(property.initializer)
        )
      ) {
        return [];
      }
      return [
        [property.name.getText(source), Number(property.initializer.text)],
      ];
    }),
  );
  assert.equal(values.get("duration"), 0);
  assert.equal(values.get("stagger"), 0);

  const streamdown = findChatStreamdown();
  assert.ok(streamdown, "chat <Streamdown> is missing");
  assert.equal(
    jsxAttribute(streamdown, "mode")?.initializer?.getText(source),
    '"streaming"',
  );
  assert.equal(
    jsxAttribute(streamdown, "animated")?.initializer?.getText(source),
    "{STREAMDOWN_IMMEDIATE_UPDATES}",
  );
});

test("token updates keep Streamdown's expensive configuration props stable", () => {
  const streamdown = findChatStreamdown();
  assert.ok(streamdown, "chat <Streamdown> is missing");

  assert.equal(
    jsxAttribute(streamdown, "plugins")?.initializer?.getText(source).replace(/\s+/g, " "),
    "{ highlightCode ? STREAMDOWN_PLUGINS : STREAMDOWN_PLAIN_CODE_PLUGINS }",
    "plugins must retain object identity while raw tokens arrive",
  );
  for (const [attribute, constant] of [
    ["controls", "STREAMDOWN_CONTROLS"],
    ["shikiTheme", "STREAMDOWN_SHIKI_THEME"],
  ]) {
    assert.equal(
      jsxAttribute(streamdown, attribute)?.initializer?.getText(source),
      `{${constant}}`,
      `${attribute} must retain object identity while raw tokens arrive`,
    );
  }
});

test("stream updates are paint-coalesced with a cost-aware cadence and no length throttle", () => {
  const markdownSource = source.getText();
  const hookStart = markdownSource.indexOf(
    "function useCoalescedStreamingText",
  );
  const hookEnd = markdownSource.indexOf("const MarkdownTextImpl", hookStart);
  assert.ok(hookStart >= 0 && hookEnd > hookStart);
  const hook = markdownSource.slice(hookStart, hookEnd);

  assert.ok(hook.includes("requestAnimationFrame"));
  assert.ok(!hook.includes("setTimeout"));
  assert.match(hook, /if \(frameTime > frameAtRef\.current\) \{\s*frameIntervalRef\.current = Math\.min\(\s*frameIntervalRef\.current,\s*frameTime - frameAtRef\.current,\s*\);\s*frameAtRef\.current = frameTime;\s*\}/, "the shortest gap between frames is the display's frame interval");
  assert.match(
    hook,
    /frameTime - renderedAtRef\.current <\s*Math\.max\(\s*16 - Math\.min\(frameIntervalRef\.current, 16\) \/ 2,\s*STREAMING_RENDER_DUTY \* costRef\.current,\s*\)/,
    "frame times come in whole frames, so the 16 ms floor allows half a frame or a 75, 90 or 144 Hz display renders every second or third frame",
  );
  assert.match(
    hook,
    /afterFrame\(\(\) => \{\s*const startedAt = performance\.now\(\);\s*flushSync\(\(\) => setDisplayed\(pendingRef\.current\)\);\s*costRef\.current = performance\.now\(\) - startedAt;\s*\}\);/,
    "the cost is the render alone, in the task after the frame, so an unrelated long task is not charged to the stream and the text resumes as soon as it ends",
  );

  // A running message can be replaced rather than appended to, as the audio
  // path does when it swaps its placeholder for the player, so holding the last
  // painted text has to be gated on the new text extending it.
  //
  // The spelling is pinned, not left open. `startsWith` scans a growing reply,
  // 74 ms against 1.6 ms over a 60,000 character stream, and the two spellings
  // are behaviourally identical, so no output test can tell them apart and this
  // source check is the only thing standing between the hot path and a quiet
  // revert. Written as `slice || startsWith` it accepted both and passed on the
  // previous code unchanged, which is to say it measured nothing.
  assert.ok(
    hook.includes("text.length >= displayed.text.length"),
    "the coalescer must reject a shorter replacement on length first",
  );
  assert.ok(
    hook.includes("text.slice(0, displayed.text.length) === displayed.text"),
    "the coalescer must hold the painted text only when the new text extends it",
  );
  assert.ok(
    !hook.includes("text.startsWith(displayed.text)"),
    "the coalescer must not scan the reply to decide the new text extends it",
  );
});

test("streaming reparses only the active Markdown tail", () => {
  const streamdown = findChatStreamdown();
  assert.ok(streamdown, "chat <Streamdown> is missing");
  assert.equal(
    jsxAttribute(streamdown, "parseIncompleteMarkdown")?.initializer?.getText(
      source,
    ),
    "{!incrementalRender}",
  );
  assert.equal(
    jsxAttribute(streamdown, "remend")?.initializer?.getText(source),
    "{pendingLinkRepair ? LITERAL_LINK_REMEND : undefined}",
  );
  assert.equal(
    jsxAttribute(streamdown, "parseMarkdownIntoBlocksFn")?.initializer?.getText(
      source,
    ),
    "{\n" +
      "              incrementalRender?.parseMarkdownIntoBlocks ??\n" +
      "              parseMarkdownIntoRenderableBlocks\n" +
      "            }",
  );
});

test("retained block and sandbox changes move Streamdown's render identity", () => {
  // Streamdown also ignores rehypePlugins changes when memoizing.
  const streamdown = findChatStreamdown();
  assert.ok(streamdown, "chat <Streamdown> is missing");
  assert.equal(
    jsxAttribute(streamdown, "key")?.initializer?.getText(source),
    "{`${messageId}:${incrementalCache.renderGeneration}:${renderKey}:${sandboxScopeKey}`}",
  );
  assert.match(
    source.getText(),
    /const sandboxScopeKey = JSON.stringify\(\[threadId, projectId\]\)/,
  );
});
