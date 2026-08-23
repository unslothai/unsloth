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
const REASONING_PATH = new URL(
  "../src/components/assistant-ui/reasoning.tsx",
  import.meta.url,
);

const OVERSIZED_CODE_PATH = new URL(
  "../src/components/assistant-ui/oversized-streaming-code-block.ts",
  import.meta.url,
);
const oversizedCodeSource = readFileSync(OVERSIZED_CODE_PATH, "utf8");
const reasoningSource = readFileSync(REASONING_PATH, "utf8");
const source = ts.createSourceFile(
  MARKDOWN_TEXT_PATH.pathname,
  readFileSync(MARKDOWN_TEXT_PATH, "utf8"),
  ts.ScriptTarget.ESNext,
  true,
  ts.ScriptKind.TSX,
);

function findObjectConfig(name: string): ts.ObjectLiteralExpression | null {
  let config: ts.ObjectLiteralExpression | null = null;
  const visit = (node: ts.Node): void => {
    if (
      ts.isVariableDeclaration(node) &&
      node.name.getText(source) === name &&
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
  const config = findObjectConfig("STREAMDOWN_IMMEDIATE_UPDATES");
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

  const plugins = jsxAttribute(streamdown, "plugins")?.initializer?.getText(source);
  assert.ok(plugins);
  assert.match(plugins, /codeHighlighting === "syntax"/);
  assert.match(plugins, /STREAMDOWN_SYNTAX_PLUGINS/);
  assert.match(plugins, /STREAMDOWN_PLAIN_CODE_PLUGINS/);
});

test("reasoning selects stable plain-code plugins without disabling rich Markdown", () => {
  // Pagination is the flag's business; see reasoning-pagination-flag.test.ts.
  assert.match(
    reasoningSource,
    /<MarkdownText\s+codeHighlighting="plain"/,
    "reasoning must select plain code",
  );

  const syntaxPlugins = findObjectConfig("STREAMDOWN_SYNTAX_PLUGINS");
  const plainPlugins = findObjectConfig("STREAMDOWN_PLAIN_CODE_PLUGINS");
  assert.ok(syntaxPlugins);
  assert.ok(plainPlugins);
  const syntaxKeys = syntaxPlugins.properties.map((property) =>
    property.name?.getText(source),
  );
  const plainKeys = plainPlugins.properties.map((property) =>
    property.name?.getText(source),
  );
  assert.deepEqual(syntaxKeys, ["code", "math", "mermaid", "renderers"]);
  assert.deepEqual(plainKeys, ["math", "mermaid", "renderers"]);

  const markdownSource = source.getText();
  assert.equal(
    markdownSource.match(/codeHighlighting === "syntax"/g)?.length,
    3,
    "plugin, persistent/global, and direct terminal/oversized paths must all gate Shiki",
  );
  assert.match(
    markdownSource,
    /codeHighlighting === "plain" \|\|\s*isOversizedStreamingCode/,
    "plain reasoning code must use the one-text-node renderer, not per-line token spans",
  );

});

test("stream updates use the paint-aware bounded presentation scheduler", () => {
  const markdownSource = source.getText();
  const hookStart = markdownSource.indexOf(
    "function useCoalescedStreamingText",
  );
  const hookEnd = markdownSource.indexOf("const MarkdownTextImpl", hookStart);
  assert.ok(hookStart >= 0 && hookEnd > hookStart);
  const hook = markdownSource.slice(hookStart, hookEnd);

  assert.ok(hook.includes("createStreamingTextPresentationScheduler"));
  assert.ok(hook.includes("scheduler.schedule(text.length, pending)"));
  assert.ok(hook.includes("scheduler.flush(pending)"));
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

test("one Streamdown provider shell owns only the final mutable block", () => {
  const streamdown = findChatStreamdown();
  assert.ok(streamdown, "chat <Streamdown> is missing");
  assert.equal(
    jsxAttribute(streamdown, "parseIncompleteMarkdown")?.initializer?.getText(
      source,
    ),
    "{false}",
  );
  assert.equal(
    jsxAttribute(streamdown, "parseMarkdownIntoBlocksFn")?.initializer?.getText(
      source,
    ),
    "{parseProviderShellBlock}",
  );
  assert.equal(
    jsxAttribute(streamdown, "BlockComponent")?.initializer?.getText(source),
    "{PartitionedStreamdownShell}",
  );
});

test("completion keeps the same provider shell and committed chunks are memoized", () => {
  const streamdown = findChatStreamdown();
  assert.ok(streamdown, "chat <Streamdown> is missing");
  assert.equal(
    jsxAttribute(streamdown, "key")?.initializer?.getText(source),
    "{messageId}",
    "completion must not remount the document",
  );

  const markdownSource = source.getText();
  const chunksStart = markdownSource.indexOf(
    "const StableCommittedChunks = memo(",
  );
  const chunksEnd = markdownSource.indexOf(
    "function PartitionedStreamdownBlock",
    chunksStart,
  );
  assert.ok(chunksStart >= 0 && chunksEnd > chunksStart);
  const chunksMemo = markdownSource.slice(chunksStart, chunksEnd);
  assert.match(chunksMemo, /previous\.chunks === next\.chunks/);
  assert.doesNotMatch(
    chunksMemo,
    /previous\.shellProps === next\.shellProps/,
    "the provider shell creates a new wrapper props object per token",
  );

  const chunkStart = markdownSource.indexOf(
    "const StableCommittedChunk = memo(",
  );
  const chunkEnd = markdownSource.indexOf(
    "type StableCommittedChunksProps",
    chunkStart,
  );
  assert.ok(chunkStart >= 0 && chunkEnd > chunkStart);
  assert.match(
    markdownSource.slice(chunkStart, chunkEnd),
    /previous\.chunk === next\.chunk/,
  );

  const partitionStart = markdownSource.indexOf(
    "function PartitionedStreamdownBlock",
  );
  const partitionEnd = markdownSource.indexOf(
    "const PartitionedStreamdownShell",
    partitionStart,
  );
  assert.ok(partitionStart >= 0 && partitionEnd > partitionStart);
  assert.doesNotMatch(
    markdownSource.slice(partitionStart, partitionEnd),
    /plan\.chunks\.map/,
    "a token-only context update must not map the committed chunk list",
  );
});

test("completed moderate fences keep the deferred highlighter on the persistent path", () => {
  const markdownSource = source.getText();
  const rendererStart = markdownSource.indexOf(
    "function PersistentOversizedCodeRenderer",
  );
  const rendererEnd = markdownSource.indexOf(
    "const presentCompletedCodeFences",
    rendererStart,
  );
  assert.ok(rendererStart >= 0 && rendererEnd > rendererStart);
  const renderer = markdownSource.slice(rendererStart, rendererEnd);

  assert.match(renderer, /codeHighlighting === "syntax"/);
  assert.match(renderer, /\? prepareOversizedCodeHighlight/);
  assert.match(renderer, /: undefined/);
});

test("syntactic fence openness is independent from stream and action state", () => {
  const markdownSource = source.getText();
  const tailStart = markdownSource.indexOf("function TerminalCodeTail");
  const tailEnd = markdownSource.indexOf(
    "function PartitionedStreamdownBlock",
    tailStart,
  );
  assert.ok(tailStart >= 0 && tailEnd > tailStart);
  const tail = markdownSource.slice(tailStart, tailEnd);

  assert.match(tail, /const isFenceOpen = !codeTail\.isClosed/);
  assert.match(tail, /const isStreaming = shellProps\.isIncomplete/);
  assert.match(tail, /const actionsDisabled = isFenceOpen && isStreaming/);
  assert.match(tail, /actionsDisabled=\{actionsDisabled\}/);
  assert.match(tail, /isFenceOpen=\{isFenceOpen\}/);
  assert.match(tail, /isIncomplete=\{actionsDisabled\}/);

  const blockStart = markdownSource.indexOf("function StreamdownBlockContent");
  const blockEnd = markdownSource.indexOf(
    "const StreamdownBlock = memo",
    blockStart,
  );
  assert.ok(blockStart >= 0 && blockEnd > blockStart);
  const block = markdownSource.slice(blockStart, blockEnd);
  assert.match(block, /isFenceOpen=\{isFenceOpen\}/);

  assert.match(
    block,
    /!isFenceOpen && !props\.isIncomplete && isSvgFence/,
  );
  assert.match(
    block,
    /!messageHasRenderableRenderHtmlTool &&\s*!isFenceOpen &&\s*!props\.isIncomplete/,
  );
  // The action bar moved inside `FenceBlock` when the fence branch was composed with the
  // off-screen deferral gate; the state it is driven from did not.
  assert.match(
    block,
    /actionsDisabled=\{actionsDisabled \?\? props\.isIncomplete\}/,
  );

  const persistentStart = markdownSource.indexOf(
    "function PersistentOversizedCodeRenderer",
  );
  const persistentEnd = markdownSource.indexOf(
    "const presentCompletedCodeFences",
    persistentStart,
  );
  assert.ok(persistentStart >= 0 && persistentEnd > persistentStart);
  const persistent = markdownSource.slice(persistentStart, persistentEnd);
  assert.match(persistent, /isFenceOpen=\{false\}/);
  assert.match(persistent, /disabled=\{isIncomplete\}/);

  assert.match(oversizedCodeSource, /if \(\s*isFenceOpen \|\|/);
  assert.match(oversizedCodeSource, /return createElement\(PlainCodeBlock, \{ isFenceOpen/);
  assert.doesNotMatch(
    oversizedCodeSource,
    /if \(\s*isIncomplete \|\|/,
    "message completion must not let a still-open fence schedule Shiki",
  );
});