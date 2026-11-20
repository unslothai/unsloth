// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Source navigation keeps the quote anchor authoritative: resolve inside the
// source message, reveal the virtualized row, restore parked content, then
// paint a temporary exact Range. These tests pin pure decisions and the DOM
// orchestration contracts that the real browser exercises.

import assert from "node:assert/strict";
import test from "node:test";
import ts from "@typescript/typescript6";
import { readSrc } from "./helpers/kit.ts";
import {
  createQuoteAnchor,
  resolveQuoteAnchor,
  type QuoteAnchor,
} from "../src/components/assistant-ui/message-quote-model.ts";
import {
  detectQuoteHighlightSupport,
  quoteHighlightBoxes,
  quoteRangeStillMatches,
  QUOTE_HIGHLIGHT_NAME,
} from "../src/components/assistant-ui/message-quote-highlight.ts";
import {
  foldText,
  type FindTextIndex,
} from "../src/features/find-in-page/lib/find-text-index.ts";

const navigationSource = readSrc(
  "components/assistant-ui/message-quote-navigation.ts",
);
const highlightSource = readSrc(
  "components/assistant-ui/message-quote-highlight.ts",
);
const collectionSource = readSrc(
  "components/assistant-ui/message-quote-collection.tsx",
);
const captureSource = readSrc(
  "components/assistant-ui/message-quote-capture.ts",
);
const windowSource = readSrc(
  "components/assistant-ui/message-window-controller.ts",
);
const cssSource = readSrc("index.css");

function liftedNode(name: string, classNode = false): string {
  const source = ts.createSourceFile(
    "message-quote-navigation.ts",
    navigationSource,
    ts.ScriptTarget.Latest,
    true,
    ts.ScriptKind.TS,
  );
  const node = source.statements.find((statement) =>
    classNode
      ? ts.isClassDeclaration(statement) && statement.name?.text === name
      : ts.isFunctionDeclaration(statement) && statement.name?.text === name,
  );
  assert.ok(node, `missing ${name}`);
  return ts.transpileModule(
    node.getText(source).replace(/^export\s+/, ""),
    {
      compilerOptions: {
        target: ts.ScriptTarget.ES2022,
        module: ts.ModuleKind.None,
      },
    },
  ).outputText;
}

function lifted(name: string): string {
  return liftedNode(name);
}

function liftedClass(name: string): string {
  return liftedNode(name, true);
}

const pure = new Function(
  `${lifted("quoteSourceAvailable")}\n${lifted("quoteScrollDelta")}
return { quoteSourceAvailable, quoteScrollDelta };`,
)() as {
  quoteSourceAvailable: (
    messages: readonly { id?: string; role?: string }[],
    messageId: string,
  ) => boolean;
  quoteScrollDelta: (
    passage: { top: number; bottom: number },
    viewport: { top: number; height: number; paddingTop?: number },
    inset?: number,
  ) => number;
};

function fakeIndex(text: string): FindTextIndex {
  const folded = foldText(text);
  return {
    text: folded,
    segments: [],
    truncated: false,
    seams: new Set<number>(),
    rootLength: folded.length,
    surfaces: [],
  };
}

function anchorAt(
  index: FindTextIndex,
  text: string,
  start: number,
): QuoteAnchor {
  const anchor = createQuoteAnchor({
    messageId: "assistant-1",
    turn: 2,
    text,
    markdown: text,
    index,
    start,
    end: start + text.length,
  });
  assert.ok(anchor);
  return anchor;
}

test("source availability accepts only an assistant message in this thread", () => {
  const messages = [
    { id: "assistant-1", role: "assistant" },
    { id: "user-1", role: "user" },
  ];
  assert.equal(pure.quoteSourceAvailable(messages, "assistant-1"), true);
  assert.equal(pure.quoteSourceAvailable(messages, "user-1"), false);
  assert.equal(pure.quoteSourceAvailable(messages, "deleted"), false);
  assert.equal(
    pure.quoteSourceAvailable([{ id: "other", role: "assistant" }], "assistant-1"),
    false,
  );
});

test("two passages in one source message resolve to their own offsets", () => {
  const index = fakeIndex("Alpha. Beta. Alpha.");
  const first = anchorAt(index, "Alpha", 0);
  const second = anchorAt(index, "Alpha", 13);
  assert.deepEqual(resolveQuoteAnchor(index, first), {
    start: 0,
    end: 5,
    status: "exact",
  });
  assert.deepEqual(resolveQuoteAnchor(index, second), {
    start: 13,
    end: 18,
    status: "exact",
  });
  assert.notEqual(
    resolveQuoteAnchor(index, first)?.start,
    resolveQuoteAnchor(index, second)?.start,
  );
});

test("ambiguous repeated text without context does not resolve", () => {
  const index = fakeIndex("one one");
  const anchor: QuoteAnchor = {
    ...anchorAt(index, "one", 0),
    start: 100,
    end: 103,
    prefix: "",
    suffix: "",
    occurrence: 1,
  };
  assert.equal(resolveQuoteAnchor(index, anchor), null);
});

test("a changed passage does not resolve to a different matching phrase", () => {
  const captured = fakeIndex("unique phrase and other text");
  const anchor = anchorAt(captured, "unique phrase", 0);
  const changed = fakeIndex("other text with a different phrase");
  assert.equal(resolveQuoteAnchor(changed, anchor), null);
});

test("passage alignment centers an off-screen passage in the upper third", () => {
  const viewport = { top: 100, height: 600, paddingTop: 30 };
  assert.ok(pure.quoteScrollDelta({ top: -300, bottom: -280 }, viewport) < 0);
  assert.ok(pure.quoteScrollDelta({ top: 900, bottom: 920 }, viewport) > 0);
  assert.equal(
    pure.quoteScrollDelta({ top: 250, bottom: 280 }, viewport),
    0,
  );
});

test("the highlight predicate rejects detached or changed ranges", () => {
  const range = (text: string, connected = true) =>
    ({
      startContainer: { isConnected: connected },
      endContainer: { isConnected: connected },
      collapsed: false,
      toString: () => text,
    }) as unknown as Range;

  assert.equal(quoteRangeStillMatches(range("delta echo"), "delta echo"), true);
  assert.equal(quoteRangeStillMatches(range("delta echo"), "other"), false);
  assert.equal(
    quoteRangeStillMatches(range("delta echo", false), "delta echo"),
    false,
  );
});

test("highlight support requires both CSS.highlights and Highlight", () => {
  const scope = (css: boolean, highlight: boolean) => ({
    document: {},
    CSS: css ? { highlights: {} } : undefined,
    Highlight: highlight ? class {} : undefined,
  });
  assert.equal(detectQuoteHighlightSupport(scope(true, true) as never), true);
  assert.equal(detectQuoteHighlightSupport(scope(false, true) as never), false);
  assert.equal(detectQuoteHighlightSupport(scope(true, false) as never), false);
  assert.equal(
    detectQuoteHighlightSupport({ CSS: { highlights: {} }, Highlight: class {} } as never),
    false,
  );
});

test("overlay geometry filters empty and invalid rectangles", () => {
  assert.deepEqual(
    quoteHighlightBoxes([
      { left: 10, top: 20, width: 30, height: 8 },
      { left: 0, top: 0, width: 0, height: 8 },
      { left: 0, top: 0, width: 8, height: -1 },
    ]),
    [{ left: 10, top: 20, width: 30, height: 8 }],
  );
});

test("navigation uses the source message, progressive reveal, and body restore", () => {
  assert.match(navigationSource, /quoteSourceAvailable/);
  assert.match(navigationSource, /revealProgressiveRow/);
  assert.match(navigationSource, /completeProgressiveMounts\(/);
  assert.match(navigationSource, /session\.abort\.signal/);
  assert.match(navigationSource, /restoreMessageWindowBody/);
  assert.match(navigationSource, /buildTextIndex/);
  assert.match(navigationSource, /resolveQuoteAnchor/);
  assert.match(navigationSource, /rangeForMatch/);
  assert.match(navigationSource, /quoteRangeForContent/);
  assert.doesNotMatch(navigationSource, /querySelectorAll\([^)]*\)\.find\(/);
});

test("capture canonicalizes block separators and paint validates the range text", () => {
  assert.match(captureSource, /canonicalQuoteText\(captured, spanText\)/);
  assert.match(
    captureSource,
    /quoteSearchIndex\(index\)\.text\.slice\(start, end\)/,
  );
  assert.match(
    navigationSource,
    /expectedText: normalizeQuoteText\(finalRange\.toString\(\)\)/,
  );
});

test("capture and navigation share quote chrome filtering", () => {
  assert.match(captureSource, /QUOTE_PRUNE_SELECTOR/);
  assert.match(navigationSource, /QUOTE_PRUNE_SELECTOR/);
  assert.match(
    captureSource,
    /buildTextIndex\([\s\S]*?\[\],\s*QUOTE_PRUNE_SELECTOR/,
  );
  assert.match(
    navigationSource,
    /buildTextIndex\([\s\S]*?\[\],\s*QUOTE_PRUNE_SELECTOR/,
  );
});

test("navigation checks its module token and thread identity after awaits", () => {
  assert.match(navigationSource, /let navigationGeneration = 0/);
  assert.match(navigationSource, /\+\+navigationGeneration/);
  assert.match(navigationSource, /activeSession === session/);
  assert.match(navigationSource, /activeThreadId === session\.activeThreadId/);
  assert.match(navigationSource, /readThreadKey\?\.\(\)/);
  assert.match(collectionSource, /readThreadKey: \(\) => threadRef\.current/);
  assert.match(navigationSource, /isSessionCurrent\(session\)/);
  assert.ok(
    (navigationSource.match(/isSessionCurrent\(session\)/g) ?? []).length >= 6,
    "every async boundary must recheck the session",
  );
  assert.match(navigationSource, /export function cancelQuoteNavigation/);
  assert.match(navigationSource, /activeSession\?\.threadKey !== threadKey/);
  assert.match(navigationSource, /session\.abort\.abort\(\)/);
  assert.match(navigationSource, /clearQuoteAnnouncement\(\)/);
});

test("collapsed sections are tracked, user-touched, and restored", () => {
  assert.match(navigationSource, /class CollapsedSession/);
  assert.match(navigationSource, /data-slot="collapsible"\]\[data-state="closed"/);
  assert.match(navigationSource, /details:not\(\[open\]\)/);
  assert.match(navigationSource, /event\.isTrusted/);
  assert.match(navigationSource, /userTouched/);
  assert.match(navigationSource, /tracker\.root\.isConnected/);
  assert.match(navigationSource, /tracker\.opener\.click\(\)/);
  assert.match(navigationSource, /this\.trackers\.length = 0/);
  assert.match(navigationSource, /this\.restored \|\| !isActive\(\)/);
  assert.match(
    navigationSource,
    /openClosed\(content, \(\) =>\s*isSessionCurrent\(session\)/,
  );
});

test("restore closes details that navigation opened, not still-closed ones", () => {
  class FakeDetailsElement {
    open = false;
    parentElement: HTMLElement | null = null;
    isConnected = true;
    addEventListener() {}
    removeEventListener() {}
    querySelector() {
      return null;
    }
    getAttribute() {
      return null;
    }
  }
  const Session = new Function(
    "HTMLDetailsElement",
    "nextFrame",
    "closedQuoteRoots",
    `${liftedClass("CollapsedSession")}\n${lifted("isClosedRoot")}
     return CollapsedSession;`,
  )(FakeDetailsElement, () => Promise.resolve(), () => []) as new () => {
    trackers: Array<{
      root: HTMLElement;
      opener: HTMLElement | null;
      details: boolean;
      userTouched: boolean;
      markUserTouched: () => void;
    }>;
    restore(): void;
  };

  const opened = new FakeDetailsElement();
  const stillClosed = new FakeDetailsElement();
  const session = new Session();
  session.trackers.push(
    {
      root: opened as unknown as HTMLElement,
      opener: null,
      details: true,
      userTouched: false,
      markUserTouched: () => undefined,
    },
    {
      root: stillClosed as unknown as HTMLElement,
      opener: null,
      details: true,
      userTouched: false,
      markUserTouched: () => undefined,
    },
  );
  opened.open = true;
  session.restore();
  assert.equal(opened.open, false);
  assert.equal(stillClosed.open, false);
  assert.equal(session.trackers.length, 0);
});

test("restore clicks a controlled collapsible closed unless the user touched it", () => {
  class FakeDetailsElement {}
  const Session = new Function(
    "HTMLDetailsElement",
    "nextFrame",
    "closedQuoteRoots",
    `${liftedClass("CollapsedSession")}\n${lifted("isClosedRoot")}
     return CollapsedSession;`,
  )(FakeDetailsElement, () => Promise.resolve(), () => []) as new () => {
    trackers: Array<{
      root: HTMLElement;
      opener: HTMLElement | null;
      details: boolean;
      userTouched: boolean;
      markUserTouched: () => void;
    }>;
    restore(): void;
  };

  const makeRoot = (state: string) =>
    ({
      isConnected: true,
      parentElement: null,
      addEventListener() {},
      removeEventListener() {},
      getAttribute: () => state,
    }) as unknown as HTMLElement;
  let clicks = 0;
  const opener = {
    isConnected: true,
    click() {
      clicks += 1;
    },
  } as unknown as HTMLElement;

  const session = new Session();
  session.trackers.push(
    {
      root: makeRoot("open"),
      opener,
      details: false,
      userTouched: false,
      markUserTouched: () => undefined,
    },
    {
      root: makeRoot("open"),
      opener,
      details: false,
      userTouched: true,
      markUserTouched: () => undefined,
    },
  );
  session.restore();
  assert.equal(clicks, 1);
});

test("highlight observes the stable content boundary and validates the range", () => {
  assert.match(highlightSource, /quoteRangeStillMatches/);
  assert.match(highlightSource, /contentObserver/);
  assert.match(highlightSource, /boundaryObserver/);
  assert.match(highlightSource, /boundaryObserver\.observe\(row/);
  assert.match(highlightSource, /subtree: true/);
  assert.match(highlightSource, /if \(row !== null && !row\.isConnected\)/);
  assert.match(highlightSource, /requestAnimationFrame/);
  assert.match(highlightSource, /QUOTE_HIGHLIGHT_DURATION_MS = 4000/);
  assert.match(highlightSource, /CSS\?\.highlights/);
  assert.match(highlightSource, /typeof scope\.Highlight/);
  assert.match(highlightSource, /resolveRange/);
  assert.match(highlightSource, /paintRange\(replacement\)/);
  assert.match(highlightSource, /observeContentRoot\(\)/);
  assert.match(highlightSource, /liveRegionTimer = setTimeout/);
  assert.match(highlightSource, /data-quote-navigation-status/);
  assert.match(highlightSource, /if \(liveRegion === null\)/);
  assert.match(highlightSource, /session\.overlaySetRange\?\.\(session\.range\)/);
});

test("highlight cleanup removes registry, overlay, observers, and frames", () => {
  assert.match(highlightSource, /cancelAnimationFrame\(session\.frame\)/);
  assert.match(highlightSource, /cancelAnimationFrame\(session\.validationFrame\)/);
  assert.match(highlightSource, /session\.contentObserver\?\.disconnect\(\)/);
  assert.match(highlightSource, /removeRegisteredHighlight/);
  assert.match(highlightSource, /session\.overlay\.remove\(\)/);
  assert.match(highlightSource, /removeEventListener\("scroll"/);
  assert.match(highlightSource, /removeEventListener\("resize"/);
  assert.match(cssSource, /::highlight\(unsloth-quote\)/);
  assert.equal(QUOTE_HIGHLIGHT_NAME, "unsloth-quote");
});

test("navigation rows are native buttons and remove is a sibling action", () => {
  assert.match(collectionSource, /data-quote-navigate/);
  assert.match(collectionSource, /data-quote-remove/);
  assert.match(collectionSource, /onNavigate=\{navigate\}/);
  assert.match(collectionSource, /event\.stopPropagation\(\)/);
  assert.match(collectionSource, /onRemove\(quote\.id\)/);
  assert.doesNotMatch(collectionSource, /<li[^>]*onClick=/);
  assert.match(collectionSource, /announceQuoteNavigation/);
  assert.match(collectionSource, /cancelQuoteNavigation\(threadId\)/);
});

test("navigation cleanup follows thread changes, quote removal, and unmount", () => {
  assert.match(
    collectionSource,
    /highlightedQuoteRef\.current = null;\s*cancelQuoteNavigation\(threadId\);\s*return \(\) => cancelQuoteNavigation\(threadId\);/s,
  );
  assert.match(
    collectionSource,
    /if \(highlightedQuoteRef\.current === quoteId\) \{\s*cancelQuoteNavigation\(threadId\)/s,
  );
  assert.match(collectionSource, /highlightedQuoteRef/);
  assert.match(collectionSource, /!quotes\.some\(\(quote\) => quote\.id === highlighted\)/);
  assert.match(collectionSource, /quoteSourceMissing/);
  assert.match(collectionSource, /quotePassageMissing/);
  assert.match(collectionSource, /pillRef\.current\?\.focus\(\)/);
});

test("message window exposes a synchronous restore path", () => {
  assert.match(windowSource, /revealMessage\(messageId: string\)/);
  assert.match(windowSource, /restoreMessageWindowBody/);
  assert.match(windowSource, /this\.restore\(found, false\)/);
  assert.match(
    windowSource,
    /found\.saved === null && found\.frozen === null/,
  );
  assert.match(windowSource, /return true/);
});
