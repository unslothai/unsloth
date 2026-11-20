// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Quote anchors are pure text selectors, so these tests need no DOM and prove
// that no Range or element reference is part of the stored source.

import assert from "node:assert/strict";
import test from "node:test";

import {
  canonicalQuoteText,
  createQuoteAnchor,
  normalizeQuoteText,
  quoteSearchIndex,
  resolveQuoteAnchor,
  type QuoteAnchor,
} from "../src/components/assistant-ui/message-quote-model.ts";
import {
  foldText,
  type FindTextIndex,
} from "../src/features/find-in-page/lib/find-text-index.ts";

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

function anchorFor(
  index: FindTextIndex,
  text: string,
  start: number,
): QuoteAnchor {
  const anchor = createQuoteAnchor({
    messageId: "message-1",
    turn: 2,
    text,
    markdown: text,
    index,
    start,
    end: start + text.length,
  });
  assert.ok(anchor, `expected an anchor for ${JSON.stringify(text)}`);
  return anchor;
}

test("quote text normalises clipboard whitespace", () => {
  assert.equal(normalizeQuoteText(" A\u00a0B\r\nC "), "A B\nC");
});

test("quote anchors capture text, offsets, context, and occurrence", () => {
  const anchor = anchorFor(
    fakeIndex("Alpha passage. Beta passage."),
    "Alpha passage",
    0,
  );

  assert.equal(anchor.messageId, "message-1");
  assert.equal(anchor.turn, 2);
  assert.equal(anchor.text, "Alpha passage");
  assert.equal(anchor.markdown, "Alpha passage");
  assert.equal(anchor.start, 0);
  assert.equal(anchor.end, 13);
  assert.equal(anchor.prefix, "");
  assert.equal(anchor.suffix, ". beta passage.");
  assert.equal(anchor.occurrence, 1);

  // The source is plain serializable data, not a DOM reference.
  assert.deepEqual(JSON.parse(JSON.stringify(anchor)), anchor);
});

test("repeated passages store the right occurrence", () => {
  const anchor = anchorFor(fakeIndex("one two one"), "one", 8);

  assert.equal(anchor.occurrence, 2);
  assert.equal(anchor.prefix, "one two ");
  assert.equal(anchor.suffix, "");
});

test("an unchanged message resolves at the captured offsets", () => {
  const index = fakeIndex("Alpha passage. Beta passage.");
  const anchor = anchorFor(index, "Alpha passage", 0);

  assert.deepEqual(resolveQuoteAnchor(index, anchor), {
    start: 0,
    end: 13,
    status: "exact",
  });
});

test("content inserted earlier relocates the passage by text", () => {
  const anchor = anchorFor(
    fakeIndex("Alpha passage. Beta passage."),
    "Alpha passage",
    0,
  );

  assert.deepEqual(
    resolveQuoteAnchor(
      fakeIndex("Intro. Alpha passage. Beta passage."),
      anchor,
    ),
    { start: 7, end: 20, status: "relocated" },
  );
});

test("context picks the intended repeated passage", () => {
  const second = anchorFor(
    fakeIndex("Repeat this. Repeat this."),
    "Repeat this",
    13,
  );

  assert.equal(second.occurrence, 2);
  assert.deepEqual(
    resolveQuoteAnchor(
      fakeIndex("New. Repeat this. Repeat this."),
      second,
    ),
    { start: 18, end: 29, status: "relocated" },
  );
});

test("a missing passage marks the anchor stale", () => {
  const anchor = anchorFor(fakeIndex("Alpha passage."), "Alpha passage", 0);

  assert.equal(
    resolveQuoteAnchor(fakeIndex("Completely different."), anchor),
    null,
  );
});

test("repeated text without context is treated as ambiguous", () => {
  const ambiguous: QuoteAnchor = {
    messageId: "message-1",
    turn: 1,
    text: "same",
    markdown: "same",
    start: 100,
    end: 104,
    prefix: "",
    suffix: "",
    occurrence: 1,
  };

  assert.equal(
    resolveQuoteAnchor(fakeIndex("same same"), ambiguous),
    null,
  );
});

test("block separators do not break cross-paragraph anchors", () => {
  const index = fakeIndex("first\u0000second");
  const anchor = anchorFor(index, "first\nsecond", 0);

  assert.deepEqual(resolveQuoteAnchor(index, anchor), {
    start: 0,
    end: 12,
    status: "exact",
  });
});

test("captured block selections canonicalize to the indexed span", () => {
  const cases = [
    {
      captured: "alpha onebeta two",
      span: "alpha one\nbeta two",
      index: "alpha one\u0000beta two",
    },
    {
      captured: "line oneline two",
      span: "line one\nline two",
      index: "line one\u0000line two",
    },
    {
      captured: "oneTwoThree",
      span: "one\nTwo\nThree",
      index: "one\u0000Two\u0000Three",
    },
    {
      captured: "HeadingBody",
      span: "Heading\nBody",
      index: "Heading\u0000Body",
    },
  ];

  for (const { captured, span, index: text } of cases) {
    const index = fakeIndex(text);
    const canonical = canonicalQuoteText(captured, span);
    assert.equal(canonical, span);
    const anchor = createQuoteAnchor({
      messageId: "message-1",
      turn: 2,
      text: canonical,
      markdown: canonical,
      index,
      start: 0,
      end: canonical.length,
    });
    assert.ok(anchor);
    assert.deepEqual(resolveQuoteAnchor(index, anchor), {
      start: 0,
      end: canonical.length,
      status: "exact",
    });
  }
});

test("single-block capture text is unchanged", () => {
  const captured = "alpha one";
  const text = "alpha one";
  const index = fakeIndex(text);
  const canonical = canonicalQuoteText(captured, text);
  assert.equal(canonical, captured);
  const anchor = anchorFor(index, canonical, 0);
  assert.deepEqual(resolveQuoteAnchor(index, anchor), {
    start: 0,
    end: canonical.length,
    status: "exact",
  });
});

test("block boundary canonicalization accepts CRLF but not extra spaces", () => {
  assert.equal(
    canonicalQuoteText("alpha\u00a0one\r\nbeta two", "alpha one\nbeta two"),
    "alpha one\nbeta two",
  );
  assert.equal(
    canonicalQuoteText(
      "line one\nline twoline three",
      "line one\nline two\nline three",
    ),
    "line one\nline two\nline three",
  );
  assert.equal(
    canonicalQuoteText("alpha  onebeta two", "alpha one\nbeta two"),
    "alpha  onebeta two",
  );
});

test("repeated multi-block passages keep context-based resolution", () => {
  const passage = "repeat.\ntail.";
  const index = fakeIndex(
    "Intro.\nRepeat.\nTail. More.\nRepeat.\nTail.",
  );
  const search = quoteSearchIndex(index);
  const first = search.text.indexOf(passage);
  const second = search.text.indexOf(passage, first + 1);
  assert.ok(first >= 0 && second > first);

  const text = canonicalQuoteText("repeat.tail.", passage);
  const anchor = createQuoteAnchor({
    messageId: "message-1",
    turn: 2,
    text,
    markdown: text,
    index,
    start: second,
    end: second + text.length,
  });
  assert.ok(anchor);
  assert.equal(anchor.occurrence, 2);

  const shifted = fakeIndex(`New. ${index.text}`);
  assert.deepEqual(resolveQuoteAnchor(shifted, anchor), {
    start: second + 5,
    end: second + 5 + text.length,
    status: "relocated",
  });
});

test("ambiguous repeated multi-block passages still fail", () => {
  const index = fakeIndex("Repeat.\nTail. Repeat.\nTail.");
  const text = canonicalQuoteText("Repeat.Tail.", "Repeat.\nTail.");
  const anchor: QuoteAnchor = {
    messageId: "message-1",
    turn: 1,
    text,
    markdown: text,
    start: 100,
    end: 100 + text.length,
    prefix: "",
    suffix: "",
    occurrence: 1,
  };

  assert.equal(resolveQuoteAnchor(index, anchor), null);
});

test("empty or inverted capture spans produce no anchor", () => {
  const index = fakeIndex("Alpha passage.");

  assert.equal(
    createQuoteAnchor({
      messageId: "message-1",
      turn: 1,
      text: "   ",
      markdown: "",
      index,
      start: 0,
      end: 5,
    }),
    null,
  );
  assert.equal(
    createQuoteAnchor({
      messageId: "message-1",
      turn: 1,
      text: "Alpha",
      markdown: "Alpha",
      index,
      start: 5,
      end: 2,
    }),
    null,
  );
  assert.equal(
    createQuoteAnchor({
      messageId: "",
      turn: 1,
      text: "Alpha",
      markdown: "Alpha",
      index,
      start: 0,
      end: 5,
    }),
    null,
  );
});
