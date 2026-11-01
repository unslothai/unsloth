// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { markerRule } from "../src/features/chat/rich-composer/input-rules.ts";
import {
  composerSchema,
  parseMarkdown,
  serializeMarkdown,
} from "../src/features/chat/rich-composer/markdown.ts";
import {
  EditorState,
  Selection,
} from "../src/vendor/prosemirror-state/index.js";

type Rule = {
  match: RegExp;
  handler: (
    state: EditorState,
    match: RegExpMatchArray,
    start: number,
    end: number,
  ) => { doc: EditorState["doc"] } | null;
};

const { bold, italic, code } = composerSchema.marks;
const { paragraph, hard_break: hardBreak } = composerSchema.nodes;

function typed(text: string) {
  const content = text
    .split("\n")
    .flatMap((line, i) => [
      ...(i > 0 ? [hardBreak.create()] : []),
      ...(line ? [composerSchema.text(line)] : []),
    ]);
  return composerSchema.node("doc", null, [paragraph.create(null, content)]);
}

function type(
  before: string,
  char: string,
  marker: string,
  mark = bold,
): string | null {
  const doc = typed(before);
  const state = EditorState.create({ doc, selection: Selection.atEnd(doc) });
  const rule = markerRule(marker, mark) as unknown as Rule;
  const { $from } = state.selection;
  const text =
    $from.parent.textBetween(0, $from.parentOffset, undefined, "￼") + char;
  const match = rule.match.exec(text);
  if (!match) return null;
  const tr = rule.handler(
    state,
    match,
    $from.pos - (match[0].length - char.length),
    $from.pos,
  );
  return tr ? `${serializeMarkdown(tr.doc)} | ${tr.doc.textContent}` : null;
}

test("a closing marker turns the text between into the mark", () => {
  assert.equal(type("say **bold*", "*", "**"), "say **bold** | say bold");
  assert.equal(type("**two words*", "*", "**"), "**two words** | two words");
  assert.equal(type("an *it", "*", "*", italic), "an *it* | an it");
  assert.equal(
    type("run `ls -la", "`", "`", code),
    "run `ls -la` | run ls -la",
  );
});

test("markers that would not read back the same are left alone", () => {
  assert.equal(type("x**bold*", "*", "**"), null);
  assert.equal(type("say ** bold*", "*", "**"), null);
  assert.equal(type("say **bold *", "*", "**"), null);
  assert.equal(type("say **bold*", "*", "*", italic), null);
  assert.equal(type("a\n**b", "*", "**"), null);
});

test("a rule fires at the start of a line after a break", () => {
  assert.equal(
    type("first\n**second*", "*", "**"),
    "first\n**second** | first\nsecond",
  );
});
