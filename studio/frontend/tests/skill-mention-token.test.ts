// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Accepting a skill from the @ popover replaces the whole token under the caret. Before this,
// both composers replaced only the text up to the caret, so arrowing back into "@calculator"
// and pressing Enter produced "@calculator tor".

import assert from "node:assert/strict";
import test from "node:test";
import {
  mentionTokenAt,
  replaceMentionToken,
} from "../src/components/assistant-ui/skill-mention-token.ts";
import { readSrc } from "./helpers/kit.ts";

test("the token runs from the @ to the end of the name, whichever side of the caret", () => {
  assert.deepEqual(mentionTokenAt("@calc", 5), { start: 0, end: 5, query: "calc" });
  assert.deepEqual(mentionTokenAt("@calculator", 5), { start: 0, end: 11, query: "calc" });
  assert.deepEqual(mentionTokenAt("@calculator", 1), { start: 0, end: 11, query: "" });
  assert.deepEqual(mentionTokenAt("use @calculator now", 8), {
    start: 4,
    end: 15,
    query: "cal",
  });
  // The query is what was typed before the caret; the name after it is not a filter.
  assert.equal(mentionTokenAt("@calculator", 1)?.query, "");
});

test("no token without an @ at a word start, or with a name character the rule refuses", () => {
  assert.equal(mentionTokenAt("mail foo@example", 16), null);
  assert.equal(mentionTokenAt("plain text", 5), null);
  assert.equal(mentionTokenAt("@calc.md", 8), null);
  // A caret right after the @ still counts: that is the arrow-back case.
  assert.notEqual(mentionTokenAt("x @b", 3), null);
});

test("replacing the token keeps one space after the mention and puts the caret past it", () => {
  const token = mentionTokenAt("@calculator", 5);
  assert.ok(token);
  assert.deepEqual(replaceMentionToken("@calculator", token, "@calculator"), {
    text: "@calculator ",
    caret: 12,
  });
  const mid = mentionTokenAt("ask @calc for it", 9);
  assert.ok(mid);
  assert.deepEqual(replaceMentionToken("ask @calc for it", mid, "@calculator"), {
    text: "ask @calculator for it",
    caret: 16,
  });
  const wholeWord = mentionTokenAt("@calculator tor", 11);
  assert.ok(wholeWord);
  assert.equal(
    replaceMentionToken("@calculator tor", wholeWord, "@calculator").text,
    "@calculator tor",
  );
});

// Both composers go through the same token rule: the plain textarea one directly, the
// assistant-ui one through the select override the library offers.
test("both composers replace the whole token", () => {
  const mentions = readSrc("components/assistant-ui/skill-mention-token.ts");
  assert.match(mentions, /const tail = \/\^\[a-z0-9-\]\*\/i\.exec\(text\.slice\(caret\)\)/);
  const popover = readSrc("components/assistant-ui/skill-mentions.tsx");
  assert.match(popover, /const next = mentionTokenAt\(nextText, caret\);/);
  assert.match(popover, /const next = replaceMentionToken\(text, range, directive\);/);
  assert.match(
    popover,
    /registerSelectItemOverride\(\(item\) => \{\n\s*const input = document\.activeElement;\n\s*if \(!\(input instanceof HTMLTextAreaElement\)\) return false;/,
  );
  assert.match(popover, /aui\.composer\(\)\.setText\(next\.text\);\n\s*setCursorPosition\(next\.caret\);/);
  assert.match(popover, /<MentionTokenReplacer \/>/);
});
