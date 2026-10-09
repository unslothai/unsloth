// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


import assert from "node:assert/strict";
import test from "node:test";
import {
  mentionTokenAt,
  mentionableSkills,
  rankMentionSkills,
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
  assert.equal(mentionTokenAt("@calculator", 1)?.query, "");
});

test("no token without an @ at a word start, or with a name character the rule refuses", () => {
  assert.equal(mentionTokenAt("mail foo@example", 16), null);
  assert.equal(mentionTokenAt("plain text", 5), null);
  assert.equal(mentionTokenAt("@calc.md", 8), null);
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

const skill = (name: string, description = "", extra = {}) => ({
  name,
  description,
  valid: true,
  shadowed: false,
  enabled: true,
  ...extra,
});

test("only enabled, valid, winning skills are offered", () => {
  const offered = mentionableSkills([
    skill("on"),
    skill("off", "", { enabled: false }),
    skill("broken", "", { valid: false }),
    skill("shadowed", "", { shadowed: true }),
  ]);
  assert.deepEqual(
    offered.map((entry) => entry.name),
    ["on"],
  );
});

test("suggestions rank name prefixes, then names, then descriptions", () => {
  const skills = [
    skill("release-notes", "Write a changelog"),
    skill("notes", "Plain notes"),
    skill("pdf", "Read notes from a PDF"),
    skill("unrelated", "Nothing here"),
  ];
  const names = (query: string, limit = 50) =>
    rankMentionSkills(skills, query, limit).map((entry) => entry.name);
  assert.deepEqual(names("notes"), ["notes", "release-notes", "pdf"]);
  assert.deepEqual(names("NOTES"), ["notes", "release-notes", "pdf"]);
  assert.deepEqual(names("re"), ["release-notes", "unrelated", "pdf"]);
  // An empty query is the bare @: every skill in catalog order, capped.
  assert.deepEqual(names(""), skills.map((entry) => entry.name));
  assert.deepEqual(names("", 2), ["release-notes", "notes"]);
  assert.deepEqual(names("zzz"), []);
});

test("picking a suggestion inserts the exact token the send path resolves", () => {
  const token = mentionTokenAt("summarize with @rel", 19);
  assert.ok(token);
  const [picked] = rankMentionSkills([skill("release-notes")], token.query, 50);
  const next = replaceMentionToken("summarize with @rel", token, `@${picked.name}`);
  assert.equal(next.text, "summarize with @release-notes ");
  assert.equal(next.caret, next.text.length);
});

test("both composers share the ranked filter", () => {
  const popover = readSrc("components/assistant-ui/skill-mentions.tsx");
  assert.equal(popover.match(/rankMentionSkills\(/g)?.length, 2);
  assert.equal(popover.match(/mentionableSkills\(skills\)/g)?.length, 2);
});
