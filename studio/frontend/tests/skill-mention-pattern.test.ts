// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

// The module imports the auth layer, so the pattern is lifted from source; the assertion keeps the copy honest.
const SKILLS_API_SOURCE = readFileSync(
  new URL("../src/features/chat/api/skills-api.ts", import.meta.url),
  "utf8",
);
const declaration = /export const SKILL_MENTION_PATTERN =\s*(\/.*\/g);/.exec(
  SKILLS_API_SOURCE,
);
assert.ok(declaration, "SKILL_MENTION_PATTERN declaration not found");
const SKILL_MENTION_PATTERN = new RegExp(
  declaration[1].slice(1, declaration[1].lastIndexOf("/")),
  "g",
);

const names = (text: string): string[] =>
  [...text.matchAll(SKILL_MENTION_PATTERN)].map((match) => match[2] ?? "");

test("spec-shaped names at a word boundary are mentions", () => {
  assert.deepEqual(names("@probe-alpha"), ["probe-alpha"]);
  assert.deepEqual(names("use @probe-alpha."), ["probe-alpha"]);
  // No whitespace before the first @, so only the second one is a mention.
  assert.deepEqual(names("(@probe-alpha), then @beta2!"), ["beta2"]);
  assert.deepEqual(names("@a @b9 x"), ["a", "b9"]);
  assert.deepEqual(names(`say "@probe-alpha"`), []);
  assert.deepEqual(names("first line\n@probe-alpha\tnext"), ["probe-alpha"]);
});

test("emails, underscores, uppercase and dangling hyphens are not mentions", () => {
  assert.deepEqual(names("mail foo@example.com"), []);
  assert.deepEqual(names("@example.com please"), []);
  // Digit-led names are valid per spec, so `@3pm` stays a candidate; the send path re-reads under a deadline.
  assert.deepEqual(names("meet @3pm"), ["3pm"]);
  assert.deepEqual(names("@probe_alpha"), []);
  assert.deepEqual(names("@Probe-Alpha"), []);
  assert.deepEqual(names("@probe-alpha-"), []);
  assert.deepEqual(names("@-probe"), []);
});

test("a name is at most 64 characters", () => {
  const sixtyFour = "a".repeat(64);
  assert.deepEqual(names(`@${sixtyFour}`), [sixtyFour]);
  assert.deepEqual(names(`@${sixtyFour}a`), []);
});

test("the picker formatter and the send path share one pattern", () => {
  const mentions = readFileSync(
    new URL("../src/components/assistant-ui/skill-mentions.tsx", import.meta.url),
    "utf8",
  );
  assert.match(mentions, /text\.matchAll\(SKILL_MENTION_PATTERN\)/);
  assert.doesNotMatch(mentions, /const MENTION_PATTERN =/);
});
