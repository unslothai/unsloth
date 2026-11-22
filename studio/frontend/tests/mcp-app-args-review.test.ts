// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import * as mcpUi from "../src/features/chat/mcp-apps/mcp-ui.ts";

const frame = readFileSync(
  new URL("../src/features/chat/mcp-apps/mcp-app-frame.tsx", import.meta.url),
  "utf8",
);

// Read loosely: on a tree without the export, the assertion below is the failure.
const review = (mcpUi as Record<string, unknown>).toolArgsForReview as
  | ((args: Record<string, unknown>) => string | null)
  | undefined;

test("an argument placed after 600 characters reaches the approval card", () => {
  assert.equal(typeof review, "function", "mcp-ui.ts exports no toolArgsForReview");
  const args = { note: "x".repeat(620), path: "/home/user/.ssh/id_ed25519" };
  const shown = review?.(args);
  assert.equal(shown, JSON.stringify(args, null, 2));
  assert.match(shown ?? "", /id_ed25519/);
});

test("arguments too long to show whole say how much is not shown", () => {
  assert.equal(typeof review, "function", "mcp-ui.ts exports no toolArgsForReview");
  const max = (mcpUi as Record<string, unknown>).MAX_REVIEW_ARGS_CHARS as number;
  const shown = review?.({ blob: "y".repeat(max) }) ?? "";
  assert.match(shown, /more characters are sent but not shown$/);
  assert.equal(review?.({}), "");
});

test("the frame shows the reviewed arguments, not a silent 600-character cut", () => {
  assert.doesNotMatch(frame, /\.slice\(0, 600\)/);
  assert.match(frame, /const preview = toolArgsForReview\(args\);/);
});

const BS = String.fromCharCode(92);

test("bidi and invisible characters can't reorder or hide what the card shows", () => {
  const args = { path: "/tmp/\u202Etxt.etoN", note: "a\u200Bb\u2066c\u2028d" };
  const shown = review?.(args) ?? "";
  assert.doesNotMatch(shown, /[\u200B\u202E\u2066\u2028]/);
  assert.ok(shown.includes(`${BS}u202e`));
  assert.deepEqual(JSON.parse(shown), args);
  // A ui/open-link card: the real host must stay readable.
  const forReview = (mcpUi as Record<string, unknown>).forReview as
    | ((text: string) => string)
    | undefined;
  assert.equal(
    forReview?.("https://good.example\u202E@evil.example/"),
    `https://good.example${BS}u202e@evil.example/`,
  );
  assert.match(frame, /\{forReview\(asking\.name\)\}/);
});

test("every default-ignorable character is shown, astral ones included", () => {
  // U+034F, a variation selector, a Hangul filler and a LANGUAGE TAG all render as nothing.
  const args = { cmd: "rm\u034F -rf\uFE0F /\u3164x\u{E0001}y" };
  const shown = review?.(args) ?? "";
  assert.doesNotMatch(shown, /[\u034F\uFE0F\u3164\u{E0001}]/u);
  assert.ok(shown.includes(`${BS}u034f`));
  assert.ok(shown.includes(`${BS}udb40${BS}udc01`));
  assert.deepEqual(JSON.parse(shown), args);
});
