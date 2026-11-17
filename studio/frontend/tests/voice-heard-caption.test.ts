// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  lastUserUtterance,
  showsHeardCaption,
} from "../src/features/chat/voice/heard-caption.ts";

test("the caption is the last user turn's text", () => {
  const messages = [
    { role: "user", content: [{ type: "text", text: "First question" }] },
    { role: "assistant", content: [{ type: "text", text: "An answer." }] },
    {
      role: "user",
      content: [
        { type: "image" },
        { type: "text", text: "  What is\nthe time? " },
      ],
    },
    { role: "assistant", content: [] },
  ];
  assert.equal(lastUserUtterance(messages), "What is the time?");
  assert.equal(
    lastUserUtterance([
      { role: "assistant", content: [{ type: "text", text: "Hi" }] },
    ]),
    "",
  );
});

test("it shows while the turn is answered, not while the next one is heard", () => {
  for (const state of [
    "thinking",
    "generating",
    "synthesizing",
    "speaking",
    "paused",
  ] as const) {
    assert.ok(showsHeardCaption(state), state);
  }
  for (const state of [
    "listening",
    "hearing",
    "transcribing",
    "loading",
    "downloading",
    null,
  ] as const) {
    assert.ok(!showsHeardCaption(state), String(state));
  }
});

test("the orb gives the caption the chip's slot, and the decision chip keeps it when on", () => {
  const orb = readFileSync(
    new URL("../src/components/assistant-ui/voice-orb.tsx", import.meta.url),
    "utf8",
  );
  assert.match(
    orb,
    /chip: \(showOverlay && showCaption && hasDecision\) \|\| showHeard,/,
  );
  assert.match(
    orb,
    /const showHeard = showOverlay && !\(showCaption && hasDecision\) && showsHeardCaption\(orbState\)/,
  );
});
