// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  type AutoReadMessage,
  decideAutoRead,
} from "../src/features/chat/utils/auto-read-aloud.ts";

import { readSrc } from "./helpers/kit.ts";

function reply(
  status: string,
  content: AutoReadMessage["content"] = [
    { type: "text", text: "Hello there." },
  ],
): AutoReadMessage {
  return { id: "a1", role: "assistant", status: { type: status }, content };
}

const user: AutoReadMessage = {
  id: "u1",
  role: "user",
  content: [{ type: "text", text: "Hi" }],
};

test("a finished reply with text is spoken (#12303)", () => {
  assert.deepEqual(decideAutoRead([user, reply("complete")]), {
    kind: "speak",
    messageId: "a1",
  });
});

test("a reply still settling waits instead of being dropped", () => {
  assert.deepEqual(decideAutoRead([user, reply("running")]), { kind: "wait" });
});

test("stopped, failed and tool-approval replies stay silent", () => {
  for (const status of ["incomplete", "requires-action"]) {
    assert.deepEqual(decideAutoRead([user, reply(status)]), { kind: "skip" });
  }
});

test("a reply Studio marked cut off settles as complete but stays silent", () => {
  for (const reason of [
    "interrupted",
    "length",
    "cancelled",
    "context_window",
  ]) {
    const cut = {
      ...reply("complete"),
      metadata: { custom: { incomplete: { reason } } },
    };
    assert.deepEqual(decideAutoRead([user, cut]), { kind: "skip" }, reason);
  }
});

test("empty, reasoning-only and user-last threads stay silent", () => {
  assert.deepEqual(decideAutoRead([]), { kind: "skip" });
  assert.deepEqual(decideAutoRead([user]), { kind: "skip" });
  assert.deepEqual(
    decideAutoRead([
      user,
      reply("complete", [
        { type: "reasoning", text: "thinking" },
        { type: "text", text: "  " },
      ]),
    ]),
    { kind: "skip" },
  );
});

test("auto read-aloud is opt-in and only a saved true turns it on", () => {
  const store = readSrc("features/settings/stores/voice-settings-store.ts");
  assert.match(store, /autoReadAloud: false,/);
  assert.match(store, /autoReadAloud: saved\?\.autoReadAloud === true,/);
});

test("the thread mounts it only where a composer is shown, not in Compare panes", () => {
  const thread = readSrc("components/assistant-ui/thread.tsx");
  assert.match(thread, /\{!hideComposer && <AutoReadAloud \/>\}/);
});

test("a hidden chat stays silent and leaving a chat stops its own playback", () => {
  const src = readSrc("components/assistant-ui/auto-read-aloud.tsx");
  assert.match(src, /decision\.kind !== "speak" \|\| !active/);
  assert.match(src, /threads\.getById\(\s*threadId,?\s*\)/);
  assert.match(src, /speech\?\.messageId === spoken\)[\s\S]*stopSpeaking\(\)/);
});
