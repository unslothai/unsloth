// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { doc, node } from "./flows-editor-fixtures.ts";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const C = await import("../src/features/flows/player/chat/chat-state.ts");
const { rerollTargets } = await import(
  "../src/features/flows/player/player-api.ts"
);

test("a flow plays as a chat when it has an enabled Chat Output", () => {
  const d = doc([
    { ...node("in", "chat.input"), params: { welcome: "Ask away" } },
    node("out", "chat.output"),
  ]);
  assert.deepEqual(C.chatEntry(d), { output: "out", welcome: "Ask away" });
  assert.equal(C.chatEntry(doc([node("t", "text.template")])), null);
  assert.equal(
    C.chatEntry(doc([{ ...node("out", "chat.output"), disabled: true }])),
    null,
  );
  assert.equal(
    C.chatEntry(doc([node("out", "chat.output")]))?.welcome,
    "Hi! How can I help?",
  );
});

test("each message carries the earlier turns, capped", () => {
  const turns = Array.from({ length: 60 }, (_, i) => ({
    role: (i % 2 ? "assistant" : "user") as "user" | "assistant",
    content: String(i),
  }));
  const p = C.turnParams(turns, "next");
  assert.equal(p.message, "next");
  assert.equal(p.history.length, C.HISTORY_SENT);
  assert.equal(p.history.at(-1)?.content, "59");
});

test("a saved chat round-trips and a foreign save starts fresh", () => {
  const turns = [
    { role: "user" as const, content: "hi" },
    { role: "assistant" as const, content: "hello" },
  ];
  assert.deepEqual(C.restoreChat(C.serializeChat(turns)), turns);
  assert.deepEqual(C.restoreChat({ v: 99, turns }), []);
  assert.deepEqual(
    C.restoreChat({ v: 1, turns: [{ role: "system", content: "x" }, 3] }),
    [],
  );
  assert.deepEqual(C.restoreChat(null), []);
});

test("a re-roll never runs the chat entry point", () => {
  const d = doc([node("out", "chat.output"), node("w", "world.generator")]);
  assert.deepEqual(rerollTargets(d), ["w"]);
});

test("a saved chat keeps a bounded tail of the conversation", () => {
  const turns = Array.from({ length: C.SAVED_TURNS + 50 }, (_, i) => ({
    role: (i % 2 ? "assistant" : "user") as "user" | "assistant",
    content: i === C.SAVED_TURNS + 49 ? "x".repeat(50000) : `t${i}`,
  }));
  const saved = C.serializeChat(turns) as { turns: { content: string }[] };
  assert.equal(saved.turns.length, C.SAVED_TURNS);
  assert.equal(saved.turns[0].content, "t50");
  assert.ok(saved.turns[saved.turns.length - 1].content.length <= 20000);
  assert.equal(C.restoreChat(saved).length, C.SAVED_TURNS);
});

test("a failed load keeps the chat blocked and offers Retry instead of opening an empty history", async () => {
  const { readSrc } = await import("./helpers/kit.ts");
  const src = readSrc("features/flows/player/chat/chat-player.tsx");
  assert.ok(
    !src.includes(
      ".catch(() => undefined)\n      .finally(() => {\n        if (live) setLoaded(true);",
    ),
  );
  assert.match(src, /label: "Retry", onClick: \(\) => setAttempt/);
});
