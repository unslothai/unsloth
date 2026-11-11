// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { byTestId, doc, flushMicrotasks, node } from "./helpers/flows.ts";
import { fakeReact } from "./helpers/fake-react.ts";
import { registerBundlerResolver } from "./helpers/kit.ts";
import { elementsOf, loadWithStubs, stubEveryImport, stubJsxRuntime } from "./helpers/module-stubs.ts";

registerBundlerResolver();

const C = await import("../src/features/flows/player/chat/chat-state.ts");
const { rerollTargets } = await import("../src/features/flows/player/player-api.ts");
/** `n` alternating user/assistant turns. */
const turnsOf = (n: number, content: (i: number) => string) =>
  Array.from({ length: n }, (_, i) => ({
    role: (i % 2 ? "assistant" : "user") as "user" | "assistant",
    content: content(i),
  }));

test("a flow plays as a chat when it has an enabled Chat Output", () => {
  const d = doc([
    node("in", "chat.input", [0, 0], { welcome: "Ask away" }),
    node("out", "chat.output"),
  ]);
  assert.deepEqual(C.chatEntry(d), { output: "out", welcome: "Ask away" });
  assert.equal(C.chatEntry(doc([node("t", "text.template")])), null);
  assert.equal(C.chatEntry(doc([{ ...node("out", "chat.output"), disabled: true }])), null);
  assert.equal(C.chatEntry(doc([node("out", "chat.output")]))?.welcome, "Hi! How can I help?");
});

test("each message carries the earlier turns, capped", () => {
  const p = C.turnParams(turnsOf(60, String), "next");
  assert.equal(p.message, "next");
  assert.equal(p.history.length, C.HISTORY_SENT);
  assert.equal(p.history.at(-1)?.content, "59");
});

test("a saved chat round-trips and a foreign save starts fresh", () => {
  const turns = turnsOf(2, (i) => ["hi", "hello"][i]);
  assert.deepEqual(C.restoreChat(C.serializeChat(turns)), turns);
  assert.deepEqual(C.restoreChat({ v: 99, turns }), []);
  assert.deepEqual(C.restoreChat({ v: 1, turns: [{ role: "system", content: "x" }, 3] }), []);
  assert.deepEqual(C.restoreChat(null), []);
});

test("a re-roll never runs the chat entry point", () => {
  const d = doc([node("out", "chat.output"), node("w", "story.generator")]);
  assert.deepEqual(rerollTargets(d), ["w"]);
});

test("a saved chat keeps a bounded tail of the conversation", () => {
  const turns = turnsOf(C.SAVED_TURNS + 50, (i) => (i === C.SAVED_TURNS + 49 ? "x".repeat(50000) : `t${i}`));
  const saved = C.serializeChat(turns) as { turns: { content: string }[] };
  assert.equal(saved.turns.length, C.SAVED_TURNS);
  assert.equal(saved.turns[0].content, "t50");
  assert.ok(saved.turns[saved.turns.length - 1].content.length <= 20000);
  assert.equal(C.restoreChat(saved).length, C.SAVED_TURNS);
});

type Toast = { message: string; opts?: { action?: { label: string; onClick: () => void } } };

/** ChatPlayer rendered with deterministic hooks; `loadSession` answers the saved-chat load. */
function chatHarness(loadSession: () => Promise<unknown>) {
  const hooks = fakeReact();
  const toasts: Toast[] = [];
  const { ChatPlayer } = loadWithStubs<{ ChatPlayer: (p: unknown) => unknown }>(
    new URL("../src/features/flows/player/chat/chat-player.tsx", import.meta.url),
    stubEveryImport({
      react: hooks.react,
      "react/jsx-runtime": stubJsxRuntime(),
      sonner: { toast: (message: string, opts?: never) => toasts.push({ message, opts }) },
      "./chat-state": C,
    }),
  );
  let loads = 0;
  const api = {
    loadSession: () => {
      loads++;
      return loadSession();
    },
    saveSession: () => Promise.resolve(),
  };
  const props = { flowId: "f", api, entry: { output: "out", welcome: "Hi" }, title: "T", onExit: () => {} };
  const render = () => elementsOf(hooks.render(ChatPlayer, props));
  const resetDisabled = () => byTestId(render(), "chat-reset")?.props.disabled;
  return { render, resetDisabled, toasts, loads: () => loads };
}

test("a failed load keeps the chat blocked and offers Retry instead of opening an empty history", async () => {
  let fail = true;
  const h = chatHarness(() =>
    fail ? Promise.reject(new Error("offline")) : Promise.resolve({ state: C.serializeChat([{ role: "user", content: "hi" }]) }),
  );
  h.render();
  await flushMicrotasks();
  assert.equal(h.resetDisabled(), true, "still blocked after the failure");
  const retry = h.toasts.at(-1)?.opts?.action;
  assert.equal(retry?.label, "Retry");
  fail = false;
  retry?.onClick();
  h.render();
  assert.equal(h.loads(), 2, "Retry loads again");
  await flushMicrotasks();
  assert.equal(h.resetDisabled(), false, "opens once the load succeeds");
});

test("New chat waits for the saved conversation so a late load cannot restore it", async () => {
  let finish: (v: unknown) => void = () => {};
  const h = chatHarness(() => new Promise((r) => (finish = r)));
  assert.equal(h.resetDisabled(), true);
  finish(null);
  await flushMicrotasks();
  assert.equal(h.resetDisabled(), false);
});
