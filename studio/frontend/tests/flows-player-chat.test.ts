// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { doc, flushMicrotasks, node } from "./helpers/flows.ts";
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
  const d = doc([node("out", "chat.output"), node("w", "story.generator")]);
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

// ChatPlayer rendered with stubbed React hooks: state persists across renders, effects run when
// their deps change, and the returned element tree is inspected directly.
function chatHarness(loadSession: () => Promise<unknown>) {
  const { elementsOf, loadWithStubs, stubEveryImport, stubJsxRuntime } =
    harnessDeps;
  const slots: unknown[] = [];
  const effectDeps: (unknown[] | undefined)[] = [];
  const cleanups: ((() => void) | undefined)[] = [];
  const toasts: { message: string; opts?: { action?: { label: string; onClick: () => void } } }[] = [];
  let slot = 0;
  let effect = 0;
  let pending: (() => void)[] = [];
  const react = {
    useState(init: unknown) {
      const i = slot++;
      if (!(i in slots)) slots[i] = init;
      return [slots[i], (v: unknown) => {
        slots[i] = typeof v === "function" ? (v as (x: unknown) => unknown)(slots[i]) : v;
      }];
    },
    useRef(init: unknown) {
      const i = slot++;
      if (!(i in slots)) slots[i] = { current: init };
      return slots[i];
    },
    useEffect(fn: () => (() => void) | undefined, deps?: unknown[]) {
      const i = effect++;
      const prev = effectDeps[i];
      if (prev && deps && prev.length === deps.length && prev.every((d, k) => Object.is(d, deps[k]))) return;
      effectDeps[i] = deps;
      pending.push(() => {
        cleanups[i]?.();
        cleanups[i] = fn() ?? undefined;
      });
    },
  };
  const { ChatPlayer } = loadWithStubs<{ ChatPlayer: (p: unknown) => unknown }>(
    new URL("../src/features/flows/player/chat/chat-player.tsx", import.meta.url),
    stubEveryImport({
      react,
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
  const render = () => {
    slot = 0;
    effect = 0;
    pending = [];
    const tree = ChatPlayer(props);
    for (const run of pending) run();
    return elementsOf(tree);
  };
  const resetButton = (els: ReturnType<typeof render>) =>
    els.find((e) => e.props["data-testid"] === "chat-reset");
  return { render, resetButton, toasts, loads: () => loads };
}

const harnessDeps = await import("./helpers/module-stubs.ts");

test("a failed load keeps the chat blocked and offers Retry instead of opening an empty history", async () => {
  let fail = true;
  const h = chatHarness(() =>
    fail ? Promise.reject(new Error("offline")) : Promise.resolve({ state: C.serializeChat([{ role: "user", content: "hi" }]) }),
  );
  h.render();
  await flushMicrotasks();
  let els = h.render();
  assert.equal(h.resetButton(els)?.props.disabled, true, "still blocked after the failure");
  const retry = h.toasts.at(-1)?.opts?.action;
  assert.equal(retry?.label, "Retry");
  fail = false;
  retry?.onClick();
  h.render();
  assert.equal(h.loads(), 2, "Retry loads again");
  await flushMicrotasks();
  els = h.render();
  assert.equal(h.resetButton(els)?.props.disabled, false, "opens once the load succeeds");
});

test("New chat waits for the saved conversation so a late load cannot restore it", async () => {
  let finish: (v: unknown) => void = () => {};
  const h = chatHarness(() => new Promise((r) => (finish = r)));
  const els = h.render();
  assert.equal(h.resetButton(els)?.props.disabled, true);
  finish(null);
  await flushMicrotasks();
  assert.equal(h.resetButton(h.render())?.props.disabled, false);
});
