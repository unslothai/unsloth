// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import { test } from "node:test";

Object.assign(globalThis, { window: globalThis });

type Answer = { installed: boolean; events: unknown[] };
type Call = { command: Record<string, unknown>; answer: (value: Answer) => void; fail: () => void };
const calls: Call[] = [];
(globalThis as { nativeViewCall?: unknown }).nativeViewCall = (
  _name: string,
  args: { command: Record<string, unknown> },
) =>
  new Promise<Answer>((answer, reject) =>
    calls.push({ command: args.command, answer, fail: () => reject(new Error("no such tab")) }),
  );

register("./helpers/browser-store-resolver.mjs", import.meta.url);
register("./helpers/native-view-resolver.mjs", import.meta.url);
const { startNativeAnnotate } = await import("../src/features/browser/native-annotate.ts");

const settle = () => new Promise((resolve) => setTimeout(resolve, 5));
const sent = () => calls.map(({ command }) => command.command);

test("commands reach the page one at a time, in order, and only checked reports come back", async () => {
  calls.length = 0;
  const events: unknown[] = [];
  const channel = startNativeAnnotate("t1", (event) => events.push(event));
  try {
    channel.send({ command: "annotateInstall", code: "return {}" });
    channel.send({ command: "annotate", on: true, color: "#fff" });
    await settle();
    // Start waits for install.
    assert.deepEqual(sent(), ["install"]);
    calls[0]?.answer({ installed: true, events: [] });
    await settle();
    assert.deepEqual(sent(), ["install", "start"]);
    assert.deepEqual(calls[1]?.command, { command: "start", color: "#fff" });
    calls[1]?.answer({
      installed: true,
      events: [
        { type: "annotate", event: "up" },
        { type: "annotate", event: "mark", id: 1, rect: { left: 1, top: 2, width: 3, height: 4 }, quote: "Hi" },
        // Forged by the page: dropped.
        { type: "annotate", event: "mark", id: "x" },
        { type: "navigate", url: "https://example.com" },
        "junk",
      ],
    });
    await settle();
    assert.deepEqual(events, [
      { kind: "up" },
      { kind: "mark", id: 1, rect: { left: 1, top: 2, width: 3, height: 4 }, quote: "Hi", image: false, alt: "", area: false },
    ]);
  } finally {
    channel.stop();
  }
});

test("a page that loses the code (it navigated) asks for it again, once", async () => {
  calls.length = 0;
  const events: unknown[] = [];
  const channel = startNativeAnnotate("t4", (event) => events.push(event));
  const answerNext = async (installed: boolean) => {
    for (let wait = 0; wait < 100 && calls.length === 0; wait++) await settle();
    const call = calls.shift();
    call?.answer({ installed, events: [] });
    await settle();
  };
  try {
    // Never installed yet: no ready.
    await answerNext(false);
    assert.deepEqual(events, []);
    await answerNext(true);
    await answerNext(false);
    await answerNext(false);
    assert.deepEqual(events, [{ kind: "ready" }]);
  } finally {
    channel.stop();
    await settle();
  }
  assert.ok(calls.some(({ command }) => command.command === "stop"), "stopping turns the page's annotate off");
});

test("a closing channel's stop runs before a new channel's commands for the tab", async () => {
  calls.length = 0;
  const old = startNativeAnnotate("t2", () => undefined);
  await new Promise((resolve) => setTimeout(resolve, 100));
  // A poll hangs (the page is navigating) while the layer remounts.
  assert.deepEqual(sent(), ["poll"]);
  old.stop();
  const fresh = startNativeAnnotate("t2", () => undefined);
  try {
    fresh.send({ command: "annotateInstall", code: "return {}" });
    fresh.send({ command: "annotate", on: true, color: "#fff" });
    for (let at = 0; at < 3; at++) {
      calls[at]?.answer({ installed: at > 0, events: [] });
      await settle();
    }
    assert.deepEqual(sent().slice(0, 4), ["poll", "stop", "install", "start"]);
  } finally {
    fresh.stop();
    await settle();
  }
});

test("an install or start that failed is sent again after the next answer", async () => {
  calls.length = 0;
  const channel = startNativeAnnotate("t3", () => undefined);
  try {
    channel.send({ command: "annotateInstall", code: "return {}" });
    channel.send({ command: "annotate", on: true, color: "#fff" });
    await settle();
    calls[0]?.fail();
    await settle();
    calls[1]?.fail();
    // The next poll answers: both go again, in order.
    for (let wait = 0; wait < 100 && calls.length < 3; wait++) await settle();
    calls[2]?.answer({ installed: false, events: [] });
    await settle();
    calls[3]?.answer({ installed: true, events: [] });
    await settle();
    assert.deepEqual(sent().slice(0, 5), ["install", "start", "poll", "install", "start"]);
  } finally {
    channel.stop();
    await settle();
  }
});
