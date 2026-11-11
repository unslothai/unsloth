// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { stripTypeScriptTypes } from "node:module";
import test from "node:test";
import { readSrc } from "./helpers/kit.ts";

const composer = readSrc("features/chat/shared-composer.tsx");

function slice(from: string, to: string): string {
  const start = composer.indexOf(from);
  const end = composer.indexOf(to, start);
  assert.ok(start >= 0 && end > start, `${from} .. ${to}`);
  return composer.slice(start, end);
}

function asyncBody(source: string): string {
  const wrapped = stripTypeScriptTypes(`async function f() {\n${source}\n}`);
  return wrapped.slice(wrapped.indexOf("{") + 1, wrapped.lastIndexOf("}"));
}

const registerHandle = new Function(
  "handlesRef",
  "name",
  "aui",
  asyncBody(
    slice(
      "const currentHandles = handlesRef.current;",
      "return () => {\n      delete currentHandles[name];",
    ),
  ),
);

const AsyncFunction = Object.getPrototypeOf(async () => {}).constructor;
const appendBoth = new AsyncFunction(
  "handlesRef",
  "content",
  `${asyncBody(
    slice('const handle1 = handlesRef.current["model1"];', "const name1 ="),
  )}\nreturn { handle1, handle2 };`,
);

function newCompareThread(paneId: string) {
  const listeners = new Set<() => void>();
  const state = { messages: [] as { id: string; role: string }[], isRunning: false };
  const runs: (string | null)[] = [];
  const aui = {
    subscribe(listener: () => void) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    threadListItem: () => ({ getState: () => ({ id: paneId, remoteId: null }) }),
    thread: () => ({
      getState: () => state,
      append() {
        setImmediate(() => {
          state.messages = [...state.messages, { id: `${paneId}-user`, role: "user" }];
          for (const listener of listeners) listener();
        });
      },
      startRun({ parentId }: { parentId: string | null }) {
        runs.push(parentId);
      },
    }),
  };
  return { aui, runs };
}

test("the first message of a new compare chat starts pane 1 on the user message", async () => {
  const handlesRef = { current: {} as Record<string, unknown> };
  const pane1 = newCompareThread("pane1");
  const pane2 = newCompareThread("pane2");
  registerHandle(handlesRef, "model1", pane1.aui);
  registerHandle(handlesRef, "model2", pane2.aui);

  const { handle1, handle2 } = await appendBoth(handlesRef, [
    { type: "text", text: "What is 2+2?" },
  ]);
  await "ready";
  handle1.startRun();
  handle2.startRun();

  assert.deepEqual(pane1.runs, ["pane1-user"]);
  assert.deepEqual(pane2.runs, ["pane2-user"]);
});
