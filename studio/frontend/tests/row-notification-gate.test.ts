// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import type { AssistantClient } from "@assistant-ui/react";
import { createRowNotificationGate } from "../src/components/assistant-ui/row-notification-gate.ts";

type State = Record<string, unknown>;

// Stand-in client: enumerable scopes with `getState`, one channel, the composer inside thread state.
function fakeClient() {
  const messages: unknown[] = [
    { id: "m0" },
    { id: "m1" },
    { id: "m2" },
    { id: "m3" },
  ];
  let composer: State = {
    text: "",
    isEmpty: true,
    attachments: [],
    dictation: undefined,
    queue: [],
  };
  let thread: State = { isRunning: false, messages, composer };
  let threads: State = { mainThreadId: "t1", threadItems: {}, main: thread };
  let tools: State = { tools: {} };
  let repository: unknown[] = [];
  const listeners = new Set<() => void>();
  let parentSubscriptions = 0;
  const client = {
    subscribe(listener: () => void) {
      listeners.add(listener);
      parentSubscriptions += 1;
      return () => {
        listeners.delete(listener);
        parentSubscriptions -= 1;
      };
    },
    on() {
      return () => {};
    },
    threads: () => ({ getState: () => threads }),
    thread: () => ({
      getState: () => thread,
      __internal_getRuntime: () => ({
        getState: () => ({ messages: repository }),
      }),
    }),
    composer: () => ({ getState: () => composer }),
    tools: () => ({ getState: () => tools }),
  };
  const publish = (next: {
    composer?: State;
    thread?: State;
    tools?: State;
    repositoryWrite?: boolean;
  }) => {
    if (next.repositoryWrite) repository = [...repository];
    if (next.composer) composer = next.composer;
    thread = { ...thread, ...(next.thread ?? {}), composer };
    threads = { ...threads, main: thread };
    if (next.tools) tools = next.tools;
    for (const listener of listeners) listener();
  };
  return {
    client: client as unknown as AssistantClient,
    publish,
    composer: () => composer,
    thread: () => thread,
    parentSubscriptions: () => parentSubscriptions,
  };
}

function subscribeRows(fake: ReturnType<typeof fakeClient>, count: number) {
  const gate = createRowNotificationGate(fake.client);
  const heard = Array.from({ length: count }, () => 0);
  const release = heard.map((_, index) =>
    gate.row(index).subscribe(() => {
      heard[index] += 1;
    }),
  );
  return {
    gate,
    heard,
    release: () => {
      for (const r of release) r();
    },
  };
}

function withMessage(fake: ReturnType<typeof fakeClient>, index: number) {
  const messages = [...(fake.thread().messages as unknown[])];
  messages[index] = { ...(messages[index] as object), touched: true };
  return messages;
}

test("a keystroke reaches no row, any other composer or scope change reaches every row", () => {
  const fake = fakeClient();
  const rows = subscribeRows(fake, 4);

  fake.publish({
    composer: { ...fake.composer(), text: "h", isEmpty: false, queue: [] },
  });
  fake.publish({ composer: { ...fake.composer(), text: "hi", queue: [] } });
  assert.deepEqual(rows.heard, [0, 0, 0, 0]);

  fake.publish({
    composer: { ...fake.composer(), attachments: [{ id: "a" }] },
  });
  assert.deepEqual(
    rows.heard,
    [1, 1, 1, 1],
    "an attachment change is not a keystroke",
  );

  fake.publish({
    composer: { ...fake.composer(), dictation: { inputDisabled: true } },
  });
  assert.deepEqual(
    rows.heard,
    [2, 2, 2, 2],
    "dictation state is not a keystroke",
  );

  fake.publish({ thread: { isRunning: true } });
  assert.deepEqual(
    rows.heard,
    [3, 3, 3, 3],
    "a run starting reaches every row",
  );

  fake.publish({ tools: { tools: { search: {} } } });
  assert.deepEqual(
    rows.heard,
    [4, 4, 4, 4],
    "a scope the gate knows nothing about is compared by identity",
  );

  fake.publish({ composer: { ...fake.composer(), text: "hi there" } });
  assert.deepEqual(rows.heard, [4, 4, 4, 4]);
  rows.release();
});

test("a streamed token reaches the last two rows only", () => {
  const fake = fakeClient();
  const rows = subscribeRows(fake, 4);
  fake.publish({ thread: { messages: withMessage(fake, 3) } });
  assert.deepEqual(rows.heard, [0, 0, 1, 1]);
  rows.release();
});

test("a rebuilt messages array holding the same messages reaches every row only after a repository write", () => {
  const fake = fakeClient();
  const rows = subscribeRows(fake, 4);
  const rebuilt = () => [...(fake.thread().messages as unknown[])];
  fake.publish({ thread: { messages: rebuilt() } });
  assert.deepEqual(rows.heard, [0, 0, 0, 0], "a store-only rebuild");
  // A write on a hidden branch rebuilds the array without touching a visible message.
  fake.publish({ thread: { messages: rebuilt() }, repositoryWrite: true });
  assert.deepEqual(rows.heard, [1, 1, 1, 1]);
  rows.release();
});

test("a rebuilt messages array reaches every row when the repository cannot be read", () => {
  const fake = fakeClient();
  const client = {
    ...(fake.client as unknown as Record<string, unknown>),
    thread: () => ({ getState: () => fake.thread() }),
  } as unknown as AssistantClient;
  const gate = createRowNotificationGate(client);
  const heard = [0, 0];
  const release = heard.map((_, index) =>
    gate.row(index).subscribe(() => {
      heard[index] += 1;
    }),
  );
  fake.publish({
    thread: { messages: [...(fake.thread().messages as unknown[])] },
  });
  assert.deepEqual(heard, [1, 1]);
  for (const r of release) r();
});

test("a non-empty queue change reaches every row", () => {
  const fake = fakeClient();
  const rows = subscribeRows(fake, 2);
  fake.publish({ composer: { ...fake.composer(), queue: [{ id: "q" }] } });
  assert.deepEqual(rows.heard, [1, 1]);
  rows.release();
});

test("a changed message reaches its own row, the one before it and every later row", () => {
  const fake = fakeClient();
  const rows = subscribeRows(fake, 4);
  fake.publish({ thread: { messages: withMessage(fake, 2) } });
  assert.deepEqual(rows.heard, [0, 1, 1, 1]);
  fake.publish({ thread: { messages: withMessage(fake, 0) } });
  assert.deepEqual(rows.heard, [1, 2, 2, 2]);
  rows.release();
});

test("a message added or removed reaches every row", () => {
  const fake = fakeClient();
  const rows = subscribeRows(fake, 4);
  fake.publish({
    thread: {
      messages: [...(fake.thread().messages as unknown[]), { id: "m4" }],
    },
  });
  assert.deepEqual(rows.heard, [1, 1, 1, 1]);
  fake.publish({
    thread: { messages: (fake.thread().messages as unknown[]).slice(1) },
  });
  assert.deepEqual(rows.heard, [2, 2, 2, 2]);
  rows.release();
});

test("a fingerprint that throws reaches every row", () => {
  const fake = fakeClient();
  let broken = false;
  const client = new Proxy(fake.client as object, {
    ownKeys(target) {
      if (broken) throw new Error("boom");
      return Reflect.ownKeys(target);
    },
  }) as AssistantClient;
  const gate = createRowNotificationGate(client);
  const heard = [0, 0];
  const release = heard.map((_, index) =>
    gate.row(index).subscribe(() => {
      heard[index] += 1;
    }),
  );
  const original = console.error;
  console.error = () => {};
  try {
    broken = true;
    fake.publish({ composer: { ...fake.composer(), text: "x" } });
  } finally {
    console.error = original;
  }
  assert.deepEqual(heard, [1, 1]);
  for (const r of release) r();
});

test("the gate holds the parent subscription only while a row is subscribed", () => {
  const fake = fakeClient();
  const gate = createRowNotificationGate(fake.client);
  assert.equal(fake.parentSubscriptions(), 0);
  const a = gate.row(0).subscribe(() => {});
  const b = gate.row(1).subscribe(() => {});
  assert.equal(fake.parentSubscriptions(), 1);
  a();
  assert.equal(fake.parentSubscriptions(), 1);
  b();
  assert.equal(fake.parentSubscriptions(), 0);
});

test("a row client is stable per index and inherits every scope from its parent", () => {
  const fake = fakeClient();
  const gate = createRowNotificationGate(fake.client);
  assert.equal(gate.row(2), gate.row(2));
  const row = gate.row(2) as unknown as Record<
    string,
    () => { getState: () => State }
  >;
  assert.equal(row.thread().getState(), fake.thread());
  assert.notEqual(
    (row as unknown as AssistantClient).subscribe,
    fake.client.subscribe,
  );
});
