// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import type { AssistantClient } from "@assistant-ui/react";
import { createComposerTextGatedClient } from "../src/components/assistant-ui/composer-text-gate.ts";

type State = Record<string, unknown>;

// A stand-in for the assistant-ui client: enumerable scope accessors returning `{ getState }`,
// one notification channel, and the thread state embedding its composer as the real one does.
function fakeClient() {
  const messages: unknown[] = [{ id: "m1" }];
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
    thread: () => ({ getState: () => thread }),
    composer: () => ({ getState: () => composer }),
    tools: () => ({ getState: () => tools }),
  };
  const publish = (next: {
    composer?: State;
    thread?: State;
    tools?: State;
  }) => {
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

test("a keystroke reaches no row, any other change does", () => {
  const fake = fakeClient();
  const gated = createComposerTextGatedClient(fake.client);
  let calls = 0;
  const unsubscribe = gated.subscribe(() => {
    calls += 1;
  });

  // Typing rebuilds composer, thread and thread-list states, with a fresh `queue: []`.
  fake.publish({
    composer: { ...fake.composer(), text: "h", isEmpty: false, queue: [] },
  });
  fake.publish({ composer: { ...fake.composer(), text: "hi", queue: [] } });
  assert.equal(calls, 0);

  fake.publish({
    composer: { ...fake.composer(), attachments: [{ id: "a" }] },
  });
  assert.equal(calls, 1, "an attachment change is not a keystroke");

  fake.publish({
    composer: { ...fake.composer(), dictation: { inputDisabled: true } },
  });
  assert.equal(calls, 2, "dictation state is not a keystroke");

  fake.publish({
    thread: {
      messages: [...(fake.thread().messages as unknown[]), { id: "m2" }],
    },
  });
  assert.equal(calls, 3, "a new message reaches the rows");

  fake.publish({ thread: { isRunning: true } });
  assert.equal(calls, 4, "a run starting reaches the rows");

  fake.publish({ tools: { tools: { search: {} } } });
  assert.equal(
    calls,
    5,
    "a scope the gate knows nothing about is compared by identity",
  );

  fake.publish({ composer: { ...fake.composer(), text: "hi there" } });
  assert.equal(calls, 5);

  unsubscribe();
});

test("the gate holds the parent subscription only while a row is subscribed", () => {
  const fake = fakeClient();
  const gated = createComposerTextGatedClient(fake.client);
  assert.equal(fake.parentSubscriptions(), 0);
  const a = gated.subscribe(() => {});
  const b = gated.subscribe(() => {});
  assert.equal(fake.parentSubscriptions(), 1);
  a();
  assert.equal(fake.parentSubscriptions(), 1);
  b();
  assert.equal(fake.parentSubscriptions(), 0);
});

test("the gated client inherits every scope and event from its parent", () => {
  const fake = fakeClient();
  const gated = createComposerTextGatedClient(fake.client) as unknown as Record<
    string,
    () => { getState: () => State }
  >;
  assert.equal(gated.thread().getState(), fake.thread());
  assert.notEqual(
    (gated as unknown as AssistantClient).subscribe,
    fake.client.subscribe,
  );
});
