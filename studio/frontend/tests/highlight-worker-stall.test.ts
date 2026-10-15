// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test, { type TestContext } from "node:test";

import { loadWithStubs } from "./helpers/module-stubs.ts";

type Client = {
  requestFullHighlight: (
    source: string,
    language: string | null,
    onResult: (result: unknown) => void,
    speculative?: boolean,
    patient?: boolean,
  ) => (() => void) | null;
  highlightWorkerState: () => string;
};

class FakeWorker {
  static made: FakeWorker[] = [];
  posted: Record<string, unknown>[] = [];
  terminated = false;
  onmessage: ((event: { data: unknown }) => void) | null = null;
  onerror: ((event: { message: string; type: string }) => void) | null = null;

  constructor() {
    FakeWorker.made.push(this);
  }

  postMessage(data: Record<string, unknown>): void {
    this.posted.push(data);
  }

  terminate(): void {
    this.terminated = true;
  }

  send(data: unknown): void {
    this.onmessage?.({ data });
  }

  requests(): number[] {
    return this.posted.flatMap((message) =>
      typeof message.client === "number" ? [message.client] : [],
    );
  }
}

const RESULT = { tokens: [[]], fg: "#000", bg: "#fff" };

const MESSAGES = loadWithStubs<Record<string, unknown>>(
  new URL("../src/components/assistant-ui/reasoning-highlight.ts", import.meta.url),
  {},
);

function setup(t: TestContext): { client: Client; drain: () => void } {
  FakeWorker.made = [];
  const scope = globalThis as { Worker?: unknown };
  scope.Worker = FakeWorker;
  t.after(() => {
    delete scope.Worker;
  });
  t.mock.method(console, "warn", () => {});
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const client = loadWithStubs<Client>(
    new URL("../src/components/assistant-ui/use-reasoning-highlight.ts", import.meta.url),
    {
      "@/features/chat": { useChatRuntimeStore: {} },
      "./reasoning-highlight": MESSAGES,
      "./stream-activity": { createStreamActivity: () => ({ active: () => false }) },
    },
  );
  const drain = (): void => {
    for (let round = 0; round < 10; round += 1) t.mock.timers.tick(1);
  };
  return { client, drain };
}

async function settleChannels(client: Client): Promise<void> {
  for (let turn = 0; turn < 50 && client.highlightWorkerState() !== "stalled"; turn += 1) {
    await new Promise<void>((resolve) => setImmediate(resolve));
  }
}

async function stallBoot(t: TestContext, client: Client): Promise<void> {
  t.mock.timers.tick(5_000);
  await settleChannels(client);
  assert.equal(client.highlightWorkerState(), "stalled");
}

test("a whole-fence request waits out a slow boot and the worker that was late answers it", async (t) => {
  const { client, drain } = setup(t);
  const fence: unknown[] = [];
  const speculation: unknown[] = [];
  assert.ok(client.requestFullHighlight("<html></html>", "html", (result) => fence.push(result)));
  assert.ok(client.requestFullHighlight("<p></p>", "html", (result) => speculation.push(result), true));
  const [booting] = FakeWorker.made;
  await stallBoot(t, client);
  drain();
  assert.deepEqual(speculation, [null], "the speculator hears of the stall, so it can pause");
  assert.deepEqual(fence, [], "the fence is not failed over to the main thread");
  const mounted: unknown[] = [];
  assert.ok(
    client.requestFullHighlight("<div></div>", "html", (result) => mounted.push(result)),
    "a fence that mounts during the stall queues on the same worker",
  );
  assert.equal(client.requestFullHighlight("<b></b>", "html", () => {}, true), null, "speculation is not sent to a late worker");
  assert.equal(FakeWorker.made.length, 1);
  booting.send({ ready: true });
  assert.equal(client.highlightWorkerState(), "ready");
  const [waited, , queued] = booting.requests();
  booting.send({ client: waited, revision: 1, lines: [], result: RESULT });
  booting.send({ client: queued, revision: 1, lines: [], result: RESULT });
  assert.deepEqual(fence, [RESULT]);
  assert.deepEqual(mounted, [RESULT]);
  t.mock.timers.tick(10_000);
  assert.equal(booting.terminated, true, "with nothing left waiting, the recovered worker idles out");
});

test("a whole-fence request cancelled during the stall is never answered and lets the worker idle", async (t) => {
  const { client, drain } = setup(t);
  const fence: unknown[] = [];
  const cancel = client.requestFullHighlight("<html></html>", "html", (result) => fence.push(result));
  assert.ok(cancel);
  const [booting] = FakeWorker.made;
  await stallBoot(t, client);
  drain();
  cancel();
  assert.deepEqual(booting.posted.at(-1), { cancel: booting.requests()[0] });
  booting.send({ ready: true });
  booting.send({ client: booting.requests()[0], revision: 1, lines: [], result: RESULT });
  assert.deepEqual(fence, []);
  t.mock.timers.tick(10_000);
  assert.equal(booting.terminated, true);
});

test("a crashed worker fails every waiting request, whole-fence ones included, and is never rebuilt", (t) => {
  const { client, drain } = setup(t);
  const fence: unknown[] = [];
  client.requestFullHighlight("<html></html>", "html", (result) => fence.push(result));
  const [crashed] = FakeWorker.made;
  crashed.onerror?.({ message: "boom", type: "error" });
  assert.equal(client.highlightWorkerState(), "unavailable");
  assert.equal(crashed.terminated, true);
  drain();
  assert.deepEqual(fence, [null]);
  assert.equal(client.requestFullHighlight("<html></html>", "html", () => {}), null);
  assert.equal(FakeWorker.made.length, 1);
});

test("a worker that fails a whole-fence request answers it once with a failure", (t) => {
  const { client, drain } = setup(t);
  const fence: unknown[] = [];
  client.requestFullHighlight("<html></html>", "html", (result) => fence.push(result));
  const [ready] = FakeWorker.made;
  ready.send({ ready: true });
  const [id] = ready.requests();
  ready.send({ client: id, revision: 1, lines: [], failed: true });
  ready.send({ client: id, revision: 1, lines: [], result: RESULT });
  drain();
  assert.deepEqual(fence, [null]);
  assert.equal(client.highlightWorkerState(), "ready");
});

test("a fence the main thread could tokenize does not wait out a late boot", async (t) => {
  const { client, drain } = setup(t);
  const short: unknown[] = [];
  const long: unknown[] = [];
  assert.ok(client.requestFullHighlight("<p></p>", "html", (result) => short.push(result), false, false));
  assert.ok(client.requestFullHighlight("<html></html>", "html", (result) => long.push(result)));
  await stallBoot(t, client);
  drain();
  assert.deepEqual(short, [null], "handed back, so it is tokenized on the main thread as it would be without a worker");
  assert.deepEqual(long, [], "a fence too long for the main thread keeps waiting");
  assert.equal(client.requestFullHighlight("<b></b>", "html", () => {}, false, false), null, "while the worker is late, short fences never queue on it");
});

test("a ready worker that sends nothing while a fence waits is treated as late until it answers", async (t) => {
  const { client, drain } = setup(t);
  const short: unknown[] = [];
  const long: unknown[] = [];
  client.requestFullHighlight("<p></p>", "html", (result) => short.push(result), false, false);
  client.requestFullHighlight("<html></html>", "html", (result) => long.push(result));
  const [silent] = FakeWorker.made;
  silent.send({ ready: true });
  t.mock.timers.tick(14_999);
  await settleChannels(client);
  assert.equal(client.highlightWorkerState(), "ready", "fifteen seconds of silence, not less");
  t.mock.timers.tick(1);
  await settleChannels(client);
  assert.equal(client.highlightWorkerState(), "stalled");
  drain();
  assert.deepEqual(short, [null]);
  assert.deepEqual(long, []);
  const [, waiting] = silent.requests();
  silent.send({ client: waiting, revision: 1, lines: [], result: RESULT });
  assert.equal(client.highlightWorkerState(), "ready", "the first word from the worker ends the stall");
  assert.deepEqual(long, [RESULT]);
  assert.equal(silent.terminated, false);
});

test("a busy worker that keeps answering is never declared silent", async (t) => {
  const { client } = setup(t);
  const results: unknown[] = [];
  for (const source of ["<a></a>", "<b></b>", "<i></i>"]) {
    client.requestFullHighlight(source, "html", (result) => results.push(result), false, false);
  }
  const [busy] = FakeWorker.made;
  busy.send({ ready: true });
  for (const id of busy.requests()) {
    t.mock.timers.tick(10_000);
    await settleChannels(client);
    assert.equal(client.highlightWorkerState(), "ready");
    busy.send({ client: id, revision: 1, lines: [], result: RESULT });
  }
  assert.deepEqual(results, [RESULT, RESULT, RESULT]);
  t.mock.timers.tick(30_000);
  await settleChannels(client);
  assert.equal(client.highlightWorkerState(), "ready", "with nothing waiting, silence means nothing");
});

test("a late worker that idles out is replaced, so the next long fence is coloured again", async (t) => {
  const { client } = setup(t);
  const cancel = client.requestFullHighlight("<html></html>", "html", () => {});
  assert.ok(cancel);
  const [late] = FakeWorker.made;
  late.send({ ready: true });
  t.mock.timers.tick(15_000);
  await settleChannels(client);
  assert.equal(client.highlightWorkerState(), "stalled");
  cancel();
  t.mock.timers.tick(5_000);
  assert.equal(client.requestFullHighlight("<p></p>", "html", () => {}, false, false), null);
  t.mock.timers.tick(5_000);
  assert.equal(late.terminated, true, "with nothing left waiting, the late worker idles out, however many callers it turned away meanwhile");
  assert.equal(client.highlightWorkerState(), "untested");
  assert.ok(client.requestFullHighlight("<html></html>", "html", () => {}), "the next long fence gets a worker again");
  assert.equal(FakeWorker.made.length, 2);
});
