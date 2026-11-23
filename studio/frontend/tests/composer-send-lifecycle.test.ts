// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// These tests exercise the installed @assistant-ui/core public wrapper, not a
// Promise-returning mock. In 0.1.17 ComposerRuntimeImpl.send() returns void and
// the success signal is the public `send` event.

import assert from "node:assert/strict";
import test, { after } from "node:test";
import type { AttachmentAdapter, PendingAttachment } from "@assistant-ui/core";
import {
  BaseComposerRuntimeCore,
  ThreadComposerRuntimeImpl,
} from "@assistant-ui/core/internal";
import {
  ComposerAttachmentPreparationError,
  noteComposerSendMessage,
  observeComposerSend,
  reportComposerAttachmentFailure,
} from "../src/components/assistant-ui/composer-send-lifecycle.ts";
import {
  closeTapMessageChannels,
  composerStoreAccessor,
} from "./helpers/composer-store-accessor.ts";

after(() => closeTapMessageChannels());

class TestComposerCore extends BaseComposerRuntimeCore {
  readonly path = { ref: "thread.composer", threadId: "thread-1" };
  readonly sent: Array<{ metadata?: { custom?: Record<string, unknown> } }> =
    [];
  adapter: AttachmentAdapter | undefined;

  protected getAttachmentAdapter(): AttachmentAdapter | undefined {
    return this.adapter;
  }
  protected getDictationAdapter(): undefined {
    return undefined;
  }
  get canCancel(): boolean {
    return false;
  }
  protected handleSend(message: {
    metadata?: { custom?: Record<string, unknown> };
  }): void {
    this.sent.push(message);
  }
  protected handleCancel(): void {
    // The lifecycle tests never cancel an in-flight send.
  }

  seedPendingAttachment(): void {
    const attachment: PendingAttachment = {
      id: "pending-1",
      type: "file",
      name: "notes.txt",
      contentType: "text/plain",
      file: new File(["hello"], "notes.txt", { type: "text/plain" }),
      status: { type: "running", reason: "uploading", progress: 0 },
    };
    this.setAttachments([attachment]);
  }
}

function composerFor(core: TestComposerCore): ThreadComposerRuntimeImpl {
  return new ThreadComposerRuntimeImpl({
    path: core.path,
    getState: () => core,
    subscribe: (callback: () => void) => core.subscribe(callback),
  } as never);
}

function accessorFor(core: TestComposerCore) {
  const runtime = composerFor(core);
  const fixture = composerStoreAccessor(runtime);
  return { accessor: fixture.accessor, dispose: fixture.dispose, runtime };
}

function observing(registration: ReturnType<typeof observeComposerSend>) {
  assert.equal(registration.kind, "observing");
  if (registration.kind !== "observing") {
    throw new Error("expected an observing registration");
  }
  return registration.attempt;
}

async function setComposerText(
  core: TestComposerCore,
  text: string,
): Promise<void> {
  core.setText(text);
  // The store client recomputes its memoized state on the tap scheduler's
  // macrotask.
  await new Promise<void>((resolve) => setTimeout(resolve, 0));
}

const quote = {
  messageId: "__unsloth_quote_context_v1__",
  text: "unsloth-quote-context:v1\n{}",
};

test("the installed public send() returns void and reports the send event", async () => {
  const core = new TestComposerCore();
  core.setText("hello");
  const { accessor, dispose, runtime } = accessorFor(core);
  assert.equal(
    typeof (accessor as { unstable_on?: unknown }).unstable_on,
    "undefined",
  );
  assert.equal(typeof accessor.__internal_getRuntime, "function");
  assert.equal(accessor.__internal_getRuntime?.(), runtime);
  let sent = 0;
  let failed = 0;
  const attempt = observing(
    observeComposerSend(
      accessor,
      ["thread-1"],
      Symbol("reservation"),
      { onSent: () => sent++, onFailed: () => failed++ },
      { expectedContextId: null },
    ),
  );
  assert.equal(attempt.isActive(), true);

  const result = accessor.send();
  assert.equal(result, undefined);
  await Promise.resolve();
  await Promise.resolve();

  assert.equal(sent, 1);
  assert.equal(failed, 0);
  assert.equal(core.sent.length, 1);
  assert.equal(attempt.isActive(), false);
  attempt.stop();
  dispose();
});

test("a quoted send settles only after the exact context reaches history", async () => {
  const core = new TestComposerCore();
  core.setText("quoted");
  core.setQuote(quote);
  const { accessor, dispose } = accessorFor(core);
  let sent = 0;
  const attempt = observing(
    observeComposerSend(
      accessor,
      ["thread-1"],
      Symbol("reservation"),
      { onSent: () => sent++, onFailed: () => undefined },
      { expectedContextId: "ctx-1" },
    ),
  );

  accessor.send();
  await Promise.resolve();
  await Promise.resolve();
  assert.equal(
    sent,
    0,
    "the payload-less event alone cannot attribute a quoted send",
  );
  assert.equal(core.sent.length, 1);
  assert.deepEqual(core.sent[0]?.metadata?.custom?.quote, quote);

  noteComposerSendMessage("ctx-1");
  assert.equal(sent, 1);
  noteComposerSendMessage("ctx-1");
  assert.equal(sent, 1, "settlement is idempotent");
  assert.equal(attempt.isActive(), false);
  attempt.stop();
  dispose();
});

test("a second registration is rejected without canceling the active attempt", async () => {
  const core = new TestComposerCore();
  core.setText("first");
  const first = accessorFor(core);
  const secondFixture = composerStoreAccessor(first.runtime);
  assert.equal(
    secondFixture.accessor.__internal_getRuntime?.(),
    first.accessor.__internal_getRuntime?.(),
    "separate accessors resolve to the same ownership identity",
  );
  const active = observing(
    observeComposerSend(
      first.accessor,
      ["thread-1"],
      Symbol("reservation"),
      { onSent: () => undefined, onFailed: () => undefined },
      { expectedContextId: null },
    ),
  );
  const second = observeComposerSend(
    secondFixture.accessor,
    ["thread-1"],
    Symbol("other-reservation"),
    { onSent: () => undefined, onFailed: () => undefined },
    { expectedContextId: null },
  );

  assert.equal(second.kind, "busy");
  assert.equal(
    active.isActive(),
    true,
    "the valid in-flight send is untouched",
  );

  first.accessor.send();
  await Promise.resolve();
  await Promise.resolve();
  assert.equal(active.isActive(), false);

  // Once the valid attempt has terminated, a legitimate later send can own
  // the composer again.
  await setComposerText(core, "later");
  const later = observing(
    observeComposerSend(
      secondFixture.accessor,
      ["thread-1"],
      Symbol("later-reservation"),
      { onSent: () => undefined, onFailed: () => undefined },
      { expectedContextId: null },
    ),
  );
  later.stop();
  first.dispose();
  secondFixture.dispose();
});

test("an empty composer is rejected before it can occupy activeByComposer", async () => {
  const core = new TestComposerCore();
  const { accessor, dispose } = accessorFor(core);
  const ignored = observeComposerSend(
    accessor,
    ["thread-1"],
    Symbol("reservation"),
    { onSent: () => undefined, onFailed: () => undefined },
    { expectedContextId: null },
  );
  assert.equal(ignored.kind, "ignored");
  if (ignored.kind === "ignored") {
    assert.equal(ignored.reason, "empty");
  }

  // BaseComposerRuntimeCore.send() would return without emitting `send`.
  // The composer must still be available for a real send afterwards.
  await setComposerText(core, "later");
  const later = observing(
    observeComposerSend(
      accessor,
      ["thread-1"],
      Symbol("later-reservation"),
      { onSent: () => undefined, onFailed: () => undefined },
      { expectedContextId: null },
    ),
  );
  later.stop();
  dispose();
});

test("observer teardown unsubscribes once and releases ownership", () => {
  const core = new TestComposerCore();
  core.setText("teardown");
  const { accessor, dispose } = accessorFor(core);
  const runtime = accessor.__internal_getRuntime?.();
  assert.ok(runtime);
  const original = runtime.unstable_on;
  let unsubscribed = 0;
  runtime.unstable_on = ((
    event: "send",
    callback: (payload: unknown) => void,
  ) => {
    const unsubscribe = original.call(runtime, event, callback);
    return () => {
      unsubscribed += 1;
      unsubscribe();
    };
  }) as typeof runtime.unstable_on;
  try {
    const attempt = observing(
      observeComposerSend(
        accessor,
        ["thread-1"],
        Symbol("teardown-token"),
        { onSent: () => undefined, onFailed: () => undefined },
        { expectedContextId: null },
      ),
    );
    attempt.stop();
    attempt.stop();
    assert.equal(unsubscribed, 1);
    assert.equal(attempt.isActive(), false);
  } finally {
    runtime.unstable_on = original;
  }

  const later = observing(
    observeComposerSend(
      accessor,
      ["thread-1"],
      Symbol("later-token"),
      { onSent: () => undefined, onFailed: () => undefined },
      { expectedContextId: null },
    ),
  );
  later.stop();
  dispose();
});

test("an unavailable runtime is non-observing but still permits the send", async () => {
  const core = new TestComposerCore();
  core.setText("send without observer");
  const { accessor, dispose } = accessorFor(core);
  const runtime = accessor.__internal_getRuntime?.();
  assert.ok(runtime);
  const original = runtime.unstable_on;
  runtime.unstable_on = undefined as never;
  try {
    const unavailable = observeComposerSend(
      accessor,
      ["thread-1"],
      Symbol("reservation"),
      { onSent: () => undefined, onFailed: () => undefined },
      { expectedContextId: null },
    );
    assert.equal(unavailable.kind, "unavailable");

    accessor.send();
    await Promise.resolve();
    await Promise.resolve();
    assert.equal(core.sent.length, 1);
  } finally {
    runtime.unstable_on = original;
    dispose();
  }
});

test("a throwing subscription rolls back every registration", () => {
  const core = new TestComposerCore();
  core.setText("rollback");
  const { accessor, dispose } = accessorFor(core);
  const runtime = accessor.__internal_getRuntime?.();
  assert.ok(runtime);
  const original = runtime.unstable_on;
  let failures = 0;
  const token = Symbol("rollback-token");
  runtime.unstable_on = (() => {
    throw new Error("subscribe failed");
  }) as typeof runtime.unstable_on;
  try {
    const unavailable = observeComposerSend(
      accessor,
      ["thread-1"],
      token,
      {
        onSent: () => undefined,
        onFailed: () => {
          failures += 1;
        },
      },
      { expectedContextId: "ctx-rollback" },
    );
    assert.equal(unavailable.kind, "unavailable");
  } finally {
    runtime.unstable_on = original;
  }

  // Token and thread registrations were rolled back, so the failure reporter
  // cannot find a stale attempt.
  reportComposerAttachmentFailure(
    token,
    ["thread-1"],
    new Error("stale lookup"),
  );
  reportComposerAttachmentFailure(
    Symbol("unknown-token"),
    ["thread-1"],
    new Error("stale thread lookup"),
  );
  assert.equal(failures, 0);

  const later = observing(
    observeComposerSend(
      accessor,
      ["thread-1"],
      Symbol("later-token"),
      { onSent: () => undefined, onFailed: () => undefined },
      { expectedContextId: null },
    ),
  );
  later.stop();
  dispose();
});

test("a synchronous send event during subscription is not lost or leaked", () => {
  const core = new TestComposerCore();
  core.setText("sync event");
  const { accessor, dispose } = accessorFor(core);
  const runtime = accessor.__internal_getRuntime?.();
  assert.ok(runtime);
  const original = runtime.unstable_on;
  let unsubscribed = 0;
  let sent = 0;
  runtime.unstable_on = ((_event: string, callback: () => void) => {
    callback();
    return () => {
      unsubscribed += 1;
    };
  }) as typeof runtime.unstable_on;
  try {
    const result = observeComposerSend(
      accessor,
      ["thread-1"],
      Symbol("sync-token"),
      {
        onSent: () => {
          sent += 1;
        },
        onFailed: () => undefined,
      },
      { expectedContextId: null },
    );
    assert.equal(sent, 1, "the synchronous event settled the attempt");
    assert.equal(unsubscribed, 1, "the late unsubscribe was invoked");
    assert.equal(result.kind, "unavailable");
  } finally {
    runtime.unstable_on = original;
  }

  const later = observing(
    observeComposerSend(
      accessor,
      ["thread-1"],
      Symbol("later-token"),
      { onSent: () => undefined, onFailed: () => undefined },
      { expectedContextId: null },
    ),
  );
  later.stop();
  dispose();
});

test("attachment preparation failure is observed before handleSend and history", async () => {
  const token = Symbol("reservation");
  const core = new TestComposerCore();
  core.setText("attached");
  core.adapter = {
    accept: "*",
    add: async () => {
      throw new Error("unused");
    },
    remove: async () => undefined,
    send: async () => {
      throw reportComposerAttachmentFailure(
        token,
        ["thread-1"],
        new Error("upload failed"),
      );
    },
  };
  core.seedPendingAttachment();
  const { accessor, dispose } = accessorFor(core);
  let sent = 0;
  let failure: unknown = null;
  const attempt = observing(
    observeComposerSend(
      accessor,
      ["thread-1"],
      token,
      {
        onSent: () => sent++,
        onFailed: (error) => {
          failure = error;
        },
      },
      { expectedContextId: null },
    ),
  );

  // The public wrapper drops the core Promise; use the core directly so the
  // test does not manufacture an unhandled rejection.
  await assert.rejects(core.send(), /upload failed/);

  assert.equal(sent, 0);
  assert.ok(failure instanceof ComposerAttachmentPreparationError);
  assert.equal(core.sent.length, 0, "history/send must not run");
  assert.equal(attempt.isActive(), false);

  await setComposerText(core, "later");
  const later = observing(
    observeComposerSend(
      accessor,
      ["thread-1"],
      Symbol("later-reservation"),
      { onSent: () => undefined, onFailed: () => undefined },
      { expectedContextId: null },
    ),
  );
  later.stop();
  dispose();
});

test("only tagged in-flight attachment failures are suppressed", () => {
  const listeners = new Set<(event: Event) => void>();
  const fakeWindow = {
    addEventListener: (_: string, callback: (event: Event) => void) => {
      listeners.add(callback);
    },
    removeEventListener: (_: string, callback: (event: Event) => void) => {
      listeners.delete(callback);
    },
  };
  const previous = (globalThis as { window?: unknown }).window;
  (globalThis as { window?: unknown }).window = fakeWindow;
  try {
    const token = Symbol("reservation");
    const core = new TestComposerCore();
    core.setText("guarded");
    const { accessor, dispose } = accessorFor(core);
    const attempt = observing(
      observeComposerSend(
        accessor,
        ["thread-1"],
        token,
        { onSent: () => undefined, onFailed: () => undefined },
        { expectedContextId: null },
      ),
    );
    assert.equal(listeners.size, 1);
    const [handler] = [...listeners];
    assert.ok(handler);

    let taggedPrevented = false;
    handler({
      reason: new ComposerAttachmentPreparationError(
        token,
        new Error("upload failed"),
      ),
      preventDefault: () => {
        taggedPrevented = true;
      },
    } as unknown as Event);
    assert.equal(taggedPrevented, true);

    let untaggedPrevented = false;
    handler({
      reason: new Error("unrelated"),
      preventDefault: () => {
        untaggedPrevented = true;
      },
    } as unknown as Event);
    assert.equal(untaggedPrevented, false);

    attempt.stop();
    assert.equal(listeners.size, 0);
    dispose();
  } finally {
    if (previous === undefined) {
      (globalThis as { window?: unknown }).window = undefined;
    } else {
      (globalThis as { window?: unknown }).window = previous;
    }
  }
});
