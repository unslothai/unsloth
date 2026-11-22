// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// These tests exercise the installed @assistant-ui/core public wrapper, not a
// Promise-returning mock. In 0.1.17 ComposerRuntimeImpl.send() returns void and
// the success signal is the public `send` event.

import assert from "node:assert/strict";
import test from "node:test";
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

function observing(registration: ReturnType<typeof observeComposerSend>) {
  assert.equal(registration.kind, "observing");
  if (registration.kind !== "observing") {
    throw new Error("expected an observing registration");
  }
  return registration.attempt;
}

const quote = {
  messageId: "__unsloth_quote_context_v1__",
  text: "unsloth-quote-context:v1\n{}",
};

test("the installed public send() returns void and reports the send event", async () => {
  const core = new TestComposerCore();
  core.setText("hello");
  const composer = composerFor(core);
  let sent = 0;
  let failed = 0;
  const attempt = observing(
    observeComposerSend(
      composer,
      ["thread-1"],
      Symbol("reservation"),
      { onSent: () => sent++, onFailed: () => failed++ },
      { expectedContextId: null },
    ),
  );
  assert.equal(attempt.isActive(), true);

  const result = composer.send();
  assert.equal(result, undefined);
  await Promise.resolve();
  await Promise.resolve();

  assert.equal(sent, 1);
  assert.equal(failed, 0);
  assert.equal(core.sent.length, 1);
  assert.equal(attempt.isActive(), false);
  attempt.stop();
});

test("a quoted send settles only after the exact context reaches history", async () => {
  const core = new TestComposerCore();
  core.setText("quoted");
  core.setQuote(quote);
  const composer = composerFor(core);
  let sent = 0;
  const attempt = observing(
    observeComposerSend(
      composer,
      ["thread-1"],
      Symbol("reservation"),
      { onSent: () => sent++, onFailed: () => undefined },
      { expectedContextId: "ctx-1" },
    ),
  );

  composer.send();
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
});

test("a second registration is rejected without canceling the active attempt", async () => {
  const core = new TestComposerCore();
  core.setText("first");
  const composer = composerFor(core);
  const active = observing(
    observeComposerSend(
      composer,
      ["thread-1"],
      Symbol("reservation"),
      { onSent: () => undefined, onFailed: () => undefined },
      { expectedContextId: null },
    ),
  );
  const second = observeComposerSend(
    composer,
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

  composer.send();
  await Promise.resolve();
  await Promise.resolve();
  assert.equal(active.isActive(), false);

  // Once the valid attempt has terminated, a legitimate later send can own
  // the composer again.
  core.setText("later");
  const later = observing(
    observeComposerSend(
      composer,
      ["thread-1"],
      Symbol("later-reservation"),
      { onSent: () => undefined, onFailed: () => undefined },
      { expectedContextId: null },
    ),
  );
  later.stop();
});

test("an empty composer is rejected before it can occupy activeByComposer", () => {
  const core = new TestComposerCore();
  const composer = composerFor(core);
  const ignored = observeComposerSend(
    composer,
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
  core.setText("later");
  const later = observing(
    observeComposerSend(
      composer,
      ["thread-1"],
      Symbol("later-reservation"),
      { onSent: () => undefined, onFailed: () => undefined },
      { expectedContextId: null },
    ),
  );
  later.stop();
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
  const composer = composerFor(core);
  let sent = 0;
  let failure: unknown = null;
  const attempt = observing(
    observeComposerSend(
      composer,
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

  core.setText("later");
  const later = observing(
    observeComposerSend(
      composer,
      ["thread-1"],
      Symbol("later-reservation"),
      { onSent: () => undefined, onFailed: () => undefined },
      { expectedContextId: null },
    ),
  );
  later.stop();
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
    const composer = composerFor(core);
    const attempt = observing(
      observeComposerSend(
        composer,
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
  } finally {
    if (previous === undefined) {
      (globalThis as { window?: unknown }).window = undefined;
    } else {
      (globalThis as { window?: unknown }).window = previous;
    }
  }
});
