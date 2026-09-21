// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { createDeepLinkIntentGate } from "../src/features/deep-links/deep-link-intent.ts";
import { parseUnslothDeepLink } from "../src/features/deep-links/parse-deep-link.ts";
import type * as Receiver from "../src/features/share-run-configs/receive-link.ts";
import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

registerBundlerResolver();
installLocalStorageFake();
const links = await import("../src/features/share-run-configs/links.ts");
const { createRunConfigInbox } = await import(
  "../src/features/share-run-configs/inbox.ts"
);
const events = await import("../src/features/auth/session-events.ts");
const sessionMark = "unsloth_auth_session_mark";
const run = "unsloth://run?model=owner/model&nParallel=3";

function harness() {
  const browser = installLocalStorageFake();
  const recovery = new Map<string, string>();
  const sessionStorage = {
    getItem: (key: string) => recovery.get(key) ?? null,
    setItem: (key: string, value: string) => recovery.set(key, value),
    removeItem: (key: string) => recovery.delete(key),
  };
  Object.assign(globalThis, { sessionStorage });
  window.location.href = "http://localhost/chat";
  let signedIn = false;
  let nextId = 0;
  const errors: string[] = [];
  const cleared: string[] = [];
  const loadDocument = () => {
    const inbox = createRunConfigInbox();
    const receiver = loadWithStubs<typeof Receiver>(
      new URL(
        "../src/features/share-run-configs/receive-link.ts",
        import.meta.url,
      ),
      {
        "@/lib/toast": {
          toast: { error: (message: string) => errors.push(message) },
        },
        "@/features/auth": {
          ...events,
          AUTH_SESSION_MARK_KEY: sessionMark,
          hasAuthToken: () => signedIn,
        },
        "../deep-links/deep-link-intent": { createDeepLinkIntentGate },
        "../deep-links/parse-deep-link": { parseUnslothDeepLink },
        "../model-picker/model-config/model-config-draft": {
          markModelConfigDraftEdited: () => undefined,
        },
        "../model-picker/model-config/model-config-handoff": {
          clearModelConfigHandoff: (id: string) => cleared.push(id),
          createModelConfigHandoffRequestId: () => `request-${++nextId}`,
        },
        "./inbox": { runConfigInbox: inbox },
        "./links": links,
      },
    );
    const dispose = receiver.subscribeRunConfigSession(() => undefined);
    return { inbox, receiver, dispose };
  };
  return {
    loadDocument,
    recovery,
    errors,
    cleared,
    signIn: (session = "first") => {
      signedIn = true;
      localStorage.setItem(sessionMark, session);
      browser.fireWindowEvent(events.AUTH_SESSION_STORED_EVENT, {});
    },
    signOut: () => {
      signedIn = false;
      localStorage.removeItem(sessionMark);
      browser.fireWindowEvent(events.AUTH_SESSION_CLEARED_EVENT, {});
    },
  };
}

test("login recovery survives the account purge and document replacement exactly once", () => {
  for (const native of [true, false]) {
    const app = harness();
    const before = app.loadDocument();
    if (native) before.receiver.receiveSharedRunConfigUrls([run]);
    else
      before.receiver.receiveRunConfigUrl(
        "http://localhost/chat#run?model=owner/model&nParallel=3",
      );
    app.recovery.clear();
    app.signIn();
    assert.equal(app.recovery.size, 1);
    before.dispose();
    const after = app.loadDocument();
    after.receiver.receiveStartupRunConfigUrl(window.location.href);
    assert.equal(after.inbox.getSnapshot()?.value.config.nParallel, 3);
    assert.equal(after.inbox.getSnapshot()?.replaceHistory, !native);
    assert.equal(app.recovery.size, 0);
    after.dispose();
    const next = app.loadDocument();
    next.receiver.receiveStartupRunConfigUrl(window.location.href);
    assert.equal(next.inbox.getSnapshot(), null);
    next.dispose();
    assert.deepEqual(app.errors, []);
  }
});

test("sign-out discards both pending and recoverable links before another session", () => {
  const app = harness();
  const doc = app.loadDocument();
  doc.receiver.receiveSharedRunConfigUrls([run]);
  app.signIn();
  const id = doc.inbox.getSnapshot()?.id;
  app.signOut();
  assert.equal(doc.inbox.getSnapshot(), null);
  assert.equal(app.recovery.size, 0);
  assert.deepEqual(app.cleared, [id]);
  app.signIn("second");
  assert.equal(app.recovery.size, 0);
  doc.receiver.receiveSharedRunConfigUrls([run]);
  assert.equal(doc.inbox.getSnapshot()?.value.config.nParallel, 3);
  doc.dispose();
});

for (const change of ["session", "expired", "invalid", "newer"] as const) {
  test(`login recovery rejects ${change} stored intents`, () => {
    const app = harness();
    const before = app.loadDocument();
    before.receiver.receiveSharedRunConfigUrls([run]);
    app.signIn();
    before.dispose();
    for (const [key, raw] of app.recovery) {
      const saved = JSON.parse(raw);
      if (change === "session") saved.session = "other-account";
      if (change === "expired") saved.expiresAt = 0;
      if (change === "invalid") saved.url = "unsloth://run?hfToken=secret";
      app.recovery.set(key, JSON.stringify(saved));
    }
    const after = app.loadDocument();
    if (change === "newer")
      after.receiver.receiveSharedRunConfigUrls([
        "unsloth://open_from_hf?model=owner/other",
      ]);
    after.receiver.receiveStartupRunConfigUrl(window.location.href);
    assert.equal(after.inbox.getSnapshot(), null);
    assert.equal(app.recovery.size, 0);
    after.dispose();
  });
}

test("binding or cancelling an in-document import removes login recovery", () => {
  for (const bind of [true, false]) {
    const app = harness();
    const doc = app.loadDocument();
    doc.receiver.receiveSharedRunConfigUrls([run]);
    app.signIn();
    const pending = doc.inbox.getSnapshot();
    assert.ok(pending);
    if (bind) doc.inbox.bind(pending.id, "draft");
    else doc.inbox.clear(pending.id);
    assert.equal(app.recovery.size, 0);
    doc.dispose();
  }
});
