// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { createDeepLinkIntentGate } from "../src/features/deep-links/deep-link-intent.ts";
import { parseUnslothDeepLink } from "../src/features/deep-links/parse-deep-link.ts";
import type * as Receiver from "../src/features/model-picker/sharing/receive-link.ts";
import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

registerBundlerResolver();
installLocalStorageFake();
const links = await import("../src/features/model-picker/sharing/links.ts");
const { createRunConfigInbox } = await import(
  "../src/features/model-picker/sharing/inbox.ts"
);
const events = await import("../src/features/auth/session-events.ts");
const sessionMark = "unsloth_auth_session_mark";
const run = "unsloth://run?v=1&model=owner/model&nParallel=3";
const browserRun =
  "http://localhost/chat#run?v=1&model=owner/model&nParallel=3";
const otherBrowserRun =
  "http://localhost/chat#run?v=1&model=owner/other&nParallel=4";

function harness({ url = "http://localhost/chat", desktop = false } = {}) {
  const browser = installLocalStorageFake();
  const recovery = new Map<string, string>();
  const sessionStorage = {
    getItem: (key: string) => recovery.get(key) ?? null,
    setItem: (key: string, value: string) => recovery.set(key, value),
    removeItem: (key: string) => recovery.delete(key),
  };
  Object.assign(globalThis, { sessionStorage });
  window.location.href = url;
  const historyState = { key: "existing-entry" };
  Object.assign(window, {
    history: {
      state: historyState,
      replaceState: (state: unknown, _title: string, href: string) => {
        assert.equal(state, historyState);
        window.location.href = href;
      },
    },
  });
  let signedIn = false;
  let nextId = 0;
  const errors: string[] = [];
  const cleared: string[] = [];
  const loadDocument = () => {
    const inbox = createRunConfigInbox();
    const receiver = loadWithStubs<typeof Receiver>(
      new URL(
        "../src/features/model-picker/sharing/receive-link.ts",
        import.meta.url,
      ),
      {
        "@/lib/api-base": { isTauri: desktop },
        "@/lib/toast": {
          toast: { error: (message: string) => errors.push(message) },
        },
        "@/features/auth": {
          ...events,
          AUTH_SESSION_MARK_KEY: sessionMark,
          hasAuthToken: () => signedIn,
        },
        "@/features/deep-links": {
          createDeepLinkIntentGate,
          parseUnslothDeepLink,
        },
        "../model-config/model-config-draft": {
          markModelConfigDraftEdited: () => undefined,
        },
        "../model-config/model-config-handoff": {
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
    const app = harness({ url: native ? "http://localhost/chat" : browserRun });
    const before = app.loadDocument();
    if (native) before.receiver.receiveSharedRunConfigUrls([run]);
    else before.receiver.receiveStartupRunConfigUrl();
    app.recovery.clear();
    app.signIn();
    assert.equal(app.recovery.size, 1);
    before.dispose();
    const after = app.loadDocument();
    after.receiver.receiveStartupRunConfigUrl();
    assert.equal(after.inbox.getSnapshot()?.value.config.nParallel, 3);
    assert.equal(after.inbox.getSnapshot()?.replaceHistory, !native);
    assert.equal(app.recovery.size, 0);
    after.dispose();
    const next = app.loadDocument();
    next.receiver.receiveStartupRunConfigUrl();
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
      if (change === "invalid") saved.url = "unsloth://run?v=1&hfToken=secret";
      app.recovery.set(key, JSON.stringify(saved));
    }
    const after = app.loadDocument();
    if (change === "newer")
      after.receiver.receiveSharedRunConfigUrls([
        "unsloth://open_from_hf?model=owner/other",
      ]);
    after.receiver.receiveStartupRunConfigUrl();
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

test("startup imports the captured URL once even after a router redirect or anchor change", () => {
  for (const current of ["http://localhost/login", otherBrowserRun]) {
    const app = harness({ url: browserRun });
    const doc = app.loadDocument();
    window.location.href = current;
    doc.receiver.receiveStartupRunConfigUrl();
    assert.equal(window.location.href, current);
    const pending = doc.inbox.getSnapshot();
    assert.equal(pending?.value.config.nParallel, 3);
    assert.equal(pending?.replaceHistory, true);
    assert.ok(pending);
    doc.inbox.clear(pending.id);
    doc.receiver.receiveStartupRunConfigUrl();
    assert.equal(doc.inbox.getSnapshot(), null);
    assert.deepEqual(app.errors, []);
    doc.dispose();
  }
});

test("startup consumes valid and invalid fragments so dismissal and reload cannot replay them", () => {
  for (const query of [
    "?run=1&keep=value",
    "?keep=value",
    "?run=2&keep=value",
  ]) {
    for (const fragment of [
      "#run?v=1&nParallel=3",
      "#run?v=1&unknown=true",
      "#run",
    ]) {
      const app = harness({ url: `http://localhost/chat${query}${fragment}` });
      app.signIn();
      const doc = app.loadDocument();
      doc.receiver.receiveStartupRunConfigUrl();
      assert.equal(
        window.location.href,
        `http://localhost/chat${query.replace("run=1&", "")}`,
      );
      const pending = doc.inbox.getSnapshot();
      if (fragment.includes("unknown") || fragment === "#run") {
        assert.equal(pending, null);
        assert.equal(app.errors.length, 1);
      } else {
        assert.ok(pending);
        doc.inbox.clear(pending.id);
        assert.deepEqual(app.errors, []);
      }
      doc.dispose();
      const reloaded = app.loadDocument();
      reloaded.receiver.receiveStartupRunConfigUrl();
      assert.equal(reloaded.inbox.getSnapshot(), null);
      reloaded.dispose();
    }
  }
});

test("unrelated startup fragments and query parameters are left intact", () => {
  const url = "http://localhost/chat?run=1&keep=value#unrelated";
  const app = harness({ url });
  const doc = app.loadDocument();
  doc.receiver.receiveStartupRunConfigUrl();
  assert.equal(window.location.href, url);
  assert.equal(doc.inbox.getSnapshot(), null);
  doc.dispose();
});

for (const payload of [
  "selectedGpuIds=%5B0%5D",
  "llamaExtraArgs=%5B%22--threads%22%2C%224%22%5D",
  "selectedGpuIds=%5B1%2C1%5D",
  "nParallel=%7B",
  "nParallel=%7D",
]) {
  test(`router decoding cannot replay startup settings: ${payload}`, () => {
    const app = harness({
      url: `http://localhost/chat?run=1#run?v=1&${payload}`,
    });
    const doc = app.loadDocument();
    window.location.href = window.location.href.replace(
      /%5B|%5D|%7B|%7D/gi,
      decodeURIComponent,
    );
    doc.receiver.receiveStartupRunConfigUrl();
    assert.equal(window.location.href, "http://localhost/chat");
    const pending = doc.inbox.getSnapshot();
    if (
      payload === "selectedGpuIds=%5B0%5D" ||
      payload.startsWith("llamaExtraArgs=")
    ) {
      assert.ok(pending);
      doc.inbox.clear(pending.id);
      assert.deepEqual(app.errors, []);
    } else {
      assert.equal(pending, null);
      assert.equal(app.errors.length, 1);
    }
    const errorCount = app.errors.length;
    doc.dispose();
    const reloaded = app.loadDocument();
    reloaded.receiver.receiveStartupRunConfigUrl();
    assert.equal(reloaded.inbox.getSnapshot(), null);
    assert.equal(app.errors.length, errorCount);
    reloaded.dispose();
  });
}

test("anchors added before or after startup intake cannot create an import", () => {
  for (const url of [browserRun, `${browserRun}&unknown=true`]) {
    const app = harness();
    const doc = app.loadDocument();
    window.location.href = url;
    doc.receiver.receiveStartupRunConfigUrl();
    assert.equal(doc.inbox.getSnapshot(), null);
    doc.receiver.receiveStartupRunConfigUrl();
    assert.equal(doc.inbox.getSnapshot(), null);
    assert.deepEqual(app.errors, []);
    doc.dispose();
  }
});

for (const origin of [
  "http://tauri.localhost",
  "tauri://localhost",
  "http://localhost:1420",
]) {
  test(`desktop startup ignores web fragments at ${origin} and still accepts native links`, () => {
    const app = harness({
      url: `${origin}/chat#run?v=1&model=owner/model&nParallel=4`,
      desktop: true,
    });
    const doc = app.loadDocument();
    doc.receiver.receiveStartupRunConfigUrl();
    assert.equal(doc.inbox.getSnapshot(), null);
    doc.receiver.receiveSharedRunConfigUrls([run]);
    assert.equal(doc.inbox.getSnapshot()?.value.config.nParallel, 3);
    assert.equal(doc.inbox.getSnapshot()?.replaceHistory, false);
    assert.deepEqual(app.errors, []);
    doc.dispose();
  });
}

for (const newer of [
  run,
  "unsloth://run?v=1&unknown=true",
  "unsloth://open_from_hf?model=owner/other",
]) {
  test(`startup cannot supersede the newer native intent ${newer}`, () => {
    const app = harness({ url: otherBrowserRun });
    const doc = app.loadDocument();
    doc.receiver.receiveSharedRunConfigUrls([newer]);
    const pending = doc.inbox.getSnapshot();
    doc.receiver.receiveStartupRunConfigUrl();
    assert.equal(doc.inbox.getSnapshot(), pending);
    assert.equal(
      pending?.value.config.nParallel,
      newer === run ? 3 : undefined,
    );
    assert.equal(app.errors.length, newer.includes("unknown") ? 1 : 0);
    doc.dispose();
  });
}

test("a login remount preserves recovery until binding or rejection", () => {
  const app = harness({ url: browserRun });
  const doc = app.loadDocument();
  doc.receiver.receiveStartupRunConfigUrl();
  app.signIn();
  const pending = doc.inbox.getSnapshot();
  doc.dispose();
  doc.receiver.receiveStartupRunConfigUrl();
  assert.equal(doc.inbox.getSnapshot(), pending);
  assert.equal(app.recovery.size, 1);
  window.location.href = "http://localhost/chat";
  const reloaded = app.loadDocument();
  reloaded.receiver.receiveStartupRunConfigUrl();
  assert.equal(reloaded.inbox.getSnapshot()?.value.config.nParallel, 3);
  assert.equal(app.recovery.size, 0);
  reloaded.dispose();
});
