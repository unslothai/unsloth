// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { createDeepLinkIntentGate } from "../src/features/deep-links/deep-link-intent.ts";
import { parseUnslothDeepLink } from "../src/features/deep-links/parse-deep-link.ts";
import * as linkAddress from "../src/features/model-picker/sharing/link-address.ts";
import type * as Receiver from "../src/features/model-picker/sharing/receive-link.ts";
import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

registerBundlerResolver();
installLocalStorageFake();
const links = await import("./helpers/sharing-links.ts");
const { createRunConfigInbox } = await import(
  "../src/features/model-picker/sharing/inbox.ts"
);
const events = await import("../src/features/auth/session-events.ts");
const sessionMark = "unsloth_auth_session_mark";
const recoveryKey = "unsloth.run-config-login.v1";
const run = "unsloth://run?v=1&model=owner/model&nParallel=3";
const browserRun =
  "http://localhost/chat#run?v=1&model=owner/model&nParallel=3";
const otherBrowserRun =
  "http://localhost/chat#run?v=1&model=owner/other&nParallel=4";

const settle = () => new Promise<void>((resolve) => setImmediate(resolve));

function deferredParser() {
  let resolve!: (value: typeof links) => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<typeof links>((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return { promise, resolve, reject };
}

function harness({
  url = "http://localhost/chat",
  desktop = false,
  loadParser = () => links,
}: {
  url?: string;
  desktop?: boolean;
  loadParser?: () => typeof links | Promise<typeof links>;
} = {}) {
  const browser = installLocalStorageFake();
  const storage = new Map<string, string>();
  const sessionStorage = {
    getItem: (key: string) => storage.get(key) ?? null,
    setItem: (key: string, value: string) => storage.set(key, value),
    removeItem: (key: string) => storage.delete(key),
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
  let parserLoads = 0;
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
        "./link-address": linkAddress,
        get "./runtime"() {
          parserLoads += 1;
          return loadParser();
        },
      },
    );
    const dispose = receiver.subscribeRunConfigSession(() => undefined);
    return { inbox, receiver, dispose };
  };
  return {
    loadDocument,
    storage,
    readRecovery: () => {
      const raw = storage.get(recoveryKey);
      assert.ok(raw);
      return JSON.parse(raw);
    },
    errors,
    cleared,
    parserLoads: () => parserLoads,
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

test("login recovery survives the account purge and document replacement exactly once", async () => {
  for (const native of [true, false]) {
    const app = harness({ url: native ? "http://localhost/chat" : browserRun });
    const before = app.loadDocument();
    if (native) {
      before.receiver.receiveSharedRunConfigUrls([run]);
      await settle();
    } else await before.receiver.receiveStartupRunConfigUrl();
    app.storage.clear();
    app.signIn();
    assert.equal(app.storage.has(recoveryKey), true);
    before.dispose();
    const after = app.loadDocument();
    await after.receiver.receiveStartupRunConfigUrl();
    assert.equal(after.inbox.getSnapshot()?.value.config.nParallel, 3);
    assert.equal(after.inbox.getSnapshot()?.replaceHistory, !native);
    assert.equal(app.storage.has(recoveryKey), false);
    after.dispose();
    const next = app.loadDocument();
    await next.receiver.receiveStartupRunConfigUrl();
    assert.equal(next.inbox.getSnapshot(), null);
    next.dispose();
    assert.deepEqual(app.errors, []);
  }
});

test("sign-out discards both pending and recoverable links before another session", async () => {
  const app = harness();
  const doc = app.loadDocument();
  doc.receiver.receiveSharedRunConfigUrls([run]);
  await settle();
  app.signIn();
  const id = doc.inbox.getSnapshot()?.id;
  app.signOut();
  assert.equal(doc.inbox.getSnapshot(), null);
  assert.equal(app.storage.has(recoveryKey), false);
  assert.deepEqual(app.cleared, [id]);
  app.signIn("second");
  assert.equal(app.storage.has(recoveryKey), false);
  doc.receiver.receiveSharedRunConfigUrls([run]);
  await settle();
  assert.equal(doc.inbox.getSnapshot()?.value.config.nParallel, 3);
  doc.dispose();
});

test("an account purge cannot replay a previously handled desktop link", async () => {
  const app = harness({ desktop: true });
  app.signIn();
  const before = app.loadDocument();
  before.receiver.receiveSharedRunConfigUrls([run], "startup");
  await settle();
  const pending = before.inbox.getSnapshot();
  assert.ok(pending);
  before.inbox.bind(pending.id, "draft");
  app.signOut();
  app.storage.clear();
  app.signIn("second");
  before.dispose();

  const after = app.loadDocument();
  assert.equal(
    after.receiver.receiveSharedRunConfigUrls([run], "startup"),
    "ignored",
  );
  await after.receiver.receiveStartupRunConfigUrl();
  assert.equal(after.inbox.getSnapshot(), null);
  assert.equal(app.parserLoads(), 1);
  after.receiver.receiveSharedRunConfigUrls([run], "event");
  await settle();
  assert.equal(after.inbox.getSnapshot()?.value.config.nParallel, 3);
  after.dispose();
});

for (const recoveryFirst of [false, true]) {
  test(`desktop login recovery imports once regardless of startup callback order: recovery first ${recoveryFirst}`, async () => {
    const app = harness({ desktop: true });
    const before = app.loadDocument();
    before.receiver.receiveSharedRunConfigUrls([run], "startup");
    await settle();
    app.storage.clear();
    app.signIn();
    before.dispose();

    const after = app.loadDocument();
    if (recoveryFirst) {
      await after.receiver.receiveStartupRunConfigUrl();
    }
    assert.equal(
      after.receiver.receiveSharedRunConfigUrls([run], "startup"),
      "ignored",
    );
    if (!recoveryFirst) {
      await after.receiver.receiveStartupRunConfigUrl();
    }
    assert.equal(after.inbox.getSnapshot()?.value.config.nParallel, 3);
    assert.equal(app.parserLoads(), 2);
    after.dispose();

    const next = app.loadDocument();
    next.receiver.receiveSharedRunConfigUrls([run], "startup");
    await next.receiver.receiveStartupRunConfigUrl();
    assert.equal(next.inbox.getSnapshot(), null);
    assert.equal(app.parserLoads(), 2);
    next.dispose();
    assert.deepEqual(app.errors, []);
  });
}

test("pending links survive repeated login reloads before signing in", async (t) => {
  let now = 1_000;
  t.mock.method(Date, "now", () => now);
  for (const native of [true, false]) {
    const app = harness({ url: native ? "http://localhost/chat" : browserRun });
    let doc = app.loadDocument();
    if (native) {
      doc.receiver.receiveSharedRunConfigUrls([run]);
      await settle();
    } else await doc.receiver.receiveStartupRunConfigUrl();
    assert.equal(app.storage.has(recoveryKey), true);
    const saved = app.readRecovery();
    for (let reload = 0; reload < 2; reload += 1) {
      now += 60_000;
      doc.dispose();
      window.location.href = "http://localhost/login";
      doc = app.loadDocument();
      await doc.receiver.receiveStartupRunConfigUrl();
      assert.equal(doc.inbox.getSnapshot()?.value.config.nParallel, 3);
      assert.equal(doc.inbox.getSnapshot()?.replaceHistory, !native);
      assert.equal(app.storage.has(recoveryKey), true);
      assert.equal(app.readRecovery().expiresAt, saved.expiresAt);
    }
    app.storage.clear();
    app.signIn();
    doc.dispose();
    const signedIn = app.loadDocument();
    await signedIn.receiver.receiveStartupRunConfigUrl();
    assert.equal(signedIn.inbox.getSnapshot()?.value.config.nParallel, 3);
    assert.equal(app.storage.has(recoveryKey), false);
    signedIn.dispose();
    assert.deepEqual(app.errors, []);
  }
});

test("clearing absent credentials during an auth redirect preserves a pre-login link", async () => {
  const app = harness({ url: browserRun });
  const before = app.loadDocument();
  await before.receiver.receiveStartupRunConfigUrl();
  app.signOut();
  before.dispose();
  window.location.href = "http://localhost/login";
  const after = app.loadDocument();
  await after.receiver.receiveStartupRunConfigUrl();
  assert.equal(after.inbox.getSnapshot()?.value.config.nParallel, 3);
  after.dispose();
});

test("pre-login recovery cannot outlive its expiry or cross a session boundary", async () => {
  for (const change of ["expired", "session"] as const) {
    const app = harness({ url: browserRun });
    const before = app.loadDocument();
    await before.receiver.receiveStartupRunConfigUrl();
    before.dispose();
    assert.equal(app.storage.has(recoveryKey), true);
    const saved = app.readRecovery();
    if (change === "expired") saved.expiresAt = 0;
    else saved.session = "other-account";
    app.storage.set(recoveryKey, JSON.stringify(saved));
    window.location.href = "http://localhost/login";
    const after = app.loadDocument();
    await after.receiver.receiveStartupRunConfigUrl();
    assert.equal(after.inbox.getSnapshot(), null);
    assert.equal(app.storage.has(recoveryKey), false);
    after.dispose();
  }
});

for (const change of ["session", "expired", "invalid", "newer"] as const) {
  test(`login recovery rejects ${change} stored intents`, async () => {
    const app = harness();
    const before = app.loadDocument();
    before.receiver.receiveSharedRunConfigUrls([run]);
    await settle();
    app.signIn();
    before.dispose();
    const saved = app.readRecovery();
    if (change === "session") saved.session = "other-account";
    if (change === "expired") saved.expiresAt = 0;
    if (change === "invalid") saved.url = "unsloth://run?v=1&hfToken=secret";
    app.storage.set(recoveryKey, JSON.stringify(saved));
    const after = app.loadDocument();
    if (change === "newer")
      after.receiver.receiveSharedRunConfigUrls([
        "unsloth://open_from_hf?model=owner/other",
      ]);
    await settle();
    await after.receiver.receiveStartupRunConfigUrl();
    assert.equal(after.inbox.getSnapshot(), null);
    assert.equal(app.storage.has(recoveryKey), false);
    after.dispose();
  });
}

test("binding or cancelling an in-document import removes login recovery", async () => {
  for (const bind of [true, false]) {
    const app = harness();
    const doc = app.loadDocument();
    doc.receiver.receiveSharedRunConfigUrls([run]);
    await settle();
    app.signIn();
    const pending = doc.inbox.getSnapshot();
    assert.ok(pending);
    if (bind) doc.inbox.bind(pending.id, "draft");
    else doc.inbox.clear(pending.id);
    assert.equal(app.storage.has(recoveryKey), false);
    doc.dispose();
  }
});

test("startup imports the captured URL once even after a router redirect or anchor change", async () => {
  for (const current of ["http://localhost/login", otherBrowserRun]) {
    const app = harness({ url: browserRun });
    const doc = app.loadDocument();
    window.location.href = current;
    await doc.receiver.receiveStartupRunConfigUrl();
    assert.equal(window.location.href, current);
    const pending = doc.inbox.getSnapshot();
    assert.equal(pending?.value.config.nParallel, 3);
    assert.equal(pending?.replaceHistory, true);
    assert.ok(pending);
    doc.inbox.clear(pending.id);
    await doc.receiver.receiveStartupRunConfigUrl();
    assert.equal(doc.inbox.getSnapshot(), null);
    assert.deepEqual(app.errors, []);
    doc.dispose();
  }
});

test("startup consumes valid and invalid fragments so dismissal and reload cannot replay them", async () => {
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
      await doc.receiver.receiveStartupRunConfigUrl();
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
      await reloaded.receiver.receiveStartupRunConfigUrl();
      assert.equal(reloaded.inbox.getSnapshot(), null);
      reloaded.dispose();
    }
  }
});

test("unrelated startup fragments and query parameters are left intact", async () => {
  const url = "http://localhost/chat?run=1&keep=value#unrelated";
  const app = harness({ url });
  const doc = app.loadDocument();
  await doc.receiver.receiveStartupRunConfigUrl();
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
  test(`router decoding cannot replay startup settings: ${payload}`, async () => {
    const app = harness({
      url: `http://localhost/chat?run=1#run?v=1&${payload}`,
    });
    const doc = app.loadDocument();
    window.location.href = window.location.href.replace(
      /%5B|%5D|%7B|%7D/gi,
      decodeURIComponent,
    );
    await doc.receiver.receiveStartupRunConfigUrl();
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
    await reloaded.receiver.receiveStartupRunConfigUrl();
    assert.equal(reloaded.inbox.getSnapshot(), null);
    assert.equal(app.errors.length, errorCount);
    reloaded.dispose();
  });
}

test("anchors added before or after startup intake cannot create an import", async () => {
  for (const url of [browserRun, `${browserRun}&unknown=true`]) {
    const app = harness();
    const doc = app.loadDocument();
    window.location.href = url;
    await doc.receiver.receiveStartupRunConfigUrl();
    assert.equal(doc.inbox.getSnapshot(), null);
    await doc.receiver.receiveStartupRunConfigUrl();
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
  test(`desktop startup ignores web fragments at ${origin} and still accepts native links`, async () => {
    const app = harness({
      url: `${origin}/chat#run?v=1&model=owner/model&nParallel=4`,
      desktop: true,
    });
    const doc = app.loadDocument();
    await doc.receiver.receiveStartupRunConfigUrl();
    assert.equal(doc.inbox.getSnapshot(), null);
    doc.receiver.receiveSharedRunConfigUrls([run]);
    await settle();
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
  test(`startup cannot supersede the newer native intent ${newer}`, async () => {
    const app = harness({ url: otherBrowserRun });
    const doc = app.loadDocument();
    doc.receiver.receiveSharedRunConfigUrls([newer]);
    await settle();
    const pending = doc.inbox.getSnapshot();
    await doc.receiver.receiveStartupRunConfigUrl();
    assert.equal(doc.inbox.getSnapshot(), pending);
    assert.equal(
      pending?.value.config.nParallel,
      newer === run ? 3 : undefined,
    );
    assert.equal(app.errors.length, newer.includes("unknown") ? 1 : 0);
    doc.dispose();
  });
}

test("a login remount preserves recovery until binding or rejection", async () => {
  const app = harness({ url: browserRun });
  const doc = app.loadDocument();
  await doc.receiver.receiveStartupRunConfigUrl();
  app.signIn();
  const pending = doc.inbox.getSnapshot();
  doc.dispose();
  await doc.receiver.receiveStartupRunConfigUrl();
  assert.equal(doc.inbox.getSnapshot(), pending);
  assert.equal(app.storage.has(recoveryKey), true);
  window.location.href = "http://localhost/chat";
  const reloaded = app.loadDocument();
  await reloaded.receiver.receiveStartupRunConfigUrl();
  assert.equal(reloaded.inbox.getSnapshot()?.value.config.nParallel, 3);
  assert.equal(app.storage.has(recoveryKey), false);
  reloaded.dispose();
});

test("ordinary startup, unrelated links and expired recovery never load the parser", async () => {
  const app = harness({ url: "http://localhost/chat#unrelated" });
  app.storage.set(
    recoveryKey,
    JSON.stringify({
      url: run,
      expiresAt: 0,
      session: null,
    }),
  );
  const doc = app.loadDocument();
  await doc.receiver.receiveStartupRunConfigUrl();
  assert.equal(doc.receiver.receiveSharedRunConfigUrls([browserRun]), false);
  assert.equal(
    doc.receiver.receiveSharedRunConfigUrls([
      "unsloth://open_from_hf?model=owner/other",
    ]),
    false,
  );
  await settle();
  assert.equal(app.parserLoads(), 0);
  assert.equal(doc.inbox.getSnapshot(), null);
  doc.dispose();
});

for (const newer of [
  "unsloth://run?v=1&model=owner/newer&nParallel=4",
  "unsloth://open_from_hf?model=owner/newer",
]) {
  test(`a delayed parser cannot restore a superseded link: ${newer}`, async () => {
    const parser = deferredParser();
    const app = harness({ url: browserRun, loadParser: () => parser.promise });
    app.signIn();
    const doc = app.loadDocument();
    const startup = doc.receiver.receiveStartupRunConfigUrl();
    assert.equal(window.location.href, "http://localhost/chat");
    doc.receiver.receiveSharedRunConfigUrls([newer]);
    parser.resolve(links);
    await startup;
    await settle();
    assert.equal(
      doc.inbox.getSnapshot()?.value.model,
      newer.includes("nParallel") ? "owner/newer" : undefined,
    );
    assert.deepEqual(app.errors, []);
    doc.dispose();
  });
}

test("sign-out retires a parser load before another account can receive its settings", async () => {
  const parser = deferredParser();
  const app = harness({ loadParser: () => parser.promise });
  app.signIn();
  const doc = app.loadDocument();
  assert.equal(doc.receiver.receiveSharedRunConfigUrls([run]), true);
  app.signOut();
  app.signIn("second");
  parser.resolve(links);
  await settle();
  assert.equal(doc.inbox.getSnapshot(), null);
  assert.equal(app.storage.has(recoveryKey), false);
  assert.deepEqual(app.errors, []);
  doc.dispose();
});

test("login recovery is saved even before the parser chunk arrives", async () => {
  const parser = deferredParser();
  const app = harness({ url: browserRun, loadParser: () => parser.promise });
  const before = app.loadDocument();
  const startup = before.receiver.receiveStartupRunConfigUrl();
  app.signOut();
  app.signIn();
  assert.equal(app.storage.has(recoveryKey), true);
  before.dispose();
  window.location.href = "http://localhost/chat";
  const after = app.loadDocument();
  const recovered = after.receiver.receiveStartupRunConfigUrl();
  parser.resolve(links);
  await Promise.all([startup, recovered]);
  assert.equal(after.inbox.getSnapshot()?.value.config.nParallel, 3);
  assert.equal(app.storage.has(recoveryKey), false);
  assert.deepEqual(app.errors, []);
  after.dispose();
});

test("a failed parser chunk reports an error and permits reopening the same native link", async () => {
  const parser = deferredParser();
  let retry = false;
  const app = harness({ loadParser: () => (retry ? links : parser.promise) });
  app.signIn();
  const doc = app.loadDocument();
  doc.receiver.receiveSharedRunConfigUrls([run]);
  await settle();
  parser.reject(new Error("Chunk unavailable"));
  await settle();
  assert.equal(app.errors.length, 1);
  assert.equal(doc.inbox.getSnapshot(), null);
  retry = true;
  doc.receiver.receiveSharedRunConfigUrls([run]);
  await settle();
  assert.equal(doc.inbox.getSnapshot()?.value.config.nParallel, 3);
  doc.dispose();
});
