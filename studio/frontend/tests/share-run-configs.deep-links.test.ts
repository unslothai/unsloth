// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import type { DeepLinkHandler as Handler } from "../src/features/deep-links/deep-link-handler.tsx";
import { createDeepLinkIntentGate } from "../src/features/deep-links/deep-link-intent.ts";
import { parseUnslothDeepLink } from "../src/features/deep-links/parse-deep-link.ts";
import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

registerBundlerResolver();
installLocalStorageFake();
const { receiverHarness, settle } = await import(
  "./helpers/sharing-receiver.ts"
);

const hub = "unsloth://open_from_hf?model=owner/model";
const run = "unsloth://run?v=1&model=owner/model&nParallel=3";
const invalid =
  "unsloth://run?v=1&llamaExtraArgs=%5B%22--host%22%2C%220.0.0.0%22%5D";

function desktopSession() {
  const values = new Map<string, string>();
  const storage = {
    getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => values.set(key, value),
    removeItem: (key: string) => values.delete(key),
  };
  Object.assign(globalThis, { sessionStorage: storage });
  return storage;
}

function harness(sharedLinks = true) {
  const { inbox, errors, receiver } = receiverHarness({ desktop: true });
  const navigations: Array<{
    to: string;
    search: { model: string; intent: number };
  }> = [];
  const commands: string[] = [];
  let listener: ((urls: string[]) => void) | undefined;
  let cleanup: (() => void) | undefined;
  let subscriptions = 0;
  let unsubscriptions = 0;
  let releaseStartup!: (urls: string[] | null) => void;
  const startup = new Promise<string[] | null>((resolve) => {
    releaseStartup = resolve;
  });
  const { DeepLinkHandler } = loadWithStubs<{
    DeepLinkHandler: typeof Handler;
  }>(
    new URL(
      "../src/features/deep-links/deep-link-handler.tsx",
      import.meta.url,
    ),
    {
      "@/lib/api-base": { isTauri: true },
      "@tanstack/react-router": {
        useNavigate:
          () => async (destination: (typeof navigations)[number]) => {
            navigations.push(destination);
          },
      },
      react: {
        useEffect: (effect: () => () => void) => {
          cleanup = effect();
        },
      },
      "./deep-link-intent": { createDeepLinkIntentGate },
      "./parse-deep-link": { parseUnslothDeepLink },
      "@tauri-apps/api/core": {
        invoke: async (command: string) => {
          commands.push(command);
        },
      },
      "@tauri-apps/plugin-deep-link": {
        getCurrent: () => startup,
        onOpenUrl: async (callback: typeof listener) => {
          subscriptions += 1;
          listener = callback;
          return () => {
            unsubscriptions += 1;
          };
        },
      },
    },
  );
  DeepLinkHandler(
    sharedLinks ? { onOpenUrls: receiver.receiveSharedRunConfigUrls } : {},
  );
  return {
    inbox,
    navigations,
    commands,
    errors,
    releaseStartup,
    emit: (urls: string[]) => {
      assert.ok(listener);
      listener(urls);
    },
    cleanup: () => cleanup?.(),
    counts: () => ({ subscriptions, unsubscriptions }),
  };
}

for (const delivery of ["startup", "live"] as const) {
  for (const url of [run, invalid]) {
    test(`a ${delivery} run link is not replayed by later desktop documents: ${url}`, async () => {
      desktopSession();
      const before = harness();
      await settle();
      before.releaseStartup(delivery === "startup" ? [url] : null);
      await settle();
      if (delivery === "live") before.emit([url]);
      await settle();
      assert.deepEqual(before.commands, ["reveal_main_window"]);
      assert.equal(before.errors.length, url === invalid ? 1 : 0);
      assert.equal(before.inbox.getSnapshot() !== null, url === run);
      before.cleanup();

      for (let reload = 0; reload < 2; reload += 1) {
        const after = harness();
        await settle();
        after.releaseStartup([hub, url]);
        await settle();
        assert.equal(after.inbox.getSnapshot(), null);
        assert.deepEqual(after.navigations, []);
        assert.deepEqual(after.commands, []);
        assert.deepEqual(after.errors, []);
        if (reload === 1) {
          after.emit([url]);
          await settle();
          assert.deepEqual(after.commands, ["reveal_main_window"]);
          assert.equal(after.errors.length, url === invalid ? 1 : 0);
          assert.equal(after.inbox.getSnapshot() !== null, url === run);
        }
        after.cleanup();
      }
    });
  }
}

test("a different startup run link and a fresh desktop session are still accepted", async () => {
  desktopSession();
  const first = harness();
  await settle();
  first.releaseStartup([run]);
  await settle();
  first.cleanup();

  const newer = harness();
  await settle();
  newer.releaseStartup([run.replace("nParallel=3", "nParallel=4")]);
  await settle();
  assert.equal(newer.inbox.getSnapshot()?.value.config.nParallel, 4);
  assert.deepEqual(newer.commands, ["reveal_main_window"]);
  newer.cleanup();

  desktopSession();
  const restarted = harness();
  await settle();
  restarted.releaseStartup([run]);
  await settle();
  assert.equal(restarted.inbox.getSnapshot()?.value.config.nParallel, 3);
  assert.deepEqual(restarted.commands, ["reveal_main_window"]);
  restarted.cleanup();
});

test("blocked session storage does not prevent opening native run links", async (t) => {
  const storage = desktopSession();
  t.mock.method(storage, "getItem", () => {
    throw new Error("Storage blocked");
  });
  t.mock.method(storage, "setItem", () => {
    throw new Error("Storage blocked");
  });
  const app = harness();
  await settle();
  app.releaseStartup([run]);
  await settle();
  assert.equal(app.inbox.getSnapshot()?.value.config.nParallel, 3);
  assert.deepEqual(app.commands, ["reveal_main_window"]);
  assert.deepEqual(app.errors, []);
  app.cleanup();
});

for (const sharedLinks of [false, true]) {
  test(`ordinary Hub links retain routing and deduplication with sharing ${sharedLinks ? "enabled" : "absent"}`, async () => {
    const app = harness(sharedLinks);
    await settle();
    app.releaseStartup([hub]);
    await settle();
    app.emit([hub]);
    app.emit(["https://example.invalid/unrelated"]);
    app.emit([hub]);
    await settle();
    assert.equal(app.navigations.length, 1);
    assert.equal(app.navigations[0].to, "/hub");
    assert.equal(app.navigations[0].search.model, "owner/model");
    assert.equal(app.inbox.getSnapshot(), null);
    assert.deepEqual(app.commands, ["reveal_main_window"]);
    app.cleanup();
    assert.deepEqual(app.counts(), { subscriptions: 1, unsubscriptions: 1 });
  });
}

for (const url of [run, invalid]) {
  test(`a ${url === run ? "valid" : "rejected"} shared link retires the previous Hub duplicate without resetting its sequence`, async () => {
    const app = harness();
    await settle();
    app.releaseStartup([hub]);
    await settle();
    app.emit([url]);
    await settle();
    app.emit([hub]);
    await settle();
    assert.equal(app.navigations.length, 2);
    assert.equal(
      app.navigations[1].search.intent,
      app.navigations[0].search.intent + 1,
    );
    assert.equal(app.inbox.getSnapshot(), null);
    assert.equal(app.errors.length, url === invalid ? 1 : 0);
    app.cleanup();
  });
}

test("the newest recognized intent wins in mixed native batches", async () => {
  const app = harness();
  await settle();
  app.releaseStartup(null);
  await settle();
  app.emit([hub, run]);
  await settle();
  assert.equal(app.navigations.length, 0);
  assert.equal(app.inbox.getSnapshot()?.value.config.nParallel, 3);
  app.emit([run, hub]);
  assert.equal(app.navigations.length, 1);
  assert.equal(app.inbox.getSnapshot(), null);
  app.emit([hub, invalid]);
  await settle();
  assert.equal(app.navigations.length, 1);
  assert.equal(app.errors.length, 1);
  app.cleanup();
});

test("desktop intake ignores web fragments without replacing pending native intents", async () => {
  const app = harness();
  await settle();
  app.releaseStartup(null);
  await settle();
  app.emit([run]);
  await settle();
  const pending = app.inbox.getSnapshot();
  assert.ok(pending);
  for (const url of [
    "https://example.invalid/chat#run?v=1&model=owner/other&nParallel=4",
    "http://localhost/chat#run?v=1&model=owner/other&nParallel=4",
    "https://example.invalid/chat#run?v=1&unknown=true",
  ]) {
    app.emit([url]);
    assert.equal(app.inbox.getSnapshot(), pending);
  }
  app.emit([hub, "https://example.invalid/chat#run?v=1&nParallel=4"]);
  assert.equal(app.inbox.getSnapshot(), null);
  assert.equal(app.navigations.length, 1);
  assert.deepEqual(app.errors, []);
  app.cleanup();
});

test("live shared links supersede delayed desktop startup URLs and disposal ignores later events", async () => {
  const app = harness();
  await settle();
  app.emit([run]);
  await settle();
  const pending = app.inbox.getSnapshot();
  app.releaseStartup([hub]);
  await settle();
  assert.equal(app.inbox.getSnapshot(), pending);
  assert.equal(app.navigations.length, 0);
  app.cleanup();
  app.emit([hub]);
  app.emit([invalid]);
  await settle();
  assert.equal(app.inbox.getSnapshot(), pending);
  assert.equal(app.navigations.length, 0);
  assert.deepEqual(app.errors, []);
  assert.deepEqual(app.counts(), { subscriptions: 1, unsubscriptions: 1 });
});
