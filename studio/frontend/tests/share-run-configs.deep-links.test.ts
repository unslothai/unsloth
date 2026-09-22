// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { createDeepLinkIntentGate } from "../src/features/deep-links/deep-link-intent.ts";
import { parseUnslothDeepLink } from "../src/features/deep-links/parse-deep-link.ts";
import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

registerBundlerResolver();
installLocalStorageFake();
const { createRunConfigInbox } = await import(
  "../src/features/model-picker/sharing/inbox.ts"
);
const { parseRunConfigLink, createRunConfigLink } = await import(
  "../src/features/model-picker/sharing/links.ts"
);

const hub = "unsloth://open_from_hf?model=owner/model";
const run = "unsloth://run?v=1&model=owner/model&nParallel=3";
const invalid =
  "unsloth://run?v=1&llamaExtraArgs=%5B%22--host%22%2C%220.0.0.0%22%5D";
const settle = () => new Promise<void>((resolve) => setImmediate(resolve));

function harness(sharedLinks = true) {
  const inbox = createRunConfigInbox();
  const navigations: Array<{
    to: string;
    search: { model: string; intent: number };
  }> = [];
  const commands: string[] = [];
  const errors: string[] = [];
  let nextId = 0;
  let listener: ((urls: string[]) => void) | undefined;
  let cleanup: (() => void) | undefined;
  let subscriptions = 0;
  let unsubscriptions = 0;
  let releaseStartup!: (urls: string[] | null) => void;
  const startup = new Promise<string[] | null>((resolve) => {
    releaseStartup = resolve;
  });
  const { receiveSharedRunConfigUrls } = loadWithStubs<{
    receiveSharedRunConfigUrls: (urls: string[]) => boolean;
  }>(
    new URL(
      "../src/features/model-picker/sharing/receive-link.ts",
      import.meta.url,
    ),
    {
      "@/lib/api-base": { isTauri: true },
      "@/lib/toast": {
        toast: { error: (message: string) => errors.push(message) },
      },
      "@/features/deep-links": {
        parseUnslothDeepLink,
        createDeepLinkIntentGate,
      },
      "../model-config/model-config-draft": {
        markModelConfigDraftEdited: () => undefined,
      },
      "../model-config/model-config-handoff": {
        clearModelConfigHandoff: () => undefined,
        createModelConfigHandoffRequestId: () => `request-${++nextId}`,
      },
      "./inbox": { runConfigInbox: inbox },
      "./links": { parseRunConfigLink, createRunConfigLink },
      "@/features/auth": { hasAuthToken: () => true },
    },
  );
  const { DeepLinkHandler } = loadWithStubs<{
    DeepLinkHandler: (props: {
      onOpenUrls?: (urls: string[]) => boolean;
    }) => null;
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
    sharedLinks ? { onOpenUrls: receiveSharedRunConfigUrls } : {},
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
  assert.equal(app.navigations.length, 0);
  assert.equal(app.inbox.getSnapshot()?.value.config.nParallel, 3);
  app.emit([run, hub]);
  assert.equal(app.navigations.length, 1);
  assert.equal(app.inbox.getSnapshot(), null);
  app.emit([hub, invalid]);
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
