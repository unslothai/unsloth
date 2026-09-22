// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import type * as CachedTarget from "../src/features/model-picker/sharing/cached-target.ts";
import type * as Lifecycle from "../src/features/model-picker/sharing/link-lifecycle.ts";
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
const { modelConfigDraftKey } = await import(
  "../src/features/model-picker/model-config/model-config-draft.ts"
);
const { modelConfigHandoffForDestination } = await import(
  "../src/features/model-picker/model-config/model-config-handoff.ts"
);
const { resolveRunConfigTarget } = await import("./helpers/sharing-target.ts");
const { RunConfigResolutionError } = loadWithStubs<typeof CachedTarget>(
  new URL(
    "../src/features/model-picker/sharing/cached-target.ts",
    import.meta.url,
  ),
  { "@/features/auth": {}, "@/features/chat": {}, "@/features/hub": {} },
);

type Target = NonNullable<ReturnType<typeof resolveRunConfigTarget>>;
function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<T>((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return { promise, resolve, reject };
}
const settle = () => new Promise<void>((resolve) => setImmediate(resolve));

function harness() {
  const inbox = createRunConfigInbox();
  inbox.submit({
    id: "first",
    value: { model: "owner/Model-GGUF", config: { nParallel: 3 } },
  });
  const pending = inbox.getSnapshot();
  assert.ok(pending);
  const calls: unknown[] = [];
  const errors: string[] = [];
  const loading = new Map<number, string>();
  let nextToastId = 0;
  const lookups: {
    target: Target;
    signal: AbortSignal;
    result: ReturnType<typeof deferred<Target>>;
  }[] = [];
  const navigationResult = deferred<void>();
  const navigation: { current: Lifecycle.RunConfigNavigation | null } = {
    current: null,
  };
  const runtime = {
    params: { checkpoint: "" },
    activeGgufVariant: null,
    loadedIsGguf: null,
    activeNativePathToken: null,
    activeLoadId: null,
    models: [],
    loras: [],
    setActiveThreadId: (id: null) => calls.push(["thread", id]),
    setActiveProjectId: (id: null) => calls.push(["project", id]),
    setIncognito: (value: boolean) => calls.push(["incognito", value]),
  };
  const lifecycle = loadWithStubs<typeof Lifecycle>(
    new URL(
      "../src/features/model-picker/sharing/link-lifecycle.ts",
      import.meta.url,
    ),
    {
      "@/features/chat": {
        clearNewChatDraft: () => calls.push("clear draft"),
        useChatRuntimeStore: { getState: () => runtime },
      },
      "@/lib/toast": {
        toast: {
          error: (message: string) => errors.push(message),
          loading: (message: string) => {
            const id = ++nextToastId;
            loading.set(id, message);
            return id;
          },
          dismiss: (id: number) => loading.delete(id),
        },
      },
      "../model-config/model-config-draft": {
        modelConfigDraftKey,
      },
      "../model-config/model-config-handoff": {
        clearModelConfigHandoff: (id: string) =>
          calls.push(["clear handoff", id]),
        requestModelConfigHandoff: (target: Target & { requestId: string }) => {
          assert.equal(
            inbox.getSnapshot()?.draftKey,
            modelConfigDraftKey(target.id, target.meta.ggufVariant),
          );
          calls.push(["handoff", target]);
        },
      },
      "./cached-target": {
        RunConfigResolutionError,
        resolveCachedRunConfigTarget: (
          target: Target,
          options: { signal: AbortSignal },
        ) => {
          const result = deferred<Target>();
          lookups.push({ target, signal: options.signal, result });
          return result.promise;
        },
      },
      "./inbox": { runConfigInbox: inbox },
      "./target": { resolveRunConfigTarget },
    },
  );
  const context = {
    pending,
    canOpen: true,
    settingsHydrated: true,
    currentModel: "",
    location: { href: "/hub", pathname: "/hub", searchStr: "" },
  };
  const destination = {
    href: "/chat?new=first",
    pathname: "/chat",
    searchStr: "?new=first",
  };
  const nav = {
    ...context,
    navigation,
    navigate: (options: unknown) => {
      calls.push(["navigate", options]);
      return navigationResult.promise;
    },
  };
  const open = {
    ...context,
    location: destination,
    routeReady: true,
    inventoryVersion: 0,
  };
  const target: Target = {
    id: pending.value.model!,
    meta: {
      source: "hub",
      isLora: false,
      isGguf: true,
      ggufVariant: "Q4_K_M",
      loadId: "/cache/snapshot",
      isDownloaded: true,
    },
  };
  const prepare = async () => {
    lifecycle.openRunConfigTarget({ ...open, location: context.location });
    lookups.at(-1)!.result.resolve(target);
    await settle();
    const prepared = inbox.getSnapshot();
    assert.ok(prepared?.target);
    nav.pending = open.pending = prepared;
  };
  return {
    runtime,
    prepare,
    ...lifecycle,
    inbox,
    calls,
    errors,
    loading,
    lookups,
    navigationResult,
    navigation,
    nav,
    open,
    target,
    destination,
  };
}

for (const newChatId of [null, "current-draft"]) {
  test(`review keeps the current new chat and its draft: ${newChatId}`, async () => {
    const app = harness();
    app.runtime.params.checkpoint = "owner/Model-GGUF";
    delete app.nav.pending.value.model;
    const location = {
      href: newChatId ? `/chat?new=${newChatId}` : "/chat",
      pathname: "/chat",
      searchStr: newChatId ? `?new=${newChatId}` : "",
    };
    const context = { location, currentModel: app.runtime.params.checkpoint };
    app.navigateRunConfig({ ...app.nav, ...context });
    app.openRunConfigTarget({ ...app.open, ...context });
    app.lookups[0].result.resolve(app.target);
    await settle();
    const pending = app.inbox.getSnapshot();
    assert.equal(pending?.newChatId, newChatId);
    app.navigateRunConfig({ ...app.nav, ...context, pending });
    app.openRunConfigTarget({ ...app.open, ...context, pending });
    const handoff = { requestId: "first", newChatId, ...app.target };
    assert.deepEqual(app.calls, [["handoff", handoff]]);
    assert.equal(
      modelConfigHandoffForDestination(handoff, { active: true, newChatId }),
      handoff,
    );
    for (const destination of [
      { active: false, newChatId },
      { active: true, newChatId: "another-chat" },
      { active: true, newChatId, threadId: "thread" },
      { active: true, newChatId, compareId: "compare" },
      { active: true, newChatId, projectId: "project" },
    ]) {
      assert.equal(
        modelConfigHandoffForDestination(handoff, destination),
        null,
      );
    }
    app.inbox.clear("first");
    assert.deepEqual(app.calls, [["handoff", handoff]]);
  });
}

for (const replaceHistory of [false, true]) {
  test(`navigation waits for availability and respects history replacement: ${replaceHistory}`, async () => {
    const app = harness();
    app.nav.pending.replaceHistory = replaceHistory;
    app.navigateRunConfig(app.nav);
    assert.deepEqual(app.calls, []);
    await app.prepare();
    app.navigateRunConfig(app.nav);
    app.navigateRunConfig(app.nav);
    assert.deepEqual(app.calls, [
      [
        "navigate",
        { to: "/chat", search: { new: "first" }, replace: replaceHistory },
      ],
    ]);
    assert.equal(app.lookups.length, 1);
    app.navigateRunConfig({ ...app.nav, location: app.destination });
    assert.equal(app.navigation.current?.from, app.destination.href);
    app.navigateRunConfig(app.nav);
    assert.equal(app.inbox.getSnapshot(), null);
  });
}

for (const reason of [
  "login",
  "settings",
  "model",
  "bound",
  "superseded",
] as const) {
  test(`navigation and lookup do nothing while ${reason} blocks the import`, () => {
    const app = harness();
    if (reason === "login") app.nav.canOpen = app.open.canOpen = false;
    if (reason === "settings")
      app.nav.settingsHydrated = app.open.settingsHydrated = false;
    if (reason === "model") delete app.nav.pending.value.model;
    if (reason === "bound") app.nav.pending.draftKey = "already bound";
    if (reason === "superseded") app.inbox.clear("first");
    app.navigateRunConfig(app.nav);
    assert.equal(app.openRunConfigTarget(app.open), undefined);
    assert.deepEqual(app.calls, []);
    assert.deepEqual(app.lookups, []);
    assert.equal(app.loading.size, 0);
  });
}

for (const reason of [
  "route loading",
  "another page",
  "another chat",
] as const) {
  test(`handoff waits at ${reason}`, async () => {
    const app = harness();
    await app.prepare();
    if (reason === "route loading") app.open.routeReady = false;
    if (reason === "another page") app.open.location = app.nav.location;
    if (reason === "another chat")
      app.open.location = { ...app.destination, searchStr: "?new=other" };
    assert.equal(app.openRunConfigTarget(app.open), undefined);
    assert.deepEqual(app.calls, []);
  });
}

test("availability binds the canonical draft before handing off the editor", async () => {
  const app = harness();
  app.openRunConfigTarget({ ...app.open, location: app.nav.location });
  assert.deepEqual([...app.loading.values()], ["Resolving shared model…"]);
  assert.equal(app.inbox.getSnapshot()?.draftKey, undefined);
  assert.deepEqual(app.calls, []);
  app.lookups[0].result.resolve(app.target);
  await settle();
  assert.equal(app.loading.size, 0);
  assert.deepEqual(app.calls, []);
  app.openRunConfigTarget({ ...app.open, pending: app.inbox.getSnapshot() });
  assert.deepEqual(app.calls, [
    "clear draft",
    ["thread", null],
    ["project", null],
    ["incognito", false],
    ["handoff", { requestId: "first", ...app.target }],
  ]);
  assert.equal(
    app.inbox.getSnapshot()?.draftKey,
    modelConfigDraftKey(app.target.id, "Q4_K_M"),
  );
});

test("a recipient's local model choice carries a settings-only import through navigation and handoff", async () => {
  const app = harness();
  const pending = {
    ...app.nav.pending,
    selectedModel: "C:\\Models\\model.gguf",
    value: { config: { nParallel: 3 } },
  };
  app.inbox.submit(pending);
  app.navigateRunConfig({ ...app.nav, pending });
  assert.deepEqual(app.calls, []);
  app.openRunConfigTarget({ ...app.open, pending, location: app.nav.location });
  assert.equal(app.lookups.length, 1);
  const target = app.lookups[0].target;
  assert.equal(target.id, pending.selectedModel);
  assert.equal(target.meta.source, "local");
  assert.equal(target.meta.isGguf, true);
  app.lookups[0].result.resolve(target);
  await settle();
  const prepared = app.inbox.getSnapshot();
  app.navigateRunConfig({ ...app.nav, pending: prepared });
  app.openRunConfigTarget({ ...app.open, pending: prepared });
  const key = modelConfigDraftKey(pending.selectedModel, undefined);
  assert.equal(app.inbox.getSnapshot()?.draftKey, key);
  assert.deepEqual(app.calls.at(-1), [
    "handoff",
    { requestId: pending.id, ...target },
  ]);
  assert.deepEqual(app.inbox.take(pending.id, key), pending.value.config);
  assert.equal(app.inbox.take(pending.id, key), null);
});

for (const failure of [false, true]) {
  test(`effect cleanup aborts stale availability ${failure ? "failures" : "successes"}`, async () => {
    const app = harness();
    const cancel = app.openRunConfigTarget(app.open);
    cancel?.();
    assert.equal(app.loading.size, 0);
    assert.equal(app.lookups[0].signal.aborted, true);
    app.openRunConfigTarget(app.open);
    if (failure) app.lookups[0].result.reject(new Error("stale"));
    else app.lookups[0].result.resolve(app.target);
    await settle();
    assert.deepEqual(app.calls, []);
    assert.deepEqual(app.errors, []);
    assert.deepEqual([...app.loading.values()], ["Resolving shared model…"]);
    app.lookups[1].result.resolve(app.target);
    await settle();
    assert.equal(app.calls.length, 0);
    assert.equal(app.inbox.getSnapshot()?.target, app.target);
    assert.equal(app.loading.size, 0);
  });

  test(`newer links survive stale availability ${failure ? "failures" : "successes"}`, async () => {
    const app = harness();
    app.openRunConfigTarget(app.open);
    app.inbox.submit({
      id: "second",
      value: { model: "owner/Other", config: {} },
    });
    if (failure) app.lookups[0].result.reject(new Error("stale"));
    else app.lookups[0].result.resolve(app.target);
    await settle();
    assert.equal(app.inbox.getSnapshot()?.id, "second");
    assert.deepEqual(app.calls, []);
    assert.deepEqual(app.errors, []);
  });
}

for (const error of [
  new Error("Private backend details"),
  new RunConfigResolutionError(
    "Could not look up the shared GGUF model. Check your connection and access to the Hugging Face model, then reopen the link.",
  ),
  new RunConfigResolutionError(
    "The shared GGUF variant is unavailable for this model. Ask the sender for an updated link.",
  ),
]) {
  test(`unresolved model failures cancel with safe feedback: ${error.message}`, async () => {
    const app = harness();
    app.navigateRunConfig(app.nav);
    app.openRunConfigTarget({ ...app.open, location: app.nav.location });
    app.lookups[0].result.reject(error);
    await settle();
    assert.equal(app.inbox.getSnapshot(), null);
    assert.deepEqual(app.errors, [
      error instanceof RunConfigResolutionError
        ? error.message
        : "Could not resolve the shared model. Reopen the link to try again.",
    ]);
    assert.equal(app.loading.size, 0);
    assert.deepEqual(app.calls, []);
  });
}

for (const superseded of [false, true]) {
  test(`navigation failure only clears its own import: superseded=${superseded}`, async () => {
    const app = harness();
    await app.prepare();
    app.navigateRunConfig(app.nav);
    if (superseded) app.inbox.submit({ id: "second", value: { config: {} } });
    app.navigationResult.reject(new Error("navigation failed"));
    await settle();
    assert.equal(
      app.inbox.getSnapshot()?.id ?? null,
      superseded ? "second" : null,
    );
    assert.equal(app.errors.length, superseded ? 0 : 1);
    assert.equal(
      app.calls.some(
        (call) => Array.isArray(call) && call[0] === "clear handoff",
      ),
      !superseded,
    );
  });
}

for (const failure of [false, true]) {
  test(`leaving during preflight preserves chat state and ignores the late ${failure ? "failure" : "result"}`, async () => {
    const app = harness();
    app.navigateRunConfig(app.nav);
    app.openRunConfigTarget({ ...app.open, location: app.nav.location });
    app.navigateRunConfig({
      ...app.nav,
      location: { href: "/settings", pathname: "/settings", searchStr: "" },
    });
    if (failure) app.lookups[0].result.reject(new Error("offline"));
    else app.lookups[0].result.resolve(app.target);
    await settle();
    assert.equal(app.inbox.getSnapshot(), null);
    assert.deepEqual(app.calls, []);
    assert.deepEqual(app.errors, []);
  });
}
