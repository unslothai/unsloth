// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test, { type TestContext } from "node:test";
import type { ChatSearch } from "../src/features/chat/chat-page.tsx";
import type * as ImportConfig from "../src/features/model-picker/sharing/import-config.ts";
import type * as Lifecycle from "../src/features/model-picker/sharing/link-lifecycle.ts";
import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

registerBundlerResolver();
installLocalStorageFake();
const drafts = await import(
  "../src/features/model-picker/model-config/model-config-draft.ts"
);
const { modelConfigDraftKey } = drafts;
const fields = await import("../src/features/model-picker/sharing/fields.ts");
const { DEFAULT_PER_MODEL_CONFIG } = await import(
  "../src/features/model-picker/model-config/per-model-config.ts"
);
const { modelConfigHandoffForDestination } = await import(
  "../src/features/model-picker/model-config/model-config-handoff.ts"
);
const { resolveRunConfigTarget } = await import("./helpers/sharing-target.ts");
const links = await import("./helpers/sharing-links.ts");
const { receiverHarness, settle } = await import(
  "./helpers/sharing-receiver.ts"
);
const events = await import("../src/features/auth/session-events.ts");
const run = "unsloth://run?v=1&model=owner/model&nParallel=3";
const browserRun =
  "http://localhost/chat#run?v=1&model=owner/model&nParallel=3";
type Doc = Pick<ReturnType<typeof receiverHarness>, "inbox" | "receiver">;
const nParallel = (doc: Doc) => doc.inbox.getSnapshot()?.value.config.nParallel;
const sharing = (file: string) =>
  new URL(`../src/features/model-picker/sharing/${file}`, import.meta.url);

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
const at = (href: string) => {
  const url = new URL(href, "http://localhost");
  return { href, pathname: url.pathname, searchStr: url.search };
};
const navigateCall = (id?: string, replace = false) => [
  "navigate",
  { to: "/chat", search: { new: id }, replace },
];
const target: Target = {
  id: "owner/Model-GGUF",
  meta: {
    source: "hub",
    isLora: false,
    isGguf: true,
    ggufVariant: "Q4_K_M",
    loadId: "/cache/snapshot",
    isDownloaded: true,
  },
};
const key = modelConfigDraftKey(target.id, "Q4_K_M");
const handoff = (resolved: Target = target, extra = {}) => [
  "handoff",
  { requestId: "first", ...extra, ...resolved, displayName: resolved.id },
];

class ResolutionError extends Error {}

function harness(loadParser?: () => typeof links | Promise<typeof links>) {
  const calls: unknown[] = [];
  const session = sessionHarness({ loadParser });
  const { inbox, receiver } = session.loadDocument();
  const { errors, notices, cleared } = session;
  const loading = new Map<number, string>();
  let toastId = 0;
  const lookups: {
    target: Target;
    signal: AbortSignal;
    checkLocalPath?: boolean;
    result: ReturnType<typeof deferred<Target>>;
  }[] = [];
  inbox.submit({
    id: "first",
    value: { model: "owner/Model-GGUF", config: { nParallel: 3 } },
  });
  const pending = inbox.getSnapshot();
  assert.ok(pending);
  const navigationResult = deferred<void>();
  const runtime = {
    params: { checkpoint: "" },
    activeGgufVariant: null,
    loadedIsGguf: null,
    activeNativePathToken: null,
    activeLoadId: null,
    models: [] as { id: string; isGguf: boolean; isLora: boolean }[],
    loras: [],
    setActiveThreadId: (id: null) => calls.push(["thread", id]),
    setActiveProjectId: (id: null) => calls.push(["project", id]),
    setIncognito: (value: boolean) => calls.push(["incognito", value]),
  };
  const { scheduleRunConfigImport } = loadWithStubs<typeof ImportConfig>(
    sharing("import-config.ts"),
    {
      "@/lib/toast": { toast: { success: () => undefined } },
      "../model-config/model-config-draft": drafts,
      "./fields": fields,
      "./inbox": { runConfigInbox: inbox },
    },
  );
  const lifecycle = loadWithStubs<typeof Lifecycle>(
    sharing("link-lifecycle.ts"),
    {
      "@/features/chat": {
        clearNewChatDraft: () => calls.push("clear draft"),
        useChatRuntimeStore: { getState: () => runtime },
      },
      "@/lib/toast": {
        toast: {
          error: (message: string) => errors.push(message),
          loading: (message: string) =>
            loading.set(++toastId, message) && toastId,
          dismiss: (id: number) => loading.delete(id),
        },
      },
      "../model-config/model-config-draft": { modelConfigDraftKey },
      "../model-config/model-config-handoff": {
        clearModelConfigHandoff: (id: string) => cleared.push(id),
        requestModelConfigHandoff: (request: Target) => {
          assert.equal(
            inbox.getSnapshot()?.draftKey,
            modelConfigDraftKey(request.id, request.meta.ggufVariant),
          );
          calls.push(["handoff", request]);
        },
      },
      "./cached-target": {
        RunConfigResolutionError: ResolutionError,
        resolveCachedRunConfigTarget: (
          lookupTarget: Target,
          options: { signal: AbortSignal; checkLocalPath?: boolean },
        ) => {
          const result = deferred<Target>();
          lookups.push({ target: lookupTarget, ...options, result });
          return result.promise;
        },
      },
      "./inbox": { runConfigInbox: inbox },
      "./target": { resolveRunConfigTarget },
      "./receive-link": receiver,
    },
  );
  const context = {
    pending,
    canOpen: true,
    settingsHydrated: true,
    currentModel: "",
    location: at("/hub"),
  };
  const nav = {
    ...context,
    navigation: { current: null as Lifecycle.RunConfigNavigation | null },
    navigate: (options: unknown) => {
      calls.push(["navigate", options]);
      return navigationResult.promise;
    },
  };
  const open = {
    ...context,
    chatSearch: null as ChatSearch | null,
    location: at("/chat?new=first"),
    routeReady: true,
    inventoryVersion: 0,
  };
  const resolve = async (resolved = target, index = 0) => {
    lookups[index].result.resolve(resolved);
    await settle();
  };
  const lookupFromHub = (extra = {}) =>
    lifecycle.openRunConfigTarget({ ...open, ...context, ...extra });
  return {
    ...lifecycle,
    ...{ runtime, notices, cleared, receiver, inbox, calls, errors },
    ...{ loading, lookups, navigationResult, nav, open, resolve },
    lookupFromHub,
    openCurrent: (extra = {}) =>
      lifecycle.openRunConfigTarget({
        ...open,
        pending: inbox.getSnapshot(),
        ...extra,
      }),
    schedule: (onImport: () => void) =>
      scheduleRunConfigImport({
        canImport: true,
        ready: true,
        hydrated: true,
        key,
        pending: inbox.getSnapshot(),
        onImport,
      }),
    primeDraft: (t: TestContext, nParallel: number) => {
      t.after(drafts.retainModelConfigDraft(key));
      drafts.primeModelConfigDraft(
        key,
        {
          config: { ...DEFAULT_PER_MODEL_CONFIG, nParallel },
          remembered: false,
        },
        "none",
      );
    },
    prepare: async () => {
      lookupFromHub();
      assert.equal(lookups[0].checkLocalPath, false);
      assert.deepEqual([...loading.values()], ["Resolving shared model…"]);
      assert.equal(inbox.getSnapshot()?.draftKey, undefined);
      await resolve();
      assert.equal(loading.size, 0);
      const prepared = inbox.getSnapshot();
      assert.ok(prepared?.target);
      nav.pending = open.pending = prepared;
    },
  };
}

for (const [pathname, newChatId] of [
  ["/chat", null],
  ["/hub", null],
  ["/settings", "current-draft"],
] as const) {
  test(`review keeps the new chat and its draft from ${pathname}: ${newChatId}`, async () => {
    const app = harness();
    app.runtime.params.checkpoint = target.id;
    delete app.nav.pending.value.model;
    const destination = at(newChatId ? `/chat?new=${newChatId}` : "/chat");
    const context = {
      location:
        pathname === "/chat"
          ? destination
          : at(`${pathname}?new=unrelated&project=unrelated`),
      chatSearch: { new: newChatId ?? undefined },
      currentModel: target.id,
    };
    app.navigateRunConfig({ ...app.nav, ...context });
    app.openRunConfigTarget({ ...app.open, ...context });
    await app.resolve();
    const pending = app.inbox.getSnapshot();
    assert.equal(pending?.newChatId, newChatId);
    app.navigateRunConfig({ ...app.nav, ...context, pending });
    assert.deepEqual(
      app.calls.splice(0),
      pathname === "/chat" ? [] : [navigateCall(newChatId ?? undefined)],
    );
    app.openCurrent({ ...context, location: destination });
    const request = handoff(target, { newChatId })[1] as Parameters<
      typeof modelConfigHandoffForDestination
    >[0];
    assert.deepEqual(app.calls, [["handoff", request]]);
    for (const [destinationState, expected] of [
      [{ active: true, newChatId }, request],
      [{ active: true, ...(newChatId && { newChatId }) }, request],
      [{ active: false, newChatId }, null],
      [{ active: true, newChatId: "another-chat" }, null],
      [{ active: true, newChatId, threadId: "thread" }, null],
      [{ active: true, newChatId, compareId: "compare" }, null],
      [{ active: true, newChatId, projectId: "project" }, null],
    ] as const) {
      assert.equal(
        modelConfigHandoffForDestination(request, destinationState),
        expected,
      );
    }
    app.inbox.clear("first");
    assert.equal(app.calls.length, 1);
  });
}

for (const chatSearch of [
  null,
  { thread: "saved-chat" },
  { compare: "comparison" },
  { project: "project" },
]) {
  test(`availability binds the canonical draft in a fresh chat from ${JSON.stringify(chatSearch)}`, async () => {
    const app = harness();
    app.open.chatSearch = chatSearch && { new: "previous", ...chatSearch };
    await app.prepare();
    assert.equal(app.inbox.getSnapshot()?.newChatId, undefined);
    app.navigateRunConfig(app.nav);
    app.openRunConfigTarget({ ...app.open, routeReady: false });
    assert.deepEqual(app.calls, [navigateCall("first")]);
    app.openRunConfigTarget(app.open);
    assert.equal(app.inbox.take("first", "another draft"), null);
    assert.deepEqual(app.calls, [
      navigateCall("first"),
      "clear draft",
      ["thread", null],
      ["project", null],
      ["incognito", false],
      handoff(),
    ]);
    assert.equal(app.inbox.getSnapshot()?.draftKey, key);
    const superseded = chatSearch !== null;
    if (superseded) app.inbox.submit({ id: "second", value: { config: {} } });
    app.navigationResult.reject(new Error("navigation failed"));
    await settle();
    assert.equal(
      app.inbox.getSnapshot()?.id,
      superseded ? "second" : undefined,
    );
    assert.equal(app.errors.length, superseded ? 0 : 1);
    assert.deepEqual(app.cleared, superseded ? [] : ["first"]);
  });
}

test("navigation waits for availability, replaces history once, and ends when the user leaves", async () => {
  const app = harness();
  app.nav.pending.replaceHistory = true;
  app.navigateRunConfig(app.nav);
  assert.deepEqual(app.calls, []);
  await app.prepare();
  app.navigateRunConfig(app.nav);
  app.navigateRunConfig(app.nav);
  assert.deepEqual(app.calls, [navigateCall("first", true)]);
  app.navigateRunConfig({ ...app.nav, location: app.open.location });
  assert.equal(app.nav.navigation.current?.from, app.open.location.href);
  app.navigateRunConfig(app.nav);
  assert.equal(app.inbox.getSnapshot(), null);
});

for (const reason of ["login", "settings", "model", "bound", "superseded"]) {
  test(`navigation and lookup do nothing while ${reason} blocks the import`, () => {
    const app = harness();
    const patch = {
      canOpen: reason !== "login",
      settingsHydrated: reason !== "settings",
    };
    if (reason === "model") delete app.nav.pending.value.model;
    if (reason === "bound") app.nav.pending.draftKey = "already bound";
    if (reason === "superseded") app.inbox.clear("first");
    app.navigateRunConfig({ ...app.nav, ...patch });
    assert.equal(app.openRunConfigTarget({ ...app.open, ...patch }), undefined);
    assert.deepEqual([app.calls, app.lookups, app.loading.size], [[], [], 0]);
  });
}

test("failed lookups cancel the import with safe feedback", async () => {
  for (const [error, shown] of [
    [
      new Error("Private backend details"),
      "Could not resolve the shared model. Reopen the link to try again.",
    ],
    [new ResolutionError("Variant unavailable."), "Variant unavailable."],
  ] as const) {
    const app = harness();
    app.lookupFromHub();
    app.lookups[0].result.reject(error);
    await settle();
    assert.deepEqual([app.inbox.getSnapshot(), app.loading.size], [null, 0]);
    assert.deepEqual([app.errors, app.calls], [[shown], []]);
  }
});

test("a chosen model known to be safetensors is refused before any lookup", () => {
  const app = harness();
  app.runtime.models = [{ id: "owner/native", isGguf: false, isLora: false }];
  app.inbox.submit({
    id: "settings-only",
    value: { config: { nParallel: 3 } },
    selectedModel: "owner/native",
  });
  const cancel = app.lookupFromHub({ pending: app.inbox.getSnapshot() });
  assert.deepEqual(
    [cancel, app.inbox.getSnapshot(), app.lookups],
    [undefined, null, []],
  );
  assert.deepEqual(app.errors, [
    "Shared run settings apply only to GGUF models. Reopen the link and choose a GGUF model.",
  ]);
});

for (const failure of [false, true]) {
  for (const interrupt of ["cleanup", "newer link"] as const) {
    test(`stale availability ${failure ? "failures" : "successes"} after ${interrupt} are ignored`, async () => {
      const app = harness();
      app.navigateRunConfig(app.nav);
      const cancel = app.lookupFromHub();
      if (interrupt === "cleanup") {
        cancel?.();
        assert.equal(app.loading.size, 0);
        assert.equal(app.lookups[0].signal.aborted, true);
        app.lookupFromHub();
      } else app.inbox.submit({ id: "second", value: { config: {} } });
      if (failure) app.lookups[0].result.reject(new Error("stale"));
      else app.lookups[0].result.resolve(target);
      await settle();
      assert.deepEqual([app.calls, app.errors], [[], []]);
      const cleanup = interrupt === "cleanup";
      assert.equal(app.loading.size, cleanup ? 1 : 0);
      if (cleanup) await app.resolve(target, 1);
      assert.equal(app.loading.size, 0);
      const snapshot = app.inbox.getSnapshot();
      assert.equal(snapshot?.target, cleanup ? target : undefined);
      assert.equal(snapshot?.id, cleanup ? "first" : "second");
    });
  }
}

for (const phase of [
  "parsing",
  "resolving",
  "resolved",
  "scheduled",
  "other quant",
] as const) {
  test(`edits during a shared import: ${phase}`, async (t) => {
    const parser = deferred<typeof links>();
    const app = harness(() => parser.promise);
    const matching = phase !== "other quant";
    app.primeDraft(t, matching ? 1 : 7);
    const edit = (when: string) => {
      if (when !== phase && matching) return;
      app.receiver.cancelRunConfigImportForEdit(
        matching ? key : modelConfigDraftKey(target.id, "Q8_0"),
      );
      if (!matching) return;
      drafts.markModelConfigDraftEdited(key);
      drafts.patchModelConfigDraft(key, { nParallel: 7 });
    };
    app.receiver.receiveSharedRunConfigUrls([
      "unsloth://run?v=1&model=owner/Model-GGUF&nParallel=3",
    ]);
    assert.equal(app.inbox.getSnapshot(), null);
    edit("parsing");
    parser.resolve(links);
    await settle();
    app.openCurrent();
    edit("resolving");
    await app.resolve();
    edit("resolved");
    if (app.inbox.getSnapshot()) {
      app.openCurrent({ location: at("/chat?new=request-1") });
      app.schedule(() => assert.ok(!matching, "A newer edit was overwritten"));
      edit("scheduled");
      await settle();
    }
    assert.equal(
      drafts.readModelConfigDraft(key)?.config.nParallel,
      matching ? 7 : 3,
    );
    assert.equal(drafts.isModelConfigDraftEdited(key), true);
    assert.equal(app.inbox.getSnapshot(), null);
    assert.deepEqual(
      app.notices.map(({ message }) => message),
      matching ? ["Run settings import cancelled"] : [],
    );
    assert.deepEqual([app.errors, app.loading.size], [[], 0]);
  });
}

function sessionHarness({
  url = browserRun,
  desktop = false,
  loadParser = (): typeof links | Promise<typeof links> => links,
} = {}) {
  const browser = installLocalStorageFake();
  const storage = new Map<string, string>();
  const getItem = (name: string) => storage.get(name) ?? null;
  const setItem = storage.set.bind(storage);
  Object.assign(globalThis, { sessionStorage: { getItem, setItem } });
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
  let parserLoads = 0;
  const errors: string[] = [];
  const notices: { message: string; description?: string }[] = [];
  const cleared: string[] = [];
  const signal = (value: boolean) => {
    signedIn = value;
    const { AUTH_SESSION_STORED_EVENT: on, AUTH_SESSION_CLEARED_EVENT: off } =
      events;
    browser.fireWindowEvent(value ? on : off, {});
  };
  const loadDocument = () => {
    const { inbox, receiver } = receiverHarness({
      desktop,
      signedIn: () => signedIn,
      drafts,
      ...{ errors, notices, cleared },
      loadParser: () => (parserLoads++, loadParser()),
    });
    const dispose = receiver.subscribeRunConfigSession(() => undefined);
    return { inbox, receiver, dispose };
  };
  return {
    ...{ loadDocument, storage, errors, notices, cleared },
    parserLoads: () => parserLoads,
    signIn: () => signal(true),
    signOut: () => signal(false),
    receive: async (doc: Doc, source: "startup" | "event" = "event") => {
      if (desktop) {
        doc.receiver.receiveSharedRunConfigUrls([run], source);
        await settle();
      } else await doc.receiver.receiveStartupRunConfigUrl();
    },
  };
}

for (const [desktop, before, after] of [
  [true, false, false],
  [false, false, false],
  [true, true, false],
  [false, false, true],
] as const) {
  test(`session changes keep or cancel a pending link once: desktop=${desktop} signedIn=${before}/${after}`, async () => {
    const app = sessionHarness({ desktop });
    if (before) app.signIn();
    const doc = app.loadDocument();
    await app.receive(doc, "startup");
    if (after) app.signIn();
    const pending = doc.inbox.getSnapshot();
    assert.ok(pending);
    if (before) doc.inbox.bind(pending.id, "draft");
    const live = before || after;
    app.signOut();
    assert.equal(doc.inbox.getSnapshot(), live ? null : pending);
    assert.deepEqual(app.cleared, live ? [pending.id] : []);
    assert.equal(app.notices.length, live ? 1 : 0);
    app.storage.clear();
    app.signOut();
    app.signIn();
    assert.equal(doc.inbox.getSnapshot(), live ? null : pending);
    assert.equal(app.notices.length, live ? 1 : 0);
    if (live) {
      assert.equal(app.notices[0].message, "Run settings import cancelled");
      assert.match(
        app.notices[0].description ?? "",
        /session.*Reopen the link/is,
      );
    }
    doc.dispose();
    const reloaded = app.loadDocument();
    if (desktop) {
      const replay = reloaded.receiver.receiveSharedRunConfigUrls(
        [run],
        "startup",
      );
      assert.equal(replay, "ignored");
    }
    await reloaded.receiver.receiveStartupRunConfigUrl();
    assert.equal(reloaded.inbox.getSnapshot(), null);
    assert.equal(app.parserLoads(), 1);
    if (desktop) {
      await app.receive(doc);
      assert.equal(nParallel(doc), 3);
      assert.notEqual(doc.inbox.getSnapshot()?.id, pending.id);
    }
    window.location.href = browserRun;
    const reopened = desktop ? reloaded : app.loadDocument();
    await app.receive(reopened);
    assert.equal(nParallel(reopened), 3);
    assert.deepEqual(app.errors, []);
  });
}

const argsQuery = "llamaExtraArgs=%5B%22--threads%22%2C%224%22%5D";
for (const [url, cleaned, valid, decode] of [
  ["?run=1&keep=value#run?v=1&nParallel=3", "?keep=value", true, false],
  ["?keep=value#run?v=1&unknown=true", "?keep=value", false, false],
  ["?run=2&keep=value#run", "?run=2&keep=value", false, false],
  ["?run=1#run?v=1&selectedGpuIds=%5B0%5D", "", true, true],
  [`?run=1#run?v=1&${argsQuery}`, "", true, true],
  ["?run=1#run?v=1&selectedGpuIds=%5B1%2C1%5D", "", false, true],
] as const) {
  test(`startup consumes ${url} once (router decoding: ${decode})`, async () => {
    const app = sessionHarness({ url: `http://localhost/chat${url}` });
    const doc = app.loadDocument();
    if (decode) {
      window.location.href = window.location.href.replace(
        /%5B|%5D|%7B|%7D/gi,
        decodeURIComponent,
      );
    }
    await doc.receiver.receiveStartupRunConfigUrl();
    assert.equal(window.location.href, `http://localhost/chat${cleaned}`);
    const pending = doc.inbox.getSnapshot();
    assert.equal(pending?.replaceHistory, valid || undefined);
    assert.equal(app.errors.length, valid ? 0 : 1);
    doc.inbox.clear(pending?.id);
    doc.dispose();
    const reloaded = app.loadDocument();
    await reloaded.receiver.receiveStartupRunConfigUrl();
    assert.equal(reloaded.inbox.getSnapshot(), null);
    assert.equal(app.errors.length, valid ? 0 : 1);
  });
}

test("desktop startup ignores web fragments and still accepts native links", async () => {
  const app = sessionHarness({
    url: "tauri://localhost/chat#run?v=1&model=owner/model&nParallel=4",
    desktop: true,
  });
  const doc = app.loadDocument();
  await doc.receiver.receiveStartupRunConfigUrl();
  assert.equal(doc.inbox.getSnapshot(), null);
  for (const url of [browserRun, "unsloth://open_from_hf?model=owner/other"]) {
    assert.equal(doc.receiver.receiveSharedRunConfigUrls([url]), false);
  }
  const urls = [run, "unsloth://unrelated"];
  assert.equal(doc.receiver.receiveSharedRunConfigUrls(urls), true);
  await settle();
  assert.deepEqual(
    [nParallel(doc), doc.inbox.getSnapshot()?.replaceHistory, app.errors],
    [3, false, []],
  );
});

const newerRun = "unsloth://run?v=1&model=owner/newer&nParallel=4";
const newerHub = "unsloth://open_from_hf?model=owner/newer";
for (const [name, native, signIn, during, model] of [
  ["a newer run link wins", false, true, newerRun, "owner/newer"],
  ["a newer hub link wins", false, true, newerHub, null],
  ["sign-out retires the load", true, true, "relogin", null],
  ["web edits survive", false, true, null, "owner/model"],
  ["desktop edits survive", true, true, null, "owner/model"],
  ["a pre-login link survives", false, false, "relogin", "owner/model"],
] as const) {
  test(`a delayed parser: ${name}`, async () => {
    const parser = deferred<typeof links>();
    const app = sessionHarness({
      desktop: native,
      loadParser: () => parser.promise,
    });
    if (signIn) app.signIn();
    const doc = app.loadDocument();
    doc.receiver.cancelRunConfigImportForEdit("older-draft");
    const intake = native
      ? doc.receiver.receiveSharedRunConfigUrls([run])
      : doc.receiver.receiveStartupRunConfigUrl();
    if (native) assert.equal(intake, true);
    else assert.equal(window.location.href, "http://localhost/chat");
    assert.equal(doc.inbox.getSnapshot(), null);
    doc.receiver.cancelRunConfigImportForEdit("edited-draft");
    if (during === "relogin") {
      app.signOut();
      app.signIn();
    } else if (during) doc.receiver.receiveSharedRunConfigUrls([during]);
    parser.resolve(links);
    await intake;
    await settle();
    assert.equal(doc.inbox.getSnapshot()?.value.model ?? null, model);
    assert.equal(doc.inbox.wasEdited("edited-draft"), model === "owner/model");
    assert.equal(doc.inbox.wasEdited("older-draft"), false);
    assert.deepEqual(app.errors, []);
  });
}

test("a failed parser chunk reports an error and permits reopening the same native link", async () => {
  const parser = deferred<typeof links>();
  let retry = false;
  const app = sessionHarness({
    desktop: true,
    loadParser: () => (retry ? links : parser.promise),
  });
  app.signIn();
  const doc = app.loadDocument();
  await app.receive(doc);
  parser.reject(new Error("Chunk unavailable"));
  await settle();
  assert.equal(app.errors.length, 1);
  assert.equal(doc.inbox.getSnapshot(), null);
  retry = true;
  await app.receive(doc);
  assert.equal(nParallel(doc), 3);
});
