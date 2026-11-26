// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";
import { setImmediate } from "node:timers/promises";
import test, { type TestContext } from "node:test";

import { loadWithStubs } from "./helpers/module-stubs.ts";

const data = new Map<string, string>();
const quota = 5 * 1024 * 1024;
const localStorage = {
  getItem: (key: string) => data.get(key) ?? null,
  setItem(key: string, value: string) {
    if (key.length + value.length > quota) {
      throw new DOMException("Storage quota exceeded", "QuotaExceededError");
    }
    data.set(key, value);
  },
};
const window = Object.assign(new EventTarget(), {
  localStorage,
  setInterval,
  clearInterval,
});
const document = Object.assign(new EventTarget(), { visibilityState: "visible" });
Object.assign(globalThis, { localStorage, window, document });

const layouts = await import("../src/features/chat/sidebar-layout.ts");
const stores = await import("../src/features/chat/stores/sidebar-organization-store.ts");
const store = stores.useSidebarOrganizationStore;
type Layout = ReturnType<typeof layouts.sidebarLayout>;
type Edit = {
  operationId: string;
  base: Layout | null;
  layout: Layout;
  orderMoves?: unknown[];
};

function mount(
  t: TestContext,
  authFetch: (url: string, init: RequestInit) => Promise<Response>,
) {
  store.setState(store.getInitialState(), true);
  store.setState({ pendingLayoutEdit: undefined, syncedLayout: undefined });
  const hook = loadWithStubs<{ useSidebarSectionsSync(): void }>(
    new URL("../src/features/chat/hooks/use-sidebar-sections-sync.ts", import.meta.url),
    {
      react: { useEffect: (effect: () => () => void) => t.after(effect()) },
      sonner: { toast: { error: () => {} } },
      "@/features/auth": {
        AUTH_SESSION_STORED_EVENT: "auth-stored",
        authFetch,
        ensureAuthSessionMark: () => "test-session",
        getAuthSessionMark: () => "test-session",
        hasSettledAuthSession: () => true,
      },
      "@/lib/account-transition": {
        accountTransitionPending: () => false,
        BROWSER_ACCOUNT_FENCE_KEY: "auth-fence",
        browserAccountKey: () => "test-account",
      },
      "@/lib/desktop-update-activity": { isBackendDownForDesktopUpdate: () => false },
      "@/lib/navigation-nonce": { createNavigationNonce: () => "next-edit" },
      "../sidebar-layout": layouts,
      "../stores/sidebar-organization-store": stores,
    },
  );
  hook.useSidebarSectionsSync();
}

function writeFromPeer(update: Record<string, unknown>) {
  const key = stores.SIDEBAR_ORGANIZATION_STORAGE_KEY;
  const saved = JSON.parse(localStorage.getItem(key) ?? "{}");
  localStorage.setItem(key, JSON.stringify({ ...saved, state: { ...saved.state, ...update } }));
  window.dispatchEvent(Object.assign(new Event("storage"), { key }));
}

test("a large saved sidebar keeps its pending sync operation within browser storage quota", () => {
  const key = stores.SIDEBAR_ORGANIZATION_STORAGE_KEY;
  const section = { id: randomUUID(), name: "Research", sort: "manual" as const };
  const chatIds = Array.from({ length: 20_000 }, () => randomUUID());
  const legacy = {
    customSections: [section],
    sectionByChatId: Object.fromEntries(chatIds.map((id) => [id, section.id])),
    manualOrder: { [stores.customSectionScope(section.id)]: chatIds },
  };
  localStorage.setItem(key, JSON.stringify({ state: legacy, version: 0 }));
  store.persist.rehydrate();
  const syncedLayout = JSON.stringify(layouts.sidebarLayout(store.getState()));
  store.setState({ syncedLayout });
  store.getState().renameCustomSection(section.id, "Field notes");
  const pendingLayoutEdit = {
    id: "large-edit",
    layout: JSON.stringify(layouts.sidebarLayout(store.getState())),
  };
  store.setState({ pendingLayoutEdit });
  const persisted = localStorage.getItem(key)!;
  assert.ok(persisted.length < quota);

  store.setState(store.getInitialState(), true);
  localStorage.setItem(key, persisted);
  store.persist.rehydrate();
  assert.ok(store.getState().syncedLayout === syncedLayout);
  assert.equal(store.getState().pendingLayoutEdit?.id, pendingLayoutEdit.id);
  assert.ok(store.getState().pendingLayoutEdit?.layout === pendingLayoutEdit.layout);
  assert.equal(store.getState().customSections[0].name, "Field notes");
  assert.equal(store.getState().sectionByChatId[chatIds.at(-1)!], section.id);
});

test("a peer can upload a later edit after retrying the closed tab's pending operation", async (t) => {
  const posts: Edit[] = [];
  const initial = layouts.sidebarLayout(store.getInitialState());
  mount(t, async (_url, init) => {
    if (init.method !== "POST") return Response.json(initial);
    const post = JSON.parse(String(init.body));
    posts.push(post);
    return Response.json(post.layout);
  });
  await setImmediate();

  const research = { name: "Research", sort: "manual" as const };
  const creation = {
    ...initial,
    sections: { research },
    sectionOrder: ["pinned", "research", "projects"],
  };
  writeFromPeer({
    customSections: [{ ...research, id: "research" }],
    sectionOrder: creation.sectionOrder,
    pendingLayoutEdit: { id: "first-edit", layout: JSON.stringify(creation) },
  });
  writeFromPeer({ sectionByChatId: { chat: "research" } });
  await setImmediate();
  assert.equal(posts.length, 0);

  window.dispatchEvent(new Event("focus"));
  await setImmediate();
  assert.equal(posts.length, 2);
  assert.equal(posts[0].operationId, "first-edit");
  assert.deepEqual(posts[1].base, JSON.parse(JSON.stringify(creation)));
  assert.deepEqual(posts[1].layout.sectionByChatId, { chat: "research" });
});

test("a surviving peer uploads a mirrored edit that never acquired a pending operation", async (t) => {
  let now = 1_000;
  t.mock.method(Date, "now", () => now);
  const posts: Edit[] = [];
  const initial = layouts.sidebarLayout(store.getInitialState());
  mount(t, async (_url, init) => {
    if (init.method !== "POST") return Response.json(initial);
    const post: Edit = JSON.parse(String(init.body));
    posts.push(post);
    return Response.json(post.layout);
  });
  await setImmediate();
  writeFromPeer({
    customSections: [{ name: "Research", sort: "manual", id: "research" }],
    sectionByChatId: { chat: "research" },
    sectionOrder: ["pinned", "research", "projects"],
  });
  assert.equal(store.getState().pendingLayoutEdit, undefined);

  window.dispatchEvent(new Event("focus"));
  await setImmediate();
  assert.equal(posts.length, 0);

  now += 30_001;
  window.dispatchEvent(new Event("focus"));
  await setImmediate();
  assert.equal(posts.length, 1);
  assert.deepEqual(posts[0].layout.sectionByChatId, { chat: "research" });
});

test("a reorder during the initial load uses an edit instead of a legacy import", async (t) => {
  const posts: Edit[] = [];
  const initial = layouts.sidebarLayout(store.getInitialState());
  const remote = {
    ...initial,
    sections: { research: { name: "Research", sort: "manual" as const } },
    sectionOrder: ["pinned", "research", "projects"],
  };
  let resolveGet!: (response: Response) => void;
  const initialGet = new Promise<Response>((resolve) => {
    resolveGet = resolve;
  });
  mount(t, async (_url, init) => {
    if (init.method !== "POST") return initialGet;
    const post = JSON.parse(String(init.body));
    posts.push(post);
    return Response.json({ ...remote, sectionOrder: ["projects", "pinned", "research"] });
  });

  store.getState().moveSection("projects", "pinned", "top");
  await setImmediate();
  resolveGet(Response.json(remote));
  await setImmediate();

  assert.equal(posts.length, 1);
  assert.deepEqual(posts[0].base, JSON.parse(JSON.stringify(initial)));
  assert.deepEqual(posts[0].layout.sectionOrder, ["projects", "pinned"]);
  assert.deepEqual(posts[0].orderMoves, [{
    sectionId: null, id: "projects", targetId: "pinned", edge: "top",
  }]);
  assert.equal(store.getState().sectionOrder[0], "projects");
});

test("an unrelated save drops moves that were undone while the last write was pending", async (t) => {
  const posts: Edit[] = [];
  const initial = {
    ...layouts.sidebarLayout(store.getInitialState()),
    sections: { research: { name: "Research", sort: "manual" as const } },
    sectionOrder: ["pinned", "research", "projects"],
  };
  let resolveFirstPost!: (response: Response) => void;
  const firstPost = new Promise<Response>((resolve) => {
    resolveFirstPost = resolve;
  });
  mount(t, async (_url, init) => {
    if (init.method !== "POST") return Response.json(initial);
    const post: Edit = JSON.parse(String(init.body));
    posts.push(post);
    return posts.length === 1 ? firstPost : Response.json(post.layout);
  });
  await setImmediate();
  store.getState().renameCustomSection("research", "Notes");
  await setImmediate();
  store.getState().moveSection("projects", "pinned", "top");
  store.getState().moveSection("projects", "research", "bottom");
  const remoteOrder = ["projects", "pinned", "research"];
  resolveFirstPost(Response.json({ ...posts[0].layout, sectionOrder: remoteOrder }));
  await setImmediate();
  assert.deepEqual(store.getState().sectionOrder, remoteOrder);

  store.getState().renameCustomSection("research", "Drafts");
  await setImmediate();
  assert.equal(posts.length, 2);
  assert.equal(posts[1].orderMoves, undefined);
  assert.deepEqual(posts[1].layout.sectionOrder, remoteOrder);
});
