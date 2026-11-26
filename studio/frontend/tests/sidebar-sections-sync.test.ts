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
  removeItem(key: string) { data.delete(key); },
};
const window = Object.assign(new EventTarget(), {
  localStorage,
  location: new URL("https://studio.example/"),
  setInterval,
  clearInterval,
});
const document = Object.assign(new EventTarget(), { visibilityState: "visible" });
Object.assign(globalThis, { localStorage, window, document });

const layouts = await import("../src/features/chat/sidebar-layout.ts");
const stores = await import("../src/features/chat/stores/sidebar-organization-store.ts");
const sidebarStorage = await import("../src/features/chat/stores/sidebar-organization-storage.ts");
const sidebarKeys = await import("../src/features/chat/stores/sidebar-organization-keys.ts");
const accounts = await import("../src/lib/account-transition.ts");
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
      "@/lib/account-transition": accounts,
      "@/lib/desktop-update-activity": { isBackendDownForDesktopUpdate: () => false },
      "@/lib/navigation-nonce": { createNavigationNonce: randomUUID },
      "../sidebar-layout": layouts,
      "../stores/sidebar-organization-store": stores,
      "../stores/sidebar-organization-storage": sidebarStorage,
    },
  );
  hook.useSidebarSectionsSync();
}

function writeFromPeer(update: Record<string, unknown>, notify = true) {
  const key = stores.SIDEBAR_ORGANIZATION_STORAGE_KEY;
  const saved = JSON.parse(localStorage.getItem(key) ?? "{}");
  localStorage.setItem(key, JSON.stringify({ ...saved, state: { ...saved.state, ...update } }));
  if (notify) window.dispatchEvent(Object.assign(new Event("storage"), { key }));
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

test("sidebar sync survives its account marker upgrade and rejects a different account", async (t) => {
  const key = accounts.BROWSER_ACCOUNT_KEY;
  localStorage.setItem(key, "alice");
  t.after(() => { data.delete(key); });
  const methods: string[] = [];
  const initial = layouts.sidebarLayout(store.getInitialState());
  mount(t, async (_url, init) => {
    methods.push(init.method ?? "GET");
    if (init.method !== "POST") return Response.json(initial);
    return Response.json(JSON.parse(String(init.body)).layout);
  });
  await setImmediate();
  assert.deepEqual(methods, ["GET"]);

  localStorage.setItem(key, accounts.browserAccountMarker({ username: "alice", accountId: "one" }));
  store.getState().createCustomSection("Research");
  await setImmediate();
  window.dispatchEvent(new Event("focus"));
  await setImmediate();
  assert.deepEqual(methods, ["GET", "POST", "GET"]);

  localStorage.setItem(key, accounts.browserAccountMarker({ username: "alice", accountId: "two" }));
  store.getState().createCustomSection("Other account");
  window.dispatchEvent(new Event("focus"));
  await setImmediate();
  assert.deepEqual(methods, ["GET", "POST", "GET"]);
});

test("a reload after an older tab's write restores the same account's sync baseline", async (t) => {
  const accountKey = accounts.BROWSER_ACCOUNT_KEY;
  localStorage.setItem(accountKey, accounts.browserAccountMarker({ username: "alice", accountId: "legacy-edit" }));
  t.after(() => {
    data.delete(accountKey);
    data.delete(sidebarKeys.SIDEBAR_SYNC_BASELINE_KEY);
  });
  const initial = {
    ...layouts.sidebarLayout(store.getInitialState()),
    sections: { research: { name: "Research", sort: "manual" as const } },
    sectionOrder: ["pinned", "research", "projects"],
  };
  mount(t, async () => Response.json(initial));
  await setImmediate();
  const synced = store.getState().syncedLayout;
  assert.ok(synced);
  const key = stores.SIDEBAR_ORGANIZATION_STORAGE_KEY;
  const legacy = JSON.stringify({ state: {
    customSections: [{ id: "research", name: "Renamed in older tab", sort: "manual" }],
    sectionOrder: initial.sectionOrder,
  }, version: 0 });

  store.setState(store.getInitialState(), true);
  localStorage.setItem(key, legacy);
  await store.persist.rehydrate();
  assert.equal(store.getState().customSections[0].name, "Renamed in older tab");
  assert.equal(store.getState().syncedLayout, synced);

  localStorage.setItem(accountKey, accounts.browserAccountMarker({ username: "alice", accountId: "different" }));
  store.setState(store.getInitialState(), true);
  localStorage.setItem(key, legacy);
  await store.persist.rehydrate();
  assert.equal(store.getState().syncedLayout, undefined);
});

test("browser storage restores space for the pending edit before keeping a recovery copy", (t) => {
  const accountKey = accounts.BROWSER_ACCOUNT_KEY;
  const baselineKey = sidebarKeys.SIDEBAR_SYNC_BASELINE_KEY;
  const layout = JSON.stringify(layouts.sidebarLayout(store.getInitialState()));
  localStorage.setItem(accountKey, accounts.browserAccountMarker({ username: "alice", accountId: "quota" }));
  t.after(() => { data.delete(accountKey); data.delete(baselineKey); });
  sidebarStorage.rememberSidebarSyncBaseline(layout);
  assert.ok(localStorage.getItem(baselineKey));
  const storage = sidebarStorage.sidebarOrganizationStorage();
  const key = stores.SIDEBAR_ORGANIZATION_STORAGE_KEY;
  const originalSetItem = localStorage.setItem;
  t.mock.method(localStorage, "setItem", (name: string, value: string) => {
    if (name === key && localStorage.getItem(baselineKey)) {
      throw new DOMException("Storage quota exceeded", "QuotaExceededError");
    }
    originalSetItem(name, value);
  });

  const pending = JSON.stringify({ state: { pendingLayoutEdit: { id: "pending", layout } }, version: 0 });
  storage.setItem(key, pending);
  assert.equal(localStorage.getItem(key), pending);
  assert.equal(localStorage.getItem(baselineKey), null);
  assert.equal(sidebarStorage.readSidebarSyncBaseline(), layout);
});

async function checkRejectedSidebarEdit(t: TestContext, status: number) {
  const posts: Edit[] = [];
  const initial = layouts.sidebarLayout(store.getInitialState());
  let rejectFirst: ((response: Response) => void) | undefined;
  mount(t, async (_url, init) => {
    if (init.method !== "POST") return Response.json(initial);
    const edit: Edit = JSON.parse(String(init.body));
    posts.push(edit);
    return posts.length === 1
      ? new Promise<Response>((resolve) => { rejectFirst = resolve; })
      : Response.json(edit.layout);
  });
  await setImmediate();
  const rejected = store.getState().createCustomSection("Rejected section")!;
  const rows = ["row-a", "row-b"];
  store.getState().setChatsSection(rows, rejected);
  store.getState().setManualOrder(stores.customSectionScope(rejected), rows, [
    { id: rows[0], targetId: rows[1], edge: "top" },
  ]);
  store.getState().moveSection("projects", "pinned", "top");
  await setImmediate();
  assert.equal(posts.length, 1);
  assert.deepEqual(posts[0].layout.rowOrder[rejected], rows);

  store.getState().deleteCustomSection(rejected);
  const corrected = store.getState().createCustomSection("Research")!;
  await setImmediate();
  rejectFirst!(new Response(null, { status }));
  await setImmediate();

  assert.equal(posts.length, 2);
  assert.notEqual(posts[1].operationId, posts[0].operationId);
  assert.deepEqual(posts[1].base, JSON.parse(JSON.stringify(initial)));
  assert.deepEqual(Object.keys(posts[1].layout.sections), [corrected]);
  assert.deepEqual(posts[1].orderMoves, [
    { id: "projects", targetId: "pinned", edge: "top", sectionId: null },
  ]);
  assert.equal(store.getState().pendingLayoutEdit, undefined);
}

for (const status of [413, 422]) {
  test(`a sidebar edit rejected with ${status} does not block the corrected layout`,
    (t) => checkRejectedSidebarEdit(t, status));
}

test("a merged layout can be reduced when its saved baseline exceeds the request limit", async (t) => {
  const gone = "research-section-with-a-uuid";
  const retained = "retained-section-with-a-uuid";
  const rowIds = Array.from({ length: 10_000 }, (_, index) =>
    `00000000-0000-4000-8000-${String(index).padStart(12, "0")}`,
  );
  const initial = {
    ...layouts.sidebarLayout(store.getInitialState()),
    sections: {
      [gone]: { name: "Research", sort: "manual" as const },
      [retained]: { name: "Inbox", sort: "manual" as const },
    },
    sectionByChatId: { ...Object.fromEntries(rowIds.map((id) => [id, gone])), inbox: retained },
    sectionByProjectId: { project: gone },
    sectionByPageId: { page: gone },
    rowOrder: { [gone]: rowIds, [retained]: ["inbox"] },
    sectionOrder: ["pinned", gone, retained, "projects"],
  };
  const corrected = {
    ...initial,
    sections: { [retained]: initial.sections[retained] },
    sectionByChatId: { inbox: retained },
    sectionByProjectId: {},
    sectionByPageId: {},
    rowOrder: { [retained]: ["inbox"] },
    sectionOrder: ["pinned", retained, "projects"],
  };
  const posts: Edit[] = [];
  mount(t, async (_url, init) => {
    if (init.method !== "POST") return Response.json(initial);
    const body = String(init.body);
    posts.push(JSON.parse(body));
    return Buffer.byteLength(body) > 1024 * 1024
      ? new Response(null, { status: 413 })
      : Response.json(corrected);
  });
  await setImmediate();

  store.getState().deleteCustomSection(gone);
  await setImmediate();
  assert.equal(posts.length, 2);
  assert.equal(posts[1].operationId, posts[0].operationId);
  assert.deepEqual(posts[1].base?.sectionByChatId, {});
  assert.deepEqual(posts[1].base?.sectionByProjectId, {});
  assert.deepEqual(posts[1].base?.sectionByPageId, {});
  assert.deepEqual(posts[1].base?.rowOrder, {});
  assert.deepEqual(JSON.parse(JSON.stringify(layouts.sidebarLayout(store.getState()))), corrected);
  assert.equal(store.getState().pendingLayoutEdit, undefined);
});

test("renaming a section can omit historical ranks from an over-limit retry", async (t) => {
  const section = "research";
  const rows = Array.from({ length: 14_000 }, (_, index) =>
    `00000000-0000-4000-8000-${String(index).padStart(12, "0")}`,
  );
  const initial = {
    ...layouts.sidebarLayout(store.getInitialState()),
    sections: { research: { name: "Research", sort: "manual" as const } },
    rowOrder: { research: rows },
    sectionOrder: ["pinned", section, "projects"],
  };
  const renamed = { ...initial, sections: { research: { ...initial.sections.research, name: "Notes" } } };
  const posts: Edit[] = [];
  mount(t, async (_url, init) => {
    if (init.method !== "POST") return Response.json(initial);
    const body = String(init.body);
    posts.push(JSON.parse(body));
    return Buffer.byteLength(body) > 1024 * 1024
      ? new Response(null, { status: 413 })
      : Response.json(renamed);
  });
  await setImmediate();

  store.getState().renameCustomSection(section, "Notes");
  await setImmediate();
  assert.equal(posts.length, 2);
  assert.equal(posts[1].operationId, posts[0].operationId);
  assert.deepEqual(posts[1].base?.rowOrder, {});
  assert.deepEqual(posts[1].layout.rowOrder, {});
  assert.deepEqual(layouts.sidebarLayout(store.getState()).rowOrder, initial.rowOrder);
  assert.equal(store.getState().customSections[0].name, "Notes");
  assert.equal(store.getState().pendingLayoutEdit, undefined);
});

test("explicit unpin placements sync even if they reuse the saved ranks", async (t) => {
  const section = "research";
  const scope = stores.customSectionScope(section);
  const rows = ["a", "b", "c", "d"];
  const initial = {
    ...layouts.sidebarLayout(store.getInitialState()),
    sections: { research: { name: "Research", sort: "manual" as const } },
    sectionByChatId: Object.fromEntries(rows.map((id) => [id, section])),
    rowOrder: { research: rows },
    sectionOrder: ["pinned", "research", "projects"],
  };
  const posts: Edit[] = [];
  let finishFirst!: (response: Response) => void;
  const first = new Promise<Response>((resolve) => { finishFirst = resolve; });
  mount(t, async (_url, init) => {
    if (init.method !== "POST") return Response.json(initial);
    const post: Edit = JSON.parse(String(init.body));
    posts.push(post);
    if (posts.length === 3 || post.operationId === posts[2]?.operationId) {
      return new Response(null, { status: 413 });
    }
    return posts.length === 1 ? first : Response.json(post.layout);
  });
  await setImmediate();

  const firstMove = { id: "b", targetId: "a", edge: "bottom", afterUnpin: true } as const;
  const secondMove = { id: "d", targetId: "c", edge: "bottom", afterUnpin: true } as const;
  store.getState().setManualOrder(scope, rows, [firstMove]);
  await setImmediate();
  assert.equal(posts.length, 1);
  assert.deepEqual(posts[0].orderMoves, [{ ...firstMove, sectionId: section }]);

  store.getState().setManualOrder(scope, rows, [secondMove]);
  finishFirst(Response.json(initial));
  await setImmediate();
  assert.equal(posts.length, 2);
  assert.deepEqual(posts[1].orderMoves, [{ ...secondMove, sectionId: section }]);

  store.getState().setManualOrder(scope, rows, [firstMove]);
  await setImmediate();
  assert.equal(posts.length, 4);
  assert.equal(posts[2].operationId, posts[3].operationId);
  assert.deepEqual(posts[3].base?.rowOrder, { [section]: rows });
  assert.deepEqual(posts[3].layout.rowOrder, { [section]: rows });
  assert.equal(store.getState().pendingLayoutEdit, undefined);
  assert.deepEqual(store.getState().orderMoves, [{ ...firstMove, sectionId: section }]);
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

test("a refresh preserves a peer filing whose storage event has not arrived", async (t) => {
  let now = 1_000;
  t.mock.method(Date, "now", () => now);
  const initial = {
    ...layouts.sidebarLayout(store.getInitialState()),
    sections: { research: { name: "Research", sort: "manual" as const } },
    sectionOrder: ["pinned", "research", "projects"],
  };
  const posts: Edit[] = [];
  mount(t, async (_url, init) => {
    if (init.method !== "POST") return Response.json(initial);
    const post: Edit = JSON.parse(String(init.body));
    posts.push(post);
    return Response.json(post.layout);
  });
  await setImmediate();

  store.getState().renameCustomSection("research", "Notes");
  writeFromPeer({ sectionByChatId: { chat: "research" } }, false);
  await setImmediate();
  assert.equal(store.getState().sectionByChatId.chat, "research");
  assert.equal(posts.length, 0);

  now += 30_001;
  window.dispatchEvent(new Event("focus"));
  await setImmediate();
  assert.equal(posts.length, 1);
  assert.deepEqual(posts[0].layout.sectionByChatId, { chat: "research" });
  assert.equal(posts[0].layout.sections.research.name, "Notes");
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
