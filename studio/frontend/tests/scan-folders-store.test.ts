// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test, { after } from "node:test";
import { installLocalStorageFake } from "./helpers/kit.ts";

register("./helpers/download-lifecycle-resolver.mjs", import.meta.url);
installLocalStorageFake();

const originalFetch = globalThis.fetch;
after(() => {
  globalThis.fetch = originalFetch;
});

const server: {
  id: number;
  path: string;
  created_at: string;
  recursive: boolean;
  label?: string | null;
  status?: string;
}[] = [{ id: 1, path: "/models/a", created_at: "", recursive: false }];
let listGate: Promise<void> | null = null;
let lists = 0;
let bareBody = false;
globalThis.fetch = (async (_input: string, init?: RequestInit) => {
  const json = (body: unknown, status = 200) =>
    new Response(JSON.stringify(body), {
      status,
      headers: { "Content-Type": "application/json" },
    });
  if (init?.method === "POST") {
    const body = JSON.parse(String(init.body));
    const known = server.find((row) => row.path === body.path);
    if (known) {
      if (body.label !== undefined) known.label = body.label;
      return json({ ...known }, 201);
    }
    const row = { id: 2, path: "/models/b", created_at: "", recursive: false };
    server.push(row);
    return json(row, 201);
  }
  if (init?.method === "PATCH") {
    const row = server[0];
    row.label = JSON.parse(String(init.body)).label || null;
    return json({ ...row });
  }
  lists += 1;
  if (bareBody) return json({});
  const snapshot = server.map((row) => ({ ...row }));
  if (listGate) await listGate;
  return json({ folders: snapshot });
}) as typeof fetch;

const store = await import(
  "../src/features/hub/inventory/scan-folders-store.ts"
);

test("an add during the first load keeps the folders the load was fetching", async () => {
  store.resetScanFoldersStoreForTests();
  let release: () => void = () => {};
  listGate = new Promise((resolve) => {
    release = resolve;
  });
  const firstLoad = store.refreshScanFolders();
  await store.addScanFolderToStore("/models/b");
  listGate = null;
  release();
  await firstLoad;
  // The superseded list is read again rather than dropped.
  await new Promise((resolve) => setTimeout(resolve, 0));
  while (store.getScanFoldersState().loading) {
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  assert.deepEqual(
    store
      .getScanFoldersState()
      .folders.map((row) => row.path)
      .sort(),
    ["/models/a", "/models/b"],
  );
  assert.equal(lists, 2);
});

test("a list body without folders reads as none instead of crashing the page", async () => {
  store.resetScanFoldersStoreForTests();
  bareBody = true;
  try {
    assert.deepEqual(await store.refreshScanFolders(), []);
  } finally {
    bareBody = false;
  }
});

test("a rename updates the row in place and keeps its last scan status", async () => {
  store.resetScanFoldersStoreForTests();
  await store.refreshScanFolders();
  const [folder] = store.getScanFoldersState().folders;
  await store.renameScanFolderInStore({ ...folder, status: "partial" }, "Jan");
  const renamed = store
    .getScanFoldersState()
    .folders.find((row) => row.id === folder.id);
  assert.equal(renamed?.label, "Jan");
  assert.equal(renamed?.status, "partial");
});

test("re-adding a folder under a new name is a change, not a duplicate", async () => {
  store.resetScanFoldersStoreForTests();
  await store.refreshScanFolders();
  const renamed = await store.addScanFolderToStore(
    "/models/a",
    undefined,
    "LM",
  );
  assert.equal(renamed.alreadyAdded, false);
  assert.equal(renamed.folder.label, "LM");
  const again = await store.addScanFolderToStore("/models/a");
  assert.equal(again.alreadyAdded, true);
});
