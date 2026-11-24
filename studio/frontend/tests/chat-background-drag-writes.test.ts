// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { stripTypeScriptTypes } from "node:module";
import test from "node:test";
import { readSrc } from "./helpers/kit.ts";

let loads = 0;
const tick = () => new Promise<void>((resolve) => setImmediate(resolve));

async function loadStorage(
  rows: Map<string, any>,
  puts: any[],
  gate?: Promise<void>,
  fails: (row: any) => boolean = () => false,
) {
  const dependencies: Record<string, any> = {
    Dexie: class {
      version() { return { stores() {} }; }
      close() {}
      table() {
        return {
          get: async (key: string) => rows.get(key),
          put: async (row: any) => {
            puts.push(row);
            if (gate) await gate;
            if (fails(row)) throw new Error("QuotaExceededError");
            rows.set(row.id, row);
          },
          delete: async (key: string) => rows.delete(key),
        };
      }
    },
    create: (initial: () => any) => {
      let state = initial();
      const store = () => state;
      store.getState = () => state;
      store.setState = (patch: any) => { state = { ...state, ...patch }; };
      return store;
    },
    useEffect() {},
    accountDatabaseName: () => "test-background-drag",
  };
  (globalThis as any).__backgroundDrag = dependencies;
  async function load(relative: string) {
    const source = stripTypeScriptTypes(readSrc(`features/${relative}`))
      .replace(/^import[\s\S]*?;\n/gm, "");
    const bindings = `// ${++loads}\nconst {${Object.keys(dependencies).join(",")}} = globalThis.__backgroundDrag;\n`;
    const module = await import(`data:text/javascript;base64,${Buffer.from(bindings + source).toString("base64")}`);
    Object.assign(dependencies, module);
    return module;
  }
  await load("chat-background/model.ts");
  return load("chat-background/storage.ts");
}

test("a slider drag stores the image once per idle write, not once per tick, and keeps the last value", async () => {
  const image = new Blob(["image"]);
  const rows = new Map<string, any>([
    ["current", { id: "current", value: { blob: image, visibility: 50 }, sha256: "abc" }],
    ["sync", { id: "sync", state: { updatedAt: 1, dirty: false } }],
  ]);
  const puts: any[] = [];
  let release!: () => void;
  const gate = new Promise<void>((resolve) => { release = resolve; });
  try {
    const storage = await loadStorage(rows, puts, gate);
    let changes = 0;
    storage.setBackgroundChangeListener(() => { changes++; });
    await storage.hydrateBackground();
    // The first tick's write starts and is held; the rest queue behind it.
    storage.patchBackground({ visibility: 1 });
    await tick();
    for (let visibility = 2; visibility <= 40; visibility++) {
      storage.patchBackground({ visibility });
      assert.equal(storage.useBackgroundStore.getState().value.visibility, visibility);
    }
    release();
    for (let i = 0; i < 20; i++) await tick();
    const imagePuts = puts.filter((row) => row.id === "current");
    assert.deepEqual(imagePuts.map((row) => row.value.visibility), [1, 40]);
    assert.equal(rows.get("current").value.visibility, 40);
    assert.equal(rows.get("current").value.blob, image);
    assert.equal(rows.get("current").sha256, "abc");
    assert.deepEqual(rows.get("sync").state, { updatedAt: 1, dirty: true });
    assert.equal(changes, 40);
    assert.equal(storage.useBackgroundStore.getState().error, null);
    assert.equal(storage.useBackgroundStore.getState().value.visibility, 40);
  } finally {
    release();
    delete (globalThis as any).__backgroundDrag;
  }
});

test("patches before an upload are still stored when the upload write fails", async () => {
  const image = new Blob(["image"]);
  const upload = new Blob(["upload"]);
  const rows = new Map<string, any>([
    ["current", { id: "current", value: { blob: image, visibility: 50 }, sha256: "abc" }],
    ["sync", { id: "sync", state: { updatedAt: 1, dirty: false } }],
  ]);
  let release!: () => void;
  const gate = new Promise<void>((resolve) => { release = resolve; });
  try {
    const storage = await loadStorage(rows, [], gate, (row) => row.value?.blob === upload);
    await storage.hydrateBackground();
    storage.patchBackground({ visibility: 1 });
    await tick();
    for (let visibility = 2; visibility <= 5; visibility++) storage.patchBackground({ visibility });
    const ticket = storage.beginBackgroundUpload();
    release();
    await storage.finishBackgroundUpload(ticket, { blob: upload, visibility: 70 });
    assert.equal(rows.get("current").value.visibility, 5);
    assert.equal(rows.get("current").value.blob, image);
    assert.equal(storage.useBackgroundStore.getState().value.visibility, 5);
    assert.equal(storage.useBackgroundStore.getState().error, "storage");
  } finally {
    release();
    delete (globalThis as any).__backgroundDrag;
  }
});
