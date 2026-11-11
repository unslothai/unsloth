// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { stripTypeScriptTypes } from "node:module";
import test from "node:test";
import { readSrc } from "./helpers/kit.ts";

const tick = () => new Promise<void>((resolve) => setImmediate(resolve));
function deferred() {
  let resolve!: () => void;
  const promise = new Promise<void>((done) => { resolve = done; });
  return { promise, resolve };
}

test("a background selected during download survives a pending IndexedDB write", async () => {
  const localWrite = deferred();
  const imageDownload = deferred();
  const downloading = deferred();
  const old = new Blob(["old"]);
  const chosen = new Blob(["chosen"]);
  const server = new Blob(["server"]);
  const rows = new Map<string, any>([
    ["current", { value: { blob: old }, sha256: "old" }],
    ["sync", { state: { updatedAt: 1, dirty: false } }],
  ]);
  const dependencies: Record<string, any> = {
    Dexie: class {
      version() { return { stores() {} }; }
      close() {}
      table() {
        return {
          get: async (key: string) => rows.get(key),
          put: async (row: any) => {
            if (row.id === "current" && row.value.blob === chosen)
              await localWrite.promise;
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
    accountDatabaseName: () => "test-background",
    loadChatBackground: async () => ({
      saved: true, updatedAt: 2, image: { sha256: "server" }, adjustments: {},
    }),
    loadChatBackgroundImage: async () => {
      downloading.resolve();
      await imageDownload.promise;
      return server;
    },
    clearChatBackground: async () => { throw new Error("Unexpected clear"); },
    saveChatBackgroundImage: async () => { throw new Error("Unexpected upload"); },
    saveChatBackgroundAdjustments: async () => { throw new Error("Unexpected save"); },
  };
  (globalThis as any).__backgroundSyncRace = dependencies;
  async function load(relative: string) {
    const source = stripTypeScriptTypes(readSrc(`features/${relative}`))
      .replace(/^import[\s\S]*?;\n/gm, "");
    const bindings = `const {${Object.keys(dependencies).join(",")}} = globalThis.__backgroundSyncRace;\n`;
    const module = await import(`data:text/javascript;base64,${Buffer.from(bindings + source).toString("base64")}`);
    Object.assign(dependencies, module);
    return module;
  }
  await load("chat-background/model.ts");
  const storage = await load("chat-background/storage.ts");
  await load("chat-background/sync-plan.ts");
  const sync = await load("chat-background/sync.ts");
  const stop = sync.startChatBackgroundSync();
  try {
    await downloading.promise;
    const save = storage.finishBackgroundUpload(
      storage.beginBackgroundUpload(), { blob: chosen },
    );
    await tick();
    imageDownload.resolve();
    await tick();
    await tick();
    localWrite.resolve();
    await save;
    await tick();
    await tick();
    assert.equal(await storage.useBackgroundStore.getState().value.blob.text(), "chosen");
    assert.equal(await rows.get("current").value.blob.text(), "chosen");
    assert.equal(rows.get("sync").state.dirty, true);
    rows.set("current", { value: { blob: server }, sha256: "server" });
    assert.equal((await storage.readBackgroundSync()).sha256, null);
  } finally {
    stop();
    localWrite.resolve();
    imageDownload.resolve();
    delete (globalThis as any).__backgroundSyncRace;
  }
});
