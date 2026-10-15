// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Issue 6369: on the desktop app, Browse for a model folder opens the OS folder dialog.
// The web app and managed accounts keep the in-app browser, and so does a desktop
// whose native dialog fails to open.

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import { readSrc, readText } from "./helpers/kit.ts";

register("./helpers/tauri-core-resolver.mjs", import.meta.url);

const MODULE = new URL("../src/features/native-intents/api.ts", import.meta.url)
  .href;

type StubControl = {
  calls: { command: string }[];
  mode: "ok" | "rejects";
  result?: unknown;
};

let generation = 0;

async function load(options: {
  tauri: boolean;
  mode?: "ok" | "rejects";
  result?: unknown;
}) {
  const control: StubControl = {
    calls: [],
    mode: options.mode ?? "ok",
    result: options.result,
  };
  const define = (name: string, value: unknown) =>
    Object.defineProperty(globalThis, name, {
      value,
      configurable: true,
      writable: true,
    });
  define("__TAURI_CORE_STUB__", control);
  define("window", {
    location: { protocol: options.tauri ? "tauri:" : "https:" },
    ...(options.tauri ? { __TAURI_INTERNALS__: {} } : {}),
  });

  generation += 1;
  const module = (await import(`${MODULE}?bust=${generation}`)) as {
    pickNativeModelFolder: (
      isOwner: boolean,
    ) => Promise<string | null | undefined>;
  };
  return { module, control };
}

test("the desktop owner gets the folder the OS dialog picked", async () => {
  const { module, control } = await load({ tauri: true, result: "D:\\models" });
  assert.equal(await module.pickNativeModelFolder(true), "D:\\models");
  assert.deepEqual(
    control.calls.map((call) => call.command),
    ["pick_native_model_folder"],
  );
});

test("cancelling the OS dialog adds nothing and opens no other browser", async () => {
  const { module } = await load({ tauri: true, result: null });
  assert.equal(await module.pickNativeModelFolder(true), null);
});

test("the web app keeps the in-app browser and never invokes", async () => {
  const { module, control } = await load({ tauri: false });
  assert.equal(await module.pickNativeModelFolder(true), undefined);
  assert.equal(control.calls.length, 0);
});

test("a managed account keeps the in-app browser", async () => {
  const { module, control } = await load({ tauri: true });
  assert.equal(await module.pickNativeModelFolder(false), undefined);
  assert.equal(control.calls.length, 0);
});

test("a native dialog that fails to open falls back to the in-app browser", async () => {
  const { module } = await load({ tauri: true, mode: "rejects" });
  assert.equal(await module.pickNativeModelFolder(true), undefined);
});

test("every model folder Browse button goes through the native picker", () => {
  const pickers = readSrc(
    "features/model-picker/components/model-selector/pickers.tsx",
  );
  assert.equal(
    pickers.match(/onClick=\{\(\) => void browseForFolder\(\)\}/g)?.length,
    3,
  );
  assert.doesNotMatch(
    pickers,
    /onClick=\{\(\) => setShowFolderBrowser\(true\)\}/,
  );
  assert.match(pickers, /pickNativeModelFolder\(isOwner\)/);

  const hub = readSrc("features/hub/catalog/on-device-folders-dialog.tsx");
  assert.match(hub, /onClick=\{\(\) => void browseLocations\(\)\}/);
  assert.doesNotMatch(hub, /onClick=\{\(\) => setBrowserOpen\(true\)\}/);
  assert.match(hub, /pickNativeModelFolder\(isOwner\)/);

  assert.match(
    readText("../../src-tauri/src/main.rs"),
    /native_intents::pick_native_model_folder,/,
  );
});
