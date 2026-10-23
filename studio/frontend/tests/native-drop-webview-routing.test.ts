// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { mockIPC, mockWindows } from "@tauri-apps/api/mocks";
import { readSrc, readText, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

class ElementStub {
  parentElement = null;
}
const zone = new ElementStub();
Object.assign(globalThis, {
  HTMLElement: ElementStub,
  window: { crypto, devicePixelRatio: 1, location: { protocol: "tauri:" } },
  document: { elementFromPoint: () => zone },
});

type Registration = {
  event: string;
  target: { kind: string; label: string };
  handler: number;
};
const registrations: Registration[] = [];
mockIPC((command, args) => {
  if (command === "plugin:event|listen") {
    registrations.push(args as unknown as Registration);
    return registrations.length;
  }
  if (command === "plugin:window|scale_factor") return 1;
});
mockWindows("main");

const { nativeDropTargetAt, registerNativeDropTarget } = await import(
  "../src/features/native-intents/native-drop-targets.ts"
);

function emitFromWebview(label: string, event: string, payload: unknown) {
  const internals = (window as unknown as {
    __TAURI_INTERNALS__: { runCallback: (id: number, event: unknown) => void };
  }).__TAURI_INTERNALS__;
  for (const listener of registrations) {
    if (
      listener.event === event &&
      listener.target.kind === "Webview" &&
      listener.target.label === label
    ) {
      internals.runCallback(listener.handler, { event, id: listener.handler, payload });
    }
  }
}

test("native targets receive main-webview drops without claiming browser-page drops", async () => {
  const delivered: string[][] = [];
  const hovered: boolean[] = [];
  const unregister = registerNativeDropTarget(zone as unknown as HTMLElement, {
    onDrop: (paths) => delivered.push(paths),
    onDragOver: (over) => hovered.push(over),
  });
  try {
    for (let i = 0; i < 100 && !nativeDropTargetAt({ x: 10, y: 10 }); i++) {
      await new Promise((resolve) => setTimeout(resolve, 5));
    }
    assert.equal(nativeDropTargetAt({ x: 10, y: 10 }), zone);
    const payload = { position: { x: 10, y: 10 }, paths: ["/tmp/probe.txt"] };
    emitFromWebview("browser-test", "tauri://drag-drop", payload);
    assert.deepEqual(delivered, []);
    emitFromWebview("main", "tauri://drag-enter", payload);
    assert.deepEqual(hovered, [true]);
    emitFromWebview("main", "tauri://drag-drop", payload);
    assert.deepEqual(delivered, [["/tmp/probe.txt"]]);
    assert.deepEqual(hovered, [true, false]);
  } finally {
    unregister();
  }
});

test("chat and training subscribe to the same webview event source", () => {
  for (const path of [
    "features/native-intents/use-native-drop.ts",
    "features/studio/sections/use-dataset-uploads.ts",
  ]) {
    const source = readSrc(path);
    assert.match(source, /getCurrentWebview\(\)\.onDragDropEvent/);
    assert.doesNotMatch(source, /(?:getCurrentWindow\(\)|currentWindow)\.onDragDropEvent/);
  }
});

test("Rust authorizes only drops on the main webview", () => {
  const main = readText("../../src-tauri/src/main.rs");
  const handler = main.slice(
    main.indexOf(".on_webview_event("),
    main.indexOf(".on_window_event("),
  );
  assert.match(handler, /webview.label\(\) == "main"/);
  assert.match(handler, /tauri::WebviewEvent::DragDrop\(tauri::DragDropEvent::Drop/);
  assert.match(handler, /note_dropped_paths\(paths\)/);
  assert.doesNotMatch(main, /tauri::WindowEvent::DragDrop/);
});
