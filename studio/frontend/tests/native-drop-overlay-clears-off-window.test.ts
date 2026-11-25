// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// #19346: a drag whose "leave" never arrived left "Drop files here" stuck over the composer.

import assert from "node:assert/strict";
import test from "node:test";

import * as dropPaths from "../src/features/native-intents/drop-paths.ts";
import type { useNativeModelDrop } from "../src/features/native-intents/use-native-drop.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

type Overlay = ReturnType<typeof useNativeModelDrop>;
type DragEvent = {
  payload: {
    type: string;
    paths: string[];
    position: { x: number; y: number };
  };
};

async function harness() {
  const eventTarget = () => {
    const listeners = new Map<string, Set<() => void>>();
    return {
      listeners,
      addEventListener: (type: string, fn: () => void) => {
        listeners.set(type, (listeners.get(type) ?? new Set()).add(fn));
      },
      removeEventListener: (type: string, fn: () => void) =>
        listeners.get(type)?.delete(fn),
    };
  };
  const window = eventTarget();
  const document = eventTarget();
  Object.assign(globalThis, { window, document });

  let overlay: Overlay = { status: "idle" };
  let dispose: (() => void) | undefined;
  let deliver!: (event: DragEvent) => Promise<void>;
  let installed!: () => void;
  const ready = new Promise<void>((resolve) => {
    installed = resolve;
  });
  const options = {
    nativePathLeasesSupported: true,
    hasActiveModel: true,
    isModelLoading: false,
    onAttachDocuments: () => {},
  };
  const hook = loadWithStubs<{ useNativeModelDrop: typeof useNativeModelDrop }>(
    new URL(
      "../src/features/native-intents/use-native-drop.ts",
      import.meta.url,
    ),
    {
      "@/lib/api-base": { isTauri: true },
      "@/lib/toast": { toast: { error: () => {} } },
      react: {
        useRef: () => ({ current: options }),
        useState: () => [
          overlay,
          (next: Overlay | ((prev: Overlay) => Overlay)) => {
            overlay = typeof next === "function" ? next(overlay) : next;
          },
        ],
        useEffect: (effect: () => (() => void) | undefined) => {
          dispose = effect();
        },
      },
      "./api": {},
      "./drop-paths": dropPaths,
      "./native-drop-targets": { nativeDropTargetAt: () => null },
      "./store": { useNativeIntentStore: () => () => {} },
      "@tauri-apps/api/webview": {
        getCurrentWebview: () => ({
          onDragDropEvent: (callback: typeof deliver) => {
            deliver = callback;
            installed();
            return Promise.resolve(() => {});
          },
        }),
      },
    },
  );
  hook.useNativeModelDrop(options);
  await ready;
  return {
    get overlay() {
      return overlay;
    },
    drag: (type: string, paths: string[] = []) =>
      deliver({ payload: { type, paths, position: { x: 0, y: 0 } } }),
    fire: (type: string) => {
      const on = type === "visibilitychange" ? document : window;
      for (const fn of on.listeners.get(type) ?? []) fn();
    },
    listening: () =>
      [...window.listeners.values(), ...document.listeners.values()].reduce(
        (n, set) => n + set.size,
        0,
      ),
    dispose: () => dispose?.(),
  };
}

for (const event of ["blur", "focus", "pointerdown", "visibilitychange"]) {
  test(`a missed drag leave does not outlive a ${event}`, async () => {
    const h = await harness();
    await h.drag("enter", ["/tmp/notes.pdf"]);
    assert.deepEqual(h.overlay, { status: "attach" });
    h.fire(event);
    assert.deepEqual(h.overlay, { status: "idle" });
    h.dispose();
  });
}

test("a drag still over the window shows the overlay again on its next move", async () => {
  const h = await harness();
  await h.drag("enter", ["/tmp/notes.pdf"]);
  h.fire("blur");
  await h.drag("over");
  assert.deepEqual(h.overlay, { status: "attach" });
  h.dispose();
  assert.equal(h.listening(), 0);
});
