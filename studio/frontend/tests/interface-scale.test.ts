// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { register } from "node:module";
import test from "node:test";

register("./helpers/tauri-webview-resolver.mjs", import.meta.url);

const MODULE = new URL(
  "../src/features/settings/stores/interface-scale-store.ts",
  import.meta.url,
).href;
const CAPABILITIES = new URL(
  "../../src-tauri/capabilities/default.json",
  import.meta.url,
);
const MAIN = new URL("../src/main.tsx", import.meta.url);
const PROVIDER = new URL("../src/app/provider.tsx", import.meta.url);
const APPEARANCE_TAB = new URL(
  "../src/features/settings/tabs/appearance-tab.tsx",
  import.meta.url,
);
const APPEARANCE_CONTROLS = new URL(
  "../src/features/settings/components/appearance-custom-controls.tsx",
  import.meta.url,
);
const GENERAL_TAB = new URL(
  "../src/features/settings/tabs/general-tab.tsx",
  import.meta.url,
);

const { NATIVE_MAC_TITLEBAR_HEIGHT_PX, NATIVE_MAC_TRAFFIC_LIGHT_INSET_PX } =
  await import("../src/features/settings/lib/interface-scale-runtime.ts");

type InterfaceScaleModule = {
  sanitizeInterfaceScale: (value: unknown) => number;
  interfaceScaleToZoom: (scale: number) => number;
  getAppliedInterfaceZoom: () => number;
  applyInterfaceScale: (scale: number) => Promise<void>;
  applyInterfaceScaleBeforeFirstPaint: (
    scale: number,
    timeoutMs?: number,
  ) => Promise<void>;
};

let generation = 0;

function define(name: string, value: unknown) {
  Object.defineProperty(globalThis, name, {
    value,
    configurable: true,
    writable: true,
  });
}

async function load(tauri: boolean) {
  const styles = new Map<string, string>();
  const windowStub: Record<string, unknown> = {
    location: { protocol: tauri ? "tauri:" : "https:" },
    localStorage: {
      getItem: () => null,
      setItem: () => undefined,
      removeItem: () => undefined,
    },
  };
  if (tauri) {
    windowStub["__TAURI_INTERNALS__"] = {};
  }
  define("window", windowStub);
  define("document", {
    documentElement: {
      style: {
        setProperty: (name: string, value: string) => styles.set(name, value),
      },
    },
  });
  const control: {
    zooms: number[];
    setZoom?: (zoom: number) => Promise<void>;
  } = { zooms: [] };
  define("__TAURI_WEBVIEW_STUB__", control);
  generation += 1;
  const mod = (await import(
    `${MODULE}?bust=${generation}`
  )) as InterfaceScaleModule;
  return { mod, control, styles };
}

test("interface scale is rounded and clamped to a usable range", async () => {
  const { mod } = await load(false);
  assert.equal(mod.sanitizeInterfaceScale(undefined), 100);
  assert.equal(mod.sanitizeInterfaceScale(10), 50);
  assert.equal(mod.sanitizeInterfaceScale(66.6), 67);
  assert.equal(mod.sanitizeInterfaceScale(500), 200);
  assert.equal(mod.interfaceScaleToZoom(55), 0.55);
  // Below the floor the zoom clamps too, rather than reaching the webview raw.
  assert.equal(mod.interfaceScaleToZoom(25), 0.5);
});

test("desktop scale sets native webview zoom", async () => {
  const { mod, control, styles } = await load(true);
  await mod.applyInterfaceScale(55);
  await mod.applyInterfaceScale(55);
  assert.deepEqual(control.zooms, [0.55]);
  assert.equal(mod.getAppliedInterfaceZoom(), 0.55);
  assert.equal(styles.get("--studio-native-titlebar-height"), `${NATIVE_MAC_TITLEBAR_HEIGHT_PX / 0.55}px`);
  assert.equal(
    styles.get("--studio-native-traffic-light-inset"),
    `${NATIVE_MAC_TRAFFIC_LIGHT_INSET_PX / 0.55}px`,
  );
});

test("the latest scale wins while an older native update is pending", async () => {
  const { mod, control, styles } = await load(true);
  let markFirstStarted: () => void = () => undefined;
  let releaseFirst: () => void = () => undefined;
  const firstStarted = new Promise<void>((resolve) => {
    markFirstStarted = resolve;
  });
  control.setZoom = (zoom) => {
    control.zooms.push(zoom);
    if (zoom !== 0.55) {
      return Promise.resolve();
    }
    markFirstStarted();
    return new Promise<void>((resolve) => {
      releaseFirst = resolve;
    });
  };

  const first = mod.applyInterfaceScale(55);
  await firstStarted;
  const latest = mod.applyInterfaceScale(75);
  releaseFirst();
  await Promise.all([first, latest]);

  assert.deepEqual(control.zooms, [0.55, 0.75]);
  assert.equal(mod.getAppliedInterfaceZoom(), 0.75);
  assert.equal(styles.get("--studio-native-titlebar-height"), `${NATIVE_MAC_TITLEBAR_HEIGHT_PX / 0.75}px`);
});

test("first paint is not held hostage by a wedged native bridge", async () => {
  const { mod, control } = await load(true);
  // Never resolves: the failure a plain catch() does not cover.
  control.setZoom = () => new Promise<void>(() => undefined);
  await mod.applyInterfaceScaleBeforeFirstPaint(75, 10);
});

test("first paint waits for the scale when the bridge answers", async () => {
  const { mod, control, styles } = await load(true);
  await mod.applyInterfaceScaleBeforeFirstPaint(75, 5_000);
  assert.deepEqual(control.zooms, [0.75]);
  assert.equal(
    styles.get("--studio-native-titlebar-height"),
    `${NATIVE_MAC_TITLEBAR_HEIGHT_PX / 0.75}px`,
  );
});

test("browser scale never calls the native webview", async () => {
  const { mod, control } = await load(false);
  await mod.applyInterfaceScale(75);
  assert.deepEqual(control.zooms, []);
});

test("desktop capability allows webview zoom", async () => {
  const capabilities = JSON.parse(await readFile(CAPABILITIES, "utf8")) as {
    permissions: unknown[];
  };
  assert.ok(
    capabilities.permissions.includes("core:webview:allow-set-webview-zoom"),
  );
});

test("startup, live changes, and both resets use the local scale", async () => {
  const [main, provider, tab, controls, general] = await Promise.all(
    [MAIN, PROVIDER, APPEARANCE_TAB, APPEARANCE_CONTROLS, GENERAL_TAB].map(
      (path) => readFile(path, "utf8"),
    ),
  );
  assert.match(
    main,
    /applyInterfaceScaleBeforeFirstPaint\(\s*useInterfaceScaleStore\.getState\(\)\.scale/,
  );
  assert.match(
    provider,
    /useInterfaceScaleStore\(\(s\) => s\.scale\)[\s\S]*applyInterfaceScale\(interfaceScale\)/,
  );
  assert.match(
    tab,
    /isTauri && \([\s\S]*settings\.appearance\.custom\.interfaceScale\.label/,
  );
  assert.match(controls, /resetAll\(\);\s*resetInterfaceScale\(\);/);
  assert.match(general, /INTERFACE_SCALE_STORAGE_KEY/);
});
