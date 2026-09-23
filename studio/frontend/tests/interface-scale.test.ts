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
  webInterfaceScaleFactor: (scale: number) => number;
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
        removeProperty: (name: string) => styles.delete(name),
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
  assert.equal(
    styles.get("--studio-native-titlebar-height"),
    `${NATIVE_MAC_TITLEBAR_HEIGHT_PX / 0.55}px`,
  );
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
  assert.equal(
    styles.get("--studio-native-titlebar-height"),
    `${NATIVE_MAC_TITLEBAR_HEIGHT_PX / 0.75}px`,
  );
});

test("first paint is not held hostage by a wedged native bridge", async () => {
  const { mod, control } = await load(true);
  // Park the abandoned call INSIDE setZoom, not on the dynamic import ahead of it. The stub
  // reads `__TAURI_WEBVIEW_STUB__` when setZoom is called, so a call still waiting on its
  // import when this test ends resumes against the NEXT test's control and records a zoom
  // there. That is one leaked 0.75 in a later assertion, blamed on the test it lands in.
  let markWedged: () => void = () => undefined;
  const wedgedEntered = new Promise<void>((resolve) => {
    markWedged = resolve;
  });
  // Never resolves: the failure a plain catch() does not cover.
  control.setZoom = () => {
    markWedged();
    return new Promise<void>(() => undefined);
  };
  await mod.applyInterfaceScaleBeforeFirstPaint(75, 10);
  await wedgedEntered;
});

// Timed, because the regression these two cover is a queue that never drains: without the
// release the awaits below hang rather than fail, and an unbounded hang wedges the runner
// instead of reporting.
test(
  "a scale change after a wedged first paint still applies",
  { timeout: 5_000 },
  async () => {
    const { mod, control, styles } = await load(true);
    let wedged = true;
    let markWedged: () => void = () => undefined;
    const wedgedEntered = new Promise<void>((resolve) => {
      markWedged = resolve;
    });
    control.setZoom = (zoom) => {
      if (wedged) {
        markWedged();
        return new Promise<void>(() => undefined);
      }
      control.zooms.push(zoom);
      return Promise.resolve();
    };

    await mod.applyInterfaceScaleBeforeFirstPaint(75, 10);
    // Wait for the abandoned call to be parked inside setZoom before unwedging. Without this
    // the deadline can fire while it is still on its dynamic import, and it then reaches a
    // setZoom that is no longer wedged and records the stale 0.75 the assertion forbids.
    await wedgedEntered;
    // The bridge comes back. Nothing about the abandoned call may keep the queue closed.
    wedged = false;
    await mod.applyInterfaceScale(125);

    assert.deepEqual(control.zooms, [1.25]);
    assert.equal(mod.getAppliedInterfaceZoom(), 1.25);
    assert.equal(
      styles.get("--studio-native-titlebar-height"),
      `${NATIVE_MAC_TITLEBAR_HEIGHT_PX / 1.25}px`,
    );
  },
);

test(
  "a late wedged call restores the requested native zoom",
  { timeout: 5_000 },
  async () => {
    const { mod, control, styles } = await load(true);
    let nativeZoom = 1;
    let releaseWedged: () => void = () => undefined;
    let markRestored: () => void = () => undefined;
    const restored = new Promise<void>((resolve) => {
      markRestored = resolve;
    });
    control.setZoom = (zoom) => {
      control.zooms.push(zoom);
      if (zoom === 0.75) {
        return new Promise<void>((resolve) => {
          releaseWedged = () => {
            nativeZoom = zoom;
            resolve();
          };
        });
      }
      nativeZoom = zoom;
      if (control.zooms.length === 3) {
        markRestored();
      }
      return Promise.resolve();
    };

    await mod.applyInterfaceScaleBeforeFirstPaint(75, 10);
    await mod.applyInterfaceScale(125);
    releaseWedged();
    await restored;
    await mod.applyInterfaceScale(125);

    assert.equal(nativeZoom, 1.25);
    assert.deepEqual(control.zooms, [0.75, 1.25, 1.25]);
    assert.equal(mod.getAppliedInterfaceZoom(), 1.25);
    assert.equal(
      styles.get("--studio-native-titlebar-height"),
      `${NATIVE_MAC_TITLEBAR_HEIGHT_PX / 1.25}px`,
    );
  },
);

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

test("browser scale resizes the UI through the tokens", async () => {
  // The page cannot zoom itself: CSS zoom on the root overflows every
  // viewport unit. It multiplies the font scale instead, which spacing and
  // icons derive from.
  const { mod, styles } = await load(false);
  await mod.applyInterfaceScale(125);
  assert.equal(styles.get("--ui-interface-scale"), "1.25");
  assert.equal(mod.webInterfaceScaleFactor(125), 1.25);
  // At 100% nothing is left behind, so the default document is unchanged.
  await mod.applyInterfaceScale(100);
  assert.equal(styles.has("--ui-interface-scale"), false);
  // Never below the floor, same as the desktop zoom.
  await mod.applyInterfaceScaleBeforeFirstPaint(10, 5_000);
  assert.equal(styles.get("--ui-interface-scale"), "0.5");
});

test("desktop scale leaves the tokens to the webview zoom", async () => {
  const { mod, styles } = await load(true);
  assert.equal(mod.webInterfaceScaleFactor(150), 1);
  await mod.applyInterfaceScale(150);
  assert.equal(styles.has("--ui-interface-scale"), false);
});

test("the scale reaches the tokens and their JS twin", async () => {
  const css = await readFile(
    new URL("../src/index.css", import.meta.url),
    "utf8",
  );
  assert.match(
    css,
    /--ui-font-scale: calc\(var\(--ui-font-size-scale, 0\.9375\) \* var\(--ui-interface-scale, 1\)\);/,
  );
  const hook = await readFile(
    new URL("../src/hooks/use-ui-space-scale.ts", import.meta.url),
    "utf8",
  );
  assert.match(hook, /webInterfaceScaleFactor\(interfaceScale\)/);
  const snapshot = await readFile(
    new URL("../public/reload-snapshot.js", import.meta.url),
    "utf8",
  );
  assert.ok(snapshot.includes('"--ui-interface-scale"'));
  assert.ok(snapshot.includes('"--ui-font-size-scale"'));
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
  // Every build: the browser scales through the tokens, desktop through zoom.
  assert.match(tab, /settings\.appearance\.custom\.interfaceScale\.label/);
  assert.doesNotMatch(tab, /isTauri && \(/);
  assert.match(controls, /resetAll\(\);\s*resetInterfaceScale\(\);/);
  assert.match(general, /INTERFACE_SCALE_STORAGE_KEY/);
});
