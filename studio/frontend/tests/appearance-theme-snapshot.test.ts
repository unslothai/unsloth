// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// snapshot.ts imports the app barrels, so it runs here with every import stubbed. Any import it
// adds that is not stubbed (for example the interface scale store) makes loading throw.

import assert from "node:assert/strict";
import test from "node:test";
import type * as SnapshotModule from "../src/features/settings/appearance-theme/snapshot.ts";
import type { AppearanceSnapshot } from "../src/features/settings/appearance-theme/application/types.ts";
import type { ThemeBackground } from "../src/features/settings/appearance-theme/domain/schema.ts";
import {
  type AppearanceCustomization,
  DEFAULT_CUSTOMIZATION,
} from "../src/features/settings/lib/appearance-customization.ts";
import type { ColorThemeId } from "../src/features/settings/lib/color-themes.ts";
import {
  type BackgroundAdjustments,
  type LensMode,
  sanitizeAdjustments,
  type WindowMaterial,
} from "../src/features/chat-background/model.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

type Call = unknown[];

type Host = ReturnType<typeof createHost>;

function createHost() {
  const calls: Call[] = [];
  const state = {
    palette: "standard" as ColorThemeId,
    customization: DEFAULT_CUSTOMIZATION as AppearanceCustomization,
    material: {
      material: "solid" as WindowMaterial,
      strength: { light: 40, dark: 15 },
      lens: "off" as LensMode,
      menus: true,
    },
    background: null as null | {
      adjustments: BackgroundAdjustments;
      blob: Blob;
    },
  };
  let ticket = 0;
  const failures = { prepare: null as string | null, convert: false };

  return {
    calls,
    state,
    failures,
    stubs: {
      "../stores/appearance-custom-store.ts": {
        useAppearanceCustomStore: {
          getState: () => ({
            customization: state.customization,
            replaceAll: (next: AppearanceCustomization) => {
              calls.push(["replaceAll", next]);
              state.customization = next;
            },
          }),
        },
      },
      "../stores/theme-store.ts": {
        getPalette: () => state.palette,
        setPalette: (next: ColorThemeId) => {
          calls.push(["setPalette", next]);
          state.palette = next;
        },
        setTheme: () => {
          throw new Error("setTheme must not be called");
        },
      },
      "@/features/chat-background": {
        sanitizeAdjustments,
        useBackgroundStore: {
          getState: () => ({ value: state.background }),
        },
        useMaterialStore: {
          getState: () => ({
            ...state.material,
            strength: { ...state.material.strength },
            setMaterial: (next: WindowMaterial) => {
              calls.push(["setMaterial", next]);
              state.material.material = next;
            },
            setStrength: (next: number, mode: "light" | "dark") => {
              calls.push(["setStrength", next, mode]);
              state.material.strength[mode] = next;
            },
            setLens: (next: LensMode) => {
              calls.push(["setLens", next]);
              state.material.lens = next;
            },
            setMenus: (next: boolean) => {
              calls.push(["setMenus", next]);
              state.material.menus = next;
            },
          }),
        },
        beginBackgroundUpload: () => {
          calls.push(["begin"]);
          ticket += 1;
          return ticket;
        },
        finishBackgroundUpload: async (
          upload: number,
          value: { blob: Blob },
        ) => {
          calls.push(["finish", upload, value]);
        },
        failBackgroundUpload: (upload: number, reason: string) => {
          calls.push(["fail", upload, reason]);
        },
        resetChatBackground: async () => {
          calls.push(["reset"]);
          return true;
        },
        dataUrlToBlob: async (dataUrl: string) => {
          calls.push(["dataUrlToBlob", dataUrl]);
          if (failures.convert) throw new Error("decode");
          return new Blob(["converted"], { type: "image/png" });
        },
        prepareBackgroundImage: async (file: Blob) => {
          calls.push(["prepare"]);
          if (failures.prepare) throw new Error(failures.prepare);
          return file;
        },
      },
    },
  };
}

function load(host: Host) {
  return loadWithStubs<typeof SnapshotModule>(
    new URL(
      "../src/features/settings/appearance-theme/snapshot.ts",
      import.meta.url,
    ),
    host.stubs,
  );
}

const NEXT = {
  palette: "cherry" as ColorThemeId,
  customization: {
    ...DEFAULT_CUSTOMIZATION,
    uiFont: "Brand Sans",
    uiFontSize: 17,
  } as AppearanceCustomization,
  glass: {
    material: "frosted" as WindowMaterial,
    strength: { light: 62, dark: 33 },
    lens: "magnify" as LensMode,
    menus: false,
  },
};

test("applying a snapshot writes the store once and sets glass in order", () => {
  const host = createHost();
  load(host).applyAppearanceSnapshot(NEXT);
  assert.deepEqual(host.calls, [
    ["replaceAll", NEXT.customization],
    ["setPalette", "cherry"],
    ["setMaterial", "frosted"],
    ["setStrength", 62, "light"],
    ["setStrength", 33, "dark"],
    ["setLens", "magnify"],
    ["setMenus", false],
  ]);
});

test("reading the snapshot and applying it again restores the previous look", () => {
  const host = createHost();
  const snapshot = load(host);
  const before: AppearanceSnapshot = snapshot.readAppearanceSnapshot();
  assert.equal(before.palette, "standard");
  assert.deepEqual(before.glass, {
    material: "solid",
    strength: { light: 40, dark: 15 },
    lens: "off",
    menus: true,
  });
  assert.equal(before.background, null);

  snapshot.applyAppearanceSnapshot(NEXT);
  assert.notDeepEqual(snapshot.readAppearanceSnapshot(), before);

  snapshot.applyAppearanceSnapshot(before);
  assert.deepEqual(snapshot.readAppearanceSnapshot(), before);
  assert.equal(host.state.palette, "standard");
  assert.equal(host.state.customization, DEFAULT_CUSTOMIZATION);
});

test("restoring no background resets it; a stored one is written back", async () => {
  const host = createHost();
  const snapshot = load(host);
  await snapshot.restoreBackground(null);
  assert.deepEqual(host.calls, [["reset"]]);

  host.calls.length = 0;
  const blob = new Blob(["png"], { type: "image/png" });
  const adjustments = sanitizeAdjustments({ effect: "Dither", visibility: 40 });
  await snapshot.restoreBackground({ adjustments, blob });
  assert.deepEqual(host.calls, [
    ["begin"],
    ["finish", 1, { ...adjustments, blob }],
  ]);
  const finish = host.calls[1] as unknown[];
  assert.equal((finish[2] as { blob: Blob }).blob, blob);
});

test("applying a theme background reports why it failed", async () => {
  const host = createHost();
  const snapshot = load(host);
  const bg: ThemeBackground = {
    dataUrl: "data:image/png;base64,iVBORw0KGgo=",
    adjustments: sanitizeAdjustments({ scope: "all" }),
  };

  assert.equal(await snapshot.applyThemeBackground(bg), true);
  assert.equal(host.calls.at(-1)?.[0], "finish");

  host.calls.length = 0;
  host.failures.prepare = "dimensions";
  assert.equal(await snapshot.applyThemeBackground(bg), false);
  assert.deepEqual(host.calls.at(-1), ["fail", 2, "dimensions"]);

  host.calls.length = 0;
  host.failures.prepare = null;
  host.failures.convert = true;
  assert.equal(await snapshot.applyThemeBackground(bg), false);
  assert.deepEqual(host.calls.at(-1), ["fail", 3, "decode"]);
});
