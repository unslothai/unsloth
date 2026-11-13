// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// snapshot.ts runs with every import stubbed; an unstubbed import makes loading throw.
import assert from "node:assert/strict";
import test from "node:test";
import { sanitizeAdjustments } from "../src/features/chat-background/model.ts";
import type { ThemeApplyPlan } from "../src/features/settings/appearance-theme/application/plan-apply.ts";
import type { ThemeBackground } from "../src/features/settings/appearance-theme/domain/schema.ts";
import type * as SnapshotModule from "../src/features/settings/appearance-theme/snapshot.ts";
import { type AppearanceCustomization, DEFAULT_CUSTOMIZATION } from "../src/features/settings/lib/appearance-customization.ts";
import { decodeBase64Bytes } from "../src/lib/base64-bytes.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

function createHost() {
  const calls: unknown[][] = [];
  const record = (...entry: unknown[]) => calls.push(entry);
  const state = {
    palette: "standard",
    customization: DEFAULT_CUSTOMIZATION as AppearanceCustomization,
    material: { material: "solid", style: "frosted", strength: { light: 40, dark: 15 }, lens: "off", menus: true, nativeFailed: false } as Record<string, unknown>,
    background: null as null | { adjustments: unknown; blob: Blob },
    busy: false,
  };
  const upload = { result: true }; let ticket = 0;
  const native = { available: false };
  const stubs = {
    "../stores/appearance-custom-store.ts": {
      useAppearanceCustomStore: {
        getState: () => ({
          customization: state.customization,
          replaceAll: (next: AppearanceCustomization) => {
            record("replaceAll", next);
            state.customization = next;
          },
        }),
      },
    },
    "../stores/theme-store.ts": {
      getPalette: () => state.palette,
      setPalette: (next: string) => {
        record("setPalette", next);
        state.palette = next;
      },
      setTheme: () => {
        throw new Error("setTheme must not be called");
      },
    },
    "@/lib/base64-bytes": { decodeBase64Bytes },
    "@/features/chat-background": {
      sanitizeAdjustments,
      nativeMaterialAvailable: () => native.available,
      useBackgroundStore: { getState: () => ({ value: state.background, busy: state.busy }) },
      useMaterialStore: {
        getState: () => ({ ...state.material }),
        setState: (partial: Record<string, unknown>) => {
          record("material", partial);
          state.material = { ...state.material, ...partial };
        },
      },
      beginBackgroundUpload: () => (record("begin"), ++ticket),
      finishBackgroundUpload: async (id: number, value: unknown) => (record("finish", id, value), upload.result),
      resetChatBackground: async () => (record("reset"), upload.result),
      hydrateBackground: async () => {
        record("hydrate");
        state.background = state.background ?? { adjustments: sanitizeAdjustments({}), blob: new Blob(["stored"]) };
      },
      uploadBackgroundImage: async (blob: Blob, adjustments: unknown) => {
        record("upload", blob, adjustments);
        return upload.result;
      },
    },
  };
  return { calls, state, upload, native, stubs };
}

const load = (host: ReturnType<typeof createHost>) =>
  loadWithStubs<typeof SnapshotModule>(new URL("../src/features/settings/appearance-theme/snapshot.ts", import.meta.url), host.stubs);

const GLASS = { material: "frosted", style: "frosted", strength: { light: 62, dark: 33 }, lens: "magnify", menus: false } as const;
const NEXT: SnapshotModule.AppearanceTarget = { palette: "cherry", customization: { ...DEFAULT_CUSTOMIZATION, uiFont: "Brand Sans", uiFontSize: 17 }, glass: GLASS };

test("applying a snapshot writes the store, palette and material once each", () => {
  const host = createHost();
  load(host).applyAppearanceSnapshot(NEXT);
  // The stub's setTheme throws, so reaching the theme store's setTheme would fail this test.
  assert.deepEqual(host.calls, [
    ["replaceAll", NEXT.customization],
    ["setPalette", "cherry"],
    ["material", { ...GLASS, nativeFailed: false }],
  ]);
});

test("reading the snapshot and applying it back restores the look, glass style included", () => {
  const host = createHost();
  const snapshot = load(host);
  const before = snapshot.readAppearanceSnapshot();
  assert.deepEqual(before, {
    palette: "standard", customization: DEFAULT_CUSTOMIZATION,
    glass: { material: "solid", style: "frosted", strength: { light: 40, dark: 15 }, lens: "off", menus: true },
    background: null,
  });
  const original = structuredClone(host.state.material);
  snapshot.applyAppearanceSnapshot(NEXT);
  assert.notDeepEqual(snapshot.readAppearanceSnapshot(), before);
  snapshot.applyAppearanceSnapshot(before);
  assert.deepEqual(snapshot.readAppearanceSnapshot(), before);
  assert.deepEqual(host.state.material, original);
});

test("restoring no background resets it; a stored one is written back through begin and finish", async () => {
  const host = createHost();
  const snapshot = load(host);
  await snapshot.restoreBackground(null);
  assert.deepEqual(host.calls.splice(0), [["reset"]]);

  const blob = new Blob(["png"], { type: "image/png" });
  const adjustments = sanitizeAdjustments({ effect: "Dither", visibility: 40 });
  await snapshot.restoreBackground({ adjustments, blob });
  assert.deepEqual(host.calls, [["begin"], ["finish", 1, { ...adjustments, blob }]]);
  assert.equal((host.calls[1][2] as { blob: Blob }).blob, blob);
});

test("applyThemeBackground uploads the decoded bytes as a Blob and returns the upload result", async () => {
  const host = createHost();
  const background: ThemeBackground = {
    dataUrl: "data:image/png;base64,AAAA",
    adjustments: sanitizeAdjustments({ scope: "all" }),
  };
  assert.equal(await load(host).applyThemeBackground(background), true);
  const [, blob, adjustments] = host.calls[0] as [string, Blob, unknown];
  assert.equal(blob.type, "image/png");
  assert.equal(blob.size, 3);
  assert.deepEqual(adjustments, sanitizeAdjustments({ scope: "all" }));

  host.upload.result = false;
  assert.equal(await load(host).applyThemeBackground(background), false);
});

test("the loaded snapshot waits for the stored background; undo's guard compares adjustments too", async () => {
  const host = createHost();
  const snapshot = load(host);
  assert.equal(snapshot.readAppearanceSnapshot().background, null);
  const loaded = await snapshot.readLoadedAppearanceSnapshot();
  assert.equal(host.calls[0][0], "hydrate");
  assert.ok(loaded.background);

  const applied = loaded.background;
  assert.equal(snapshot.sameBackground({ ...applied }, applied), true);
  const edited = { ...applied, adjustments: { ...applied.adjustments, visibility: 12 } };
  assert.equal(snapshot.sameBackground(edited, applied), false);
  assert.equal(snapshot.sameBackground(null, applied), false);
});

test("restoreBackground resolves with whether the background could be stored", async () => {
  const host = createHost();
  const snapshot = load(host);
  const prev = { adjustments: sanitizeAdjustments({}), blob: new Blob(["png"]) };
  assert.equal(await snapshot.restoreBackground(prev), true);
  host.upload.result = false;
  assert.equal(await snapshot.restoreBackground(prev), false);
  assert.equal(await snapshot.restoreBackground(null), false);
});

test("off macOS a theme's native-only glass becomes Liquid 3D, and Undo then restores the old glass", async () => {
  const { planUndo } = await import("../src/features/settings/appearance-theme/application/plan-apply.ts");
  const macos = { ...NEXT, glass: { ...GLASS, material: "macos", style: "macos" } } as SnapshotModule.AppearanceTarget;
  const host = createHost();
  const snapshot = load(host);
  const before = snapshot.readAppearanceSnapshot();
  const plan = snapshot.portableGlass<ThemeApplyPlan>({ ...macos, background: "keep", changes: [], droppedFonts: [] });
  assert.equal(plan.glass.material, "liquid-3d");
  assert.equal(plan.glass.style, "liquid-3d");
  snapshot.applyAppearanceSnapshot(plan);
  // Undo with nothing touched since: the live glass matches the plan, so the old glass comes back.
  snapshot.applyAppearanceSnapshot(planUndo(before, plan, snapshot.readAppearanceSnapshot()));
  assert.equal(host.state.material.material, "solid");
  assert.equal(host.state.material.style, "frosted");
  const mac = createHost();
  mac.native.available = true;
  assert.equal(load(mac).portableGlass(macos), macos);
});

test("undo leaves the background alone while a newer upload is pending or another image is shown", async () => {
  const host = createHost();
  const snapshot = load(host);
  const prev = { adjustments: sanitizeAdjustments({}), blob: new Blob(["old"]) };
  const applied = { adjustments: sanitizeAdjustments({}), blob: new Blob(["theme"]) };
  host.state.background = { ...applied };
  host.state.busy = true;
  assert.equal(await snapshot.undoBackground(prev, applied), true);
  assert.deepEqual(host.calls.filter((c) => c[0] === "begin" || c[0] === "reset"), []);
  host.state.busy = false;
  host.state.background = { adjustments: applied.adjustments, blob: new Blob(["mine"]) };
  assert.equal(await snapshot.undoBackground(prev, applied), true);
  assert.deepEqual(host.calls.filter((c) => c[0] === "begin" || c[0] === "reset"), []);
  host.state.background = { ...applied };
  assert.equal(await snapshot.undoBackground(prev, applied), true);
  assert.equal(host.calls.filter((c) => c[0] === "begin").length, 1);
});
