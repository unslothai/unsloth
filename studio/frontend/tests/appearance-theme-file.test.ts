// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import type {
  AppearanceSnapshot as Snapshot,
  AppearanceTheme as Theme,
} from "../src/features/settings/appearance-theme/theme-file.ts";

(globalThis as { document?: unknown }).document = {
  documentElement: {
    style: { setProperty() {}, removeProperty() {} },
    setAttribute() {},
    removeAttribute() {},
    toggleAttribute() {},
    classList: { toggle() {} },
  },
};
const { DEFAULT_CUSTOMIZATION, sanitizeCustomization } = await import(
  "../src/features/settings/stores/appearance-custom-store.ts"
);
const {
  captureTheme,
  parseThemeFile,
  planThemeApply,
  THEME_FILE_KIND,
  undoTarget,
} = await import("../src/features/settings/appearance-theme/theme-file.ts");

const font = (name: string, data = "QUJD") => ({
  name,
  dataUrl: `data:font/woff2;base64,${data}`,
});
const BASE: Snapshot = {
  palette: "standard",
  customization: DEFAULT_CUSTOMIZATION,
  glass: {
    material: "solid",
    style: "frosted",
    strength: { light: 40, dark: 15 },
    lens: "off",
    menus: true,
    scope: "all",
  },
  background: null,
};
const withLook = (look: Record<string, unknown>): Snapshot => ({
  ...BASE,
  customization: sanitizeCustomization(look),
});
const file = (extra: Record<string, unknown>) =>
  JSON.stringify({ kind: THEME_FILE_KIND, schemaVersion: 1, ...extra });
function parse(text: string) {
  const result = parseThemeFile(text);
  if (!result.ok) throw new Error(result.error);
  return result;
}
const roundTrip = (s: Snapshot, includeAssets = false) =>
  parse(
    JSON.stringify(
      captureTheme(s, { name: "Dusk", includeLayout: true, includeAssets }),
    ),
  ).theme;
const apply = (
  s: Snapshot,
  theme: Theme,
  applyLayout = true,
  applyAssets = true,
) => planThemeApply(s, theme, { applyLayout, applyAssets });

const FULL = withLook({
  colors: {
    light: { accent: "#3366ff", background: "#fafafa", foreground: "#111111" },
    dark: { accent: "#ff8800" },
  },
  uiFont: "Lato",
  codeFont: "Brand",
  uiFontSize: 17,
  codeFontSize: 13,
  contrast: 72,
  elementOutline: 25,
  composerOutline: 40,
  chatWidth: "wide",
  sentAttachments: "chips",
  fontSmoothing: false,
  codeLineNumbers: true,
  pointerCursors: true,
  reduceMotion: "off",
  importedFonts: [font("Brand")],
  sidebarNav: [
    { id: "tasks", pinned: true },
    { id: "hub", pinned: false },
  ],
  sidebarNavAuto: [],
});

test("a look survives export, parse and apply, except personal fields", () => {
  const plan = apply(
    { ...BASE, palette: "cherry" },
    roundTrip({ ...FULL, palette: "cherry" }, true),
  );
  assert.deepEqual(plan.customization, {
    ...FULL.customization,
    pointerCursors: false,
    reduceMotion: "on",
  });
  const text = JSON.stringify(
    captureTheme(FULL, { name: "x", includeLayout: true, includeAssets: true }),
  );
  assert.equal(
    text.includes("pointerCursors") || text.includes("reduceMotion"),
    false,
  );
});

test("the default look applies as no change; a sparse theme resets the rest", () => {
  assert.deepEqual(apply(BASE, roundTrip(BASE)).changes, []);
  const plan = apply(
    withLook({ contrast: 30, uiFontSize: 18 }),
    parse(file({ look: { contrast: 60 } })).theme,
  );
  assert.equal(plan.customization.contrast, 60);
  assert.equal(plan.customization.uiFontSize, null);
  assert.deepEqual(plan.changes, ["colors", "typography"]);
});

test("export without assets blanks only font selections that name an imported font", () => {
  const s = withLook({
    importedFonts: [font("Wide")],
    chatWidth: "wide",
    uiFont: "wide",
    codeFont: "Fira Code",
  });
  const out = captureTheme(s, {
    name: "x",
    includeLayout: false,
    includeAssets: false,
  }) as { look: Record<string, unknown> };
  assert.deepEqual(
    [out.look.chatWidth, out.look.uiFont, out.look.codeFont],
    ["wide", null, "Fira Code"],
  );
});

test("hostile or odd input warns and falls back, never throws", () => {
  const { theme, warnings } = parse(
    `{"kind":"${THEME_FILE_KIND}","schemaVersion":9,"__proto__":{"x":1},"palette":"neon","look":{"uiFontSize":99,"bogus":1,"colors":{"light":{"accent":"#FFAA00"}}},"glass":{"material":"plasma"}}`,
  );
  assert.equal(({} as Record<string, unknown>).x, undefined);
  assert.deepEqual(
    warnings.map((w) => w.kind),
    [
      "newerVersion",
      "unknown",
      "unknownPalette",
      "unknown",
      "invalid",
      "invalid",
    ],
  );
  assert.equal(theme.customization.uiFontSize, 20);
  assert.equal(theme.customization.colors.light.accent, "#ffaa00");
  assert.equal(theme.glass.material, "solid");
  for (const text of [
    "nope",
    "{}",
    file({}).replace('"schemaVersion":1', '"schemaVersion":0'),
  ]) {
    assert.deepEqual(parseThemeFile(text), { ok: false, error: "invalid" });
  }
  assert.deepEqual(parseThemeFile(file({ name: "x".repeat(300_000) })), {
    ok: false,
    error: "tooLarge",
  });
  const many = Object.fromEntries(
    Array.from({ length: 60 }, (_, i) => [`k${i}`, 1]),
  );
  assert.deepEqual(parse(file(many)).warnings.at(-1), {
    kind: "more",
    count: 10,
  });
});

test("theme fonts replace case-variants, keep selected fonts under the cap and fix spelling", () => {
  const current = withLook({
    importedFonts: [font("First"), font("Second"), font("Brand Sans")],
  });
  const theme = parse(
    file({ look: { uiFont: "brand sans" }, fonts: [font("New"), font("NEW")] }),
  ).theme;
  assert.deepEqual(
    theme.fonts?.map((f) => f.name),
    ["New"],
  );
  const plan = apply(current, theme);
  assert.equal(plan.customization.uiFont, "Brand Sans");
  assert.deepEqual(
    plan.customization.importedFonts.map((f) => f.name),
    ["Brand Sans", "New", "First"],
  );
  assert.deepEqual(plan.droppedFonts, ["Second"]);
});

test("a selection of an imported font that is left out resets; a local family stays", () => {
  const theme = parse(
    file({
      look: { uiFont: "Brand", codeFont: "Georgia" },
      fonts: [font("Brand"), font("Georgia")],
    }),
  ).theme;
  const plan = apply(withLook({ codeFont: "Georgia" }), theme, true, false);
  assert.equal(plan.customization.uiFont, null);
  assert.equal(plan.customization.codeFont, "Georgia");
});

test("layout applies only when asked, and a solid theme keeps the remembered glass style", () => {
  const theme = parse(
    file({
      layout: { sentAttachments: "chips" },
      glass: { material: "solid" },
    }),
  ).theme;
  const current = {
    ...BASE,
    glass: { ...BASE.glass, style: "liquid" as const },
  };
  assert.equal(
    apply(current, theme, false).customization.sentAttachments,
    "auto",
  );
  assert.equal(apply(current, theme).customization.sentAttachments, "chips");
  assert.equal(apply(current, theme).glass.style, "liquid");
});

test("embedded fonts the browser cannot load are dropped with their selections", async () => {
  const { loadWithStubs } = await import("./helpers/module-stubs.ts");
  const real = await import(
    "../src/features/settings/appearance-theme/theme-file.ts"
  );
  const snapshot = loadWithStubs<
    typeof import("../src/features/settings/appearance-theme/snapshot.ts")
  >(
    new URL(
      "../src/features/settings/appearance-theme/snapshot.ts",
      import.meta.url,
    ),
    {
      "@/features/chat-background": {},
      "@/lib/base64-bytes": {},
      "@/lib/image-fit": {},
      "@/lib/native-files": {},
      "../stores/appearance-custom-store.ts": {},
      "../stores/theme-store.ts": {},
      "./theme-file.ts": real,
    },
  );
  const g = globalThis as { FontFace?: unknown };
  g.FontFace = function (name: string) {
    return {
      load: () =>
        name === "Bad"
          ? Promise.reject(new Error("decode"))
          : Promise.resolve(),
    };
  };
  const theme = parse(
    file({
      look: { uiFont: "bad", codeFont: "Good" },
      fonts: [font("Bad"), font("Good")],
    }),
  ).theme;
  const [kept, dropped] = await snapshot.withLoadableFonts(theme);
  assert.deepEqual(dropped, ["Bad"]);
  assert.deepEqual(
    kept.fonts?.map((f) => f.name),
    ["Good"],
  );
  assert.deepEqual(
    [kept.customization.uiFont, kept.customization.codeFont],
    [null, "Good"],
  );
});

test("undo restores the previous look but keeps personal settings changed since", () => {
  const before = withLook({ contrast: 30, reduceMotion: "off" });
  const now = withLook({
    contrast: 70,
    reduceMotion: "on",
    pointerCursors: true,
  });
  const target = undoTarget(before, now).customization;
  assert.deepEqual(
    [target.contrast, target.reduceMotion, target.pointerCursors],
    [30, "on", true],
  );
});

test("glass scope round-trips and counts as a glass change", () => {
  const saved = {
    ...BASE,
    glass: {
      ...BASE.glass,
      material: "frosted" as const,
      scope: "newChat" as const,
    },
  };
  const theme = roundTrip(saved);
  assert.equal(theme.glass.scope, "newChat");
  const plan = apply(
    { ...saved, glass: { ...saved.glass, scope: "all" } },
    theme,
  );
  assert.equal(plan.glass.scope, "newChat");
  assert.deepEqual(plan.changes, ["glass"]);
});
