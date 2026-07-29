// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

/** Minimal <html> stand-in: the applier only needs style, attributes, classes. */
function stubDocument() {
  const vars = new Map<string, string>();
  const attributes = new Set<string>();
  const classes = new Set<string>();
  const element = {
    style: {
      setProperty: (name: string, value: string) => vars.set(name, value),
      removeProperty: (name: string) => vars.delete(name),
    },
    setAttribute: (name: string) => attributes.add(name),
    removeAttribute: (name: string) => attributes.delete(name),
    toggleAttribute: (name: string, on: boolean) =>
      on ? attributes.add(name) : attributes.delete(name),
    classList: {
      toggle: (name: string, on: boolean) =>
        on ? classes.add(name) : classes.delete(name),
    },
  };
  // No "fonts" key, so syncImportedFonts bails before touching FontFace.
  (globalThis as { document?: unknown }).document = {
    documentElement: element,
  };
  return vars;
}

const vars = stubDocument();

const { applyCustomizationToDocument, DEFAULT_CUSTOMIZATION } = await import(
  "../src/features/settings/stores/appearance-custom-store.ts"
);

const withAccent = (accent: string | null) => ({
  ...DEFAULT_CUSTOMIZATION,
  colors: {
    light: { ...DEFAULT_CUSTOMIZATION.colors.light, accent },
    dark: { ...DEFAULT_CUSTOMIZATION.colors.dark, accent },
  },
});

test("a custom accent recolors the brand variable, not just the control one", () => {
  applyCustomizationToDocument(withAccent("#7c3aed"), "light");

  // --primary drives primary buttons, active composer pills and the meter
  // percentages; leaving it out is what stranded them on the palette green.
  assert.equal(vars.get("--primary"), "#7c3aed");
  assert.equal(vars.get("--control-accent"), "#7c3aed");
  assert.equal(vars.get("--chart-1"), "#7c3aed");
});

test("both foregrounds follow the accent so button labels stay readable", () => {
  applyCustomizationToDocument(withAccent("#7c3aed"), "light");
  const onDark = vars.get("--primary-foreground");
  assert.equal(vars.get("--control-accent-foreground"), onDark);

  applyCustomizationToDocument(withAccent("#fde68a"), "light");
  assert.notEqual(vars.get("--primary-foreground"), onDark);
});

/** WCAG relative luminance, independent of the implementation under test. */
function luminance(hex: string): number {
  const channel = (index: number) => {
    const value = Number.parseInt(hex.slice(index, index + 2), 16) / 255;
    return value <= 0.03928 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
  };
  return 0.2126 * channel(1) + 0.7152 * channel(3) + 0.0722 * channel(5);
}

function ratio(a: string, b: string): number {
  const [high, low] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return ((high ?? 0) + 0.05) / ((low ?? 0) + 0.05);
}

test("the foreground is the higher-contrast of the two, not a luminance guess", () => {
  // Mid-tone accents are the ones a fixed 0.45 cutoff got wrong.
  for (const accent of [
    "#22c55e",
    "#17b88b",
    "#339cff",
    "#f59e0b",
    "#4ade80",
    "#7c3aed",
    "#e11d48",
    "#0d0d0d",
    "#fde68a",
    "#ececec",
  ]) {
    applyCustomizationToDocument(withAccent(accent), "light");
    const chosen = vars.get("--primary-foreground") ?? "";
    const other = chosen === "#ffffff" ? "#111417" : "#ffffff";
    assert.ok(
      ratio(accent, chosen) >= ratio(accent, other),
      `${accent}: picked ${chosen} at ${ratio(accent, chosen).toFixed(2)}:1 over ${other} at ${ratio(accent, other).toFixed(2)}:1`,
    );
  }
});

test("a saturated green label clears WCAG AA instead of failing it", () => {
  applyCustomizationToDocument(withAccent("#22c55e"), "light");
  const chosen = vars.get("--primary-foreground") ?? "";
  assert.ok(
    ratio("#22c55e", chosen) >= 4.5,
    `#22c55e on ${chosen} is only ${ratio("#22c55e", chosen).toFixed(2)}:1`,
  );
});

test("no accent leaves every palette variable alone", () => {
  applyCustomizationToDocument(withAccent("#7c3aed"), "light");
  applyCustomizationToDocument(withAccent(null), "light");

  for (const name of [
    "--primary",
    "--primary-foreground",
    "--control-accent",
    "--control-accent-foreground",
    "--chart-1",
  ]) {
    assert.equal(vars.has(name), false, `${name} should be removed`);
  }
});

test("focus rings are never touched, so highlight borders stay neutral", () => {
  applyCustomizationToDocument(withAccent("#7c3aed"), "light");
  assert.equal(vars.has("--ring"), false);
  // Status green is a signal, not a theme color.
  assert.equal(vars.has("--verified"), false);
});

test("shipped palette accents are never second-guessed", () => {
  // Every accent the app itself offers already clears the bar.
  for (const [accent, mode] of [
    ["#17b88b", "light"],
    ["#17b88b", "dark"],
    ["#339cff", "light"],
    ["#4dabff", "dark"],
    ["#171717", "light"],
    ["#ededed", "dark"],
  ] as const) {
    applyCustomizationToDocument(withAccent(accent), mode);
    assert.equal(vars.get("--primary"), accent, `${accent} in ${mode}`);
  }
});

test("an accent too pale to read as text is pulled into range", () => {
  const background = "#ffffff";
  assert.ok(ratio("#fde68a", background) < 2.5);

  applyCustomizationToDocument(withAccent("#fde68a"), "light");
  const corrected = vars.get("--primary") ?? "";

  assert.notEqual(corrected, "#fde68a");
  assert.ok(
    ratio(corrected, background) >= 2.5,
    `${corrected} is only ${ratio(corrected, background).toFixed(2)}:1`,
  );
  // Darkened, not discarded: still a yellow, red channel still leads.
  const [r, g, b] = [1, 3, 5].map((i) =>
    Number.parseInt(corrected.slice(i, i + 2), 16),
  ) as [number, number, number];
  assert.ok(r > b && g > b, `${corrected} lost its hue`);
});

test("a near-black accent is lifted in dark mode instead", () => {
  applyCustomizationToDocument(withAccent("#101010"), "dark");
  const corrected = vars.get("--primary") ?? "";
  assert.notEqual(corrected, "#101010");
  assert.ok(ratio(corrected, "#181818") >= 2.5);
  // Lightened, so it reads against the dark page.
  assert.ok(
    Number.parseInt(corrected.slice(1, 3), 16) > 0x10,
    `${corrected} was not lightened`,
  );
});

test("the label follows the corrected accent, not the raw pick", () => {
  applyCustomizationToDocument(withAccent("#fde68a"), "light");
  const corrected = vars.get("--primary") ?? "";
  const chosen = vars.get("--primary-foreground") ?? "";
  const other = chosen === "#ffffff" ? "#111417" : "#ffffff";
  assert.ok(ratio(corrected, chosen) >= ratio(corrected, other));
});

test("a custom background moves the bar with it", () => {
  // On a dark custom background, a pale accent is already legible.
  const onDark = {
    ...DEFAULT_CUSTOMIZATION,
    colors: {
      light: { accent: "#fde68a", background: "#101014", foreground: null },
      dark: { ...DEFAULT_CUSTOMIZATION.colors.dark, accent: null },
    },
  };
  applyCustomizationToDocument(onDark, "light");
  assert.equal(vars.get("--primary"), "#fde68a");
});
