// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
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

function mix(foreground: string, background: string, opacity: number): string {
  const channel = (index: number) => {
    const front = Number.parseInt(foreground.slice(index, index + 2), 16);
    const back = Number.parseInt(background.slice(index, index + 2), 16);
    return Math.round(front * opacity + back * (1 - opacity))
      .toString(16)
      .padStart(2, "0");
  };
  return `#${channel(1)}${channel(3)}${channel(5)}`;
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
    const renderedAccent = vars.get("--primary") ?? "";
    const chosen = vars.get("--primary-foreground") ?? "";
    assert.ok(
      ratio(renderedAccent, chosen) >=
        Math.max(
          ratio(renderedAccent, "#111417"),
          ratio(renderedAccent, "#ffffff"),
        ),
      `${renderedAccent}: ${chosen} is not the highest-contrast foreground`,
    );
  }
});

test("accent labels always clear WCAG AA, including the ink crossover", () => {
  for (const accent of ["#22c55e", "#7a7a7a"]) {
    applyCustomizationToDocument(withAccent(accent), "light");
    const renderedAccent = vars.get("--primary") ?? "";
    const chosen = vars.get("--primary-foreground") ?? "";
    assert.ok(
      ratio(renderedAccent, chosen) >= 4.5,
      `${renderedAccent} on ${chosen} is only ${ratio(renderedAccent, chosen).toFixed(2)}:1`,
    );
  }
  assert.equal(vars.get("--primary-foreground"), "#000000");
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

test("custom accents already safe on their strongest wash stay untouched", () => {
  for (const [accent, mode] of [
    ["#17b88b", "dark"],
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
  const wash = mix(corrected, background, 0.2);
  assert.ok(
    ratio(corrected, wash) >= 2.5,
    `${corrected} is only ${ratio(corrected, wash).toFixed(2)}:1 on ${wash}`,
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

test("a custom background does not hide the unchanged elevated surfaces", () => {
  const onDark = {
    ...DEFAULT_CUSTOMIZATION,
    colors: {
      light: { accent: "#fde68a", background: "#101014", foreground: null },
      dark: { ...DEFAULT_CUSTOMIZATION.colors.dark, accent: null },
    },
  };
  applyCustomizationToDocument(onDark, "light");
  const corrected = vars.get("--primary") ?? "";
  assert.notEqual(corrected, "#fde68a");
  assert.ok(ratio(corrected, "#101014") >= 2.5);
  assert.ok(ratio(corrected, "#ffffff") >= 2.5);
});

test("the correction endpoint follows contrast rather than a luminance guess", () => {
  const midGray = {
    ...DEFAULT_CUSTOMIZATION,
    colors: {
      light: { accent: "#aaaaaa", background: "#aaaaaa", foreground: null },
      dark: { ...DEFAULT_CUSTOMIZATION.colors.dark, accent: null },
    },
  };
  applyCustomizationToDocument(midGray, "light");
  const corrected = vars.get("--primary") ?? "";
  assert.notEqual(corrected, "#ffffff");
  assert.ok(ratio(corrected, "#aaaaaa") >= 2.5);
  assert.ok(ratio(corrected, "#ffffff") >= 2.5);
});

test("a narrow valid band between custom and elevated surfaces is not skipped", () => {
  const splitSurfaces = {
    ...DEFAULT_CUSTOMIZATION,
    colors: {
      light: { ...DEFAULT_CUSTOMIZATION.colors.light, accent: null },
      dark: { accent: "#44d088", background: "#4bba47", foreground: null },
    },
  };
  applyCustomizationToDocument(splitSurfaces, "dark");
  const corrected = vars.get("--primary") ?? "";
  assert.ok(ratio(corrected, "#4bba47") >= 2.5);
  assert.ok(ratio(corrected, "#212121") >= 2.5);
});

test("loaded-model stripes stay on the semantic success color", () => {
  const hubCss = readFileSync(
    new URL("../src/features/hub/hub.css", import.meta.url),
    "utf8",
  );
  const activeBlocks = [
    ...hubCss.matchAll(/\[data-active="true"\]\s*\{([^}]+)\}/g),
  ];
  assert.equal(activeBlocks.length, 4);
  for (const [, declarations = ""] of activeBlocks) {
    assert.match(declarations, /box-shadow:[^;]+var\(--status-success\)/);
    assert.doesNotMatch(declarations, /var\(--primary\)/);
  }
});

test("resize-handle glows follow the primary token", () => {
  const source = readFileSync(
    new URL("../src/components/ui/resizable.tsx", import.meta.url),
    "utf8",
  );
  assert.doesNotMatch(source, /rgba\(23,\s*184,\s*139/);
  assert.equal(
    source.match(/color-mix\(in_srgb,var\(--primary\)_[0-9]+%,transparent\)/g)
      ?.length,
    3,
  );
});
