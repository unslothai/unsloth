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
