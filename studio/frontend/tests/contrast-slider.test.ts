// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, readText } from "./helpers/kit.ts";

// Contrast has to reach what the eye reads as contrast: the fill of a card, a
// menu, a button or an input pill, and the hairlines between them. It does that
// by owning the tokens, so these tests pin the ownership.

const CSS = readSrc("index.css");
const STORE = readSrc("features/settings/stores/appearance-custom-store.ts");
const SNAPSHOT = readText("../public/reload-snapshot.js");

/** Fills first, then lines. Every one is authored by the palettes as -base. */
const SURFACE_TOKENS = [
  "card",
  "popover",
  "secondary",
  "muted",
  "accent",
  "nav-surface-hover",
  "panel-surface-hover",
  "panel-input-surface",
];
const LINE_TOKENS = ["border", "input", "sidebar-border"];

function block(marker: string): string {
  const at = CSS.indexOf(marker);
  assert.notEqual(at, -1, `${marker} is gone from index.css`);
  const start = CSS.lastIndexOf("{", at);
  return CSS.slice(start, CSS.indexOf("\n}", at));
}

test("the palettes author base values, never the token the app reads", () => {
  for (const token of [...SURFACE_TOKENS, ...LINE_TOKENS]) {
    const authored = CSS.match(new RegExp(`^\\t--${token}-base:`, "gm")) ?? [];
    assert.ok(
      authored.length >= 2,
      `--${token}-base is not authored by the palettes`,
    );
    // A palette declaring the public token would beat the contrast rule on
    // specificity and opt itself out.
    const direct = (CSS.match(new RegExp(`^\\t--${token}: ([^;]+);`, "gm")) ?? [])
      .filter((line) => !line.includes(`var(--${token}-base)`));
    assert.deepEqual(
      direct,
      [],
      `a palette still declares --${token} instead of --${token}-base`,
    );
  }
});

test("at the default the derivation is the identity", () => {
  const identity = block("--card: var(--card-base);");
  for (const token of [...SURFACE_TOKENS, ...LINE_TOKENS]) {
    assert.ok(
      identity.includes(`--${token}: var(--${token}-base);`),
      `--${token} does not fall back to its authored value`,
    );
  }
});

test("off the default, fills and lines each take their own curve", () => {
  const adjusted = block("--card: color-mix(in oklab, var(--card-base)");
  for (const token of SURFACE_TOKENS) {
    assert.ok(
      adjusted.includes(
        `--${token}: color-mix(in oklab, var(--${token}-base), var(--contrast-target) var(--contrast-surface-mix));`,
      ),
      `--${token} is not on the surface curve`,
    );
  }
  assert.ok(
    adjusted.includes("--border: color-mix(in oklab, var(--border-base), var(--contrast-target) var(--contrast-line-mix));"),
  );
  assert.ok(
    adjusted.includes("--sidebar-border: color-mix(in oklab, var(--sidebar-border-base), var(--contrast-target) var(--contrast-line-mix));"),
  );
  // A control outline nobody can find is not low contrast, it is broken, so
  // --input stops short of the hairlines.
  assert.ok(
    adjusted.includes("--input: color-mix(in oklab, var(--input-base), var(--contrast-target) var(--contrast-control-mix));"),
  );
});

test("lowering flattens further than raising lifts", () => {
  // A card mixed far toward the foreground reads as a block, not a surface, so
  // the top of the range is a nudge and the bottom collapses into the page.
  const ceilings = (name: string) => {
    const hit = new RegExp(
      `${name}, mix\\(raising \\? (\\d+) : (\\d+)\\)`,
    ).exec(STORE);
    assert.ok(hit, `${name} is not set from the two ceilings`);
    return { raising: Number(hit[1]), lowering: Number(hit[2]) };
  };
  for (const name of [
    "CONTRAST_SURFACE_MIX_VAR",
    "CONTRAST_LINE_MIX_VAR",
    "CONTRAST_CONTROL_MIX_VAR",
  ]) {
    const { raising, lowering } = ceilings(name);
    assert.ok(raising < lowering, `${name} is symmetric`);
    assert.ok(lowering < 100, `${name} flattens all the way to the page`);
  }
});

test("the hand-written washes follow the slider, the scrims do not", () => {
  const app = readSrc("features/settings/settings-dialog.tsx");
  // Settings controls wash the page instead of taking a token, so they carry
  // the gain explicitly.
  assert.match(app, /calc\(0\.06\*var\(--contrast-wash-gain,1\)\)/);
  assert.doesNotMatch(app, /bg-white\/\[0\.0[0-9]\]/);
  // An overlay that covers content is not chrome and keeps its own alpha.
  assert.match(
    readSrc("components/assistant-ui/markdown-text.tsx"),
    /bg-black\/10/,
  );
});

test("a reload repaints at the contrast the user chose", () => {
  // reload-snapshot.js replays these onto the replacement document. A missing
  // one paints stock contrast until React catches up.
  const written = [
    ...STORE.matchAll(/setVar\((CONTRAST_\w+_VAR|"--contrast-target")/g),
  ].map((match) => match[1]);
  const names = new Set(
    written.map((name) => {
      if (name.startsWith('"')) return name.slice(1, -1);
      const declared = new RegExp(`${name} = "(--[a-z-]+)"`).exec(STORE);
      assert.ok(declared, `${name} has no variable name`);
      return declared[1];
    }),
  );
  assert.ok(names.size >= 6);
  for (const name of names) {
    assert.ok(
      SNAPSHOT.includes(`"${name}"`),
      `${name} is missing from reload-snapshot.js`,
    );
  }
});
