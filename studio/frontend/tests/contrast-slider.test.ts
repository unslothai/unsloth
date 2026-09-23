// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readdirSync } from "node:fs";
import { join } from "node:path";

import { readSrc, readText } from "./helpers/kit.ts";

// Contrast has to reach what the eye reads as contrast: the fill of a card, a
// menu, a button or an input pill, and the hairlines between them. It does that
// by owning the tokens, so these tests pin the ownership.

const CSS = readSrc("index.css");
const HUB_CSS = readSrc("features/hub/hub.css");
const STORE = readSrc("features/settings/stores/appearance-custom-store.ts");
const SNAPSHOT = readText("../public/reload-snapshot.js");
/** Every source that can carry a class, so a fixed wash cannot slip in. */
const SOURCES = (function walk(dir: string): string[] {
  return readdirSync(join(import.meta.dirname, "../src", dir), {
    withFileTypes: true,
  }).flatMap((entry) => {
    const path = dir ? `${dir}/${entry.name}` : entry.name;
    if (entry.isDirectory()) return walk(path);
    return /\.(tsx?|css)$/.test(entry.name) ? [path] : [];
  });
})("");

/** Fills first, then lines. Every one is authored by the palettes as -base. */
const SURFACE_TOKENS = [
  "card",
  // The sidebar is a surface like the rest. Left off the curve it held its
  // authored tone while its own rows washed past it, and the lit row came out
  // darker than the sidebar behind it.
  "sidebar",
  "popover",
  "panel-input-surface",
  "panel-input-surface-hover",
  "tabs-line-indicator",
];
/** Chips, secondary buttons and the muted hovers, between planes and states. */
const CHIP_TOKENS = ["secondary", "muted"];
/** Hover and selection fills, on a shorter curve so the state stays findable. */
const STATE_TOKENS = [
  "accent",
  "nav-surface-hover",
  "panel-surface-hover",
  "sidebar-accent",
  "chat-icon-bg-hover",
];
const LINE_TOKENS = ["border", "input", "sidebar-border"];
/** Where palette fills and lines head, so a custom foreground cannot steer them. */
const PANEL_TARGET = "var(--contrast-panel-target, var(--contrast-target))";
const FILL_TOKENS = [...SURFACE_TOKENS, ...CHIP_TOKENS, ...STATE_TOKENS];
/** Body copy and labels: head for pure black/white raising, into the page lowering. */
const INK_TOKENS = [
  "foreground",
  "card-foreground",
  "popover-foreground",
  "secondary-foreground",
  "accent-foreground",
  "sidebar-foreground",
  "sidebar-accent-foreground",
  "nav-fg",
];
/** Idle labels and icons, on the same curve as --muted-foreground. */
const QUIET_INK_TOKENS = ["nav-fg-muted", "nav-icon-idle", "chat-icon-fg"];

function block(marker: string): string {
  const at = CSS.indexOf(marker);
  assert.notEqual(at, -1, `${marker} is gone from index.css`);
  const start = CSS.lastIndexOf("{", at);
  return CSS.slice(start, CSS.indexOf("\n}", at));
}

test("the palettes author base values, never the token the app reads", () => {
  for (const token of [
    ...FILL_TOKENS,
    ...LINE_TOKENS,
    ...INK_TOKENS,
    ...QUIET_INK_TOKENS,
  ]) {
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
  for (const token of [
    ...FILL_TOKENS,
    ...LINE_TOKENS,
    ...INK_TOKENS,
    ...QUIET_INK_TOKENS,
  ]) {
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
        `--${token}: color-mix(in oklab, var(--${token}-base), ${PANEL_TARGET} var(--contrast-surface-mix));`,
      ),
      `--${token} is not on the surface curve`,
    );
  }
  for (const token of CHIP_TOKENS) {
    assert.ok(
      adjusted.includes(
        `--${token}: color-mix(in oklab, var(--${token}-base), ${PANEL_TARGET} var(--contrast-fill-mix, var(--contrast-surface-mix)));`,
      ),
      `--${token} is not on the fill curve`,
    );
  }
  // A hover nobody can find is the same failure as an outline nobody can find.
  for (const token of STATE_TOKENS) {
    assert.ok(
      adjusted.includes(
        `--${token}: color-mix(in oklab, var(--${token}-base), ${PANEL_TARGET} var(--contrast-state-mix, var(--contrast-surface-mix)));`,
      ),
      `--${token} is not on the state curve`,
    );
  }
  assert.ok(
    adjusted.includes(`--border: color-mix(in oklab, var(--border-base), ${PANEL_TARGET} var(--contrast-line-mix));`),
  );
  assert.ok(
    adjusted.includes(`--sidebar-border: color-mix(in oklab, var(--sidebar-border-base), ${PANEL_TARGET} var(--contrast-line-mix));`),
  );
  // A control outline nobody can find is not low contrast, it is broken, so
  // --input stops short of the hairlines.
  assert.ok(
    adjusted.includes(`--input: color-mix(in oklab, var(--input-base), ${PANEL_TARGET} var(--contrast-control-mix));`),
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
    "CONTRAST_FILL_MIX_VAR",
    "CONTRAST_LINE_MIX_VAR",
    "CONTRAST_CONTROL_MIX_VAR",
    "CONTRAST_STATE_MIX_VAR",
  ]) {
    const { raising, lowering } = ceilings(name);
    assert.ok(raising < lowering, `${name} is symmetric`);
    assert.ok(lowering < 100, `${name} flattens all the way to the page`);
  }
});

test("a lit row never sinks below the surface it sits on", () => {
  // Both fall toward the page when contrast drops. While they shared a ceiling
  // they fell together, and since the fills start closer to the page the lit
  // row reached it first and went under: at the bottom of the slider the active
  // sidebar row was darker than the sidebar. A shorter curve keeps the order.
  const ceiling = (name: string) => {
    const hit = new RegExp(`${name}, mix\\(raising \\? \\d+ : (\\d+)\\)`).exec(
      STORE,
    );
    assert.ok(hit, `${name} is not set from the two ceilings`);
    return Number(hit[1]);
  };
  assert.ok(
    ceiling("CONTRAST_STATE_MIX_VAR") < ceiling("CONTRAST_SURFACE_MIX_VAR"),
    "hover fills flatten as fast as the surfaces under them",
  );
});

test("raising widens the step from a row to the lit row", () => {
  // The lit row and the surface under it both head for the foreground. While
  // they shared a ceiling the step between them held still, and in light mode
  // it closed: raising the slider made hover and selection harder to see. The
  // states take the steepest curve, the chips the next, the planes a nudge.
  const raising = (name: string) => {
    const hit = new RegExp(`${name}, mix\\(raising \\? (\\d+) :`).exec(STORE);
    assert.ok(hit, `${name} is not set from the two ceilings`);
    return Number(hit[1]);
  };
  const surface = raising("CONTRAST_SURFACE_MIX_VAR");
  const fill = raising("CONTRAST_FILL_MIX_VAR");
  const state = raising("CONTRAST_STATE_MIX_VAR");
  assert.ok(surface < fill, "chips lift no further than the planes");
  assert.ok(fill < state, "hovers lift no further than the chips under them");
  assert.ok(state >= 2 * surface, "the lit row barely separates from its row");
});

test("ink follows the slider", () => {
  // Contrast is first of all text against its background. Held fixed, a
  // raised slider only greyed the surfaces behind the text and lowered it.
  const adjusted = block("--card: color-mix(in oklab, var(--card-base)");
  assert.ok(
    adjusted.includes(
      "--foreground: color-mix(in oklab, var(--foreground-base), var(--contrast-ink-target, transparent) var(--contrast-ink-mix, 0%));",
    ),
    "--foreground is not on the ink curve",
  );
  for (const token of INK_TOKENS.filter((token) => token !== "foreground")) {
    assert.ok(
      adjusted.includes(
        `--${token}: color-mix(in oklab, var(--${token}-base), var(--contrast-panel-ink-target, var(--contrast-ink-target, transparent)) var(--contrast-ink-mix, 0%));`,
      ),
      `--${token} is not on the panel ink curve`,
    );
  }
  for (const token of QUIET_INK_TOKENS) {
    // Nav ink sits on palette surfaces; chat icons sit on the page.
    const target = token.startsWith("nav-") ? PANEL_TARGET : "var(--contrast-target)";
    assert.ok(
      adjusted.includes(
        `--${token}: color-mix(in oklab, var(--${token}-base), ${target} var(--contrast-text-mix));`,
      ),
      `--${token} is not on the text curve`,
    );
  }
  // Lowering stops well short of the page, or body copy stops being legible.
  const hit = /CONTRAST_INK_MIX_VAR, mix\(raising \? (\d+) : (\d+)\)/.exec(
    STORE,
  );
  assert.ok(hit, "the ink mix is not set from the two ceilings");
  assert.ok(Number(hit[2]) <= 35, "lowering washes the ink into the page");
  // A custom foreground is written as the base, so it takes the curve too.
  assert.match(STORE, /setVar\("--foreground-base", colors\.foreground\)/);
  assert.doesNotMatch(STORE, /setVar\("--foreground", colors\.foreground\)/);
  assert.ok(SNAPSHOT.includes('"--foreground-base"'));
});

test("raising pushes ink its own way, never toward its surface", () => {
  // Picked from the page, a dark custom background in light mode sent card,
  // menu and sidebar text toward white on their white surfaces (2.3:1).
  assert.match(
    STORE,
    /CONTRAST_PANEL_INK_TARGET_VAR,\s*raising \? palettePole : "var\(--background\)"/,
  );
  assert.match(
    STORE,
    /const palettePole = resolved === "light" \? "#000000" : "#ffffff";/,
  );
  // --foreground heads away from its page: #767676 on white is nearer white
  // by luminance, and pushed there it fell from 4.5:1 to 1.5:1.
  assert.match(
    STORE,
    /raising\s*\?\s*colors\.foreground\s*\?\s*inkPole\(\s*colors\.foreground,\s*colors\.background \?\? paletteSurfaces\.background,?\s*\)\s*:\s*palettePole/,
  );
  assert.match(
    STORE,
    /hexLuminance\(ink\) <= hexLuminance\(page\) \? "#000000" : "#ffffff"/,
  );
  assert.doesNotMatch(STORE, /\.background \?\? PALETTE_SURFACES\[resolved\]\.background;\s*const page/);
  assert.ok(SNAPSHOT.includes('"--contrast-panel-ink-target"'));
});

test("raising lifts palette fills away from their surfaces under any foreground", () => {
  // A white custom foreground on a dark custom page in light mode sent the
  // white sidebar's lit row toward white: its step fell from 1.18:1 to 1.16:1.
  assert.match(
    STORE,
    /CONTRAST_PANEL_TARGET_VAR,\s*raising && colors\.foreground \? palettePole : null/,
  );
  assert.match(STORE, /setVar\(CONTRAST_PANEL_TARGET_VAR, null\);/);
  assert.ok(SNAPSHOT.includes('"--contrast-panel-target"'));
  // The dark find bar stands in for --card, so it follows the same target.
  assert.ok(
    CSS.includes(
      "background-color: color-mix(in oklab, #2c2c2c, var(--contrast-panel-target, var(--contrast-target, transparent)) var(--contrast-surface-mix, 0%));",
    ),
  );
  assert.equal(
    (CSS.match(/--(?:border|input): color-mix\(in oklab, var\(--(?:border|input)-base\), var\(--contrast-panel-target, var\(--contrast-target\)\)/g) ?? []).length,
    4,
    "a scoped line re-derivation lost the panel target",
  );
});

test("the sidebar section labels follow the slider", () => {
  // Authored as fixed greys, "Recents" and "Train" held still while every
  // label around them moved.
  for (const grey of ["#80868b", "#9aa0a6"]) {
    assert.ok(
      CSS.includes(
        `color: color-mix(in oklab, ${grey}, var(--contrast-panel-target, var(--contrast-target, transparent)) var(--contrast-text-mix, 0%));`,
      ),
      `${grey} is not on the text curve`,
    );
  }
});

test("a dark selection fill takes the token, not a wash", () => {
  // A wash is scaled by --contrast-wash-gain, which still falls further at the
  // bottom of the range than --contrast-state-mix does. Painted as washes, the
  // selected tab and the selected quant sank to within a couple of levels of
  // the surface under them there.
  assert.match(
    HUB_CSS,
    /html\.dark \.hub-tab-toggle-pill,\s*html\.dark \.hub-tab-toggle-pill:hover \{[^}]*background-color: var\(--accent\)/,
    "the selected segment is not on --accent",
  );
  for (const file of [
    "features/hub/catalog/gguf-download-card.tsx",
    "features/hub/catalog/models-table.tsx",
  ]) {
    const source = readSrc(file);
    assert.match(
      source,
      /dark:(data-\[selected\]:)?bg-accent/,
      `${file} does not paint its selection with --accent`,
    );
    // Scoped to the selection utility itself; resting chips and progress
    // tracks in these files carry the gain on purpose.
    assert.doesNotMatch(
      source,
      /dark:data-\[selected\]:bg-\[[^\]]*contrast-wash-gain/,
      `${file} still paints a selection with a wash`,
    );
  }
  // The quant row's hover is --accent held back, so it cannot reach the
  // selected row: as its own wash it closed to a few levels at high contrast.
  assert.match(
    readSrc("features/hub/catalog/gguf-download-card.tsx"),
    /dark:hover:bg-\[color-mix\(in_srgb,var\(--accent\)_\d+%,transparent\)\]/,
    "the quant row hover is not derived from --accent",
  );
});

test("panel sliders move with the slider too", () => {
  // .panel-slider repaints the track, fill and thumb with !important, so the
  // component's own gain-aware colours never reach them.
  assert.match(
    CSS,
    /\.panel-slider \[data-slot="slider-track"\] \{[^}]*rgb\(0 0 0 \/ calc\(0\.025 \* var\(--contrast-wash-gain, 1\)\)\)/,
  );
  assert.match(
    CSS,
    /\.dark \.panel-slider \[data-slot="slider-track"\] \{[^}]*rgb\(255 255 255 \/ calc\(0\.025 \* var\(--contrast-wash-gain, 1\)\)\)/,
  );
  assert.ok(CSS.includes("--panel-slider-fg: var(--panel-slider-fg-base);"));
  assert.match(
    CSS,
    /--panel-slider-fg: color-mix\(\s*in oklab,\s*var\(--panel-slider-fg-base\),\s*var\(--contrast-target\) var\(--contrast-text-mix\)/,
  );
});

test("a resting wash and its hover twin share the gain", () => {
  // One scaled and one fixed alpha invert at the top of the range: the hover
  // ends up fainter than the resting fill.
  for (const file of [
    "features/profile/components/profile-personalization-panel.tsx",
    "components/assistant-ui/chat-dictation-bar.tsx",
  ]) {
    const source = readSrc(file);
    const scaled = source.match(/dark:(hover:)?bg-\[rgb\(255_255_255/g) ?? [];
    assert.equal(scaled.length, 2, `${file} scales only one of the pair`);
    assert.doesNotMatch(
      source,
      /dark:hover:bg-white\//,
      `${file} still has a fixed hover wash`,
    );
  }
  // Same for the borders that carry the edge gain: a fixed hover or drag
  // outline is a jump out of the setting, not a state change.
  for (const file of [
    "features/studio/sections/dataset-upload.tsx",
    "features/settings/components/color-picker.tsx",
  ]) {
    assert.doesNotMatch(
      readSrc(file),
      /border-(white|black)\/(\[0?\.\d+\]|\d+)/,
      `${file} still has a fixed border alpha`,
    );
  }
});

test("a light wash follows the slider as its dark twin does", () => {
  // bg-foreground/[x] is a fixed alpha, so a row that dimmed on hover in dark
  // mode stayed put in light mode. The mix carries the gain instead, and
  // resolves to the same colour at the default. Both spellings count, the
  // arbitrary one and Tailwind's shorthand.
  // index.css authors the same washes in @apply, in black and white rather
  // than the token, and both of those take an arbitrary alpha or Tailwind's
  // shorthand, so all four spellings are swept.
  // The arbitrary value is a fifth spelling: the colour and its alpha both sit
  // inside the brackets, so neither /alpha form above sees it.
  const FIXED_WASH =
    /(bg-foreground\/(\[[\d.]+\]|\d+)|bg-(white|black)\/(\[0?\.\d+\]|\d+)|bg-\[rgba?\([^\]]*[\s,\/]0?\.\d+\s*\)\])/;
  // A stylesheet can also write the wash out longhand, with no class in sight.
  const RAW_WASH =
    /background(-color)?:\s*rgba?\([\d\s,]+[\s,\/]+0?\.\d+\s*\)/;
  // Two fills are not chrome: the dark button's own surface, and the snippet
  // highlight that sits beside amber and red siblings on no curve at all.
  const NOT_CHROME = new Set([
    "components/ui/button.tsx",
    "features/security/components/remote-code-consent-dialog.tsx",
  ]);
  // These paint over content rather than tinting chrome: the media viewers
  // and their controls, the image hover scrims, the selection markers. They
  // stage a picture at a contrast of their own and keep it.
  const OVER_CONTENT = new Set([
    "components/assistant-ui/attachment-preview.tsx",
    "components/assistant-ui/image.tsx",
    "components/assistant-ui/markdown-text.tsx",
    "components/assistant-ui/tool-ui-image-generation.tsx",
    "features/images/images-page.tsx",
    "features/video/video-page.tsx",
  ]);
  // Scrims and opaque stages cover content too, wherever they are declared.
  const SCRIM = /(overlayClassName|bg-(black|white)\/(\[0?\.[3-9]\d*\]|[3-9]\d|100))/;
  const hits = SOURCES.filter((file) => {
    if (NOT_CHROME.has(file) || OVER_CONTENT.has(file)) return false;
    const source = readSrc(file);
    return source
      .split("\n")
      .some(
        (line) =>
          (FIXED_WASH.test(line) || RAW_WASH.test(line)) && !SCRIM.test(line),
      );
  });
  assert.deepEqual(hits, [], "these washes ignore the contrast setting");

  // Hairlines are the part of a control the eye reads first, so they follow
  // the edge gain the same way, drawn as a border or as a ring.
  const FIXED_LINE = /(border|ring)-foreground\/(\[[\d.]+\]|\d+)/;
  const lines = SOURCES.filter((file) => FIXED_LINE.test(readSrc(file)));
  assert.deepEqual(lines, [], "these outlines ignore the contrast setting");
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
