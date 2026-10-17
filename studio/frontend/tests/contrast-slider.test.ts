// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readdirSync } from "node:fs";
import { join } from "node:path";

import { readSrc, readText } from "./helpers/kit.ts";

// Contrast works by owning the surface and line tokens, so these tests pin that ownership.

const CSS = readSrc("index.css");
const HUB_CSS = readSrc("features/hub/hub.css");
const STORE = readSrc("features/settings/stores/appearance-custom-store.ts");
const SNAPSHOT = readText("../public/reload-snapshot.js");
const SOURCES = (function walk(dir: string): string[] {
  return readdirSync(join(import.meta.dirname, "../src", dir), {
    withFileTypes: true,
  }).flatMap((entry) => {
    const path = dir ? `${dir}/${entry.name}` : entry.name;
    if (entry.isDirectory()) return walk(path);
    return /\.(tsx?|css)$/.test(entry.name) ? [path] : [];
  });
})("");

const SURFACE_TOKENS = [
  "card",
  // The sidebar must be on the curve, or its lit row ends up darker than the sidebar.
  "sidebar",
  "popover",
  "panel-input-surface",
  "panel-input-surface-hover",
  "tabs-line-indicator",
];
const CHIP_TOKENS = ["secondary", "muted"];
const STATE_TOKENS = [
  "accent",
  "nav-surface-hover",
  "panel-surface-hover",
  "sidebar-accent",
  "chat-icon-bg-hover",
];
const LINE_TOKENS = ["border", "input", "sidebar-border"];
const PANEL_TARGET = "var(--contrast-panel-target, var(--contrast-target))";
const FILL_TOKENS = [...SURFACE_TOKENS, ...CHIP_TOKENS, ...STATE_TOKENS];
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
    // A palette declaring the public token would win on specificity and opt out.
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
  for (const token of STATE_TOKENS) {
    const target = token === "chat-icon-bg-hover" ? "var(--contrast-target)" : PANEL_TARGET;
    assert.ok(
      adjusted.includes(
        `--${token}: color-mix(in oklab, var(--${token}-base), ${target} var(--contrast-state-mix, var(--contrast-surface-mix)));`,
      ),
      `--${token} is not on the state curve`,
    );
  }
  assert.ok(
    adjusted.includes("--border: color-mix(in oklab, var(--border-base), var(--contrast-target) var(--contrast-line-mix));"),
  );
  assert.ok(
    adjusted.includes(`--sidebar-border: color-mix(in oklab, var(--sidebar-border-base), ${PANEL_TARGET} var(--contrast-line-mix));`),
  );
  assert.ok(
    adjusted.includes("--input: color-mix(in oklab, var(--input-base), var(--contrast-target) var(--contrast-control-mix));"),
  );
});

test("lowering flattens further than raising lifts", () => {
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
  // A shorter curve keeps the lit sidebar row from sinking below the sidebar at low contrast.
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
  // States take the steepest curve, chips the next, planes a nudge, so hover stays visible.
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
    const target = token.startsWith("nav-") ? PANEL_TARGET : "var(--contrast-target)";
    assert.ok(
      adjusted.includes(
        `--${token}: color-mix(in oklab, var(--${token}-base), ${target} var(--contrast-text-mix));`,
      ),
      `--${token} is not on the text curve`,
    );
  }
  const hit = /CONTRAST_INK_MIX_VAR, mix\(raising \? (\d+) : (\d+)\)/.exec(
    STORE,
  );
  assert.ok(hit, "the ink mix is not set from the two ceilings");
  assert.ok(Number(hit[2]) <= 35, "lowering washes the ink into the page");
  assert.match(STORE, /setVar\("--foreground-base", colors\.foreground\)/);
  assert.doesNotMatch(STORE, /setVar\("--foreground", colors\.foreground\)/);
  assert.ok(SNAPSHOT.includes('"--foreground-base"'));
});

test("raising pushes ink its own way, never toward its surface", () => {
  assert.match(
    STORE,
    /CONTRAST_PANEL_INK_TARGET_VAR,\s*raising \? palettePole : "var\(--background\)"/,
  );
  assert.match(
    STORE,
    /const palettePole = resolved === "light" \? "#000000" : "#ffffff";/,
  );
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
  assert.match(
    STORE,
    /CONTRAST_PANEL_TARGET_VAR,\s*raising && colors\.foreground \? palettePole : null/,
  );
  assert.match(STORE, /setVar\(CONTRAST_PANEL_TARGET_VAR, null\);/);
  assert.ok(SNAPSHOT.includes('"--contrast-panel-target"'));
  assert.ok(
    CSS.includes("background-color: color-mix(in srgb, var(--card), white 8%);"),
  );
});

test("text and lines head away from whatever they are drawn on", () => {
  const scopes = CSS.slice(
    CSS.indexOf("/* Body text, muted text and lines on palette surfaces"),
    CSS.indexOf("/* Code font size"),
  );
  const rules = [...scopes.matchAll(/\{([^}]*)\}/g)].map((m) => m[1] ?? "");
  assert.equal(rules.length, 6);
  for (const rule of rules) {
    const target = /--muted-foreground: color-mix\(\s*in oklab,\s*var\(--panel-surface-fg-muted\),\s*(var\(--contrast-panel-target, var\(--contrast-target\)\)|var\(--contrast-target\)) var/.exec(rule)?.[1];
    assert.ok(target, "scope without muted text");
    assert.ok(rule.includes(`--border: color-mix(in oklab, var(--border-base), ${target} var(--contrast-line-mix));`));
    assert.ok(rule.includes(`--input: color-mix(in oklab, var(--input-base), ${target} var(--contrast-control-mix));`));
    const ink =
      target === PANEL_TARGET
        ? "var(--contrast-panel-ink-target, var(--contrast-ink-target, transparent))"
        : "var(--contrast-ink-target, transparent)";
    assert.ok(
      rule.includes(`--foreground: color-mix(in oklab, var(--foreground-base), ${ink} var(--contrast-ink-mix, 0%));`),
      "a surface scope leaves body text on the other surface's target",
    );
  }
  assert.match(
    CSS,
    /html\[data-contrast-adjust\]:not\(\.dark\) \.diffusion-surface \{\s*--border: color-mix\(in oklab, var\(--border-base\), var\(--contrast-target\) var\(--contrast-line-mix\)\);\s*--input: color-mix\(in oklab, var\(--input-base\), var\(--contrast-target\) var\(--contrast-control-mix\)\);/,
  );
});

test("the sidebar section labels follow the slider", () => {
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
    assert.doesNotMatch(
      source,
      /dark:data-\[selected\]:bg-\[[^\]]*contrast-wash-gain/,
      `${file} still paints a selection with a wash`,
    );
  }
  assert.match(
    readSrc("features/hub/catalog/gguf-download-card.tsx"),
    /dark:hover:bg-\[color-mix\(in_srgb,var\(--accent\)_\d+%,transparent\)\]/,
    "the quant row hover is not derived from --accent",
  );
});

test("panel sliders move with the slider too", () => {
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
  // Sweeps every fixed-alpha wash spelling: /alpha, Tailwind shorthand, @apply black/white,
  // and arbitrary values with the colour and alpha inside the brackets.
  const FIXED_WASH =
    /(bg-foreground\/(\[[\d.]+\]|\d+)|bg-(white|black)\/(\[0?\.\d+\]|\d+)|bg-\[rgba?\([^\]]*[\s,\/]0?\.\d+\s*\)\])/;
  const RAW_WASH =
    /background(-color)?:\s*rgba?\([\d\s,]+[\s,\/]+0?\.\d+\s*\)/;
  const NOT_CHROME = new Set([
    "components/ui/button.tsx",
    "features/security/components/remote-code-consent-dialog.tsx",
  ]);
  const OVER_CONTENT = new Set([
    "components/assistant-ui/attachment-preview.tsx",
    "components/assistant-ui/image.tsx",
    "components/assistant-ui/markdown-text.tsx",
    "components/assistant-ui/tool-ui-image-generation.tsx",
    "features/images/images-page.tsx",
    "features/video/video-page.tsx",
  ]);
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

  const FIXED_LINE = /(border|ring)-foreground\/(\[[\d.]+\]|\d+)/;
  const lines = SOURCES.filter((file) => FIXED_LINE.test(readSrc(file)));
  assert.deepEqual(lines, [], "these outlines ignore the contrast setting");
});

test("the hand-written washes follow the slider, the scrims do not", () => {
  const app = readSrc("features/settings/settings-dialog.tsx");
  assert.match(app, /calc\(0\.06\*var\(--contrast-wash-gain,1\)\)/);
  assert.doesNotMatch(app, /bg-white\/\[0\.0[0-9]\]/);
  assert.match(
    readSrc("components/assistant-ui/markdown-text.tsx"),
    /bg-black\/10/,
  );
});

test("a reload repaints at the contrast the user chose", () => {
  // reload-snapshot.js replays these onto the replacement document; a missing one flashes.
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

test("muted text on a palette surface heads away from that surface", () => {
  const panel = CSS.match(
    /html\[data-contrast-adjust\] :is\(\.bg-card, \.bg-popover, \.bg-sidebar[^)]*\) \{([^}]*)\}/,
  );
  assert.ok(panel, "palette surfaces do not re-derive muted text");
  assert.match(
    panel[1] ?? "",
    /--muted-foreground: color-mix\(\s*in oklab,\s*var\(--panel-surface-fg-muted\),\s*var\(--contrast-panel-target, var\(--contrast-target\)\) var\(--contrast-text-mix\)/,
  );
  const pane = CSS.match(
    /html\[data-contrast-adjust\] :is\(\.bg-card, \.bg-popover, \.bg-sidebar[^)]*\) \.bg-background:not\([^{]*\) \{([^}]*)\}/,
  );
  assert.ok(pane, "page panes inside palette surfaces keep the panel target");
  assert.match(pane[1] ?? "", /var\(--contrast-target\) var\(--contrast-text-mix\)/);
  assert.doesNotMatch(pane[1] ?? "", /--contrast-panel-target/);
});

test("every near-opaque card spelling takes the panel muted text", () => {
  const scoped = CSS.match(
    /html\[data-contrast-adjust\] :is\(([^)]*)\),\s*html\[data-contrast-adjust\]\.dark :is\(([^)]*)\) \{([^}]*)\}/,
  );
  assert.ok(scoped, "opacity and dark card spellings are not scoped");
  assert.match(scoped[3] ?? "", /var\(--contrast-panel-target, var\(--contrast-target\)\)/);
  const light = (scoped[1] ?? "").replaceAll("\\", "");
  const dark = (scoped[2] ?? "").replaceAll("\\", "");
  for (const file of SOURCES.filter((f) => f.endsWith(".tsx"))) {
    for (const [spelling, isDark, alpha] of readSrc(file).matchAll(
      /(?<![\w:-])(dark:)?bg-card(?:\/(\d+))?(?![\w/-])/g,
    )) {
      if (alpha === undefined ? !isDark : Number(alpha) < 50) continue;
      assert.ok(
        (isDark ? dark : light).includes(`.${spelling}`),
        `${file} paints ${spelling} outside the panel scope`,
      );
    }
  }
});

test("surfaces the stylesheet paints take the panel muted text too", () => {
  const always = CSS.match(
    /html\[data-contrast-adjust\] :is\(\.bg-card, \.bg-popover, \.bg-sidebar, ([^)]*)\) \{/,
  );
  assert.ok(always);
  for (const surface of [
    ".menu-soft-surface",
    ".menu-soft-surface-up",
    ".hub-download-panel",
    ".hub-download-fab",
  ]) {
    assert.ok(always[1]?.includes(surface), `${surface} is not a panel`);
    assert.match(
      CSS + HUB_CSS,
      new RegExp(
        `^\\s*(?:\\.[\\w-]+ )?${surface.replace(".", "\\.")}[,\\s{][^}]*(bg-popover|var\\(--(popover|card)\\))`,
        "m",
      ),
      `${surface} no longer paints a palette surface`,
    );
  }
  assert.match(
    CSS,
    /html\[data-contrast-adjust\]:not\(\.dark\) \.hub-page \.hub-result-row \{[^}]*var\(--contrast-panel-target, var\(--contrast-target\)\)/,
  );
  assert.match(HUB_CSS, /^  \.hub-page \.hub-result-row \{[^}]*background-color: var\(--card\)/m);
  assert.match(
    HUB_CSS,
    /html\.dark \.hub-page \.hub-result-row \{\s*background-color: color-mix\(in srgb, var\(--foreground\) [^;]*, var\(--background\)\);/,
  );
  assert.doesNotMatch(always[1] ?? "", /hub-result-row/);
  const dark = [".settings-surface", ".chat-composer-surface", ".unsloth-composer-surface", ".unsloth-plus-menu", ".dialog-soft-surface"];
  const reset = CSS.match(
    /html\[data-contrast-adjust\]\.dark :is\(([^)]*)\) \.bg-background:not\(\[class\*="dark:bg-"\]:not\(\[class\*="dark:bg-background"\]\)\) \{([^}]*)\}/,
  );
  assert.ok(reset, "dark card surfaces keep their page panes on the panel target");
  assert.match(reset[2] ?? "", /var\(--contrast-target\) var\(--contrast-text-mix\)/);
  const cards = CSS.match(/html\[data-contrast-adjust\]\.dark :is\(\.dark\\:bg-card([^)]*)\) \{/);
  for (const surface of dark) {
    assert.ok(reset[1]?.includes(surface), `${surface} page panes are not reset`);
    assert.ok(cards?.[1]?.includes(surface), `${surface} is not a panel in dark`);
  }
  assert.ok(CSS.indexOf(cards?.[0] ?? "@") > CSS.indexOf(reset[0]));
});

test("a pane that repaints itself in dark is not reset to the page there", () => {
  assert.ok(
    CSS.includes(
      '.menu-soft-surface-up, .menu-soft-surface) .bg-background:not(.dark [class*="dark:bg-"]:not([class*="dark:bg-background"])) {',
    ),
  );
  assert.match(
    readSrc("features/settings/settings-dialog.tsx"),
    /bg-background[^"]*dark:bg-\[rgb\(255_255_255/,
  );
});

test("the dark Hub's popover repaint is scoped only where it wins", () => {
  assert.match(HUB_CSS, /@layer base \{[\s\S]*html\.dark \.hub-page \[class\*="bg-card"\],\s*html\.dark \.hub-page \[class\*="bg-background"\] \{\s*background-color: var\(--popover\);/);
  const rule = CSS.match(
    /html\[data-contrast-adjust\]\.dark \.hub-page :is\(\[class\*="bg-background"\], \[class\*="bg-card"\]\):not\(\[class\^="bg-"\], \[class\*=" bg-"\], \[class\*="dark:bg-"\], :hover\) \{([^}]*)\}/,
  );
  assert.ok(rule, "the repainted Hub popovers are not scoped");
  assert.match(rule[1] ?? "", /var\(--contrast-panel-target, var\(--contrast-target\)\) var\(--contrast-text-mix\)/);
});
