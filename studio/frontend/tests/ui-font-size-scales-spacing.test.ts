// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readdirSync } from "node:fs";
import { join } from "node:path";

import { readSrc } from "./helpers/kit.ts";

const SOURCES = (function walk(dir: string): string[] {
  return readdirSync(join(import.meta.dirname, "../src", dir), {
    withFileTypes: true,
  }).flatMap((entry) => {
    const path = dir ? `${dir}/${entry.name}` : entry.name;
    if (entry.isDirectory()) return walk(path);
    return /\.(tsx?|css)$/.test(entry.name) ? [path] : [];
  });
})("");

// --ui-space-scale is the one multiplier every font-size-dependent length goes through.

const CSS = readSrc("index.css");
const SIDEBAR = readSrc("components/app-sidebar.tsx");
const PROVIDER = readSrc("app/provider.tsx");

const SCALED = /calc\([\d.]+(?:px|rem)\s*\*\s*var\(--ui-space-scale,\s*1\)\)/;

test("the spacing scale is the font-size preference, normalised at the default", () => {
  // --ui-font-scale is against 16px and the default is 15px, so dividing keeps default spacing.
  assert.match(
    CSS,
    /--ui-space-scale:\s*calc\(var\(--ui-font-scale, 1\) \/ 0\.9375\);/,
  );
  assert.match(
    CSS,
    /--ui-font-scale:\s*calc\(var\(--ui-font-size-scale, 0\.9375\) \* var\(--ui-interface-scale, 1\)\);/,
  );
});

test("every tailwind spacing utility goes through it", () => {
  // The @theme passthrough is commented out, so it is not a declaration.
  const declarations = (CSS.match(/--spacing:[^;]+;/g) ?? []).filter(
    (declaration) => !declaration.includes("var(--spacing)"),
  );
  assert.equal(declarations.length, 2, "light and dark each declare --spacing");
  for (const declaration of declarations) {
    assert.match(
      declaration,
      /--spacing:\s*calc\(0\.25rem \* var\(--ui-space-scale, 1\)\);/,
    );
  }
});

test("a measured cutoff moves with the box it measures", () => {
  // min-h-9 scales, so a fixed 44px threshold misreads an empty chip field as wrapped.
  const chips = readSrc("features/recipe-studio/components/chip-input.tsx");
  assert.match(chips, /element\.clientHeight > 44 \* uiSpaceScale/);
  assert.match(chips, /\}, \[values\.length, draft, uiSpaceScale\]\);/);
});

test("em lengths are left alone, they already follow the text", () => {
  // rem needs the multiplier; em follows the element font size, so scaling it applies twice.
  const withEm = SOURCES.filter((file) =>
    /[^a-z0-9.]-?[\d.]+em\s*\*\s*var\(--ui-space-scale/.test(readSrc(file)),
  );
  assert.deepEqual(withEm, [], "these em lengths take the scale twice");
  const pickers = readSrc(
    "features/model-picker/components/model-selector/pickers.tsx",
  );
  assert.match(pickers, /mx-\[-0\.1em\]/);
  assert.match(pickers, /ml-\[0\.14em\]/);
});

test("chat chrome that holds text scales, window chrome does not", () => {
  for (const name of [
    "--studio-chat-header-height",
    "--studio-chat-control-height",
    "--studio-chat-header-padding-top",
  ]) {
    for (const source of [CSS, PROVIDER]) {
      const declaration = new RegExp(
        `${name}"?:\\s*"?(calc\\([^;\\n"]*\\)|[^;,\\n]+)`,
      ).exec(source);
      assert.ok(declaration, `${name} is gone`);
      assert.match(declaration[1], SCALED, `${name} must follow the UI scale`);
    }
  }

  // The title bar and traffic lights belong to the OS window and must not move.
  for (const name of [
    "--studio-desktop-titlebar-height",
    "--studio-content-top-inset",
  ]) {
    const declaration = new RegExp(`${name}":\\s*"([^"]+)"`).exec(PROVIDER);
    if (declaration) {
      assert.doesNotMatch(declaration[1], /--ui-space-scale/, name);
    }
  }
});

test("the sidebar's hand-set spacing follows the scale", () => {
  const bare = [
    ...SIDEBAR.matchAll(
      /(?<![\w-])-?(?:p|px|py|pt|pb|pl|pr|m|mx|my|mt|mb|ml|mr|gap|gap-x|gap-y)-\[(\d*\.?\d+)px\]/g,
    ),
  ].filter((match) => Number(match[1]) > 1);
  assert.deepEqual(
    bare.map((match) => match[0]),
    [],
    "these sidebar paddings ignore the UI font size",
  );
});

test("fixed slots that hold scaled content scale with it", () => {
  // These heights live in JS, so the CSS variable cannot reach them.
  const HOOK = "useUiSpaceScale";
  const rows = readSrc("features/hub/catalog/models-catalog-rows.tsx");
  assert.ok(rows.includes(HOOK), "the virtualizer ignores the UI font size");
  assert.match(rows, /estimateSize: \(\) => slotHeight/);
  assert.match(rows, /height: `\$\{slotHeight\}px`/);
  // estimateSize is cached, so a new scale has to invalidate the sizes.
  assert.match(rows, /virtualizer\.measure\(\)/);

  const carousel = readSrc("features/hub/catalog/hub-section-row.tsx");
  assert.ok(carousel.includes(HOOK), "the carousel slot ignores it");
  assert.match(carousel, /itemHeight=\{cardHeight\}/);
  // The stride is measured in JS against gap-4 and pt-2, which scale.
  const strip = readSrc("features/hub/catalog/card-carousel.tsx");
  assert.ok(strip.includes(HOOK), "the carousel stride ignores it");
  assert.match(strip, /const gapPx = CARD_GAP_PX \* scale;/);
  assert.match(strip, /const topPaddingPx = CAROUSEL_TOP_PADDING_PX \* scale;/);
  // The resize observer only watches the scroller's box, so the sizes are deps.
  assert.match(
    strip,
    /\}, \[updateArrows, items, itemWidth, itemHeight, gapPx\]\);/,
  );

  const lists = readSrc("features/hub/catalog/models-catalog-lists.tsx");
  assert.ok(lists.includes(HOOK), "the pinned grid ignores it");
  // The pinned block shares lanes with the virtualized rows, so no gutter of its own.
  assert.match(
    lists,
    /Math\.round\(CATALOG_COLUMN_GAP_PX \* pinnedScale\)/,
  );
  assert.doesNotMatch(lists, /columnGap: 12,/);
});

test("the titlebar reserves room for its controls, which stay in the band", () => {
  const titlebar = readSrc("components/tauri/window-titlebar.tsx");
  // The band is a fixed 34px and clips nothing, so buttons must not grow.
  assert.match(titlebar, /inline-flex size-\[30px\] shrink-0/);
  assert.match(titlebar, /aria-hidden="true" className="size-\[30px\] shrink-0"/);
  assert.match(titlebar, /inline-flex h-full w-\[46px\] shrink-0/);
  // max(): the slot may grow with scaled padding but never below the fixed buttons' width.
  assert.match(
    titlebar,
    /max\(7rem, calc\(7rem \* var\(--ui-space-scale, 1\)\)\)/,
  );
});

test("the composer's one-row clamp is one row at any size", () => {
  // The editor box wins over the input's min-height, so all three clamps and the JS floor move together.
  const clamps = CSS.match(/calc\(40px \* var\(--ui-space-scale, 1\)\)/g) ?? [];
  assert.equal(clamps.length, 3, "a 40px composer clamp is still fixed");
  const thread = readSrc("components/assistant-ui/thread.tsx");
  assert.match(thread, /const oneRowHeight = Math\.round\(40 \* uiSpaceScale\);/);
  assert.doesNotMatch(thread, /Math\.max\(40, editorHeight\)/);
});

test("overlays that scale cannot outgrow the screen", () => {
  // w-* follows --spacing, so floating surfaces must cap themselves for small screens.
  for (const file of [
    "components/ui/popover.tsx",
    "components/ui/dropdown-menu.tsx",
    "components/ui/select.tsx",
  ]) {
    assert.match(
      readSrc(file),
      /max-w-\[calc\(100vw-32px\)\]/,
      `${file} can overflow a narrow viewport`,
    );
  }
});

test("the stylesheet's comments stay comments", () => {
  // A "*/" inside a CSS comment ends it early and postcss eats the next declaration.
  const stripped = CSS.replace(/\/\*[\s\S]*?\*\//g, "");
  assert.doesNotMatch(
    stripped,
    /\butility follows\b|\bTailwind's\b/,
    "prose leaked out of a comment",
  );
  assert.ok(
    stripped.includes("--spacing: calc(0.25rem * var(--ui-space-scale, 1));"),
    "the spacing step is no longer a declaration of its own",
  );
});

test("no hand-set length above a hairline skips the scale", () => {
  // Exempt: hairlines, 1-3px nudges, images, the desktop titlebar, and the JS-measured activity grid.
  const FIXED_BY_DESIGN = new Set([
    "components/tauri/window-titlebar.tsx",
    "components/assistant-ui/image.tsx",
    "components/assistant-ui/search-image.tsx",
    "components/assistant-ui/tool-ui-image-generation.tsx",
    "features/profile/components/stats/token-activity-card.tsx",
  ]);
  const LENGTH =
    /(?<=[\s"'`:!(\[])-?(?:size|w|h|min-w|min-h|max-w|max-h|basis|gap|gap-x|gap-y|space-x|space-y|p|px|py|pt|pb|pl|pr|ps|pe|m|mx|my|mt|mb|ml|mr|ms|me|top|bottom|left|right|inset|inset-x|inset-y|start|end|translate-x|translate-y)-\[(\d*\.?\d+)(px|rem)\]/g;
  const bare = SOURCES.filter(
    (file) => /\.tsx?$/.test(file) && !FIXED_BY_DESIGN.has(file),
  ).flatMap((file) =>
    [...readSrc(file).matchAll(LENGTH)]
      .filter((m) => Number(m[1]) * (m[2] === "rem" ? 16 : 1) > 3)
      .map((m) => `${file}: ${m[0]}`),
  );
  assert.deepEqual(bare, [], "these lengths ignore the UI font size");
});

test("icons grow at the rate of the text beside them", () => {
  assert.match(CSS, /--ui-icon-size: calc\(1rem \* var\(--ui-font-scale, 1\)\);/);
  assert.match(
    CSS,
    /--ui-icon-size-sm: calc\(0\.875rem \* var\(--ui-font-scale, 1\)\);/,
  );
  assert.doesNotMatch(
    CSS,
    /min\(calc\([\d.]+(?:px|rem) \* var\(--ui-font-scale/,
    "an icon is still on the half-rate curve",
  );
  for (const px of ["15", "18"]) {
    assert.ok(
      CSS.includes(
        `& svg:is(.size-\\[${px}px\\], [class~='size-[calc(${px}px*var(--ui-space-scale,1))]'])`,
      ),
      `size-[${px}px] is matched in only one spelling`,
    );
  }
});

test("named widths scale, container breakpoints do not", () => {
  for (const [name, rem] of [
    ["xs", "20rem"],
    ["md", "28rem"],
    ["2xl", "42rem"],
    ["7xl", "80rem"],
  ]) {
    assert.ok(
      CSS.includes(`--container-${name}: calc(${rem} * var(--ui-space-scale, 1));`),
      `--container-${name} ignores the UI font size`,
    );
  }
  // Declared outside @theme so Tailwind bakes the literal into @container queries.
  const theme = CSS.slice(CSS.indexOf("@theme inline {"));
  assert.doesNotMatch(
    theme.slice(0, theme.indexOf("\n}")),
    /--container-md:/,
  );
});

test("the plain stylesheets' controls follow the scale", () => {
  const HUB = readSrc("features/hub/hub.css");
  for (const rule of [".hub-action-btn {", ".hub-run-action-btn {", ".hub-download-fab {"]) {
    const at = HUB.indexOf(rule);
    assert.notEqual(at, -1, `${rule} is gone`);
    const body = HUB.slice(at, HUB.indexOf("}", at));
    assert.match(
      body,
      /(?:height|width): calc\([\d.]+rem \* var\(--ui-space-scale, 1\)\)/,
      `${rule} has a fixed size`,
    );
  }
  assert.match(
    CSS,
    /\.panel-slider \[data-slot="slider-thumb"\] \{\s*width: calc\(0\.875rem \* var\(--ui-space-scale, 1\)\) !important;/,
  );
});

test("a scaled minimum never outgrows its own cap", () => {
  // CSS lets min win over max, so a scaled minimum must be capped too.
  for (const [file, cap] of [
    ["features/chat/chat-settings-sheet.tsx", "48dvh"],
    ["features/model-picker/components/chat-template-editor-dialog.tsx", "50dvh"],
  ] as const) {
    assert.ok(
      readSrc(file).includes(
        `min-h-[min(calc(20rem*var(--ui-space-scale,1)),${cap})] max-h-[${cap}]`,
      ),
      `${file} lets its minimum pass its cap`,
    );
  }
  for (const [file, width] of [
    ["features/settings/tabs/agents-tab.tsx", "16rem"],
    ["features/hub/catalog/gguf-download-card.tsx", "300px"],
    ["features/hub/catalog/local-on-device-card.tsx", "220px"],
  ] as const) {
    assert.ok(
      readSrc(file).includes(
        `min-w-[min(calc(${width}*var(--ui-space-scale,1)),calc(100vw-32px))]`,
      ),
      `${file} can outgrow a narrow screen`,
    );
  }
});

test("a scaled dialog keeps the viewport cap it replaces", () => {
  // A call-site max-h drops DialogContent's own viewport cap, so it restates it.
  const cap = "calc(100dvh-var(--studio-window-chrome-top,0px)-2rem)";
  assert.ok(
    readSrc("features/recipe-studio/dialogs/shared/recipe-dialog-content.tsx").includes(
      `max-h-[min(calc(650px*var(--ui-space-scale,1)),${cap})]`,
    ),
    "recipe dialogs can outgrow the viewport",
  );
  for (const file of [
    "features/recipe-studio/dialogs/config-dialog.tsx",
    "features/recipe-studio/dialogs/import-dialog.tsx",
    "features/recipe-studio/dialogs/preview-dialog.tsx",
    "features/recipe-studio/dialogs/processors-dialog.tsx",
  ]) {
    assert.match(readSrc(file), /<RecipeDialogContent\b/, file);
  }
});

test("the response details sheet scales its width, not only its cap", () => {
  const sheet = readSrc("components/assistant-ui/message-response-details-sheet.tsx");
  assert.ok(sheet.includes("w-[min(calc(28rem*var(--ui-space-scale,1)),100vw)]"));
  assert.ok(sheet.includes("sm:max-w-[calc(28rem*var(--ui-space-scale,1))]"));
});

test("a sidebar row's inset scales on both sides", () => {
  assert.ok(
    SIDEBAR.includes(
      '"ps-[calc(5px*var(--ui-space-scale,1))] pe-[calc(var(--sidebar-rail,0px)+5px*var(--ui-space-scale,1))]"',
    ),
  );
  assert.ok(
    SIDEBAR.includes('"ps-1.5 pe-[calc(var(--sidebar-rail,0px)+6px*var(--ui-space-scale,1))]"'),
  );
});

test("a scaled media rail leaves the preview its minimum", () => {
  // A shrink-0 rail overflowed its split at 200%; the header column shrinks the same way.
  const rail = "var(--media-rail-width,calc(408px*var(--ui-space-scale,1)))";
  for (const file of [
    "features/images/images-page.tsx",
    "features/audio/audio-page.tsx",
    "features/video/video-page.tsx",
  ]) {
    const source = readSrc(file);
    assert.ok(
      source.includes(`@[50rem]:w-[min(${rail},calc(100%-13rem))]`),
      `${file} rail can outgrow its split`,
    );
    assert.ok(
      source.includes(`grid-cols-[minmax(0,${rail})_minmax(13rem,1fr)]`),
      `${file} header column drifts from its rail`,
    );
  }
});

test("the settings rail never takes more than half the dialog", () => {
  assert.ok(
    readSrc("features/settings/settings-dialog.tsx").includes(
      "w-[min(calc(248px*var(--ui-space-scale,1)),50%)] shrink-0",
    ),
  );
});

test("a menu's height cap keeps Radix's available height", () => {
  // A call-site max-h replaces the component's collision cap through cn.
  const menu =
    /<(DropdownMenuContent|DropdownMenuSubContent|SelectContent|ContextMenuContent|ContextMenuSubContent)\b([^>]*)>/g;
  for (const file of SOURCES) {
    for (const [, tag, props] of readSrc(file).matchAll(menu)) {
      for (const [cap] of (props ?? "").matchAll(/max-h-\S+/g)) {
        assert.match(
          cap,
          /var\(--radix-[a-z-]+-content-available-height\)/,
          `${file} <${tag}> ${cap} can run offscreen`,
        );
      }
    }
  }
});

test("settings stacks its rail where the scaled dialog is too narrow for it", () => {
  const dialog = readSrc("features/settings/settings-dialog.tsx");
  assert.match(dialog, /const width = 608 \* useUiSpaceScale\(\);/);
  assert.match(dialog, /`\(width < \$\{width \+ 32\}px\)`/);
  assert.match(dialog, /return width > 960 \|\| narrow;/);
  assert.match(dialog, /data-stacked=\{stacked \|\| undefined\}/);
  const shell = dialog.match(/max-sm:[^\s"]+/g) ?? [];
  assert.deepEqual(shell.sort(), [
    "max-sm:!max-w-none",
    "max-sm:h-[calc(100dvh-var(--studio-window-chrome-top,0px))]",
    "max-sm:rounded-none",
    "max-sm:w-dvw",
  ]);
});

test("composite settings controls shrink inside their row", () => {
  // A fixed width's min-content blocks the max-w-full cap unless each wrapper can shrink.
  assert.ok(
    readSrc("features/settings/tabs/general-tab.tsx").includes(
      '<div className="flex max-w-full items-center gap-2">\n            <div className="relative w-[calc(260px*var(--ui-space-scale,1))] min-w-0">',
    ),
  );
  assert.ok(
    readSrc("features/settings/tabs/debugging-tab.tsx").includes(
      '"flex max-w-full shrink-0 flex-wrap items-center justify-end gap-1"',
    ),
  );
  const resources = readSrc("features/settings/tabs/resources-tab.tsx");
  assert.ok(resources.includes("flex min-w-0 flex-wrap items-center justify-between"));
  assert.ok(
    resources.includes(
      '"flex w-[min(calc(392px*var(--ui-space-scale,1)),100%)] shrink-0 flex-col',
    ),
  );
});

test("a scaled sheet width stops at the viewport", () => {
  const sheet = /<SheetContent\b([^>]*)>/g;
  for (const file of SOURCES.filter((f) => f.endsWith(".tsx"))) {
    for (const [, props] of readSrc(file).matchAll(sheet)) {
      for (const [width] of (props ?? "").matchAll(/(?<![\w:-])w-\[[^\]\s"]*ui-space-scale[^\]\s"]*\]/g)) {
        assert.match(width, /^w-\[min\(.*,100vw\)\]$/, `${file} ${width} can pass the viewport`);
      }
    }
  }
});

test("resizable panels render their layout width at the browser scale", () => {
  const sidebar = readSrc("components/ui/sidebar.tsx");
  assert.equal(sidebar.match(/"--sidebar-width": `\$\{width \* widthScale\}px`/g)?.length, 2);
  assert.ok(
    readSrc("features/chat/chat-settings-sheet.tsx").includes(
      '"--chat-settings-width": `${settingsWidth * settingsScale}px`',
    ),
  );
  const handle = readSrc("components/ui/panel-resize-handle.tsx");
  assert.ok(handle.includes("paint(`${pendingRef.current * scaleRef.current}px`)"));
  assert.ok(handle.includes("paint(`${committedRef.current * scaleRef.current}px`)"));
  assert.ok(handle.includes("const next = drag.startWidth + delta / scaleRef.current"));
  // Only the browser scale: desktop zoom already scales every px.
  assert.ok(
    readSrc("features/settings/stores/interface-scale-store.ts").includes(
      "const zoom = interfaceScaleToZoom(scale);\n  setLayoutScale(zoom);",
    ),
  );
});

test("the training start overlay never spills above its host", () => {
  const overlay = readSrc("features/studio/training-start-overlay.tsx");
  assert.ok(overlay.includes("absolute inset-0 z-30 flex flex-col items-center rounded-2xl"));
  assert.ok(overlay.includes("pointer-events-auto relative my-auto flex"));
});
