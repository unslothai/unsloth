// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/*
 * No new CSS rule whose style lands AFTER a :has(): `X:has(A) Y`, `X:has(A) > Y`, Tailwind's
 * group-has-* / peer-has-*, `has-[A]:[&>Y]`, `[:has(A)>&]`. Chromium keeps one shared invalidation set
 * for every such rule: when anything changes under an X it has tested, it re-checks every element
 * below that X matching ANY rule's Y (a tag, a [data-slot], ...). With X on the workspace or a p/div
 * above the thread, a sidebar toggle or a run-settings open restyled ~9-10k thread elements at 150
 * turns (348 ms of style). WebKit re-checks the whole document whenever anything named inside such
 * a :has() changes: with :focus-visible or [data-state] in there, every keystroke paid it (typing
 * p50 59 -> 25 ms at 150 turns once they were gone).
 *
 * Fix a new one by making the :has() element the subject (`X:has(A) { --v: ... }`, or `Y:has(~ A)`
 * for a sibling), or by setting a data attribute on X from the code that already knows A
 * (chat-background/host-flags.ts, hub/lib/badge-rows.ts, ui/info-hint.tsx), or by moving the state
 * onto X as a custom property (X sets it in its own has-[...] class, Y reads it with var()). Only allowlist a
 * rule whose X can never hold the thread, whose Y is a class used only inside X and whose :has()
 * names nothing that changes while typing, after measuring a sidebar toggle (Chromium) and typing
 * (WebKit) at 150 turns.
 *
 * Static and fast: compiles src/index.css (and its imports) with the build's Tailwind, plus every
 * class token in src that mentions has, then lists the selectors.
 */

import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import test from "node:test";

import { compile, optimize } from "@tailwindcss/node";
import postcss from "postcss";
import selectorParser, { type Selector } from "postcss-selector-parser";

import { hoistHasAnchors } from "../vite-plugin-has-anchor.ts";

const SRC = path.resolve(import.meta.dirname, "../src");

/*
 * Rules allowed to keep a style after their :has(), measured harmless with all of them present: a
 * sidebar toggle and a run-settings open at 150 turns restyle ~140 / ~250 elements (Chromium), and
 * WebKit typing at 150 turns is as fast as with none of them. Their anchor never holds the thread.
 * Tailwind utilities are listed by class token, hand-written rules by selector.
 */
const ALLOWED_TAILWIND = new Set<string>([
  // Chat model controls (a notice beside the picker) and the composer's files panel.
  "max-sm:group-has-[>[data-chat-model-notice]]/chat-model-controls:!ml-0",
  "max-sm:group-has-[>[data-chat-model-notice]]/chat-model-controls:!gap-0",
  "max-sm:group-has-[>[data-chat-model-notice]]/chat-model-controls:!px-2",
  "[:has(.unsloth-files-panel)>&]:hidden",
  // shadcn/ui primitives (sidebar menu, field, input group, avatar, alert, command, combobox, number input).
  "group-has-data-[sidebar=menu-action]/menu-item:pr-8",
  "group-has-disabled/field:opacity-50",
  "group-has-data-[size=lg]/avatar-group:size-10",
  "group-has-data-[size=lg]/avatar-group:[--avatar-count-icon:calc(var(--spacing)*5)]",
  "group-has-data-[size=sm]/avatar-group:size-6",
  "group-has-data-[size=sm]/avatar-group:[--avatar-count-icon:calc(var(--spacing)*3)]",
  "group-has-data-[slot=combobox-clear]/input-group:hidden",
  "group-has-[[data-orientation=horizontal]]/field:text-balance",
  "group-has-[[data-slot=command-shortcut]]/command-item:hidden",
  "group-has-[input:disabled]/number:pointer-events-none",
  "group-has-[input:disabled]/number:opacity-0",
  "group-has-[>input]/input-group:pt-3",
  "group-has-[>input]/input-group:pb-3",
  "group-has-[>svg]/alert:col-start-2",
  "sm:group-data-[size=default]/alert-dialog-content:group-has-data-[slot=alert-dialog-media]/alert-dialog-content:col-start-2",
  "has-[>[data-slot=field-content]]:[&>[role=checkbox],[role=radio]]:mt-px",
  "@md/field-group:has-[>[data-slot=field-content]]:[&>[role=checkbox],[role=radio]]:mt-px",
  "has-[>[data-align=block-end]]:[&>input]:pt-3",
  "has-[>[data-align=block-start]]:[&>input]:pb-3",
  "has-[>[data-align=inline-end]]:[&>input]:pr-1.5",
  "has-[>[data-align=inline-start]]:[&>input]:pl-1.5",
]);

const ALLOWED_SELECTORS = new Set<string>([
  // Sidebar row hover (index.css).
  ".group\\/project-item:hover:not(:has(.sidebar-row-action:hover)) .sidebar-nav-btn",
  ".group\\/project-chat-item:hover:not(:has(.sidebar-row-action:hover)) .sidebar-nav-btn",
  ".group\\/projects-item:hover:not(:has(.sidebar-row-action:hover)) .sidebar-nav-btn",
  ".group\\/recent-item:hover:not(:has(.sidebar-row-action:hover)) .sidebar-nav-btn",
  ".group\\/run-item:hover:not(:has(.sidebar-row-action:hover)) .sidebar-nav-btn",
  ".dark .group\\/project-item:hover:not(:has(.sidebar-row-action:hover)) .sidebar-nav-btn",
  ".dark .group\\/project-chat-item:hover:not(:has(.sidebar-row-action:hover)) .sidebar-nav-btn",
  ".dark .group\\/projects-item:hover:not(:has(.sidebar-row-action:hover)) .sidebar-nav-btn",
  ".dark .group\\/recent-item:hover:not(:has(.sidebar-row-action:hover)) .sidebar-nav-btn",
  ".dark .group\\/run-item:hover:not(:has(.sidebar-row-action:hover)) .sidebar-nav-btn",
  // Composer pills, files panel and drop overlay (index.css).
  ":is([data-pill-compact=true] .composer-pill-btn:not([data-keep-label]),[data-pill-compact=first] .composer-pill-btn.composer-pill-permissions:not([data-keep-label])):has(.composer-pill-caret):hover .composer-pill-glyph",
  ":is([data-pill-compact=true] .composer-pill-btn:not([data-keep-label]),[data-pill-compact=first] .composer-pill-btn.composer-pill-permissions:not([data-keep-label])):has(.composer-pill-caret) .composer-pill-x",
  ".composer-pill-btn[data-active=true]:has(.composer-pill-x):hover .composer-pill-glyph",
  ".composer-pill-btn[data-active=true]:has(.composer-pill-x):hover .composer-pill-x",
  ".composer-pill-btn:has(>.composer-pill-glyph[role=button]):hover .composer-pill-glyph",
  ".composer-pill-btn:has(>.composer-pill-glyph[role=button]):hover .composer-pill-x",
  ".aui-composer-root:has(.unsloth-files-panel[data-dragging=true]) .aui-composer-drop-overlay",
  ".unsloth-files-panel-row:has(>.unsloth-files-panel-strip:hover)>.unsloth-files-panel-show-all",
  // Sonner toasts (index.css): only inside a toast.
  "[data-sonner-toast][data-styled=true]:not(:has([data-title])) [data-icon]",
  "[data-sonner-toast][data-styled=true][data-multiline]:not(:has([data-title])) [data-icon]",
  "[data-sonner-toast][data-styled=true]:has([data-button]) [data-icon]",
  "[data-sonner-toast][data-styled=true]:has([data-button]) [data-content]",
  "[data-sonner-toast][data-styled=true]:has([data-button]):has([data-icon]) [data-content]",
  "[data-sonner-toast][data-styled=true]:has([data-button]) [data-button]",
  "[data-sonner-toast][data-styled=true]:has([data-button]):has([data-icon]) [data-button]",
  // Sidebar footer fade (index.css): the sidebar never holds the thread.
  "html[data-glass][data-chat-wallpaper-workspace-ready] [data-slot=sidebar-inner]:has([data-slot=sidebar-footer-fade][data-visible=true]) .sidebar-scroll-fade",
  "html[data-glass=macos] [data-slot=sidebar-inner]:has([data-slot=sidebar-footer-fade][data-visible=true]) .sidebar-scroll-fade",
  "html[data-glass-clear] [data-slot=sidebar-inner]:has([data-slot=sidebar-footer-fade][data-visible=true]) .sidebar-scroll-fade",
]);

const FUNCTIONAL = new Set([
  ":is",
  ":where",
  ":not",
  ":matches",
  ":-webkit-any",
]);

const containsHas = (node: selectorParser.Container): boolean => {
  let found = false;
  node.walkPseudos((pseudo) => {
    if (pseudo.value === ":has") found = true;
  });
  return found;
};

/** True when some compound before the last one carries a :has(), directly or inside :is()/:not(). */
function styleAfterHas(selector: Selector): boolean {
  const nodes = selector.nodes;
  let lastCombinator = -1;
  nodes.forEach((node, i) => {
    if (node.type === "combinator") lastCombinator = i;
  });
  return nodes.some((node, i) => {
    if (node.type !== "pseudo") return false;
    if (node.value === ":has") return i < lastCombinator;
    if (!node.nodes.length) return false;
    if (i < lastCombinator && node.nodes.some(containsHas)) return true;
    return FUNCTIONAL.has(node.value) && node.nodes.some(styleAfterHas);
  });
}

const normalize = (selector: Selector): string => {
  const copy = selector.clone();
  copy.walk((node) => {
    node.spaces.before = "";
    node.spaces.after = "";
    if (node.type === "combinator") node.value = node.value.trim() || " ";
  });
  return String(copy)
    .trim()
    .replace(/="([\w-]+)"/g, "=$1");
};

function* files(dir: string, ext: RegExp): Generator<string> {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) yield* files(full, ext);
    else if (ext.test(entry.name)) yield full;
  }
}

/** Every whitespace/quote-delimited token Tailwind scans that could be a class using :has(). */
const FRONTEND = path.resolve(SRC, "..");
const SCANNED = /\.(tsx?|jsx?|mjs|html)$/;

/** What Tailwind's automatic source detection reads: src, tests and scripts, plus the root files. */
function* scannedFiles(): Generator<string> {
  for (const dir of ["src", "tests", "scripts"])
    yield* files(path.join(FRONTEND, dir), SCANNED);
  for (const entry of readdirSync(FRONTEND, { withFileTypes: true })) {
    if (entry.isFile() && SCANNED.test(entry.name))
      yield path.join(FRONTEND, entry.name);
  }
}

function hasCandidates(): Set<string> {
  const out = new Set<string>();
  for (const file of scannedFiles()) {
    const text = readFileSync(file, "utf8");
    if (!text.includes("has")) continue;
    for (const token of text.split(/[\s"'`]+/)) {
      if (
        token.length < 300 &&
        (token.includes("has-") || token.includes(":has("))
      ) {
        out.add(token.replace(/^[,;({]+|[,;)}]+$/g, ""));
      }
    }
  }
  return out;
}

type Offender = { key: string; selector: string };

async function builtCss(candidates: Set<string>): Promise<string> {
  const entry = path.join(SRC, "index.css");
  const compiler = await compile(readFileSync(entry, "utf8"), {
    base: SRC,
    onDependency: () => {},
  });
  // vite.config.ts runs the same hoist on every .css; optimize() flattens nesting like the build.
  const warn = console.warn;
  console.warn = () => {};
  try {
    return optimize(hoistHasAnchors(compiler.build([...candidates])), {
      minify: false,
    }).code;
  } finally {
    console.warn = warn;
  }
}

function offenders(css: string, candidates: Set<string>): Offender[] {
  const out: Offender[] = [];
  postcss.parse(css).walkRules((rule) => {
    // Join nesting first (a raw .css file may nest `.x:has(.a) { .y {} }`), skipping @media and the like.
    let full = rule.selector;
    for (
      let parent = rule.parent;
      parent && parent.type !== "root";
      parent = parent.parent
    ) {
      if (parent.type !== "rule") continue;
      const outer = (parent as postcss.Rule).selector;
      full = full.includes("&")
        ? full.replaceAll("&", `:is(${outer})`)
        : `:is(${outer}) ${full}`;
    }
    if (!full.includes(":has(")) return;
    selectorParser((root) => {
      root.each((selector) => {
        if (!styleAfterHas(selector)) return;
        let key = normalize(selector);
        selector.walkClasses((cls) => {
          if (candidates.has(cls.value)) key = cls.value;
        });
        out.push({ key, selector: String(selector).trim() });
      });
    }).processSync(full);
  });
  return out;
}

async function appOffenders(): Promise<Offender[]> {
  const candidates = hasCandidates();
  const found = offenders(await builtCss(candidates), candidates);
  // CSS files the entry does not import are bundled on their own.
  const imported = readFileSync(path.join(SRC, "index.css"), "utf8");
  for (const file of files(SRC, /\.css$/)) {
    const rel = `./${path.relative(SRC, file).split(path.sep).join("/")}`;
    if (rel === "./index.css" || imported.includes(`"${rel}"`)) continue;
    found.push(...offenders(readFileSync(file, "utf8"), candidates));
  }
  return found;
}

test("no new CSS rule puts its style after a :has() (Chromium restyles the whole thread)", async () => {
  const found = await appOffenders();
  assert.ok(
    found.length > 0,
    "the scan found nothing at all; it is no longer reading the app's CSS",
  );
  const fresh = found.filter(
    ({ key }) => !ALLOWED_TAILWIND.has(key) && !ALLOWED_SELECTORS.has(key),
  );
  assert.deepEqual(
    [...new Set(fresh.map(({ key }) => key))],
    [],
    "These rules style an element AFTER a :has() (X:has(A) Y, group-has-*, peer-has-*, has-[A]:[&>Y]).\n" +
      "Chromium then re-checks every Y-like element under each X on any change below X; with X above the\n" +
      "chat thread a sidebar toggle restyled ~9k thread elements at 150 turns. Make the :has() element the\n" +
      "subject, or set a data attribute on X from the component that knows A (see the comment at the top\n" +
      "of tests/has-subject-after.test.ts). Allowlist only after measuring.\n" +
      fresh.map(({ selector }) => `  ${selector}`).join("\n"),
  );
  // A stale entry would quietly allow a future rule with the same name.
  const keys = new Set(found.map(({ key }) => key));
  const stale = [...ALLOWED_TAILWIND, ...ALLOWED_SELECTORS].filter(
    (key) => !keys.has(key),
  );
  assert.deepEqual(
    stale,
    [],
    "allowlisted rules that no longer exist: remove them from the allowlist",
  );
});

test("no custom variant hides a :has() from the scan", () => {
  // Utilities built from these are not in the candidate list, so the scan above cannot see them.
  const variants: string[] = [];
  for (const file of files(SRC, /\.css$/)) {
    postcss
      .parse(readFileSync(file, "utf8"))
      .walkAtRules(/^(custom-variant|variant)$/, (rule) => {
        if (rule.toString().includes(":has("))
          variants.push(`${path.basename(file)}: ${rule.params}`);
      });
  }
  assert.deepEqual(
    variants,
    [],
    "a @custom-variant with :has(): scan its utilities or make the :has() the subject",
  );
});

test("the scan flags the shapes that restyled the thread and passes subject :has()", () => {
  const flagged = (css: string): number =>
    offenders(css, new Set(["group-has-x"])).length;
  const bad = [
    ":is([data-wallpaper-workspace], [data-chat-content-root]):has(> .chat-wallpaper[data-ready]) [data-chat-page-root] {}",
    ".hub-readme-prose :is(p, div):has(> a:nth-of-type(2) > img) img {}",
    ".menu-row-with-action:has([data-row-action]:hover) > [data-slot=dropdown-menu-item] {}",
    ":is(:where(.group):has([data-state=open]) *) {}",
    ".a:not(:has(.b)) .c {}",
    ".a { :where(.group):has(.b) & { opacity: 0 } }",
    ".x:has(.a) { @media (hover: hover) { .y { opacity: 0 } } }",
  ];
  for (const css of bad) assert.equal(flagged(css), 1, css);
  const good = [
    ".menu-row-with-action > [data-slot=dropdown-menu-item]:has(~ * [data-row-action]:hover) {}",
    "[data-wallpaper-workspace][data-wallpaper-ready] [data-chat-page-root] {}",
    ".composer-pill-btn:has(.composer-pill-caret) {}",
    ".a :is(.b:has(> .c)) {}",
  ];
  for (const css of good) assert.equal(flagged(css), 0, css);
});
