// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import tailwindVite from "@tailwindcss/vite";
import { compile } from "tailwindcss";

import {
  childSelectorOrder,
  parentFirstChildSelectors,
} from "../vite-plugin-child-selectors.ts";
import { readSrc, readText } from "./helpers/kit.ts";

const CHILD_POSITION_FIRST = />\s*:not\(:last-child\)\)/;

test("Tailwind's spacing and divide utilities are rewritten to test the parent first", async () => {
  const compiler = await compile(
    "@theme { --spacing: 0.25rem; --color-border: #ddd; } @tailwind utilities;",
  );
  const generated = compiler.build([
    "space-y-4",
    "-space-x-0.5",
    "space-x-2",
    "divide-y",
    "divide-border",
    "divide-border/50",
    ...["hover", "group-hover"].map((variant) => `${variant}:space-y-4`),
  ]);
  assert.match(
    generated,
    CHILD_POSITION_FIRST,
    "Tailwind still emits :where(& > :not(:last-child)), the shape this plugin rewrites",
  );
  const rewritten = parentFirstChildSelectors(generated);
  assert.doesNotMatch(
    rewritten,
    CHILD_POSITION_FIRST,
    "a :not(:last-child) Chromium evaluates first flags every element in the app, so appending to any container restyled its previous last child's whole subtree",
  );
  assert.equal(
    rewritten.split(":where(& > *):where(:not(:last-child))").length - 1,
    generated.split(":where(& > :not(:last-child))").length - 1,
  );
});

test("the rewrite keeps the match and the zero specificity, in every form Tailwind ships", () => {
  const cases: [string, string][] = [
    [
      ":where(.space-y-4>:not(:last-child)){margin:0}",
      ":where(.space-y-4 > *):where(:not(:last-child)){margin:0}",
    ],
    [
      ".space-y-4 {\n  :where(& > :not(:last-child)) {",
      ".space-y-4 {\n  :where(& > *):where(:not(:last-child)) {",
    ],
    [
      ":where(.divide-border>:not(:last-child)),:where(.divide-border\\/50>:not(:last-child)){",
      ":where(.divide-border > *):where(:not(:last-child)),:where(.divide-border\\/50 > *):where(:not(:last-child)){",
    ],
    [
      ":where(.divide-border\\/\\[0\\.06\\]>:not(:last-child)){",
      ":where(.divide-border\\/\\[0\\.06\\] > *):where(:not(:last-child)){",
    ],
    [
      ":where(.group-hover\\:space-y-2:is(:where(.group):hover *)>:not(:last-child)){",
      ":where(.group-hover\\:space-y-2:is(:where(.group):hover *) > *):where(:not(:last-child)){",
    ],
    [
      ":where(.a\\)b>:not(:last-child)){",
      ":where(.a\\)b > *):where(:not(:last-child)){",
    ],
  ];
  for (const [input, output] of cases) {
    assert.equal(parentFirstChildSelectors(input), output);
  }
});

test("selectors the rewrite cannot keep equivalent are left alone", () => {
  for (const css of [
    ".not-last\\:border-b:not(:last-child){",
    ":where(.a :not(:last-child)){",
    ":where(.a > :not(:last-child), .b){",
    ":where(.a > .b:not(:last-child)){",
    ":where(.a > :not(:first-child)){",
    ":where(> :not(:last-child)){",
    ":is(.a > :not(:last-child)){",
    ":where(.a > :not(:last-child)",
  ]) {
    assert.equal(parentFirstChildSelectors(css), css);
  }
});

test("the plugin rewrites stylesheets only, after Tailwind has generated them", () => {
  const hook = childSelectorOrder().transform as {
    filter: { id: RegExp; code: string };
    handler: (code: string) => { code: string; map: null } | null;
  };
  assert.ok(hook.filter.id.test("/app/src/index.css"));
  assert.ok(hook.filter.id.test("/app/src/index.css?direct"));
  assert.ok(!hook.filter.id.test("/app/src/main.tsx"));
  assert.ok(!hook.filter.id.test("/app/src/theme.css.ts"));
  assert.equal(hook.filter.code, ":not(:last-child)");
  assert.deepEqual(hook.handler(":where(.a>:not(:last-child)){}"), {
    code: ":where(.a > *):where(:not(:last-child)){}",
    map: null,
  });
  assert.equal(hook.handler(".a:not(:last-child){}"), null);

  const generators = tailwindVite().filter((plugin) =>
    plugin.name.includes(":generate:"),
  );
  assert.ok(generators.length > 0);
  for (const plugin of generators) {
    assert.equal(
      plugin.enforce,
      "pre",
      "Tailwind generates its CSS before normal plugins, so this one sees the utilities",
    );
  }
  assert.match(
    readText("../vite.config.ts"),
    /plugins: \[react\(\), tailwindcss\(\), childSelectorOrder\(\),/,
  );
});

type Complex = { compounds: string[]; combinators: string[] };

function topLevel(selector: string, split: (char: string) => boolean): string[] {
  const parts: string[] = [];
  let depth = 0;
  let current = "";
  for (let i = 0; i < selector.length; i++) {
    const char = selector[i];
    if (char === "\\") {
      current += char + (selector[i + 1] ?? "");
      i++;
      continue;
    }
    if (char === "(" || char === "[") depth++;
    if (char === ")" || char === "]") depth--;
    if (depth === 0 && split(char)) {
      parts.push(current, char);
      current = "";
      continue;
    }
    current += char;
  }
  parts.push(current);
  return parts;
}

function complexSelectors(css: string): string[] {
  const preludes =
    css.replace(/\/\*[\s\S]*?\*\//g, "").match(/[^{};]+(?=\{)/g) ?? [];
  return preludes
    .map((prelude) => prelude.trim())
    .filter((prelude) => prelude && !prelude.startsWith("@"))
    .flatMap((prelude) =>
      topLevel(prelude, (char) => char === ",")
        .filter((part) => part !== ",")
        .map((part) => part.trim()),
    )
    .filter((selector) => selector && !/^(?:from|to|[\d.]+%)$/.test(selector));
}

function parse(selector: string): Complex {
  const tokens = topLevel(
    selector.replace(/\s+/g, " ").replace(/\s*([>+~])\s*/g, "$1").trim(),
    (char) => char === ">" || char === "+" || char === "~" || char === " ",
  );
  return {
    compounds: tokens.filter((_, index) => index % 2 === 0),
    combinators: tokens.filter((_, index) => index % 2 === 1),
  };
}

function hasFeature(compound: string): boolean {
  const outside = topLevel(compound, () => false)
    .join("")
    .replace(/\((?:[^()]|\([^()]*\))*\)/g, "()");
  return /(?:^|[^:\\])[.#[&]|^[a-z]/i.test(outside);
}

function testsParentFirst(compound: string): boolean {
  return /^:(?:where|is)\([^()]*>\s*\*\)/.test(compound);
}

function functionalArguments(compound: string): string[] {
  const found: string[] = [];
  const opener = /:(?:is|where|not|has)\(/g;
  let match: RegExpExecArray | null;
  while ((match = opener.exec(compound))) {
    let depth = 1;
    let i = match.index + match[0].length;
    const start = i;
    for (; i < compound.length && depth > 0; i++) {
      if (compound[i] === "\\") {
        i++;
        continue;
      }
      if (compound[i] === "(") depth++;
      if (compound[i] === ")") depth--;
    }
    found.push(
      ...topLevel(compound.slice(start, i - 1), (char) => char === ",")
        .filter((part) => part !== ",")
        .map((part) => part.trim())
        .filter(Boolean),
    );
    opener.lastIndex = i;
  }
  return found;
}

function siblingOnUnnamedElement(selector: string): boolean {
  const { compounds, combinators } = parse(selector.replace(/^\s*[>+~]\s*/, ""));
  for (let i = 0; i < combinators.length; i++) {
    if (/[+~]/.test(combinators[i]) && !hasFeature(compounds[i + 1])) return true;
  }
  return compounds.some(
    (compound) =>
      !testsParentFirst(compound) &&
      functionalArguments(compound).some(siblingOnUnnamedElement),
  );
}

test("index.css never makes Chromium check a sibling or a child position on every element", () => {
  const offenders: string[] = [];
  for (const selector of complexSelectors(readSrc("index.css"))) {
    const { compounds, combinators } = parse(selector);
    const subject = compounds[compounds.length - 1];
    if (hasFeature(subject) || testsParentFirst(subject)) continue;
    const sibling = /[+~]/.test(combinators[combinators.length - 1] ?? "");
    const structural =
      combinators.length === 0 &&
      /:(?:first|last|only)-child|:nth-|-of-type\b|[+~]/.test(subject);
    if (sibling || structural) offenders.push(selector);
  }
  assert.deepEqual(
    offenders,
    [],
    "with KaTeX's MathML on the page, Chromium's universal sibling set invalidates whole subtrees: one flagged parent turned every middle insert or remove into a restyle of everything below it, <body> included",
  );
});

test("no rule tries a sibling combinator on an element it cannot name, in any position", () => {
  for (const flagged of [
    ".dark\n\t\tbody\n\t\t> :is([data-slot=\"dialog-content\"])[data-state=\"open\"]\n\t\t~ *\n\t\t:is([data-slot=\"select-content\"], .dropdown-surface)",
    ".dark body > [data-radix-popper-content-wrapper]:has(> .menu[data-state=\"open\"]) ~ * :is([data-slot=\"popover-content\"])",
    ".dark :is(body > .dialog[data-state=\"open\"] ~ *, body > .picker ~ *) :is([data-slot=\"popover-content\"]):not(.a)",
    ".a + *",
    ":where(.a ~ *)",
  ]) {
    assert.ok(siblingOnUnnamedElement(flagged), flagged);
  }
  for (const allowed of [
    ".aui-branch-chevron-btn + .aui-branch-picker-state",
    ":where(.aui-assistant-message-content > *):where(.h-0 + *)",
    ".dark [data-dropdown-glow]:not(.sidebar-menu, .browser-menu)",
    ".a:has(> .b + .c)",
  ]) {
    assert.ok(!siblingOnUnnamedElement(allowed), allowed);
  }
  for (const file of ["index.css", "features/hub/hub.css"]) {
    assert.deepEqual(
      complexSelectors(readSrc(file)).filter(siblingOnUnnamedElement),
      [],
      `${file}: a sibling combinator whose right side names nothing is tried on every ancestor of the subject, so the parents of <body>'s children get flagged and, with KaTeX's MathML on the page, every insert at the start of <body> (Radix focus guards, toasts) restyles the whole page`,
    );
  }
});

test("the message spacing, the badge follower and the branch count keep their rules in the parent-first form", () => {
  const css = readSrc("index.css");
  for (const selector of [
    ":where(.aui-assistant-message-content > *):where(:not(:first-child)) {",
    ":where(.aui-assistant-message-content > *):where(.h-0 + *) {",
    ".aui-branch-chevron-btn,\n\t.aui-branch-chevron-btn + .aui-branch-picker-state {",
    "\t.aui-branch-chevron-btn + .aui-branch-picker-state {\n",
  ]) {
    assert.ok(css.includes(selector), selector);
  }
  for (const removed of [
    ".aui-branch-chevron-btn + *",
    ".aui-branch-chevron-btn + :not(",
    ":where(.aui-assistant-message-content > :not(:first-child))",
    ":where(.aui-assistant-message-content > .h-0 + *)",
  ]) {
    assert.ok(!css.includes(removed), removed);
  }
  assert.match(
    readSrc("components/assistant-ui/thread.tsx"),
    /className="aui-branch-chevron-btn"[\s\S]{0,400}?<span className="aui-branch-picker-state[\s\S]{0,400}?className="aui-branch-chevron-btn"/,
    "the count the rules target is the element between the two chevrons",
  );
});
