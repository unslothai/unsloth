// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { optimize } from "@tailwindcss/node";
import postcss from "postcss";
import selectorParser, {
  type Node as SelectorNode,
  type Selector,
} from "postcss-selector-parser";
import { compile } from "tailwindcss";
import ts from "@typescript/typescript6";
import { gateViewportMedia } from "../vite-plugin-viewport-media.ts";
import { readSrc } from "./helpers/kit.ts";

const parsed = ts.createSourceFile(
  "sidebar.tsx",
  readSrc("components/ui/sidebar.tsx"),
  ts.ScriptTarget.Latest,
  true,
  ts.ScriptKind.TSX,
);
const component = parsed.statements.find(
  (node) =>
    ts.isFunctionDeclaration(node) && node.name?.text === "SidebarInset",
);
assert.ok(component);
const current: string[] = [];
const visit = (node: ts.Node) => {
  if (ts.isStringLiteral(node))
    current.push(
      ...node.text.split(/\s+/).filter((value) => value.startsWith("md:")),
    );
  ts.forEachChild(node, visit);
};
visit(component);
const previous = ["m-2", "ml-0", "rounded-xl", "shadow-sm"].map((utility) =>
  ["md", "peer-data-[variant=inset]", utility].join(":"),
);
previous.push(
  [
    "md",
    "peer-data-[variant=inset]",
    "peer-data-[state=collapsed]",
    "ml-2",
  ].join(":"),
);
const activeDeclaration = parsed.statements
  .filter(ts.isVariableStatement)
  .flatMap((node) => node.declarationList.declarations)
  .find((node) => node.name.getText(parsed) === "sidebarMenuPeerActiveClass");
assert.ok(activeDeclaration?.initializer);
assert.ok(ts.isTaggedTemplateExpression(activeDeclaration.initializer));
assert.equal(activeDeclaration.initializer.tag.getText(parsed), "String.raw");
assert.ok(
  ts.isNoSubstitutionTemplateLiteral(activeDeclaration.initializer.template),
);
const currentActive = activeDeclaration.initializer.template
  .getText(parsed)
  .slice(1, -1);
const previousActive = [
  "peer-data-active/menu-button",
  "text-sidebar-accent-foreground",
].join(":");
const pairs = [
  ...previous.map((value, index) => [value, current[index]]),
  [previousActive, currentActive],
];
const candidates = [...current, currentActive];
const activeVariant = postcss
  .parse(
    readFileSync(
      new URL("../src/shadcn-tailwind.css", import.meta.url),
      "utf8",
    ),
  )
  .nodes.find(
    (node) =>
      node.type === "atrule" &&
      node.name === "custom-variant" &&
      node.params === "data-active",
  );
assert.ok(activeVariant);
const generated = (async () => {
  const compiler = await compile(
    `@theme { --spacing: .25rem; --breakpoint-md: 48rem; --radius-xl: .75rem; --shadow-sm: 0 1px 3px #0002; --color-sidebar-accent-foreground: #333; } ${activeVariant} @tailwind utilities;`,
  );
  return optimize(compiler.build(pairs.flat()), { minify: true }).code;
})();

function includesClass(node: SelectorNode, candidate: string): boolean {
  if (node.type === "class") return node.value === candidate;
  return (
    "nodes" in node &&
    Array.isArray(node.nodes) &&
    node.nodes.some((child) => includesClass(child, candidate))
  );
}

function rules(css: string, candidate: string) {
  const matches: {
    selector: Selector;
    declarations: string[];
    media: string | undefined;
  }[] = [];
  postcss.parse(css).walkRules((rule) => {
    for (const selector of selectorParser().astSync(rule.selector).nodes) {
      if (!includesClass(selector, candidate)) continue;
      const declarations = (rule.nodes ?? [])
        .filter((node) => node.type === "decl")
        .map(
          (node) =>
            `${node.prop}:${node.value}${node.important ? "!important" : ""}`,
        );
      const parent = rule.parent;
      matches.push({
        selector,
        declarations,
        media:
          parent?.type === "atrule" && parent.name === "media"
            ? parent.params
            : undefined,
      });
    }
  });
  assert.ok(matches.length > 0, candidate);
  return matches;
}

type Specificity = [number, number, number];
const add = (a: Specificity, b: Specificity): Specificity => [
  a[0] + b[0],
  a[1] + b[1],
  a[2] + b[2],
];
const compare = (a: Specificity, b: Specificity): number =>
  a[0] - b[0] || a[1] - b[1] || a[2] - b[2];
function specificity(node: SelectorNode): Specificity {
  if (node.type === "id") return [1, 0, 0];
  if (node.type === "class" || node.type === "attribute") return [0, 1, 0];
  if (node.type === "tag") return [0, 0, 1];
  if (node.type === "pseudo") {
    if (node.value === ":where") return [0, 0, 0];
    if ([":is", ":not", ":has"].includes(node.value))
      return (node.nodes ?? [])
        .map(specificity)
        .reduce((a, b) => (compare(a, b) >= 0 ? a : b), [
          0, 0, 0,
        ] as Specificity);
    return node.value.startsWith("::") ? [0, 0, 1] : [0, 1, 0];
  }
  if ("nodes" in node && Array.isArray(node.nodes))
    return node.nodes.map(specificity).reduce(add, [0, 0, 0] as Specificity);
  return [0, 0, 0];
}

function unnamedSiblings(selector: Selector, candidate: string): string[] {
  const found: string[] = [];
  selector.walkCombinators((combinator) => {
    if (combinator.value.trim() !== "~") return;
    const nodes = combinator.parent!.nodes;
    const next = nodes.slice(nodes.indexOf(combinator) + 1);
    const end = next.findIndex((node) => node.type === "combinator");
    const compound = end === -1 ? next : next.slice(0, end);
    if (!compound.some((node) => includesClass(node, candidate)))
      found.push(String(combinator.parent));
  });
  return found;
}

test("sidebar sibling utilities keep declarations and specificity through compilation and viewport gating", async () => {
  assert.equal(current.length, previous.length);
  const raw = await generated;
  const gated = gateViewportMedia(raw);
  assert.deepEqual(
    gated.queries.map(({ query }) => query),
    ["(min-width:48rem)"],
  );
  for (const css of [raw, optimize(gated.css, { minify: true }).code]) {
    for (const [old, current] of pairs) {
      const before = rules(css, old);
      const after = rules(css, current);
      assert.equal(after.length, before.length);
      assert.deepEqual(
        after.map((rule) => rule.declarations),
        before.map((rule) => rule.declarations),
      );
      assert.deepEqual(
        after.map((rule) => rule.media),
        before.map((rule) => rule.media),
      );
      assert.deepEqual(
        after.map((rule) => specificity(rule.selector)),
        before.map((rule) => specificity(rule.selector)),
      );
    }
  }
});

test("every sidebar sibling selector names its target after final CSS generation", async () => {
  const raw = await generated;
  assert.ok(
    previous.some((candidate) =>
      rules(raw, candidate).some(
        ({ selector }) => unnamedSiblings(selector, candidate).length > 0,
      ),
    ),
  );
  for (const css of [
    raw,
    optimize(gateViewportMedia(raw).css, { minify: true }).code,
  ]) {
    for (const candidate of candidates) {
      for (const { selector } of rules(css, candidate))
        assert.deepEqual(
          unnamedSiblings(selector, candidate),
          [],
          String(selector),
        );
    }
  }
});
