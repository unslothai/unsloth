// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/*
 * No descendant-argument :has() on an ancestor of the thread (Chromium re-walks the whole thread
 * per insertion); child has-[>...] is cheap; other ancestors are guarded by the perf ladder.
 */

import assert from "node:assert/strict";
import test from "node:test";

import ts from "@typescript/typescript6";
import postcss from "postcss";
import selectorParser from "postcss-selector-parser";

import { openingTag } from "./helpers/tsx-ast.ts";

import { readText } from "./helpers/kit.ts";

const parse = (rel: string): ts.SourceFile =>
  ts.createSourceFile(rel, readText(rel), ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);

const stringLiterals = (source: ts.SourceFile): string[] => {
  const out: string[] = [];
  const walk = (node: ts.Node): void => {
    if (ts.isStringLiteral(node) || ts.isNoSubstitutionTemplateLiteral(node)) {
      out.push(node.text);
    }
    ts.forEachChild(node, walk);
  };
  walk(source);
  return out;
};

/*
 * has-[>[x]] is child (cheap); has-[[x]], has-data-*, has-aria-*, group-has-* are descendant.
 * Match on the class token so `has-[>...]` is not misread by a substring test.
 */
const descendantHasUtilities = (className: string): string[] =>
  className
    .split(/\s+/)
    .filter((token) => {
      const variant = token.startsWith("group-has-") ? token.slice("group-".length) : token;
      if (!variant.startsWith("has-")) return false;
      if (variant.startsWith("has-[>")) return false;
      return true;
    });

test("the sidebar wrapper, an ancestor of the whole app, has no descendant-argument :has()", () => {
  const source = parse("../src/components/ui/sidebar.tsx");
  const wrappers = stringLiterals(source).filter((s) => s.includes("group/sidebar-wrapper"));
  assert.equal(
    wrappers.length,
    1,
    "expected exactly one className carrying group/sidebar-wrapper; if this element was "
      + "restructured the assertion below is no longer looking at the ancestor of the thread",
  );
  assert.deepEqual(
    descendantHasUtilities(wrappers[0]),
    [],
    `sidebar-wrapper className carries a descendant-argument :has(): ${wrappers[0]}`,
  );
  // The rule must remain in child form; dropping it would change the inset sidebar.
  assert.ok(
    wrappers[0].includes("has-[>[data-variant=inset]]:bg-sidebar"),
    `sidebar-wrapper lost the inset background rule: ${wrappers[0]}`,
  );
});

test("the chat wrapper that contains the thread has no descendant-argument :has()", () => {
  const source = parse("../src/features/chat/chat-page.tsx");
  const wrappers: string[] = [];
  const walk = (node: ts.Node): void => {
    const tag = openingTag(node);
    if (tag?.attributes.properties.some((attr) =>
      ts.isJsxAttribute(attr) && attr.name.getText() === "data-chat-content-root",
    )) {
      for (const attr of tag.attributes.properties) {
        if (ts.isJsxAttribute(attr) && attr.name.getText() === "className") {
          assert.ok(attr.initializer && ts.isStringLiteral(attr.initializer));
          wrappers.push(attr.initializer.text);
        }
      }
    }
    ts.forEachChild(node, walk);
  };
  walk(source);
  assert.equal(wrappers.length, 1, "expected exactly one chat content root");
  assert.deepEqual(
    descendantHasUtilities(wrappers[0]),
    [],
    `the chat wrapper carries a descendant-argument :has(): ${wrappers[0]}`,
  );
});

test("index.css has no descendant-argument :has() on the panels that hold the thread", () => {
  const css = readText("../src/index.css");
  const offenders =
    css.match(/(?:#chat-thread|\.chat-artifact-split|\.chat-thread-pane)[^{},]*:has\((?!\s*>)/g) ?? [];
  assert.deepEqual(offenders, [], `descendant-argument :has() on a thread ancestor: ${offenders.join(" | ")}`);
});

// Elements that hold the whole thread: a descendant :has() on any of them re-walks every message.
const THREAD_ANCESTORS = [
  ["attribute", "data-wallpaper-workspace"],
  ["attribute", "data-chat-content-root"],
  // Same element as the workspace on Windows non-chat pages, where the chat stays mounted.
  ["attribute", "data-windows-wallpaper-page"],
  ["class", "aui-thread-viewport"],
] as const;

type Node = selectorParser.Node;

// The compound (run of simple selectors between combinators) that `node` sits in.
const compoundOf = (node: Node): Node[] => {
  const parts = node.parent?.nodes ?? [];
  const at = parts.indexOf(node as never);
  let from = at;
  while (from > 0 && parts[from - 1].type !== "combinator") from--;
  let to = at;
  while (to < parts.length - 1 && parts[to + 1].type !== "combinator") to++;
  return parts.slice(from, to + 1) as Node[];
};

// :has() nested in :not()/:is()/:where() still anchors on the outer compound.
const anchorCompound = (pseudo: selectorParser.Pseudo): Node[] => {
  let node: Node = pseudo;
  for (;;) {
    const outer: Node | undefined = node.parent?.parent as Node | undefined;
    if (!outer || !selectorParser.isPseudo(outer) || outer.value === ":has") break;
    node = outer;
  }
  return compoundOf(node);
};

const isThreadAncestor = (compound: Node[]): boolean => {
  const hits = (node: Node): boolean =>
    THREAD_ANCESTORS.some(([type, value]) =>
      node.type === type
      && (type === "attribute" ? (node as selectorParser.Attribute).attribute : node.value) === value)
    || (node.type === "pseudo" && node.value === ":is" && node.nodes.some((sel) => sel.nodes.some(hits)));
  return compound.some(hits);
};

const childOnly = (path: selectorParser.Selector): boolean => {
  if (path.first?.type !== "combinator" || path.first.value !== ">") return false;
  let ok = true;
  path.each((node) => {
    if (node.type === "combinator" && node.value !== ">") ok = false;
  });
  return ok;
};

const descendantHasOnThreadAncestors = (cssText: string): string[] => {
  const offenders: string[] = [];
  postcss.parse(cssText).walkRules((rule) => {
    if (!rule.selector.includes(":has(")) return;
    selectorParser().astSync(rule.selector).walkPseudos((pseudo) => {
      if (pseudo.value !== ":has" || !isThreadAncestor(anchorCompound(pseudo))) return;
      for (const path of pseudo.nodes) {
        if (!childOnly(path)) offenders.push(`${String(pseudo)} in ${rule.selector}`);
      }
    });
  });
  return offenders;
};

test("no descendant-argument :has() anchored on an element that holds the thread", () => {
  const offenders = descendantHasOnThreadAncestors(readText("../src/index.css"));
  assert.deepEqual(offenders, [], offenders.join("\n"));
});

test("the thread-ancestor :has() guard catches the shapes that regressed typing", () => {
  const workspace = ":is([data-wallpaper-workspace], [data-chat-content-root])";
  const bad = [
    `${workspace}:has(> .chat-wallpaper[data-ready] .chat-wallpaper-image[data-fade="false"]) .thread-bottom-fade {}`,
    `${workspace}:has(> .chat-wallpaper[data-ready]):not(:has(.aui-thread-welcome-center)) > .chat-wallpaper {}`,
    `${workspace}:has([data-tour="chat-compare-view"]) > .chat-wallpaper {}`,
    `${workspace} .aui-thread-viewport:has(.aui-thread-welcome-center) {}`,
    `html:not(.dark)[data-glass] ${workspace}:has(.aui-thread-welcome-center) .chat-wallpaper-frame {}`,
    "[data-windows-wallpaper-page]:has([data-windows-glass-frame]) > [data-slot=\"page-wallpaper-blur\"] {}",
  ];
  for (const css of bad) assert.equal(descendantHasOnThreadAncestors(css).length, 1, css);
  const good = [
    `${workspace}:has(> .chat-wallpaper[data-ready] > .chat-wallpaper-frame > .chat-wallpaper-image[data-fade="false"]) .thread-bottom-fade {}`,
    `html:not([data-chat-welcome]) ${workspace}:has(> .chat-wallpaper[data-ready]) > .chat-wallpaper {}`,
    `${workspace} .aui-thread-viewport:has(> .aui-thread-welcome-root > .aui-thread-welcome-center) {}`,
    ".composer-pill-btn:has(.composer-pill-caret) {}",
  ];
  for (const css of good) assert.deepEqual(descendantHasOnThreadAncestors(css), [], css);
});

test("the welcome and compare views stamp the root flags the wallpaper CSS reads", () => {
  const welcome = readText("../src/features/chat-background/welcome.ts");
  assert.match(welcome, /ROOT_WELCOME = "data-chat-welcome"/);
  assert.match(welcome, /ROOT_COMPARE = "data-chat-compare"/);
  const thread = readText("../src/components/assistant-ui/thread.tsx");
  const welcomeBody = thread.slice(thread.indexOf("const ThreadWelcome"));
  assert.ok(welcomeBody.includes("useMarkChatWelcome();"), "ThreadWelcome must mark the welcome flag");
  // The child-only viewport rule depends on this exact nesting.
  assert.match(welcomeBody, /className="aui-thread-welcome-root[^"]*"[\s\S]{0,200}"aui-thread-welcome-center/);
  const page = readText("../src/features/chat/chat-page.tsx");
  const shell = page.slice(page.indexOf("function CompareShell"), page.indexOf("function CompareShell") + 800);
  assert.ok(shell.includes("useMarkChatCompare();"), "CompareShell must mark the compare flag");
  assert.ok(shell.includes('data-tour="chat-compare-view"'));
});
