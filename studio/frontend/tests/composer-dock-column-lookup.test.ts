// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { runInNewContext } from "node:vm";

import ts from "@typescript/typescript6";

import { readSrc } from "./helpers/kit.ts";

const THREAD = readSrc("components/assistant-ui/thread.tsx");

const declaration = (name: string): string => {
  const parsed = ts.createSourceFile("thread.tsx", THREAD, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
  for (const statement of parsed.statements) {
    if (!ts.isVariableStatement(statement)) continue;
    const found = statement.declarationList.declarations
      .find((node) => ts.isIdentifier(node.name) && node.name.text === name);
    if (found) return `const ${found.getText(parsed)};`;
  }
  throw new Error(`${name} is not declared at module scope`);
};

class FakeElement {
  readonly children: FakeElement[];
  readonly classList: { contains: (name: string) => boolean };
  constructor(classes: string[], children: FakeElement[] = []) {
    this.children = children;
    this.classList = { contains: (name) => classes.includes(name) };
  }
}

const load = () => {
  const { outputText } = ts.transpileModule(
    `${declaration("childWithClass")} exports.childWithClass = childWithClass;`,
    { compilerOptions: { target: ts.ScriptTarget.ES2022 } },
  );
  const context = { exports: {} as { childWithClass: (parent: unknown, name: string) => unknown }, HTMLElement: FakeElement };
  runInNewContext(outputText, context);
  return context.exports.childWithClass;
};

test("the lookup returns the first direct child carrying the class and ignores descendants", () => {
  const childWithClass = load();
  const nested = new FakeElement(["aui-thread-viewport-footer"]);
  const footer = new FakeElement(["aui-thread-viewport-footer", "x"]);
  const later = new FakeElement(["aui-thread-viewport-footer"]);
  const viewport = new FakeElement(["aui-thread-viewport"], [
    new FakeElement(["row"], [nested]),
    footer,
    later,
  ]);
  assert.equal(childWithClass(viewport, "aui-thread-viewport-footer"), footer);
  assert.equal(childWithClass(viewport, "missing"), null);
  assert.equal(childWithClass(null, "aui-thread-viewport"), null);
  assert.equal(childWithClass(undefined, "aui-thread-viewport"), null);
});

test("a child that is not an HTML element is skipped even when it carries the class", () => {
  const childWithClass = load();
  class FakeSvgElement {
    readonly children: FakeElement[] = [];
    readonly classList = { contains: (name: string) => name === "aui-thread-viewport-footer" };
  }
  const footer = new FakeElement(["aui-thread-viewport-footer"]);
  const viewport = new FakeElement(["aui-thread-viewport"], [
    new FakeSvgElement() as unknown as FakeElement,
    footer,
  ]);
  assert.equal(childWithClass(viewport, "aui-thread-viewport-footer"), footer);
});

test("a nested match is not mistaken for the column box", () => {
  const childWithClass = load();
  const viewport = new FakeElement(["aui-thread-viewport"], [
    new FakeElement(["row"], [new FakeElement(["aui-thread-viewport-footer"])]),
  ]);
  assert.equal(childWithClass(viewport, "aui-thread-viewport-footer"), null);
});

test("the composer dock finds its column boxes by walking children, not by a selector over the thread", () => {
  const dock = THREAD.slice(THREAD.indexOf("const ThreadComposerDock: FC<"), THREAD.indexOf("const ThreadWelcome"));
  assert.ok(dock.includes(`childWithClass(dock?.parentElement, "aui-thread-viewport")`));
  assert.ok(dock.includes(`childWithClass(viewport, "aui-thread-viewport-footer")`));
  assert.equal(
    dock.includes(":scope >"),
    false,
    "the dock's resize observer runs on every composer height change; a :scope child selector there visits every element of the thread",
  );
});
