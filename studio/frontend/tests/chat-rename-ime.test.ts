// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import ts from "typescript";
import { readSrc } from "./helpers/kit.ts";

// Execute the shipped callbacks, following composer-submit-path.test.ts. This
// catches guards placed after preventDefault or after the rename side effects.
function handler(
  file: string,
  inline: boolean | "escape",
  deps: Record<string, unknown>,
) {
  const source = ts.createSourceFile(
    file,
    readSrc(file),
    ts.ScriptTarget.Latest,
    true,
    ts.ScriptKind.TSX,
  );
  const matches: ts.Node[] = [];
  function visit(node: ts.Node) {
    if (inline === "escape") {
      // The rename dialog's DialogContent, which wraps the commitRename input.
      if (
        ts.isJsxAttribute(node) &&
        node.name.getText(source) === "onEscapeKeyDown" &&
        node.initializer &&
        ts.isJsxExpression(node.initializer) &&
        node.initializer.expression &&
        ts.isArrowFunction(node.initializer.expression) &&
        node.parent.parent.parent.getText(source).includes("commitRename()")
      )
        matches.push(node.initializer.expression);
    } else if (inline) {
      if (
        ts.isFunctionDeclaration(node) &&
        node.name?.text === "handleInlineRenameKeyDown"
      )
        matches.push(node);
    } else if (
      ts.isJsxAttribute(node) &&
      node.name.getText(source) === "onKeyDown" &&
      node.initializer &&
      ts.isJsxExpression(node.initializer) &&
      node.initializer.expression &&
      ts.isArrowFunction(node.initializer.expression) &&
      node.initializer.expression.getText(source).includes("commitRename()")
    )
      matches.push(node.initializer.expression);
    ts.forEachChild(node, visit);
  }
  visit(source);
  assert.equal(
    matches.length,
    1,
    `${file}: unique ${inline === "escape" ? "dialog Escape" : "rename"} handler`,
  );
  const code = ts.transpileModule(`return (${matches[0].getText(source)});`, {
    compilerOptions: {
      target: ts.ScriptTarget.ES2022,
      module: ts.ModuleKind.None,
    },
  }).outputText;
  return new Function(...Object.keys(deps), code)(...Object.values(deps)) as (
    event: object,
  ) => void;
}

function fixture(file: string, inline: boolean, dirty = true) {
  const effects: string[] = [];
  const skipRenameBlurRef = { current: false };
  const onKey = handler(file, inline, {
    renameDirty: dirty,
    skipRenameBlurRef,
    commitRename: () => effects.push("save"),
    setRenamingTarget: () => effects.push("close"),
  });
  function key(
    key: string,
    isComposing = false,
    keyCode = key === "Enter" ? 13 : 27,
  ) {
    onKey({
      key,
      keyCode,
      nativeEvent: { isComposing, keyCode },
      preventDefault: () => effects.push("prevent"),
    });
  }
  return { effects, skipRenameBlurRef, key };
}

for (const [name, file, inline] of [
  ["sidebar inline", "components/app-sidebar.tsx", true],
  ["sidebar dialog", "components/app-sidebar.tsx", false],
  ["thread sidebar dialog", "features/chat/thread-sidebar.tsx", false],
] as const) {
  for (const [isComposing, keyCode] of [
    [true, 13],
    [false, 229],
    [true, 229],
  ] as const) {
    test(`${name}: IME Enter (${isComposing}, ${keyCode}) keeps editing until a separate Enter`, () => {
      const f = fixture(file, inline);
      f.key("Enter", isComposing, keyCode);
      assert.deepEqual(f.effects, []);
      assert.equal(f.skipRenameBlurRef.current, false);
      f.key("Enter");
      assert.equal(f.effects.filter((effect) => effect === "save").length, 1);
      if (inline) assert.equal(f.skipRenameBlurRef.current, true);
    });
  }
}

for (const dirty of [true, false]) {
  test(`sidebar inline: candidate Escape preserves editing (dirty=${dirty})`, () => {
    const f = fixture("components/app-sidebar.tsx", true, dirty);
    f.key("Escape", true);
    f.key("Escape", false, 229);
    assert.deepEqual(f.effects, []);
    assert.equal(f.skipRenameBlurRef.current, false);
    f.key("Escape");
    assert.deepEqual(f.effects, ["prevent", "close"]);
    assert.equal(f.skipRenameBlurRef.current, true);
  });
}

test("sidebar inline: an unchanged IME Enter does not close the input", () => {
  const f = fixture("components/app-sidebar.tsx", true, false);
  f.key("Enter", true);
  f.key("Enter", false, 229);
  assert.deepEqual(f.effects, []);
  f.key("Enter");
  assert.deepEqual(f.effects, ["prevent", "close"]);
});

for (const [name, file] of [
  ["sidebar dialog", "components/app-sidebar.tsx"],
  ["thread sidebar dialog", "features/chat/thread-sidebar.tsx"],
] as const) {
  test(`${name}: candidate Escape does not close the dialog`, () => {
    // Radix calls this from a document capture listener, before the input's
    // onKeyDown, so the input guard alone cannot keep the dialog open.
    const onEscape = handler(file, "escape", {});
    const press = (isComposing: boolean, keyCode: number) => {
      let prevented = false;
      onEscape({
        key: "Escape",
        isComposing,
        keyCode,
        preventDefault: () => (prevented = true),
      });
      return prevented;
    };
    assert.equal(press(true, 27), true);
    assert.equal(press(false, 229), true);
    assert.equal(press(false, 27), false);
  });
}
