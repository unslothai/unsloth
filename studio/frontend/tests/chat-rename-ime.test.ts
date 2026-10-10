// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import ts from "typescript";
import { readSrc } from "./helpers/kit.ts";
import {
  imeOwnsInputKeydown,
  inputImeHandlers,
  newInputImeState,
} from "../src/features/chat/utils/composer-preferences.ts";

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
  const renameImeRef = { current: newInputImeState() };
  const ime = inputImeHandlers(renameImeRef.current);
  let now = 1000;
  const onKey = handler(file, inline, {
    imeOwnsInputKeydown,
    renameImeRef,
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
    now += 1000;
    onKey({
      key,
      keyCode,
      metaKey: false,
      ctrlKey: false,
      shiftKey: false,
      altKey: false,
      timeStamp: now,
      nativeEvent: { isComposing, keyCode },
      preventDefault: () => effects.push("prevent"),
    });
  }
  // WebKit order (bug 165004): compositionend lands just before the committing 229 keydown.
  function compose(start: boolean) {
    if (start) ime.onCompositionStart();
    else ime.onCompositionEnd({ timeStamp: now + 999 });
  }
  return { effects, skipRenameBlurRef, key, compose };
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
      f.compose(true);
      if (!isComposing) f.compose(false);
      f.key("Enter", isComposing, keyCode);
      assert.deepEqual(f.effects, []);
      assert.equal(f.skipRenameBlurRef.current, false);
      if (isComposing) f.compose(false);
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

for (const [name, file, inline] of [
  ["sidebar inline", "components/app-sidebar.tsx", true],
  ["sidebar dialog", "components/app-sidebar.tsx", false],
  ["thread sidebar dialog", "features/chat/thread-sidebar.tsx", false],
] as const) {
  test(`${name}: idle macOS Pinyin Enter (229, no composition) saves (#12137)`, () => {
    const f = fixture(file, inline);
    f.key("Enter", false, 229);
    assert.equal(f.effects.filter((effect) => effect === "save").length, 1);
  });

  test(`${name}: open composition keeps a 229 Enter blocked`, () => {
    const f = fixture(file, inline);
    f.compose(true);
    f.key("Enter", false, 229);
    assert.deepEqual(f.effects, []);
  });
}

test("sidebar inline: a separate idle Pinyin Enter closes an unchanged input", () => {
  const f = fixture("components/app-sidebar.tsx", true, false);
  f.compose(true);
  f.key("Enter", true);
  f.compose(false);
  f.key("Enter", false, 229);
  assert.deepEqual(f.effects, ["prevent", "close"]);
});

for (const [name, file] of [
  ["sidebar dialog", "components/app-sidebar.tsx"],
  ["thread sidebar dialog", "features/chat/thread-sidebar.tsx"],
] as const) {
  test(`${name}: candidate Escape does not close the dialog`, () => {
    // Radix fires this before the input's onKeyDown, so the input guard alone is not enough.
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

for (const reset of ["onFocus", "onBlur"] as const) {
  test(`missing compositionend is cleared by ${reset}`, () => {
    const state = newInputImeState();
    const ime = inputImeHandlers(state);
    ime.onCompositionStart();
    assert.equal(state.open, true);
    ime[reset]();
    const enter = {
      key: "Enter",
      keyCode: 229,
      metaKey: false,
      ctrlKey: false,
      shiftKey: false,
      altKey: false,
      timeStamp: 5000,
      nativeEvent: { isComposing: false },
    };
    assert.equal(imeOwnsInputKeydown(enter, state), false);
  });
}

const plainEnter = (keyCode: number, timeStamp: number) => ({
  key: "Enter",
  keyCode,
  metaKey: false,
  ctrlKey: false,
  shiftKey: false,
  altKey: false,
  timeStamp,
  nativeEvent: { isComposing: false },
});

test("focus change clears a recent compositionend so the next idle Pinyin Enter saves", () => {
  const state = newInputImeState();
  const ime = inputImeHandlers(state);
  ime.onCompositionStart();
  ime.onCompositionEnd({ timeStamp: 1000 });
  ime.onBlur();
  ime.onFocus();
  assert.equal(imeOwnsInputKeydown(plainEnter(229, 1100), state), false);
});

test("a keyCode 13 candidate-confirming Enter inside an open composition is swallowed once", () => {
  const state = newInputImeState();
  const ime = inputImeHandlers(state);
  ime.onCompositionStart();
  assert.equal(imeOwnsInputKeydown(plainEnter(13, 2000), state), true);
  ime.onCompositionEnd({ timeStamp: 2010 });
  assert.equal(imeOwnsInputKeydown(plainEnter(229, 2100), state), false);
});

test("a Chrome-order composition does not swallow the next idle Pinyin Enter", () => {
  const state = newInputImeState();
  const ime = inputImeHandlers(state);
  ime.onCompositionStart();
  assert.equal(
    imeOwnsInputKeydown(
      {
        ...plainEnter(229, 1000),
        nativeEvent: { isComposing: true },
      },
      state,
    ),
    true,
  );
  ime.onCompositionEnd({ timeStamp: 1010 });
  assert.equal(imeOwnsInputKeydown(plainEnter(229, 1100), state), false);
});

test("every rename input resets IME state on focus and blur", () => {
  for (const file of [
    "components/app-sidebar.tsx",
    "features/chat/thread-sidebar.tsx",
  ]) {
    const source = ts.createSourceFile(
      file,
      readSrc(file),
      ts.ScriptTarget.Latest,
      true,
      ts.ScriptKind.TSX,
    );
    function visit(node: ts.Node) {
      if (ts.isJsxAttributes(node)) {
        const props = node.properties;
        const spread = props.findIndex(
          (p) =>
            ts.isJsxSpreadAttribute(p) &&
            p.expression.getText(source).startsWith("inputImeHandlers("),
        );
        if (spread >= 0) {
          for (const name of ["onFocus", "onBlur"]) {
            const override = props.findIndex(
              (p, i) =>
                i > spread &&
                ts.isJsxAttribute(p) &&
                p.name.getText(source) === name,
            );
            if (override >= 0)
              assert.match(
                props[override].getText(source),
                /resetInputIme\(renameImeRef\.current\)/,
                `${file}: ${name} override drops the IME reset`,
              );
          }
        }
      }
      ts.forEachChild(node, visit);
    }
    visit(source);
  }
});
