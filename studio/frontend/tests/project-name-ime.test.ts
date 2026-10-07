// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import ts from "typescript";
import {
  imeOwnsInputKeydown,
  inputImeHandlers,
  newInputImeState,
} from "../src/features/chat/utils/composer-preferences.ts";
import { readSrc } from "./helpers/kit.ts";

type Target = "input" | "dialog";

function assertImeHandlerSpread(file: string, target: Target) {
  const source = ts.createSourceFile(
    file,
    readSrc(file),
    ts.ScriptTarget.Latest,
    true,
    ts.ScriptKind.TSX,
  );
  const matches: ts.JsxSpreadAttribute[] = [];
  function visit(node: ts.Node) {
    if (
      (ts.isJsxOpeningElement(node) || ts.isJsxSelfClosingElement(node)) &&
      node.tagName.getText(source) ===
        (target === "input" ? "input" : "DialogContent")
    ) {
      for (const attribute of node.attributes.properties) {
        if (
          ts.isJsxSpreadAttribute(attribute) &&
          attribute.expression.getText(source) === "nameImeHandlers"
        ) {
          matches.push(attribute);
        }
      }
    }
    ts.forEachChild(node, visit);
  }
  visit(source);
  assert.equal(
    matches.length,
    1,
    `${file}: unique ${target} IME handler spread`,
  );
}

function compositionHandlers(file: string, deps: Record<string, unknown>) {
  const source = ts.createSourceFile(
    file,
    readSrc(file),
    ts.ScriptTarget.Latest,
    true,
    ts.ScriptKind.TSX,
  );
  const matches: ts.ObjectLiteralExpression[] = [];
  function visit(node: ts.Node) {
    if (
      ts.isVariableDeclaration(node) &&
      node.name.getText(source) === "nameImeHandlers" &&
      node.initializer &&
      ts.isObjectLiteralExpression(node.initializer)
    ) {
      matches.push(node.initializer);
    }
    ts.forEachChild(node, visit);
  }
  visit(source);
  assert.equal(matches.length, 1, `${file}: unique IME handler object`);
  const code = ts.transpileModule(`return (${matches[0].getText(source)});`, {
    compilerOptions: {
      target: ts.ScriptTarget.ES2022,
      module: ts.ModuleKind.None,
    },
  }).outputText;
  return new Function(...Object.keys(deps), code)(...Object.values(deps)) as {
    onFocus: () => void;
    onBlur: () => void;
    onCompositionStart: () => void;
    onCompositionEnd: (event: { timeStamp: number }) => void;
  };
}

function keyHandler(
  file: string,
  target: Target,
  action: "commitCreate" | "save",
  deps: Record<string, unknown>,
) {
  const source = ts.createSourceFile(
    file,
    readSrc(file),
    ts.ScriptTarget.Latest,
    true,
    ts.ScriptKind.TSX,
  );
  const matches: ts.ArrowFunction[] = [];
  function visit(node: ts.Node) {
    if (
      ts.isJsxAttribute(node) &&
      node.name.getText(source) === "onKeyDown" &&
      node.initializer &&
      ts.isJsxExpression(node.initializer) &&
      node.initializer.expression &&
      ts.isArrowFunction(node.initializer.expression)
    ) {
      const opening = node.parent.parent;
      if (
        (ts.isJsxOpeningElement(opening) ||
          ts.isJsxSelfClosingElement(opening)) &&
        opening.tagName.getText(source) ===
          (target === "input" ? "input" : "DialogContent") &&
        node.initializer.expression.getText(source).includes(`${action}()`)
      ) {
        matches.push(node.initializer.expression);
      }
    }
    ts.forEachChild(node, visit);
  }
  visit(source);
  assert.equal(matches.length, 1, `${file}: unique ${target} handler`);
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

function fixture(
  file: string,
  action: "commitCreate" | "save",
  target: Target = "input",
) {
  const effects: string[] = [];
  const nameImeRef = { current: newInputImeState() };
  const ime = compositionHandlers(file, { inputImeHandlers, nameImeRef });
  const onKey = keyHandler(file, target, action, {
    imeOwnsInputKeydown,
    nameImeRef,
    commitCreate: () => effects.push("submit"),
    save: () => effects.push("submit"),
  });
  let now = 1000;
  function key({
    isComposing = false,
    keyCode = 13,
    metaKey = false,
    ctrlKey = false,
    repeat = false,
    advance = 1000,
  }: {
    isComposing?: boolean;
    keyCode?: number;
    metaKey?: boolean;
    ctrlKey?: boolean;
    repeat?: boolean;
    advance?: number;
  } = {}) {
    now += advance;
    onKey({
      key: "Enter",
      keyCode,
      metaKey,
      ctrlKey,
      shiftKey: false,
      altKey: false,
      repeat,
      timeStamp: now,
      nativeEvent: { isComposing },
      preventDefault: () => effects.push("prevent"),
      stopPropagation: () => effects.push("stop"),
    });
  }
  function compose(start: boolean) {
    if (start) ime.onCompositionStart();
    else ime.onCompositionEnd({ timeStamp: now + 999 });
  }
  return {
    effects,
    key,
    compose,
    focus: ime.onFocus,
    blur: ime.onBlur,
  };
}

for (const [name, file, action] of [
  [
    "new project",
    "features/chat/components/new-project-dialog.tsx",
    "commitCreate",
  ],
  ["edit project", "features/chat/components/edit-project-dialog.tsx", "save"],
] as const) {
  test(`${name}: composition events update the state used by its key handler`, () => {
    assertImeHandlerSpread(file, name === "new project" ? "input" : "dialog");
  });

  test(`${name}: composition Enter waits for a separate Enter`, () => {
    const f = fixture(file, action);
    f.compose(true);
    f.key({ isComposing: true });
    assert.equal(f.effects.includes("submit"), false);
    f.compose(false);
    f.key({ keyCode: 229 });
    assert.equal(f.effects.filter((effect) => effect === "submit").length, 1);
  });

  test(`${name}: WebKit compositionend-before-keydown waits for a separate Enter`, () => {
    const f = fixture(file, action);
    f.compose(true);
    f.compose(false);
    f.key({ keyCode: 229 });
    assert.equal(f.effects.includes("submit"), false);
    f.key({ keyCode: 229, advance: 100 });
    assert.equal(f.effects.filter((effect) => effect === "submit").length, 1);
  });

  test(`${name}: a held candidate Enter cannot auto-repeat into submit`, () => {
    const f = fixture(file, action);
    f.compose(true);
    f.key({ isComposing: true });
    f.compose(false);
    f.key({ keyCode: 229, repeat: true, advance: 50 });
    assert.equal(f.effects.includes("submit"), false);
    f.key({ keyCode: 229, advance: 50 });
    assert.equal(f.effects.filter((effect) => effect === "submit").length, 1);
  });

  test(`${name}: idle macOS Pinyin Enter still submits`, () => {
    const f = fixture(file, action);
    f.key({ keyCode: 229 });
    assert.equal(f.effects.filter((effect) => effect === "submit").length, 1);
  });

  for (const modifier of ["metaKey", "ctrlKey"] as const) {
    test(`${name}: idle macOS Pinyin ${modifier} Enter still submits`, () => {
      const f = fixture(file, action);
      f.key({ keyCode: 229, [modifier]: true });
      assert.equal(f.effects.filter((effect) => effect === "submit").length, 1);
    });
  }

  for (const reset of ["focus", "blur"] as const) {
    test(`${name}: ${reset} clears a missing compositionend`, () => {
      const f = fixture(file, action);
      f.compose(true);
      f[reset]();
      f.key({ keyCode: 229 });
      assert.equal(f.effects.filter((effect) => effect === "submit").length, 1);
    });
  }
}

test("edit project: composition blocks the save chord from the instructions field", () => {
  const f = fixture(
    "features/chat/components/edit-project-dialog.tsx",
    "save",
    "dialog",
  );
  f.compose(true);
  f.key({ isComposing: true, metaKey: true });
  assert.equal(f.effects.includes("submit"), false);
  f.compose(false);
  f.key({ metaKey: true });
  assert.equal(f.effects.filter((effect) => effect === "submit").length, 1);
});

for (const modifier of ["metaKey", "ctrlKey"] as const) {
  test(`edit project: idle macOS Pinyin ${modifier} save chord from instructions submits`, () => {
    const f = fixture(
      "features/chat/components/edit-project-dialog.tsx",
      "save",
      "dialog",
    );
    f.key({ keyCode: 229, [modifier]: true });
    assert.equal(f.effects.filter((effect) => effect === "submit").length, 1);
  });
}

test("edit project: blocked name candidate cannot bubble into the save chord", () => {
  const f = fixture("features/chat/components/edit-project-dialog.tsx", "save");
  f.compose(true);
  f.compose(false);
  f.key({ keyCode: 229, metaKey: true });
  assert.equal(f.effects.includes("submit"), false);
  assert.equal(f.effects.includes("stop"), true);
});
