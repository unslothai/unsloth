// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import ts from "@typescript/typescript6";
import { readSrc } from "./helpers/kit.ts";

function parse(relative: string) {
  return ts.createSourceFile(
    relative,
    readSrc(relative),
    ts.ScriptTarget.Latest,
    true,
    ts.ScriptKind.TSX,
  );
}

// The `hint` expression on the Field wrapping the prompt Textarea.
function promptFieldHint(relative: string): string {
  const tree = parse(relative);
  const hints: string[] = [];
  const visit = (node: ts.Node) => {
    if (
      ts.isJsxElement(node) &&
      node.openingElement.tagName.getText(tree) === "Field" &&
      node.children.some(
        (child) =>
          (ts.isJsxSelfClosingElement(child) || ts.isJsxElement(child)) &&
          (ts.isJsxElement(child) ? child.openingElement : child)
            .getText(tree)
            .includes('data-type-to-activate="prompt"'),
      )
    ) {
      const hint = node.openingElement.attributes.properties.find(
        (prop) => ts.isJsxAttribute(prop) && prop.name.getText(tree) === "hint",
      );
      assert.ok(hint && ts.isJsxAttribute(hint) && hint.initializer, `${relative}: prompt Field has no hint`);
      const init = hint.initializer;
      assert.ok(ts.isJsxExpression(init) && init.expression);
      hints.push(init.expression.getText(tree));
    }
    ts.forEachChild(node, visit);
  };
  visit(tree);
  assert.equal(hints.length, 1, `${relative}: expected one prompt Field`);
  return hints[0];
}

function evaluate(expression: string, scope: Record<string, unknown>): unknown {
  const { outputText } = ts.transpileModule(`return (${expression});`, {
    compilerOptions: { target: ts.ScriptTarget.ES2022 },
  });
  return new Function(...Object.keys(scope), outputText)(...Object.values(scope));
}

const TIP = "Write the change you want as an instruction.";

test("the Images prompt Field shows the loaded model's tip, but not for an external model", () => {
  const hint = promptFieldHint("features/images/images-page.tsx");
  assert.equal(evaluate(hint, { external: null, status: { prompt_hint: TIP } }), TIP);
  assert.equal(evaluate(hint, { external: null, status: { prompt_hint: null } }), undefined);
  assert.equal(evaluate(hint, { external: null, status: null }), undefined);
  assert.equal(
    evaluate(hint, { external: { provider: "x" }, status: { prompt_hint: TIP } }),
    undefined,
  );
});

test("the Video prompt Field shows the loaded model's tip", () => {
  const hint = promptFieldHint("features/video/video-page.tsx");
  assert.equal(evaluate(hint, { status: { prompt_hint: TIP } }), TIP);
  assert.equal(evaluate(hint, { status: { prompt_hint: null } }), undefined);
  assert.equal(evaluate(hint, { status: { prompt_hint: "" } }), undefined);
  assert.equal(evaluate(hint, { status: null }), undefined);
});

test("both status types declare prompt_hint", () => {
  for (const file of ["features/images/api.ts", "features/video/api.ts"]) {
    assert.match(readSrc(file), /^\s*prompt_hint\?: string \| null;$/m, file);
  }
});

test("each page's Field renders InfoHint only for a truthy hint", () => {
  for (const file of ["features/images/images-page.tsx", "features/video/video-page.tsx"]) {
    const tree = parse(file);
    let body = "";
    const visit = (node: ts.Node) => {
      if (ts.isFunctionDeclaration(node) && node.name?.getText(tree) === "Field") {
        body = node.getText(tree);
      }
      ts.forEachChild(node, visit);
    };
    visit(tree);
    assert.ok(body, `${file}: no local Field`);
    const infoHints = body.match(/<InfoHint>/g) ?? [];
    assert.equal(infoHints.length, 1, file);
    assert.match(body, /\{hint && <InfoHint>\{hint\}<\/InfoHint>\}/, file);
  }
});
