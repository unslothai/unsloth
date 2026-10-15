// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A page in the browser panel relays Cmd/Ctrl+F, L, D, ... to the panel. While the page was full
// screen, the find bar or address bar opened under it, out of sight.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import ts from "typescript";

const TAB_VIEW = fileURLToPath(new URL("../src/features/browser/tab-view.tsx", import.meta.url));

function shortcutCase(): string {
  const source = ts.createSourceFile(TAB_VIEW, readFileSync(TAB_VIEW, "utf8"), ts.ScriptTarget.ESNext, true, ts.ScriptKind.TSX);
  let found: string | null = null;
  const visit = (node: ts.Node): void => {
    if (ts.isCaseClause(node) && ts.isStringLiteral(node.expression) && node.expression.text === "shortcut") found = node.getText();
    node.forEachChild(visit);
  };
  visit(source);
  assert.ok(found, "the frame message switch has a shortcut case");
  return found;
}

test("a relayed shortcut leaves full screen before acting, and acts at once otherwise", () => {
  const clause = shortcutCase();
  assert.match(clause, /if \(document\.fullscreenElement\) void document\.exitFullscreen\(\)\.catch\(\(\) => undefined\)\.then\(run\);\s*else run\(\);/);
  for (const action of ["store.focusAddress()", "requestFind(BROWSER_FIND_TARGET)", "store.bookmarkPage()", "store.newTab()"]) {
    const at = clause.indexOf(action);
    assert.ok(at > clause.indexOf("const run = () =>") && at < clause.indexOf("document.exitFullscreen()"), `${action} runs inside run()`);
  }
});
