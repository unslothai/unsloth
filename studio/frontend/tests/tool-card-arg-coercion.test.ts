// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import ts from "typescript";

import { toolArgText } from "../src/components/assistant-ui/tool-arg-text.ts";

const CARDS_DIR = "../src/components/assistant-ui/";

const sourceFile = (relative: string): ts.SourceFile => {
  const path = fileURLToPath(new URL(relative, import.meta.url));
  return ts.createSourceFile(
    path,
    readFileSync(path, "utf8"),
    ts.ScriptTarget.ESNext,
    true,
    ts.ScriptKind.TSX,
  );
};

// A model that answers `{"code": 42}` reaches `code.split("\n")`, and a throw in
// a card is caught by the router, not by the thread: all of Studio is replaced
// with "Something went wrong!". The message is persisted, so it reproduces on
// every reopen. These are the shapes local models actually send.
test("toolArgText renders whatever the model sent as text", () => {
  assert.equal(toolArgText(42), "42");
  assert.equal(toolArgText(0), "0");
  assert.equal(toolArgText(true), "true");
  assert.equal(toolArgText(["ls", "-la"]), "ls,-la");
  assert.equal(toolArgText({ cmd: "ls" }), "[object Object]");
  assert.equal(toolArgText("print(1)"), "print(1)");
});

// Absent and null both mean "the model has not written this yet", and the cards
// branch on the empty string to show their writing state.
test("toolArgText maps a missing argument to the empty string", () => {
  assert.equal(toolArgText(undefined), "");
  assert.equal(toolArgText(null), "");
});

test("a coerced argument survives the calls the cards make on it", () => {
  assert.equal(toolArgText(42).split("\n")[0], "42");
  assert.equal(toolArgText(42).slice(0, 60), "42");
  assert.equal(toolArgText(42).trim(), "42");
});

/**
 * Every `const <name> = ...` initializer declared inside `component`.
 *
 * Scoped to the component on purpose: the module-level parsing helpers in
 * tool-ui-web-search.tsx declare a `url` of their own out of a regex match, and
 * that one has nothing to do with what the model sent.
 */
function readConsts(
  relative: string,
  component: string,
  name: string,
): string[] {
  const source = sourceFile(relative);
  let body: ts.Node | undefined;
  const findComponent = (node: ts.Node): void => {
    if (ts.isVariableDeclaration(node) && node.name.getText() === component) {
      body = node.initializer;
    }
    node.forEachChild(findComponent);
  };
  source.forEachChild(findComponent);
  assert.ok(body, `${relative} does not declare ${component}`);

  const found: string[] = [];
  const visit = (node: ts.Node): void => {
    if (
      ts.isVariableDeclaration(node) &&
      node.name.getText() === name &&
      node.initializer
    ) {
      found.push(node.initializer.getText());
    }
    node.forEachChild(visit);
  };
  body.forEachChild(visit);
  return found;
}

// The argument each card calls a string method on. Asserting the declaration
// rather than the whole line lets the read be spelled `args`-cast or through a
// parsed-args object, which is how the cards already differ.
const COERCED: ReadonlyArray<
  readonly [file: string, component: string, props: readonly string[]]
> = [
  ["tool-ui-python.tsx", "PythonToolUIImpl", ["code"]],
  ["tool-ui-terminal.tsx", "TerminalToolUIImpl", ["command"]],
  ["tool-ui-knowledge-base.tsx", "KnowledgeBaseToolUIImpl", ["query"]],
  ["tool-ui-web-search.tsx", "WebSearchToolUIImpl", ["query", "url"]],
  [
    "tool-ui-code-execution.tsx",
    "CodeExecutionToolUIImpl",
    ["command", "path"],
  ],
  [
    "tool-ui-image-generation.tsx",
    "ImageGenerationToolUIImpl",
    ["prompt", "resultPrompt"],
  ],
];

const COERCION_CALL = /^toolArgText\(/;

test("every card reads its text arguments through toolArgText", () => {
  for (const [file, component, props] of COERCED) {
    for (const prop of props) {
      const [initializer, ...rest] = readConsts(
        CARDS_DIR + file,
        component,
        prop,
      );
      assert.ok(initializer, `${file} does not declare ${prop}`);
      assert.equal(rest.length, 0, `${file} declares ${prop} more than once`);
      assert.match(
        initializer,
        COERCION_CALL,
        `${file} reads ${prop} without coercing it; a model that sends a number there takes all of Studio down`,
      );
    }
  }
});

// render_html guards with `typeof === "string"` instead, which is also crash
// safe: it drops a non-string rather than rendering it. Listed so the closure
// test below stays exact.
const TYPEOF_GUARDED: ReadonlyArray<readonly [string, string]> = [
  ["tool-ui-render-html.tsx", "RenderHtmlToolUIImpl"],
];

// A new card is the way this bug comes back, so adding one has to fail here
// until somebody decides how it reads its arguments.
test("no tool card escapes the coercion policy", () => {
  const present = readdirSync(
    fileURLToPath(new URL(CARDS_DIR, import.meta.url)),
  )
    .filter((f) => f.startsWith("tool-ui-") && f.endsWith(".tsx"))
    .sort();
  const accounted = [
    ...COERCED.map(([file]) => file),
    ...TYPEOF_GUARDED.map(([file]) => file),
  ].sort();
  assert.deepEqual(
    present,
    accounted,
    "a tool card is not listed in COERCED or TYPEOF_GUARDED",
  );
});

const TYPEOF_GUARD = /typeof parsedArgs\.\w+ === "string"/;

test("the typeof-guarded card still guards", () => {
  for (const [file, component] of TYPEOF_GUARDED) {
    for (const prop of ["code", "title"]) {
      const [initializer] = readConsts(CARDS_DIR + file, component, prop);
      assert.ok(initializer, `${file} does not declare ${prop}`);
      assert.match(
        initializer,
        TYPEOF_GUARD,
        `${file} reads ${prop} without a type guard`,
      );
    }
  }
});
