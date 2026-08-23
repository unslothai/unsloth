// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Kept out of kit.ts on purpose, like tsx-ast.ts: only the few tests that drive a
// component or a hook directly should pay for loading the TypeScript compiler.

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import ts from "typescript";

/** A JSX element as the stub runtime below records it. */
export type StubElement = {
  type: unknown;
  props: Record<string, unknown>;
};

/**
 * Runs one shipped src module with every import replaced by a stub, so a test can
 * call the real source with dependencies it controls. Bare node cannot import a
 * .tsx file at all, and a hook needs a React whose renders the test drives.
 */
export function loadWithStubs<T>(
  moduleUrl: URL,
  stubs: Record<string, unknown>,
): T {
  const path = fileURLToPath(moduleUrl);
  const { outputText } = ts.transpileModule(readFileSync(path, "utf8"), {
    fileName: path,
    compilerOptions: {
      target: ts.ScriptTarget.ES2022,
      module: ts.ModuleKind.CommonJS,
      jsx: ts.JsxEmit.ReactJSX,
    },
  });
  const loaded = { exports: {} as T };
  const requireStub = (specifier: string): unknown => {
    if (specifier in stubs) return stubs[specifier];
    throw new Error(`no stub for import "${specifier}" in ${path}`);
  };
  new Function("require", "module", "exports", outputText)(
    requireStub,
    loaded,
    loaded.exports,
  );
  return loaded.exports;
}

/** A react/jsx-runtime that builds plain records instead of React elements. */
export function stubJsxRuntime(): Record<string, unknown> {
  const jsx = (type: unknown, props: Record<string, unknown>): StubElement => ({
    type,
    props,
  });
  return { jsx, jsxs: jsx, Fragment: Symbol.for("Fragment") };
}
