// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Kept out of kit.ts so only tests that need it load the TypeScript compiler.

import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";

import ts from "@typescript/typescript6";

export type StubElement = {
  type: unknown;
  props: Record<string, unknown>;
};

function resolveSibling(fromPath: string, specifier: string): URL {
  const base = new URL(specifier, pathToFileURL(fromPath));
  for (const suffix of [".ts", ".tsx", "/index.ts", "/index.tsx", ""]) {
    const candidate = new URL(base.href + suffix);
    if (existsSync(fileURLToPath(candidate))) return candidate;
  }
  throw new Error(`cannot resolve import "${specifier}" from ${fromPath}`);
}

/** Run a shipped src module with every import replaced by a stub. */
export function loadWithStubs<T>(
  moduleUrl: URL,
  stubs: Record<string, unknown>,
  options: { relativePassthrough?: boolean } = {},
): T {
  const path = fileURLToPath(moduleUrl);
  const source = readFileSync(path, "utf8").replaceAll(
    "import.meta.url",
    JSON.stringify(pathToFileURL(path).href),
  );
  const { outputText } = ts.transpileModule(source, {
    fileName: path,
    compilerOptions: {
      target: ts.ScriptTarget.ES2022,
      module: ts.ModuleKind.CommonJS,
      jsx: ts.JsxEmit.ReactJSX,
      // TypeScript 6 turns this on by default, which would wrap the stubs in __importDefault.
      esModuleInterop: false,
    },
  });
  const loaded = { exports: {} as T };
  const requireStub = (specifier: string): unknown => {
    if (specifier in stubs) return stubs[specifier];
    // Opt-in so unrelated new imports do not break tests; off by default to keep the throw.
    if (options.relativePassthrough && specifier.startsWith(".")) {
      return loadWithStubs(resolveSibling(path, specifier), stubs, options);
    }
    throw new Error(`no stub for import "${specifier}" in ${path}`);
  };
  new Function("require", "module", "exports", outputText)(
    requireStub,
    loaded,
    loaded.exports,
  );
  return loaded.exports;
}

export function stubJsxRuntime(): Record<string, unknown> {
  const jsx = (type: unknown, props: Record<string, unknown>): StubElement => ({
    type,
    props,
  });
  return { jsx, jsxs: jsx, Fragment: Symbol.for("Fragment") };
}

/** Stub table for `loadWithStubs` where any import not in `overrides` is an inert module of named no-op functions. */
export function stubEveryImport(
  overrides: Record<string, unknown> = {},
): Record<string, unknown> {
  const inert = (specifier: string) => {
    const named = new Map<PropertyKey, unknown>();
    return new Proxy({} as Record<PropertyKey, unknown>, {
      get(_target, key) {
        if (key === "__esModule") return true;
        if (!named.has(key)) {
          const name = String(key);
          const fn = { [name]: (..._args: unknown[]) => null }[name];
          named.set(key, Object.assign(fn, { stubOf: specifier }));
        }
        return named.get(key);
      },
    });
  };
  const made = new Map<string, unknown>();
  return new Proxy(overrides, {
    has: () => true,
    get(target, key) {
      if (typeof key !== "string") return undefined;
      if (key in target) return target[key];
      if (!made.has(key)) made.set(key, inert(key));
      return made.get(key);
    },
  });
}

/** Every stub element in a rendered tree, depth first (does not call nested components). */
export function elementsOf(tree: unknown): StubElement[] {
  const out: StubElement[] = [];
  const walk = (value: unknown) => {
    if (Array.isArray(value)) {
      for (const item of value) walk(item);
      return;
    }
    if (!value || typeof value !== "object" || !("props" in value)) return;
    const element = value as StubElement;
    out.push(element);
    walk(element.props?.children);
  };
  walk(tree);
  return out;
}
