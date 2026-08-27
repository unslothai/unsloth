// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * A value imported from the @/features/chat barrel must not be read while the
 * module is still loading.
 *
 * features/chat sits in an import cycle -- chat-runtime-store -> presets/
 * preset-load-config -> features/model-picker -> ... -> features/chat -- so a
 * module importing from the barrel can be evaluated while chat-runtime-store is
 * still initializing. Reading one of its `const` exports then hits the temporal
 * dead zone and throws at import time, which takes the whole page down rather
 * than failing anything locally:
 *
 *   [ansi-smoke] pageerror: Cannot access 'CHAT_GPU_MEMORY_MODE_KEY'
 *                           before initialization
 *
 * That shipped from hooks/use-model-memory.ts, whose WATCHED_STORAGE_KEYS array
 * listed the key at module scope. Reading inside a function is safe: by call
 * time every module has finished loading.
 *
 * This walks the real TypeScript AST rather than matching source text. A regex
 * version of this guard missed four separate shapes -- a second import
 * declaration in the same file (both thread.tsx and pickers.tsx have one), an
 * aliased specifier, a parenthesized read, and any expression that is not a
 * const/let initializer, such as a bare call or a static class field.
 */

import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import ts from "typescript";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SRC = path.join(HERE, "..", "src");
const BARREL = "@/features/chat";

/**
 * Local names this module binds to values pulled from the chat barrel.
 *
 * A namespace import contributes its own name: `import * as chat` makes
 * `chat.ANYTHING` a read of the namespace object, and the object itself is what
 * is uninitialized, so flagging a module-scope mention of `chat` is exactly
 * right.
 */
function barrelValueNames(source: ts.SourceFile): Set<string> {
  const names = new Set<string>();
  for (const statement of source.statements) {
    if (!ts.isImportDeclaration(statement)) continue;
    const specifier = statement.moduleSpecifier;
    if (!ts.isStringLiteral(specifier) || specifier.text !== BARREL) continue;
    const clause = statement.importClause;
    // `import type { ... }` is erased before the code runs, so it cannot trip a
    // temporal dead zone.
    if (!clause || clause.isTypeOnly) continue;
    if (clause.name) names.add(clause.name.text); // default import
    const bound = clause.namedBindings;
    if (!bound) continue;
    if (ts.isNamespaceImport(bound)) {
      names.add(bound.name.text);
      continue;
    }
    for (const element of bound.elements) {
      if (element.isTypeOnly) continue;
      // element.name is the LOCAL name, so `X as y` correctly yields `y`.
      names.add(element.name.text);
    }
  }
  return names;
}

function collectBindingNames(name: ts.BindingName, out: Set<string>): void {
  if (ts.isIdentifier(name)) {
    out.add(name.text);
    return;
  }
  for (const element of name.elements) {
    if (ts.isBindingElement(element)) collectBindingNames(element.name, out);
  }
}

/** Nodes that introduce a binding scope. */
function isScope(node: ts.Node): boolean {
  return (
    ts.isSourceFile(node) ||
    ts.isBlock(node) ||
    ts.isModuleBlock(node) ||
    ts.isCaseBlock(node) ||
    ts.isFunctionDeclaration(node) ||
    ts.isFunctionExpression(node) ||
    ts.isArrowFunction(node) ||
    ts.isMethodDeclaration(node) ||
    ts.isConstructorDeclaration(node) ||
    ts.isGetAccessorDeclaration(node) ||
    ts.isSetAccessorDeclaration(node) ||
    ts.isForStatement(node) ||
    ts.isForInStatement(node) ||
    ts.isForOfStatement(node) ||
    ts.isCatchClause(node) ||
    ts.isClassDeclaration(node) ||
    ts.isClassExpression(node)
  );
}

const declaredCache = new WeakMap<ts.Node, Set<string>>();

/** Names bound by this scope itself, not by anything nested inside it. */
function declaredIn(scope: ts.Node): Set<string> {
  const cached = declaredCache.get(scope);
  if (cached) return cached;
  const out = new Set<string>();
  const addStatements = (statements: readonly ts.Statement[]): void => {
    for (const statement of statements) {
      if (ts.isVariableStatement(statement)) {
        for (const d of statement.declarationList.declarations) collectBindingNames(d.name, out);
      } else if (ts.isFunctionDeclaration(statement) && statement.name) {
        out.add(statement.name.text);
      } else if (ts.isClassDeclaration(statement) && statement.name) {
        out.add(statement.name.text);
      }
    }
  };
  if (ts.isSourceFile(scope) || ts.isBlock(scope) || ts.isModuleBlock(scope)) {
    addStatements(scope.statements);
  } else if (ts.isCaseBlock(scope)) {
    for (const clause of scope.clauses) addStatements(clause.statements);
  } else if (ts.isCatchClause(scope)) {
    if (scope.variableDeclaration) collectBindingNames(scope.variableDeclaration.name, out);
  } else if (
    ts.isForStatement(scope) ||
    ts.isForInStatement(scope) ||
    ts.isForOfStatement(scope)
  ) {
    const initializer = scope.initializer;
    if (initializer && ts.isVariableDeclarationList(initializer)) {
      for (const d of initializer.declarations) collectBindingNames(d.name, out);
    }
  } else if (ts.isClassDeclaration(scope) || ts.isClassExpression(scope)) {
    if (scope.name) out.add(scope.name.text);
  }
  const parameters = (scope as ts.SignatureDeclarationBase).parameters;
  if (parameters) for (const p of parameters) collectBindingNames(p.name, out);
  if ((ts.isFunctionExpression(scope) || ts.isFunctionDeclaration(scope)) && scope.name) {
    out.add(scope.name.text);
  }
  declaredCache.set(scope, out);
  return out;
}

/**
 * True when an enclosing scope re-declares this name, so it is not the import.
 *
 * Resolved per occurrence rather than per file. Suppressing the name everywhere
 * once it is shadowed anywhere was simpler but cut both ways: a genuine
 * top-level read stopped being reported as soon as any function took a parameter
 * of the same name, and this tree has 4969 functions.
 */
function isShadowed(identifier: ts.Identifier): boolean {
  for (let node: ts.Node | undefined = identifier.parent; node; node = node.parent) {
    if (!isScope(node)) continue;
    // Module scope is where the import itself binds the name.
    if (ts.isSourceFile(node)) return false;
    if (declaredIn(node).has(identifier.text)) return true;
  }
  return false;
}

/** True when this identifier sits anywhere inside erased type syntax. */
function insideTypeSyntax(node: ts.Node): boolean {
  for (let current: ts.Node | undefined = node.parent; current; current = current.parent) {
    if (
      ts.isTypeNode(current) ||
      ts.isTypeAliasDeclaration(current) ||
      ts.isInterfaceDeclaration(current) ||
      ts.isTypeParameterDeclaration(current)
    ) {
      return true;
    }
  }
  return false;
}

/** True when this function is called on the spot, so its body runs eagerly. */
function isImmediatelyInvoked(node: ts.Node): boolean {
  let current: ts.Node = node;
  let parent = current.parent;
  while (parent && ts.isParenthesizedExpression(parent)) {
    current = parent;
    parent = parent.parent;
  }
  return Boolean(parent) && ts.isCallExpression(parent) && parent.expression === current;
}

/** Nodes whose bodies run when called, not when the module loads. */
function defersEvaluation(node: ts.Node): boolean {
  if (
    ts.isFunctionDeclaration(node) ||
    ts.isFunctionExpression(node) ||
    ts.isArrowFunction(node) ||
    ts.isMethodDeclaration(node) ||
    ts.isConstructorDeclaration(node) ||
    ts.isGetAccessorDeclaration(node) ||
    ts.isSetAccessorDeclaration(node)
  ) {
    // An IIFE runs during module initialization like any other expression.
    return !isImmediatelyInvoked(node);
  }
  // An instance field initializer runs at construction. A static one runs when
  // the class is defined, which is module load, so it is NOT deferred. Either
  // way the COMPUTED NAME is evaluated at class definition; the walk visits it
  // with the outer deferral for that reason.
  if (ts.isPropertyDeclaration(node)) {
    const isStatic = ts
      .getModifiers(node)
      ?.some((m) => m.kind === ts.SyntaxKind.StaticKeyword);
    return !isStatic;
  }
  return false;
}

/** True when this identifier is a name being declared or a property label. */
function isNonReference(node: ts.Identifier): boolean {
  const parent = node.parent;
  if (!parent) return false;
  // obj.NAME / {NAME: value} / label: -- not a read of the import.
  if (ts.isPropertyAccessExpression(parent) && parent.name === node) return true;
  if (ts.isPropertyAssignment(parent) && parent.name === node) return true;
  if (ts.isBindingElement(parent) && parent.propertyName === node) return true;
  // Type positions are erased before the code runs. Checked against every
  // ancestor, not just the parent: in `type T = chat.Entry` the `chat` node's
  // parent is a QualifiedName, and in `Foo<typeof K>` it is a type argument.
  if (insideTypeSyntax(node)) return true;
  return false;
}

function eagerReads(source: ts.SourceFile, names: Set<string>): string[] {
  if (names.size === 0) return [];

  const found: string[] = [];
  const visit = (node: ts.Node, deferred: boolean): void => {
    // The import declaration binds these names; it does not read them.
    if (ts.isImportDeclaration(node)) return;
    // `export { X }` re-exports the binding without evaluating it.
    if (ts.isExportDeclaration(node)) return;
    if (
      !deferred &&
      ts.isIdentifier(node) &&
      names.has(node.text) &&
      !isNonReference(node) &&
      !isShadowed(node)
    ) {
      const { line } = source.getLineAndCharacterOfPosition(node.getStart(source));
      found.push(`${node.text} (line ${line + 1})`);
    }
    const next = deferred || defersEvaluation(node);
    node.forEachChild((child) => {
      // `class C { [KEY] = 1 }` evaluates KEY when the class is defined, even
      // though the initializer waits for construction.
      const eagerName =
        ts.isPropertyDeclaration(node) &&
        child === node.name &&
        ts.isComputedPropertyName(child);
      visit(child, eagerName ? deferred : next);
    });
  };
  source.forEachChild((child) => visit(child, false));
  return found;
}

function parse(fileName: string, text: string): ts.SourceFile {
  return ts.createSourceFile(fileName, text, ts.ScriptTarget.ESNext, true);
}

function analyse(fileName: string, text: string): string[] {
  const source = parse(fileName, text);
  const names = barrelValueNames(source);
  return names.size === 0 ? [] : eagerReads(source, names);
}

function walkSources(dir: string, out: string[] = []): string[] {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) walkSources(full, out);
    else if (/\.tsx?$/.test(entry.name)) out.push(full);
  }
  return out;
}

/** Every module the barrel's own initialization can pull in, transitively. */
function barrelInitClosure(files: string[]): Set<string> {
  const sources = new Map<string, string>();
  for (const f of files) sources.set(f, readFileSync(f, "utf8"));

  const resolve = (spec: string, from: string): string | null => {
    let base: string;
    if (spec.startsWith("@/")) base = path.join(SRC, spec.slice(2));
    else if (spec.startsWith(".")) base = path.resolve(path.dirname(from), spec);
    else return null;
    // The exact path first: this codebase writes the extension on 66 imports,
    // and appending another produced "clipboard-payload.ts.ts", silently
    // dropping those edges and shrinking the at-risk set.
    const candidates = [
      base,
      `${base}.ts`,
      `${base}.tsx`,
      path.join(base, "index.ts"),
      path.join(base, "index.tsx"),
    ];
    for (const c of candidates) {
      if (sources.has(c)) return c;
    }
    return null;
  };

  const edges = (file: string): string[] => {
    const source = parse(file, sources.get(file) ?? "");
    const out: string[] = [];
    for (const st of source.statements) {
      const spec =
        (ts.isImportDeclaration(st) || ts.isExportDeclaration(st)) && st.moduleSpecifier;
      if (!spec || !ts.isStringLiteral(spec)) continue;
      // A type-only edge is erased and cannot drag a module into evaluation.
      if (ts.isImportDeclaration(st) && st.importClause?.isTypeOnly) continue;
      if (ts.isExportDeclaration(st) && st.isTypeOnly) continue;
      const target = resolve(spec.text, file);
      if (target) out.push(target);
    }
    return out;
  };

  const seen = new Set<string>();
  const stack = [path.join(SRC, "features", "chat", "index.ts")];
  while (stack.length > 0) {
    const node = stack.pop() as string;
    for (const next of edges(node)) {
      if (!seen.has(next)) {
        seen.add(next);
        stack.push(next);
      }
    }
  }
  return seen;
}

test("no module-scope read of a chat barrel value", () => {
  const files = walkSources(SRC);
  // Only a module the barrel can reach during its own initialization can be
  // caught half-initialized by the cycle. A leaf that merely imports from the
  // barrel is always evaluated after it, so an eager read there is safe and
  // flagging it would demand unrelated refactors to keep this green.
  const atRisk = barrelInitClosure(files);
  const offenders: string[] = [];
  let importers = 0;
  for (const file of files) {
    const text = readFileSync(file, "utf8");
    if (!text.includes(BARREL)) continue;
    const source = parse(file, text);
    const names = barrelValueNames(source);
    if (names.size === 0) continue;
    importers += 1;
    if (!atRisk.has(file)) continue;
    for (const hit of eagerReads(source, names)) {
      offenders.push(`${path.relative(SRC, file)}: ${hit}`);
    }
  }
  assert.deepEqual(
    offenders,
    [],
    `these read a value imported from ${BARREL} while the module is still loading, ` +
      `which throws if the import cycle re-enters before the binding is initialized. ` +
      `Move the read inside a function, as hooks/use-model-memory.ts does with ` +
      `watchedStorageKeys().\n  ${offenders.join("\n  ")}`,
  );
  // Anti-vacuity: a barrel rename would otherwise make this pass by finding nothing.
  assert.ok(importers >= 5, `only ${importers} modules import values from ${BARREL}`);
});

test("the scan catches every shape the regex version missed", () => {
  const cases: Array<[string, string]> = [
    ["plain const", `import { K } from "${BARREL}";\nconst a = [K];\n`],
    [
      "second import declaration",
      `import { A } from "${BARREL}";\nimport { K } from "${BARREL}";\nconst a = [K];\n`,
    ],
    ["aliased specifier", `import { K as k } from "${BARREL}";\nconst a = [k];\n`],
    ["parenthesized", `import { K } from "${BARREL}";\nconst a = (K);\n`],
    ["bare call expression", `import { K } from "${BARREL}";\nregister(K);\n`],
    [
      "static class field",
      `import { K } from "${BARREL}";\nclass C { static k = K; }\n`,
    ],
    [
      "namespace import",
      `import * as chat from "${BARREL}";\nconst a = chat.K;\n`,
    ],
    [
      "immediately invoked arrow",
      `import { K } from "${BARREL}";\nconst a = (() => K)();\n`,
    ],
    [
      "immediately invoked function expression",
      `import { K } from "${BARREL}";\nconst a = (function () { return K; })();\n`,
    ],
    [
      "computed instance field name",
      `import { K } from "${BARREL}";\nclass C { [K] = 1; }\n`,
    ],
    [
      "module-scope read in a file that also shadows the name in a function",
      `import { K } from "${BARREL}";\nfunction f(K) { return K; }\nconst a = [K];\n`,
    ],
    [
      "module-scope read in a file that also shadows the name in a block",
      `import { K } from "${BARREL}";\n{ const K = 1; use(K); }\nconst a = [K];\n`,
    ],
  ];
  for (const [label, code] of cases) {
    assert.equal(analyse("t.ts", code).length, 1, `${label} should be flagged`);
  }
});

test("deferred reads and non-references are left alone", () => {
  const cases: Array<[string, string]> = [
    ["arrow body", `import { K } from "${BARREL}";\nconst f = () => [K];\n`],
    [
      "function body",
      `import { K } from "${BARREL}";\nfunction f() { return K; }\n`,
    ],
    ["method body", `import { K } from "${BARREL}";\nclass C { m() { return K; } }\n`],
    [
      "instance field",
      `import { K } from "${BARREL}";\nclass C { k = K; }\n`,
    ],
    ["type-only import", `import type { K } from "${BARREL}";\nconst a: K = x;\n`],
    [
      "type-only specifier",
      `import { type K } from "${BARREL}";\nlet a: K;\n`,
    ],
    ["re-export", `import { K } from "${BARREL}";\nexport { K };\n`],
    [
      "unrelated property with the same name",
      `import { K } from "${BARREL}";\nconst f = () => obj.K;\n`,
    ],
    ["different module", `import { K } from "@/features/hub";\nconst a = [K];\n`],
    [
      "a name the module re-declares itself",
      `import { K } from "${BARREL}";\n{ const K = 1; consume(K); }\n`,
    ],
    [
      "namespace object only touched inside a function",
      `import * as chat from "${BARREL}";\nconst f = () => chat.K;\n`,
    ],
    [
      "deferred instance field initializer",
      `import { K } from "${BARREL}";\nclass C { k = K; }\n`,
    ],
    [
      "parameter of the same name, read in its own function",
      `import { K } from "${BARREL}";\nfunction f(K) { return K; }\n`,
    ],
    [
      "catch binding of the same name",
      `import { K } from "${BARREL}";\ntry { go(); } catch (K) { report(K); }\n`,
    ],
    [
      "loop binding of the same name",
      `import { K } from "${BARREL}";\nfor (const K of list) { use(K); }\n`,
    ],
    [
      "qualified type reference through a namespace import",
      `import * as chat from "${BARREL}";\ntype T = chat.PromptQueueUIEntry;\n`,
    ],
    [
      "typeof query nested in a type argument",
      `import { K } from "${BARREL}";\nlet a: Readonly<typeof K>;\n`,
    ],
    [
      "interface member typed through the namespace",
      `import * as chat from "${BARREL}";\ninterface I { e: chat.Entry }\n`,
    ],
  ];
  for (const [label, code] of cases) {
    assert.deepEqual(analyse("t.ts", code), [], `${label} should not be flagged`);
  }
});

test("the barrel closure resolves imports that spell their extension", () => {
  // The 66 such imports in this tree were invisible to the previous resolver,
  // which appended a second extension and dropped the edge.
  const closure = barrelInitClosure(walkSources(SRC));
  for (const relative of [
    "features/chat/utils/clipboard-payload.ts",
    "features/chat/stores/sidebar-organization-keys.ts",
  ]) {
    assert.ok(
      closure.has(path.join(SRC, relative)),
      `${relative} is reachable from the barrel but missing from the closure`,
    );
  }
});
