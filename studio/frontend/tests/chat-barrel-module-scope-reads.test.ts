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
function barrelValueNames(
  source: ts.SourceFile,
  // A local module that re-exports barrel values hands out the same live ESM
  // bindings, so importing from it is importing from the barrel. Callers that
  // have the module graph pass a predicate; the default is the direct case.
  carriesBarrelValues: (specifier: string) => boolean = (s) => s === BARREL,
): Set<string> {
  const names = new Set<string>();
  for (const statement of source.statements) {
    if (!ts.isImportDeclaration(statement)) continue;
    const specifier = statement.moduleSpecifier;
    if (!ts.isStringLiteral(specifier) || !carriesBarrelValues(specifier.text)) continue;
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

/** Scopes a `var` binds to: a function body or the module itself. */
function isVarScope(node: ts.Node): boolean {
  return (
    ts.isSourceFile(node) ||
    ts.isModuleBlock(node) ||
    ts.isFunctionDeclaration(node) ||
    ts.isFunctionExpression(node) ||
    ts.isArrowFunction(node) ||
    ts.isMethodDeclaration(node) ||
    ts.isConstructorDeclaration(node) ||
    ts.isGetAccessorDeclaration(node) ||
    ts.isSetAccessorDeclaration(node)
  );
}

/** Every `var` name declared under this scope, excluding nested function scopes. */
function collectHoistedVars(scope: ts.Node, out: Set<string>): void {
  const visit = (node: ts.Node): void => {
    // A nested function owns its own vars; they do not reach this scope.
    if (node !== scope && isVarScope(node)) return;
    if (
      ts.isVariableDeclarationList(node) &&
      !(node.flags & (ts.NodeFlags.Let | ts.NodeFlags.Const))
    ) {
      for (const d of node.declarations) collectBindingNames(d.name, out);
    }
    node.forEachChild(visit);
  };
  scope.forEachChild(visit);
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
  // `var` binds to the enclosing function or module, not to the block it is
  // written in, so a function scope has to collect the ones nested inside it.
  // Recording them only on the inner block left a later read in the function
  // body resolving to the import instead of the local, which reported a file
  // that never touches the barrel at that point. `let` and `const` keep block
  // scoping and are gathered by addStatements above, per block.
  if (isVarScope(scope)) collectHoistedVars(scope, out);
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

/**
 * True when this identifier sits in a heritage clause that survives to runtime,
 * i.e. `class C extends K`. `implements` is erased, and so is `interface I
 * extends J`, but a base class is an ordinary expression evaluated when the
 * class is defined. TypeScript still wraps it in an ExpressionWithTypeArguments,
 * which `ts.isTypeNode` accepts, so it has to be excluded by hand.
 */
function inRuntimeHeritage(node: ts.Node): boolean {
  // Follow the expression spine only. A type argument on the base class hangs
  // off `typeArguments` rather than `expression`, so it stays erased, while
  // `extends ns.K` and `extends makeBase(K)` are both reached.
  let current: ts.Node = node;
  let parent = current.parent;
  while (parent && !ts.isExpressionWithTypeArguments(parent)) {
    if (!ts.isExpression(parent)) return false;
    current = parent;
    parent = parent.parent;
  }
  if (!parent || parent.expression !== current) return false;
  const clause = parent.parent;
  if (!clause || !ts.isHeritageClause(clause)) return false;
  if (clause.token !== ts.SyntaxKind.ExtendsKeyword) return false;
  const declaration = clause.parent;
  if (!declaration) return false;
  if (!ts.isClassDeclaration(declaration) && !ts.isClassExpression(declaration)) return false;
  // `declare class C extends K {}` emits no JavaScript at all, so its base is
  // never evaluated and the name is never read. Without this the ambient
  // declaration is reported as an eager read and the guard rejects a file that
  // cannot crash.
  return !isAmbient(declaration);
}

/** True when this declaration is ambient, so it emits nothing to run. */
function isAmbient(node: ts.Node): boolean {
  for (let current: ts.Node | undefined = node; current; current = current.parent) {
    if (ts.isSourceFile(current)) return current.isDeclarationFile;
    const modifiers = ts.canHaveModifiers(current) ? ts.getModifiers(current) : undefined;
    if (modifiers?.some((m) => m.kind === ts.SyntaxKind.DeclareKeyword)) return true;
  }
  return false;
}

/** True when this identifier sits anywhere inside erased type syntax. */
function insideTypeSyntax(node: ts.Node): boolean {
  if (inRuntimeHeritage(node)) return false;
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
  if (!parent) return false;
  // `new (function () { ... })()` runs the body synchronously during
  // construction, so it is as eager as a plain call.
  if (
    (ts.isCallExpression(parent) || ts.isNewExpression(parent)) &&
    parent.expression === current
  ) {
    return true;
  }
  // `new Promise(executor)` calls the executor synchronously before returning,
  // so a function handed to it runs during initialization like an IIFE. Only
  // Promise: an arbitrary callback argument may be stored and called much later,
  // and treating those as eager would report reads that never happen at load.
  if (
    ts.isNewExpression(parent) &&
    ts.isIdentifier(parent.expression) &&
    parent.expression.text === "Promise" &&
    parent.arguments?.[0] === current
  ) {
    return true;
  }
  // `.call(...)` and `.apply(...)` invoke on the spot too, but they put a
  // property access between the function expression and the call, so the check
  // above reads the body as deferred and the walk never looks inside it.
  return (
    ts.isPropertyAccessExpression(parent) &&
    parent.expression === current &&
    (parent.name.text === "call" || parent.name.text === "apply") &&
    Boolean(parent.parent) &&
    ts.isCallExpression(parent.parent) &&
    parent.parent.expression === parent
  );
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
  // `class C { static K = 1 }` -- a plain member label, not a read. Computed
  // names are a different matter and are deliberately left to fall through.
  if (
    (ts.isPropertyDeclaration(parent) ||
      ts.isMethodDeclaration(parent) ||
      ts.isGetAccessorDeclaration(parent) ||
      ts.isSetAccessorDeclaration(parent)) &&
    parent.name === node
  ) {
    return true;
  }
  // Type positions are erased before the code runs. Checked against every
  // ancestor, not just the parent: in `type T = chat.Entry` the `chat` node's
  // parent is a QualifiedName, and in `Foo<typeof K>` it is a type argument.
  if (insideTypeSyntax(node)) return true;
  return false;
}

/** What an eager call or construction in this scope can be followed into. */
interface Targets {
  functions: Map<string, ts.Node>;
  classes: Map<string, ts.ClassLikeDeclaration>;
  objects: Map<string, ts.ObjectLiteralExpression>;
}

/**
 * The callables a scope's own statements declare, layered over the enclosing
 * scope's.
 *
 * Collected per scope rather than only at the top level. A helper declared
 * inside a function that is itself called during initialization is reached by
 * that outer call, so `function outer() { function inner() { return K; } return
 * inner(); }` needs `inner` visible while outer's body is walked; a map built
 * from `source.statements` alone leaves it unresolvable and the read inside it
 * unseen.
 */
function collectTargets(statements: readonly ts.Statement[], inherited?: Targets): Targets {
  const targets: Targets = {
    functions: new Map(inherited?.functions),
    classes: new Map(inherited?.classes),
    objects: new Map(inherited?.objects),
  };
  for (const statement of statements) {
    if (ts.isFunctionDeclaration(statement) && statement.name) {
      targets.functions.set(statement.name.text, statement);
      continue;
    }
    if (ts.isClassDeclaration(statement) && statement.name) {
      targets.classes.set(statement.name.text, statement);
      continue;
    }
    if (!ts.isVariableStatement(statement)) continue;
    for (const declaration of statement.declarationList.declarations) {
      const initializer = declaration.initializer;
      if (!initializer || !ts.isIdentifier(declaration.name)) continue;
      if (ts.isArrowFunction(initializer) || ts.isFunctionExpression(initializer)) {
        targets.functions.set(declaration.name.text, initializer);
      } else if (ts.isClassExpression(initializer)) {
        targets.classes.set(declaration.name.text, initializer);
      } else if (ts.isObjectLiteralExpression(initializer)) {
        // Reading a property off this object can run a getter defined on it.
        targets.objects.set(declaration.name.text, initializer);
      }
    }
  }
  return targets;
}

/** The scope a declaration belongs to. */
function declaringScopeOf(node: ts.Node): ts.Node | undefined {
  for (let current: ts.Node | undefined = node.parent; current; current = current.parent) {
    if (isScope(current)) return current;
  }
  return undefined;
}

/**
 * True when this name really refers to that target here.
 *
 * A plain "is it shadowed anywhere above" test cannot answer this once helpers
 * resolve per scope, because a nested helper's own declaration IS a binding
 * above the call and reads as a shadow of itself. So walk out and ask which
 * comes first: the scope the target is declared in, or some other scope that
 * binds the same name.
 */
function resolvesToTarget(identifier: ts.Identifier, target: ts.Node): boolean {
  const home = declaringScopeOf(target);
  if (!home) return false;
  for (let node: ts.Node | undefined = identifier.parent; node; node = node.parent) {
    if (!isScope(node)) continue;
    if (node === home) return true;
    if (declaredIn(node).has(identifier.text)) return false;
    if (ts.isSourceFile(node)) return false;
  }
  return false;
}

/** True when this node carries the given modifier. */
function hasModifier(node: ts.Node, kind: ts.SyntaxKind): boolean {
  const modifiers = ts.canHaveModifiers(node) ? ts.getModifiers(node) : undefined;
  return Boolean(modifiers?.some((m) => m.kind === kind));
}

function firstSuspensionPos(body: ts.Node): number | null {
  let earliest: number | null = null;
  const visit = (node: ts.Node): void => {
    if (node !== body && isVarScope(node) && !ts.isSourceFile(node)) return;
    const suspends =
      ts.isAwaitExpression(node) ||
      ((ts.isForOfStatement(node) && node.awaitModifier !== undefined));
    if (suspends && (earliest === null || node.getStart() < earliest)) {
      earliest = node.getStart();
    }
    node.forEachChild(visit);
  };
  visit(body);
  return earliest;
}

/** Walk only the part of a body that runs before `limit`. */
function visitUntil(
  body: ts.Node,
  limit: number,
  visit: (node: ts.Node, deferred: boolean) => void,
): void {
  const walk = (node: ts.Node): void => {
    if (node.getStart() >= limit) return;
    visit(node, false);
  };
  body.forEachChild(walk);
}

function eagerReads(source: ts.SourceFile, names: Set<string>): string[] {
  if (names.size === 0) return [];

  const moduleTargets = collectTargets(source.statements);
  // Guards against recursion, and stops a function called twice from being
  // reported twice.
  const entered = new Set<ts.Node>();

  const found: string[] = [];

  /**
   * Walk a callable an eager call reached, holding back what does not run yet.
   *
   * Defaults are evaluated on entry, before the body. A generator call only
   * builds an iterator, and an async function resumes past initialization, so
   * the body is walked only as far as its first suspension.
   */
  const enterCallable = (target: ts.Node, targets: Targets): void => {
    const fn = target as ts.FunctionLikeDeclaration;
    for (const parameter of fn.parameters ?? []) {
      if (parameter.initializer) visit(parameter.initializer, false, targets);
    }
    if (!fn.body || fn.asteriskToken) return;
    const inner = ts.isBlock(fn.body) ? collectTargets(fn.body.statements, targets) : targets;
    const suspendsAt = firstSuspensionPos(fn.body);
    if (suspendsAt === null) visit(fn.body, false, inner);
    else visitUntil(fn.body, suspendsAt, (n, d) => visit(n, d, inner));
  };

  const visit = (node: ts.Node, deferred: boolean, targets: Targets): void => {
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
    // `const value = read()` at module scope runs read's body now, so the read
    // inside it is eager even though the declaration looked deferred. Without
    // this the guard's own advice -- move the read into a function -- could be
    // followed to the letter and still leave the crash in place.
    if (!deferred && ts.isCallExpression(node) && ts.isIdentifier(node.expression)) {
      const target = targets.functions.get(node.expression.text);
      if (target && !entered.has(target) && resolvesToTarget(node.expression, target)) {
        entered.add(target);
        enterCallable(target, targets);
      }
    }
    // `new C()` runs the constructor and every instance field initializer now,
    // for a named class exactly as for an inline function expression.
    if (!deferred && ts.isNewExpression(node) && ts.isIdentifier(node.expression)) {
      const cls = targets.classes.get(node.expression.text);
      if (cls && !entered.has(cls) && resolvesToTarget(node.expression, cls)) {
        entered.add(cls);
        for (const member of cls.members) {
          if (ts.isConstructorDeclaration(member) && member.body) {
            enterCallable(member, targets);
          } else if (
            ts.isPropertyDeclaration(member) &&
            member.initializer &&
            !hasModifier(member, ts.SyntaxKind.StaticKeyword)
          ) {
            visit(member.initializer, false, targets);
          }
        }
      }
    }
    // `obj.value` runs a getter defined on obj synchronously, so an eager read
    // of the property is an eager read of whatever the accessor body touches.
    if (!deferred && ts.isPropertyAccessExpression(node) && ts.isIdentifier(node.expression)) {
      const object = targets.objects.get(node.expression.text);
      if (object && resolvesToTarget(node.expression, object)) {
        for (const property of object.properties) {
          if (
            ts.isGetAccessorDeclaration(property) &&
            ts.isIdentifier(property.name) &&
            property.name.text === node.name.text &&
            !entered.has(property)
          ) {
            entered.add(property);
            enterCallable(property, targets);
          }
        }
      }
    }
    const next = deferred || defersEvaluation(node);
    node.forEachChild((child) => {
      // A computed member name is evaluated where the member is written, even
      // when the body or initializer beneath it waits: `class C { [KEY] = 1 }`
      // and `{ [KEY]() {} }` both read KEY as the class or object is created.
      // Applies to every named member, not just fields, so a computed method
      // name cannot hide an eager read behind its own deferred body.
      // A decorator is applied where it is written, so `@decorate(K) method() {}`
      // reads K as the class is defined even though the method body waits. Same
      // reasoning as the computed name, and the same restoration of the outer
      // state, or the deferral the member sets would hide the read.
      const eagerName =
        (child === (node as ts.NamedDeclaration).name && ts.isComputedPropertyName(child)) ||
        ts.isDecorator(child);
      visit(child, eagerName ? deferred : next, targets);
    });
  };
  source.forEachChild((child) => visit(child, false, moduleTargets));
  return found;
}

/**
 * True when this import/export contributes nothing at runtime, so it cannot
 * pull the target into the barrel's initialization.
 *
 * A bare `import "./x"` is kept: a side-effect import is the one clause-less
 * form that does evaluate the target.
 */
function isErasedEdge(statement: ts.ImportDeclaration | ts.ExportDeclaration): boolean {
  if (ts.isImportDeclaration(statement)) {
    const clause = statement.importClause;
    if (!clause) return false; // side-effect import
    if (clause.isTypeOnly) return true;
    if (clause.name) return false; // default import
    const bound = clause.namedBindings;
    if (!bound || ts.isNamespaceImport(bound)) return false;
    // `import {} from "./x"` still evaluates the target, and `every` is vacuously
    // true on an empty list, so the length check is load-bearing.
    return bound.elements.length > 0 && bound.elements.every((e) => e.isTypeOnly);
  }
  if (statement.isTypeOnly) return true;
  const clause = statement.exportClause;
  if (!clause || ts.isNamespaceExport(clause)) return false;
  return clause.elements.length > 0 && clause.elements.every((e) => e.isTypeOnly);
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

type Resolver = (specifier: string, from: string) => string | null;

function makeResolver(sources: Map<string, string>): Resolver {
  return (spec, from) => {
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
}

function readAll(files: string[]): Map<string, string> {
  const sources = new Map<string, string>();
  for (const f of files) sources.set(f, readFileSync(f, "utf8"));
  return sources;
}

/**
 * Modules that hand out the barrel's own bindings, transitively.
 *
 * A bridge that does `export { K } from "@/features/chat"` re-exports the live
 * binding rather than a copy, so a consumer importing K from the bridge is
 * exposed to exactly the same dead zone. Without this the check silently stops
 * applying the moment someone introduces a bridge, which is an ordinary
 * refactor rather than an exotic one.
 */
function barrelBearingModules(
  sources: Map<string, string>,
  resolve: Resolver,
): Set<string> {
  const bearing = new Set<string>();
  let changed = true;
  while (changed) {
    changed = false;
    for (const [file, text] of sources) {
      if (bearing.has(file)) continue;
      const source = parse(file, text);
      // `import { K } from barrel; export { K }` re-exports the same live
      // binding as `export { K } from barrel`, just spelled in two statements,
      // so the local names have to be known before the exports are read.
      const imported = barrelValueNames(source, (spec) => {
        if (spec === BARREL) return true;
        const target = resolve(spec, file);
        return target !== null && bearing.has(target);
      });
      for (const statement of source.statements) {
        if (!ts.isExportDeclaration(statement)) continue;
        if (isErasedEdge(statement)) continue;
        const specifier = statement.moduleSpecifier;
        if (!specifier) {
          const clause = statement.exportClause;
          if (!clause || !ts.isNamedExports(clause)) continue;
          // The LOCAL name is what was imported; `export { K as J }` still
          // hands out the barrel's binding under a new name.
          const reExports = clause.elements.some(
            (element) => !element.isTypeOnly && imported.has((element.propertyName ?? element.name).text),
          );
          if (!reExports) continue;
          bearing.add(file);
          changed = true;
          break;
        }
        if (!ts.isStringLiteral(specifier)) continue;
        const target = resolve(specifier.text, file);
        if (specifier.text === BARREL || (target && bearing.has(target))) {
          bearing.add(file);
          changed = true;
          break;
        }
      }
    }
  }
  return bearing;
}

/** Every module the barrel's own initialization can pull in, transitively. */
function barrelInitClosure(files: string[]): Set<string> {
  const sources = readAll(files);
  const resolve = makeResolver(sources);

  const edges = (file: string): string[] => {
    const source = parse(file, sources.get(file) ?? "");
    const out: string[] = [];
    for (const st of source.statements) {
      const spec =
        (ts.isImportDeclaration(st) || ts.isExportDeclaration(st)) && st.moduleSpecifier;
      if (!spec || !ts.isStringLiteral(spec)) continue;
      // A type-only edge is erased and cannot drag a module into evaluation.
      // `import { type A, type B }` sets no declaration-level flag but is just
      // as erased as `import type { A, B }`, so check the specifiers too.
      if (isErasedEdge(st)) continue;
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
  const sources = readAll(files);
  const resolve = makeResolver(sources);
  const bearing = barrelBearingModules(sources, resolve);
  const offenders: string[] = [];
  let importers = 0;
  for (const file of files) {
    const text = sources.get(file) ?? "";
    // No cheap text prefilter on BARREL here: a module importing through a
    // bridge never spells the barrel's name, and skipping it was the hole.
    const source = parse(file, text);
    const names = barrelValueNames(source, (specifier) => {
      if (specifier === BARREL) return true;
      const target = resolve(specifier, file);
      return target !== null && bearing.has(target);
    });
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
      "function expression invoked through call",
      `import { K } from "${BARREL}";\nconst a = (function () { return K; }).call(undefined);\n`,
    ],
    [
      "function expression invoked through apply",
      `import { K } from "${BARREL}";\nconst a = (function () { return K; }).apply(undefined, []);\n`,
    ],
    [
      "computed instance field name",
      `import { K } from "${BARREL}";\nclass C { [K] = 1; }\n`,
    ],
    [
      // Defaults are evaluated on entry, before the body runs.
      "default parameter of an eagerly called function",
      `import { K } from "${BARREL}";\nfunction read(v = K) {}\nread();\n`,
    ],
    [
      // Reachable only because its caller runs at load.
      "helper declared and called inside an eagerly called function",
      `import { K } from "${BARREL}";\nfunction outer() { function inner() { return K; } return inner(); }\nouter();\n`,
    ],
    [
      "constructor of a named class built at module scope",
      `import { K } from "${BARREL}";\nclass C { constructor() { consume(K); } }\nnew C();\n`,
    ],
    [
      "instance field of a named class built at module scope",
      `import { K } from "${BARREL}";\nclass C { f = K; }\nnew C();\n`,
    ],
    [
      // The Promise constructor runs its executor before it returns.
      "promise executor",
      `import { K } from "${BARREL}";\nnew Promise(() => consume(K));\n`,
    ],
    [
      "getter run by an eager property read",
      `import { K } from "${BARREL}";\nconst obj = { get value() { return K; } };\nconst v = obj.value;\n`,
    ],
    [
      // Constructing a function expression runs its body on the spot.
      "function expression invoked with new",
      `import { K } from "${BARREL}";\nnew (function () { consume(K); })();\n`,
    ],
    [
      // The decorator is applied as the class is defined, before any call.
      "decorator argument on a deferred method",
      `import { K } from "${BARREL}";\nclass C { @decorate(K) method() {} }\n`,
    ],
    [
      "a real base class is evaluated when the class is defined",
      `import { K } from "${BARREL}";\nclass C extends K {}\n`,
    ],
    [
      "module-scope read in a file that also shadows the name in a function",
      `import { K } from "${BARREL}";\nfunction f(K) { return K; }\nconst a = [K];\n`,
    ],
    [
      "module-scope read in a file that also shadows the name in a block",
      `import { K } from "${BARREL}";\n{ const K = 1; use(K); }\nconst a = [K];\n`,
    ],
    ["class extends", `import { K } from "${BARREL}";\nclass C extends K {}\n`],
    [
      "class extends through a namespace",
      `import * as chat from "${BARREL}";\nclass C extends chat.K {}\n`,
    ],
    [
      "class extends a call on the import",
      `import { K } from "${BARREL}";\nclass C extends makeBase(K) {}\n`,
    ],
    [
      "computed object literal method name",
      `import { K } from "${BARREL}";\nconst x = { [K]() {} };\n`,
    ],
    [
      "computed accessor name",
      `import { K } from "${BARREL}";\nclass C { get [K]() { return 1; } }\n`,
    ],
    [
      "computed method name on a class",
      `import { K } from "${BARREL}";\nclass C { [K]() {} }\n`,
    ],
    [
      "module-scope call into a local function declaration",
      `import { K } from "${BARREL}";\nfunction read() { return K; }\nconst v = read();\n`,
    ],
    [
      "module-scope call into a local arrow binding",
      `import { K } from "${BARREL}";\nconst read = () => K;\nconst v = read();\n`,
    ],
    [
      "async body before the first await",
      `import { K } from "${BARREL}";\nasync function read() { const a = K; await x; }\nread();\n`,
    ],
    [
      "await only inside a nested callback does not suspend",
      `import { K } from "${BARREL}";\nasync function read() { const g = async () => { await x; }; return K; }\nread();\n`,
    ],
    [
      "eager call two functions deep",
      `import { K } from "${BARREL}";\nfunction inner() { return K; }\nfunction outer() { return inner(); }\nconst v = outer();\n`,
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
      "a getter nobody reads",
      `import { K } from "${BARREL}";\nconst obj = { get value() { return K; } };\n`,
    ],
    [
      // An ordinary callback may be stored and invoked long after load.
      "callback handed to an unknown function",
      `import { K } from "${BARREL}";\nregister(() => consume(K));\n`,
    ],
    [
      "a named class nobody constructs",
      `import { K } from "${BARREL}";\nclass C { constructor() { consume(K); } }\n`,
    ],
    [
      // The parameter shadows the module function, so the call is not that one.
      "a call through a name a parameter rebinds",
      `import { K } from "${BARREL}";\nfunction read() { return K; }\nfunction outer(read) { return read(); }\n`,
    ],
    [
      // `var` binds to the function, not the `if` block it sits in, so the read
      // below it is the local one.
      "var hoisted out of a nested block to its function scope",
      `import { K } from "${BARREL}";\n(function () { if (c) { var K = 1; } consume(K); })();\n`,
    ],
    [
      // An ambient declaration emits no JavaScript, so its base is never read.
      "ambient class heritage",
      `import { K } from "${BARREL}";\ndeclare class C extends K {}\n`,
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
    [
      "implements clause",
      `import { K } from "${BARREL}";\nclass C implements K {}\n`,
    ],
    [
      "interface extending an imported type",
      `import { K } from "${BARREL}";\ninterface I extends K {}\n`,
    ],
    [
      "type argument on a base class",
      `import { K } from "${BARREL}";\nclass C extends Base<K> {}\n`,
    ],
    [
      "computed method name deferred body",
      `import { K } from "${BARREL}";\nclass C { m() { return K; } }\n`,
    ],
    [
      "literal static field label",
      `import { K } from "${BARREL}";\nclass C { static K = 1; }\n`,
    ],
    [
      "local function that is never called at module scope",
      `import { K } from "${BARREL}";\nfunction read() { return K; }\nexport { read };\n`,
    ],
    [
      "local function called only from inside another deferred function",
      `import { K } from "${BARREL}";\nfunction read() { return K; }\nexport const f = () => read();\n`,
    ],
    [
      "called generator only builds an iterator",
      `import { K } from "${BARREL}";\nfunction* read() { yield K; }\nread();\n`,
    ],
    [
      "async body after the first await",
      `import { K } from "${BARREL}";\nasync function read() { await x; return K; }\nread();\n`,
    ],
    [
      "self-recursive function is not followed forever",
      `import { K } from "${BARREL}";\nfunction loop() { return loop(); }\nconst v = loop();\n`,
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

test("barrel bindings are tracked through local re-export bridges", () => {
  const bridge = path.join(SRC, "fake", "bridge.ts");
  const deeper = path.join(SRC, "fake", "deeper.ts");
  const plain = path.join(SRC, "fake", "plain.ts");
  const consumer = path.join(SRC, "fake", "consumer.ts");
  const sources = new Map<string, string>([
    [bridge, `export { K } from "${BARREL}";\n`],
    [deeper, `export { K } from "./bridge";\n`],
    // Re-exports something of its own, so it hands out no barrel binding.
    [plain, `export const K = 1;\n`],
    [consumer, `import { K } from "./bridge";\nconst a = [K];\n`],
  ]);
  const resolve = makeResolver(sources);
  const bearing = barrelBearingModules(sources, resolve);

  assert.ok(bearing.has(bridge), "a direct re-export of the barrel carries its bindings");
  assert.ok(bearing.has(deeper), "bearing propagates through a second hop");
  assert.ok(!bearing.has(plain), "a module exporting its own value carries nothing");

  const names = (from: string, text: string): Set<string> =>
    barrelValueNames(parse(from, text), (specifier) => {
      if (specifier === BARREL) return true;
      const target = resolve(specifier, from);
      return target !== null && bearing.has(target);
    });

  assert.deepEqual(
    [...names(consumer, sources.get(consumer) as string)],
    ["K"],
    "importing through the bridge must be treated as importing from the barrel",
  );
  assert.deepEqual(
    [...names(consumer, `import { K } from "./plain";\nconst a = [K];\n`)],
    [],
    "importing an unrelated local value must not be tracked",
  );
});

test("an erased re-export does not make a module a bridge", () => {
  const bridge = path.join(SRC, "fake", "typebridge.ts");
  const sources = new Map<string, string>([
    [bridge, `export type { T } from "${BARREL}";\nexport { type U } from "${BARREL}";\n`],
  ]);
  const bearing = barrelBearingModules(sources, makeResolver(sources));
  assert.ok(!bearing.has(bridge), "a type-only re-export carries no runtime binding");
});

test("an import-then-export bridge carries barrel bindings too", () => {
  const bridge = path.join(SRC, "fake", "twostep.ts");
  const aliased = path.join(SRC, "fake", "aliased.ts");
  const copy = path.join(SRC, "fake", "copy.ts");
  const sources = new Map<string, string>([
    [bridge, `import { K } from "${BARREL}";\nexport { K };\n`],
    [aliased, `import { K } from "${BARREL}";\nexport { K as J };\n`],
    // A fresh binding initialized from the import, not the barrel's own.
    [copy, `import { K } from "${BARREL}";\nexport const J = K;\n`],
  ]);
  const bearing = barrelBearingModules(sources, makeResolver(sources));
  assert.ok(bearing.has(bridge), "import-then-export re-exports the live binding");
  assert.ok(bearing.has(aliased), "renaming on the way out changes nothing");
  assert.ok(!bearing.has(copy), "export const is a copy, not the barrel's binding");
});
