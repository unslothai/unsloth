// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * A value imported from the @/features/chat barrel must not be read while the
 * module is still loading.
 *
 * features/chat is in an import cycle, so a module importing from the barrel
 * can be evaluated while chat-runtime-store is still initializing. Reading one
 * of its `const` exports then hits the temporal dead zone and throws at import
 * time, taking the whole page down:
 *
 *   [ansi-smoke] pageerror: Cannot access 'CHAT_GPU_MEMORY_MODE_KEY'
 *                           before initialization
 *
 * That shipped from hooks/use-model-memory.ts. Reading inside a function is
 * safe: by call time every module has finished loading.
 *
 * Walks the AST rather than the source text. A regex version missed four
 * shapes: a second import declaration, an aliased specifier, a parenthesized
 * read, and anything that is not a const/let initializer.
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
/** Marks a bridge that re-exports the whole barrel, so it carries every name. */
const STAR = "*";

/**
 * Local names this module binds to values pulled from the chat barrel.
 *
 * A namespace import contributes its own name: the namespace object is what is
 * uninitialized, so a module-scope mention of `chat` is itself the read.
 */
function barrelValueNames(
  source: ts.SourceFile,
  // A re-export hands out the same live binding, so importing from a bridge is
  // importing from the barrel. Asked per imported name, not per module: a
  // bridge also exports values of its own, and those are ordinary imports.
  // `exported` is undefined when the whole module is taken, as by `import *`.
  carriesBarrelValues: (specifier: string, exported?: string) => boolean = (s) => s === BARREL,
): Set<string> {
  const names = new Set<string>();
  for (const statement of source.statements) {
    if (!ts.isImportDeclaration(statement)) continue;
    const specifier = statement.moduleSpecifier;
    if (!ts.isStringLiteral(specifier)) continue;
    const from = specifier.text;
    const clause = statement.importClause;
    // Erased before the code runs, so it cannot trip a dead zone.
    if (!clause || clause.isTypeOnly) continue;
    if (clause.name && carriesBarrelValues(from, "default")) {
      names.add(clause.name.text);
    }
    const bound = clause.namedBindings;
    if (!bound) continue;
    if (ts.isNamespaceImport(bound)) {
      if (carriesBarrelValues(from)) names.add(bound.name.text);
      continue;
    }
    for (const element of bound.elements) {
      if (element.isTypeOnly) continue;
      // propertyName is the name on the far side; element.name is the LOCAL
      // one, so `X as y` is looked up as X and recorded as y.
      if (!carriesBarrelValues(from, (element.propertyName ?? element.name).text)) continue;
      names.add(element.name.text);
    }
  }
  return names;
}

/**
 * A source path as the allowlist writes it: relative to src, forward slashes.
 *
 * Not path.relative alone. On Windows that returns backslashes, so every entry
 * in KNOWN_DEEP_CYCLE_READS missed and the guard failed there while passing on
 * Linux -- caught on a Windows CI runner, not by reasoning about it.
 */
function toPosix(relative: string): string {
  return relative.split("\\").join("/");
}

function relativeToSrc(file: string): string {
  return toPosix(path.relative(SRC, file));
}

/** Local names bound by `import * as X`, whichever module they came from. */
function namespaceImportNames(source: ts.SourceFile): Set<string> {
  const out = new Set<string>();
  for (const statement of source.statements) {
    if (!ts.isImportDeclaration(statement)) continue;
    const clause = statement.importClause;
    if (!clause || clause.isTypeOnly) continue;
    const bound = clause.namedBindings;
    if (bound && ts.isNamespaceImport(bound)) out.add(bound.name.text);
  }
  return out;
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
  // `var` binds to the enclosing function, not its block, so recording it only
  // on the inner block left a later read resolving to the import instead of the
  // local. `let`/`const` keep block scoping via addStatements above.
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
 * Per occurrence, not per file: suppressing a name everywhere once it was
 * shadowed anywhere silenced genuine top-level reads, and this tree has 4969
 * functions to shadow from.
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
 * True when this identifier sits in a heritage clause that survives to runtime.
 *
 * A base class is evaluated when the class is defined, but TypeScript wraps it
 * in an ExpressionWithTypeArguments, which `ts.isTypeNode` accepts, so it must
 * be excluded by hand. `implements` and `interface I extends J` stay erased.
 */
function inRuntimeHeritage(node: ts.Node): boolean {
  // The expression spine only: a type argument hangs off `typeArguments`, so it
  // stays erased, while `extends ns.K` and `extends makeBase(K)` are reached.
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
  // `declare class C extends K {}` emits no JavaScript, so its base is never
  // evaluated and flagging it would reject a file that cannot crash.
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
  // `new Promise(executor)` runs the executor before returning, so it is as
  // eager as an IIFE. Promise only: an arbitrary callback may be stored and
  // called long after load, and treating those as eager would over-report.
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

/** True when this node is something with a callable body. */
function isCallableNode(node: ts.Node): node is ts.FunctionLikeDeclaration {
  return (
    ts.isFunctionDeclaration(node) ||
    ts.isFunctionExpression(node) ||
    ts.isArrowFunction(node) ||
    ts.isMethodDeclaration(node) ||
    ts.isConstructorDeclaration(node) ||
    ts.isGetAccessorDeclaration(node) ||
    ts.isSetAccessorDeclaration(node)
  );
}

/** The call or construction that invokes this function expression, if any. */
function invocationOf(node: ts.Node): ts.CallExpression | ts.NewExpression | undefined {
  let current: ts.Node = node;
  let parent = current.parent;
  while (parent && ts.isParenthesizedExpression(parent)) {
    current = parent;
    parent = parent.parent;
  }
  if (!parent) return undefined;
  if ((ts.isCallExpression(parent) || ts.isNewExpression(parent)) && parent.expression === current) {
    return parent;
  }
  // `.call(...)` / `.apply(...)`: the arguments are the callee's, shifted by the
  // receiver, so they cannot be matched positionally. Report no arguments,
  // which makes defaults conservatively eligible.
  return undefined;
}

/** Array and Object helpers that run their callback before returning. */
const SYNCHRONOUS_CALLBACK_METHODS = new Set([
  "map", "forEach", "filter", "reduce", "reduceRight", "flatMap",
  "find", "findLast", "findIndex", "findLastIndex", "some", "every", "sort",
]);

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
    // Always deferred here. An IIFE does run now, but it is entered through
    // enterCallable so that a generator or an await inside it is respected;
    // walking the body eagerly from here ignored both.
    return true;
  }
  // An instance field initializer runs at construction; a static one runs at
  // class definition, which is module load, so it is NOT deferred.
  if (ts.isPropertyDeclaration(node)) {
    const isStatic = ts
      .getModifiers(node)
      ?.some((m) => m.kind === ts.SyntaxKind.StaticKeyword);
    return !isStatic;
  }
  return false;
}

/** True when this namespace identifier is being read through, as `chat.K`. */
function isNamespaceMemberRead(node: ts.Identifier): boolean {
  const parent = node.parent;
  if (!parent) return false;
  if (ts.isPropertyAccessExpression(parent) && parent.expression === node) return true;
  if (ts.isElementAccessExpression(parent) && parent.expression === node) return true;
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
  // `class C { static K = 1 }` -- a label, not a read. Computed names fall
  // through on purpose.
  if (
    (ts.isPropertyDeclaration(parent) ||
      ts.isMethodDeclaration(parent) ||
      ts.isGetAccessorDeclaration(parent) ||
      ts.isSetAccessorDeclaration(parent)) &&
    parent.name === node
  ) {
    return true;
  }
  // Erased before the code runs. Every ancestor, not just the parent: in
  // `type T = chat.Entry` the parent is a QualifiedName.
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
 * Per scope rather than top level only: a helper declared inside a function
 * that is itself called during initialization is reached by that outer call, so
 * `inner` must be visible while outer's body is walked.
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
    // `const` only. A reassignable binding's declaration initializer is not
    // what the call reaches: `let read = () => 1; read = () => K; read()` reads
    // K, and swapping the two bodies rejects code that does not. Tracking
    // assignments would mean flow analysis, so a mutable target is simply not
    // resolved -- that loses a hazard rather than inventing one.
    if (!(statement.declarationList.flags & ts.NodeFlags.Const)) continue;
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
 * "Shadowed anywhere above" cannot answer this once helpers resolve per scope:
 * a nested helper's own declaration is a binding above the call and reads as a
 * shadow of itself. Walk out and ask which scope comes first.
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

/**
 * What a constructor hands to its base, or undefined when it writes no `super`.
 *
 * Undefined is not "no arguments": it means the walk could not tell, and every
 * base default stays eligible. A `super` inside a nested function belongs to
 * that function, so it is not this constructor's call.
 */
function superArgumentsOf(
  ctor: ts.ConstructorDeclaration,
): readonly ts.Expression[] | undefined {
  let found: readonly ts.Expression[] | undefined;
  const visit = (node: ts.Node): void => {
    if (found) return;
    if (isVarScope(node)) return;
    if (ts.isCallExpression(node) && node.expression.kind === ts.SyntaxKind.SuperKeyword) {
      found = node.arguments;
      return;
    }
    node.forEachChild(visit);
  };
  ctor.body?.forEachChild(visit);
  return found;
}

/** True when this node carries the given modifier. */
function hasModifier(node: ts.Node, kind: ts.SyntaxKind): boolean {
  const modifiers = ts.canHaveModifiers(node) ? ts.getModifiers(node) : undefined;
  return Boolean(modifiers?.some((m) => m.kind === kind));
}

/** Constructs whose body may be skipped, so an await inside is not certain. */
function isConditionalConstruct(node: ts.Node): boolean {
  return (
    ts.isIfStatement(node) ||
    ts.isSwitchStatement(node) ||
    ts.isTryStatement(node) ||
    ts.isConditionalExpression(node) ||
    // `flag && await x` and `a ?? await x` skip their right side, so an await
    // there is not certain to run either.
    (ts.isBinaryExpression(node) &&
      (node.operatorToken.kind === ts.SyntaxKind.AmpersandAmpersandToken ||
        node.operatorToken.kind === ts.SyntaxKind.BarBarToken ||
        node.operatorToken.kind === ts.SyntaxKind.QuestionQuestionToken)) ||
    ts.isForStatement(node) ||
    ts.isForInStatement(node) ||
    ts.isForOfStatement(node) ||
    ts.isWhileStatement(node) ||
    ts.isDoStatement(node)
  );
}

/**
 * Where this body first suspends for certain, or null if it never does.
 *
 * Only an await that always runs counts. `if (skip) await x; return K;` reads K
 * synchronously whenever the branch is not taken, so treating the lexically
 * first await as the boundary would skip a genuine eager read. Awaits inside
 * conditionals and loops are therefore not boundaries, which keeps the rest of
 * the body eager and errs toward reporting. Nested functions are skipped too:
 * an await in a callback declared here does not suspend its declarer.
 */
function firstSuspensionPos(body: ts.Node): number | null {
  let earliest: number | null = null;
  const visit = (node: ts.Node): void => {
    if (node !== body && isVarScope(node) && !ts.isSourceFile(node)) return;
    if (node !== body && isConditionalConstruct(node)) {
      // `for await (...)` suspends on entry, before the body is skippable.
      if (ts.isForOfStatement(node) && node.awaitModifier !== undefined) {
        if (earliest === null || node.getStart() < earliest) earliest = node.getStart();
      }
      return;
    }
    if (ts.isAwaitExpression(node) && (earliest === null || node.getStart() < earliest)) {
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

/** A function another module exports, together with that module's barrel names. */
interface ImportedHelper {
  file: string;
  source: ts.SourceFile;
  fn: ts.FunctionLikeDeclaration;
  names: Set<string>;
}

interface ScanOptions {
  /** Start at this callable instead of the module body. */
  entry?: ts.Node;
  /** Arguments of the call that reached `entry`, for its parameter defaults. */
  entryArgs?: readonly ts.Expression[];
  /** Local name -> a helper another module exports, for cross-file calls. */
  helpers?: Map<string, ImportedHelper>;
}

function eagerReads(
  source: ts.SourceFile,
  names: Set<string>,
  options: ScanOptions = {},
  // `import * as chat` binds the namespace OBJECT, which exists from
  // instantiation. Passing it around is safe; only `chat.K` reads an export
  // that may still be uninitialized.
  namespaces: Set<string> = new Set(),
): string[] {
  if (names.size === 0 && !options.helpers?.size) return [];

  const moduleTargets = collectTargets(source.statements);
  // Guards against recursion, and stops a function called twice from being
  // reported twice.
  const entered = new Set<ts.Node>();
  const helpers = options.helpers ?? new Map<string, ImportedHelper>();

  const found: string[] = [];

  /**
   * Walk a callable an eager call reached, holding back what does not run yet.
   *
   * Defaults are evaluated on entry, before the body. A generator call only
   * builds an iterator, and an async function resumes past initialization, so
   * the body is walked only as far as its first suspension.
   */
  const enterCallable = (
    target: ts.Node,
    targets: Targets,
    args?: readonly ts.Expression[],
    defaultsOnly = false,
  ): void => {
    const fn = target as ts.FunctionLikeDeclaration;
    (fn.parameters ?? []).forEach((parameter, index) => {
      if (!parameter.initializer) return;
      // A default only runs when the argument is missing or literally
      // `undefined`. `read(1)` never evaluates `value = K`, and reporting it
      // rejects code that cannot crash. A rest parameter has no single
      // argument to match, so leave those alone.
      if (args && !parameter.dotDotDotToken) {
        const supplied = args[index];
        const omitted =
          supplied === undefined ||
          (ts.isIdentifier(supplied) && supplied.text === "undefined") ||
          supplied.kind === ts.SyntaxKind.SpreadElement;
        if (!omitted) return;
      }
      visit(parameter.initializer, false, targets);
    });
    if (defaultsOnly || !fn.body || fn.asteriskToken) return;
    const inner = ts.isBlock(fn.body) ? collectTargets(fn.body.statements, targets) : targets;
    const suspendsAt = firstSuspensionPos(fn.body);
    if (suspendsAt === null) visit(fn.body, false, inner);
    else visitUntil(fn.body, suspendsAt, (n, d) => visit(n, d, inner));
  };

  /**
   * Run the getters an eager property read would invoke.
   *
   * `wanted` null means every getter on the object: a spread and a rest element
   * both read every own property, so there is no single name to match.
   */
  const enterGetters = (
    object: ts.ObjectLiteralExpression,
    targets: Targets,
    wanted: Set<string> | null,
  ): void => {
    for (const property of object.properties) {
      if (!ts.isGetAccessorDeclaration(property)) continue;
      if (!ts.isIdentifier(property.name)) continue;
      if (wanted && !wanted.has(property.name.text)) continue;
      if (entered.has(property)) continue;
      entered.add(property);
      enterCallable(property, targets);
    }
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
      !isShadowed(node) &&
      !(namespaces.has(node.text) && !isNamespaceMemberRead(node))
    ) {
      const { line } = source.getLineAndCharacterOfPosition(node.getStart(source));
      found.push(`${node.text} (line ${line + 1})`);
    }
    // An IIFE runs now, but through enterCallable so a generator body or an
    // await inside it still holds back what does not run yet. Walking it
    // eagerly reported reads that only happen after the module has loaded.
    if (!deferred && isCallableNode(node) && isImmediatelyInvoked(node)) {
      if (!entered.has(node)) {
        entered.add(node);
        enterCallable(node, targets, invocationOf(node)?.arguments);
      }
      return;
    }
    // `` tag`value` `` invokes tag synchronously, exactly like tag("value").
    if (!deferred && ts.isTaggedTemplateExpression(node) && ts.isIdentifier(node.tag)) {
      const tagged = targets.functions.get(node.tag.text);
      if (tagged && !entered.has(tagged) && resolvesToTarget(node.tag, tagged)) {
        entered.add(tagged);
        enterCallable(tagged, targets);
      }
    }
    // `f.call(...)` / `f.apply(...)` on a local function runs it now.
    if (
      !deferred &&
      ts.isCallExpression(node) &&
      ts.isPropertyAccessExpression(node.expression) &&
      (node.expression.name.text === "call" || node.expression.name.text === "apply") &&
      ts.isIdentifier(node.expression.expression)
    ) {
      const named = targets.functions.get(node.expression.expression.text);
      if (named && !entered.has(named) && resolvesToTarget(node.expression.expression, named)) {
        entered.add(named);
        enterCallable(named, targets);
      }
    }
    // `[0].map(cb)` runs cb before it returns. Only these built-ins: an
    // arbitrary callback may be stored and invoked long after load, which is
    // the same line the Promise executor case draws.
    if (
      !deferred &&
      ts.isCallExpression(node) &&
      ts.isPropertyAccessExpression(node.expression) &&
      SYNCHRONOUS_CALLBACK_METHODS.has(node.expression.name.text)
    ) {
      for (const argument of node.arguments) {
        // A named callback resolves through the target map, as a call to it
        // would; an inline one is the callable itself.
        let callee: ts.Node | undefined;
        if (isCallableNode(argument)) callee = argument;
        else if (ts.isIdentifier(argument)) {
          const named = targets.functions.get(argument.text);
          if (named && resolvesToTarget(argument, named)) callee = named;
        }
        if (!callee || entered.has(callee)) continue;
        entered.add(callee);
        enterCallable(callee, targets);
      }
    }
    // `C.read()` runs a static method now, the same as any other eager call.
    if (
      !deferred &&
      ts.isCallExpression(node) &&
      ts.isPropertyAccessExpression(node.expression) &&
      ts.isIdentifier(node.expression.expression)
    ) {
      const cls = targets.classes.get(node.expression.expression.text);
      if (cls && resolvesToTarget(node.expression.expression, cls)) {
        const wanted = node.expression.name.text;
        for (const member of cls.members) {
          if (!ts.isMethodDeclaration(member) || !member.name) continue;
          if (!ts.isIdentifier(member.name) || member.name.text !== wanted) continue;
          if (!hasModifier(member, ts.SyntaxKind.StaticKeyword)) continue;
          if (entered.has(member)) continue;
          entered.add(member);
          enterCallable(member, targets, node.arguments);
        }
      }
    }
    // `obj.read()` on a local object literal runs that method now. The getter
    // case below covers a bare `obj.value`; this is the invoked-method twin.
    if (
      !deferred &&
      ts.isCallExpression(node) &&
      ts.isPropertyAccessExpression(node.expression) &&
      ts.isIdentifier(node.expression.expression)
    ) {
      const object = targets.objects.get(node.expression.expression.text);
      if (object && resolvesToTarget(node.expression.expression, object)) {
        const wanted = node.expression.name.text;
        for (const property of object.properties) {
          const name = property.name;
          if (!name || !ts.isIdentifier(name) || name.text !== wanted) continue;
          const fn = ts.isMethodDeclaration(property)
            ? property
            : ts.isPropertyAssignment(property) && isCallableNode(property.initializer)
              ? property.initializer
              : undefined;
          if (fn && !entered.has(fn)) {
            entered.add(fn);
            enterCallable(fn, targets, node.arguments);
          }
        }
      }
    }
    // `const value = read()` at module scope runs read's body now, so the read
    // inside it is eager even though the declaration looked deferred. Without
    // this the guard's own advice -- move the read into a function -- could be
    // followed to the letter and still leave the crash in place.
    if (!deferred && ts.isCallExpression(node) && ts.isIdentifier(node.expression)) {
      const target = targets.functions.get(node.expression.text);
      if (target && resolvesToTarget(node.expression, target)) {
        // Defaults depend on THIS call's arguments, so they are re-checked even
        // when the body has already been walked: `read(1); read();` evaluates
        // the default only on the second call.
        enterCallable(target, targets, node.arguments, entered.has(target));
        entered.add(target);
      }
      // Extracting the helper into its own module is the same move as
      // extracting it into a function, and it hid the read just as well: the
      // helper only ever sees a deferred read, and the caller has no barrel
      // name of its own to match. Reported at the call site, since that is the
      // line that has to change.
      const helper = !target ? helpers.get(node.expression.text) : undefined;
      if (helper && !entered.has(helper.fn) && !isShadowed(node.expression)) {
        entered.add(helper.fn);
        const inner = eagerReads(helper.source, helper.names, {
          entry: helper.fn,
          entryArgs: node.arguments,
        });
        if (inner.length > 0) {
          const { line } = source.getLineAndCharacterOfPosition(node.getStart(source));
          const where = relativeToSrc(helper.file);
          found.push(`${node.expression.text}() (line ${line + 1}) reaches ${where}: ${inner.join(", ")}`);
        }
      }
    }
    // `new Promise(read)` calls read before it returns, exactly as an inline
    // executor does. The inline form goes through isImmediatelyInvoked; a named
    // one has to be resolved here.
    if (
      !deferred &&
      ts.isNewExpression(node) &&
      ts.isIdentifier(node.expression) &&
      node.expression.text === "Promise"
    ) {
      const executor = node.arguments?.[0];
      if (executor && ts.isIdentifier(executor)) {
        const named = targets.functions.get(executor.text);
        if (named && !entered.has(named) && resolvesToTarget(executor, named)) {
          entered.add(named);
          enterCallable(named, targets);
        }
      }
    }
    // `new C()` runs the constructor and every instance field initializer now,
    // for a named class exactly as for an inline function expression.
    if (!deferred && ts.isNewExpression(node) && ts.isIdentifier(node.expression)) {
      // A derived class runs its base's constructor and instance fields first,
      // even when it declares no constructor of its own, so walk the chain.
      // Each level carries ITS OWN arguments: the base sees what `super(...)`
      // passes, not what `new` did, so handing the outer arguments down the
      // chain both misses defaults and rejects code that supplies them.
      const chain: Array<{
        cls: ts.ClassLikeDeclaration;
        args: readonly ts.Expression[] | undefined;
      }> = [];
      let current = targets.classes.get(node.expression.text);
      let anchor: ts.Identifier | undefined = node.expression;
      let args: readonly ts.Expression[] | undefined = node.arguments ?? [];
      while (current && anchor && resolvesToTarget(anchor, current)) {
        const cls = current;
        if (chain.some((level) => level.cls === cls)) break;
        chain.unshift({ cls, args });
        const ctor = cls.members.find(
          (m): m is ts.ConstructorDeclaration =>
            ts.isConstructorDeclaration(m) && Boolean(m.body),
        );
        // No constructor of its own means an implicit `super(...args)`, which
        // forwards everything it was given.
        args = ctor === undefined ? args : superArgumentsOf(ctor);
        anchor = undefined;
        for (const clause of cls.heritageClauses ?? []) {
          if (clause.token !== ts.SyntaxKind.ExtendsKeyword) continue;
          const base = clause.types[0]?.expression;
          if (base && ts.isIdentifier(base)) anchor = base;
        }
        current = anchor ? targets.classes.get(anchor.text) : undefined;
      }
      for (const { cls, args: own } of chain) {
        if (entered.has(cls)) continue;
        entered.add(cls);
        for (const member of cls.members) {
          if (ts.isConstructorDeclaration(member) && member.body) {
            enterCallable(member, targets, own);
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
        enterGetters(object, targets, new Set([node.name.text]));
      }
      // `C.value` runs a static getter for the same reason.
      const cls = targets.classes.get(node.expression.text);
      if (cls && resolvesToTarget(node.expression, cls)) {
        for (const member of cls.members) {
          if (!ts.isGetAccessorDeclaration(member) || !ts.isIdentifier(member.name)) continue;
          if (member.name.text !== node.name.text) continue;
          if (!hasModifier(member, ts.SyntaxKind.StaticKeyword)) continue;
          if (entered.has(member)) continue;
          entered.add(member);
          enterCallable(member, targets);
        }
      }
    }
    // `const { value } = obj` performs the same property read `obj.value` does,
    // so it runs the getter too. A rest element takes every own property, which
    // is why it asks for all of them.
    if (
      !deferred &&
      ts.isVariableDeclaration(node) &&
      node.initializer &&
      ts.isIdentifier(node.initializer) &&
      ts.isObjectBindingPattern(node.name)
    ) {
      const object = targets.objects.get(node.initializer.text);
      if (object && resolvesToTarget(node.initializer, object)) {
        let wanted: Set<string> | null = new Set<string>();
        for (const element of node.name.elements) {
          if (element.dotDotDotToken) {
            wanted = null;
            break;
          }
          const key = element.propertyName ?? element.name;
          if (ts.isIdentifier(key)) wanted.add(key.text);
          else if (ts.isStringLiteral(key)) wanted.add(key.text);
          else {
            // A computed key names a property this walk cannot predict.
            wanted = null;
            break;
          }
        }
        enterGetters(object, targets, wanted);
      }
    }
    // `{ ...obj }` copies every own enumerable property, running each getter.
    if (!deferred && ts.isSpreadAssignment(node) && ts.isIdentifier(node.expression)) {
      const object = targets.objects.get(node.expression.text);
      if (object && resolvesToTarget(node.expression, object)) {
        enterGetters(object, targets, null);
      }
    }
    // `obj.value = 1` and `C.value = 1` run a setter synchronously, the write
    // twin of the getter case above.
    if (
      !deferred &&
      ts.isBinaryExpression(node) &&
      node.operatorToken.kind === ts.SyntaxKind.EqualsToken &&
      ts.isPropertyAccessExpression(node.left) &&
      ts.isIdentifier(node.left.expression)
    ) {
      const wanted = node.left.name.text;
      const object = targets.objects.get(node.left.expression.text);
      if (object && resolvesToTarget(node.left.expression, object)) {
        for (const property of object.properties) {
          if (!ts.isSetAccessorDeclaration(property)) continue;
          if (!ts.isIdentifier(property.name) || property.name.text !== wanted) continue;
          if (entered.has(property)) continue;
          entered.add(property);
          enterCallable(property, targets, [node.right]);
        }
      }
      const cls = targets.classes.get(node.left.expression.text);
      if (cls && resolvesToTarget(node.left.expression, cls)) {
        for (const member of cls.members) {
          if (!ts.isSetAccessorDeclaration(member) || !ts.isIdentifier(member.name)) continue;
          if (member.name.text !== wanted) continue;
          if (!hasModifier(member, ts.SyntaxKind.StaticKeyword)) continue;
          if (entered.has(member)) continue;
          entered.add(member);
          enterCallable(member, targets, [node.right]);
        }
      }
    }
    // `@deco class C {}` calls deco as the class is defined. The factory form
    // `@deco()` is an ordinary call and already goes through the path above.
    if (!deferred && ts.isDecorator(node) && ts.isIdentifier(node.expression)) {
      const named = targets.functions.get(node.expression.text);
      if (named && !entered.has(named) && resolvesToTarget(node.expression, named)) {
        entered.add(named);
        enterCallable(named, targets);
      }
    }
    const next = deferred || defersEvaluation(node);
    // A block, a switch case and a module body each declare callables of their
    // own. Without layering them here a helper declared inside `if (ready) {
    // ... }` never resolves, so an eager call to it reads as deferred.
    const scoped =
      ts.isBlock(node) || ts.isModuleBlock(node)
        ? collectTargets(node.statements, targets)
        : ts.isCaseClause(node) || ts.isDefaultClause(node)
          ? collectTargets(node.statements, targets)
          : targets;
    node.forEachChild((child) => {
      // A computed name and a decorator are both evaluated where they are
      // written, as the class or object is created, even though the body or
      // initializer beneath them waits. Both carry the INCOMING state, so one
      // inside a deferred function stays deferred.
      const eagerName =
        (child === (node as ts.NamedDeclaration).name && ts.isComputedPropertyName(child)) ||
        ts.isDecorator(child);
      visit(child, eagerName ? deferred : next, scoped);
    });
  };
  if (options.entry) enterCallable(options.entry, moduleTargets, options.entryArgs);
  else source.forEachChild((child) => visit(child, false, moduleTargets));
  return found;
}

/**
 * True when this import/export contributes nothing at runtime, so it cannot
 * pull the target into the barrel's initialization. A bare `import "./x"` is
 * kept: a side-effect import does evaluate the target.
 */
function isErasedEdge(statement: ts.ImportDeclaration | ts.ExportDeclaration): boolean {
  if (ts.isImportDeclaration(statement)) {
    const clause = statement.importClause;
    if (!clause) return false; // side-effect import
    if (clause.isTypeOnly) return true;
    if (clause.name) return false; // default import
    const bound = clause.namedBindings;
    if (!bound || ts.isNamespaceImport(bound)) return false;
    // `import {}` still evaluates the target and `every` is vacuously true on
    // an empty list, so the length check is load-bearing.
    return bound.elements.length > 0 && bound.elements.every((e) => e.isTypeOnly);
  }
  if (statement.isTypeOnly) return true;
  const clause = statement.exportClause;
  if (!clause || ts.isNamespaceExport(clause)) return false;
  return clause.elements.length > 0 && clause.elements.every((e) => e.isTypeOnly);
}

// Keyed on the text as well as the name: the unit cases below reparse "t.ts"
// with different sources, so caching on the name alone would return the wrong
// tree. Without this the bearing fixpoint reparses every file on each pass and
// the scan reparses each imported module once per importer.
const parseCache = new Map<string, ts.SourceFile>();

function parse(fileName: string, text: string): ts.SourceFile {
  const key = `${fileName} ${text}`;
  const cached = parseCache.get(key);
  if (cached) return cached;
  const source = ts.createSourceFile(fileName, text, ts.ScriptTarget.ESNext, true);
  parseCache.set(key, source);
  return source;
}

function analyse(fileName: string, text: string): string[] {
  const source = parse(fileName, text);
  const names = barrelValueNames(source);
  return names.size === 0 ? [] : eagerReads(source, names, {}, namespaceImportNames(source));
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
    // Exact path first: 66 imports here spell the extension, and appending a
    // second one silently dropped those edges.
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
 * A bridge re-exports the live binding rather than a copy, so its consumers sit
 * in the same dead zone. Without this the check silently stops applying the
 * moment someone adds a bridge, which is an ordinary refactor.
 */
function barrelBearingModules(
  sources: Map<string, string>,
  resolve: Resolver,
): Map<string, Set<string>> {
  // file -> the names IT exports that are the barrel's own bindings. Per name,
  // not per module: a bridge usually also exports things of its own, and
  // `export { K } from barrel; export const SAFE = 1` must not make an
  // importer of SAFE look like a barrel consumer.
  const bearing = new Map<string, Set<string>>();
  const record = (file: string, name: string): boolean => {
    let names = bearing.get(file);
    if (!names) bearing.set(file, (names = new Set()));
    if (names.has(name)) return false;
    names.add(name);
    return true;
  };
  const carries = (file: string) => (spec: string, exported?: string): boolean => {
    if (spec === BARREL) return true;
    const target = resolve(spec, file);
    if (target === null) return false;
    const names = bearing.get(target);
    if (!names) return false;
    if (names.has(STAR)) return true;
    return exported === undefined ? names.size > 0 : names.has(exported);
  };

  let changed = true;
  while (changed) {
    changed = false;
    for (const [file, text] of sources) {
      const source = parse(file, text);
      // `import { K } from barrel; export { K }` re-exports the same live
      // binding as `export { K } from barrel`, just spelled in two statements,
      // so the local names have to be known before the exports are read.
      const imported = barrelValueNames(source, carries(file));
      for (const statement of source.statements) {
        if (!ts.isExportDeclaration(statement)) continue;
        if (isErasedEdge(statement)) continue;
        const specifier = statement.moduleSpecifier;
        const clause = statement.exportClause;
        if (!specifier) {
          if (!clause || !ts.isNamedExports(clause)) continue;
          for (const element of clause.elements) {
            // The LOCAL name is what was imported; the EXPORTED name is what a
            // consumer sees, so `export { K as J }` publishes J.
            if (element.isTypeOnly) continue;
            if (!imported.has((element.propertyName ?? element.name).text)) continue;
            if (record(file, element.name.text)) changed = true;
          }
          continue;
        }
        if (!ts.isStringLiteral(specifier)) continue;
        const target = resolve(specifier.text, file);
        const fromBarrel = specifier.text === BARREL;
        const upstream = target ? bearing.get(target) : undefined;
        if (!fromBarrel && !upstream) continue;
        if (!clause) {
          // `export * from x` republishes whatever x carries. From the barrel
          // that is every one of its exports, so record a wildcard rather than
          // enumerating them; the predicate treats it as matching any name.
          if (fromBarrel) {
            if (record(file, STAR)) changed = true;
          } else {
            for (const name of upstream ?? []) if (record(file, name)) changed = true;
          }
          continue;
        }
        if (ts.isNamespaceExport(clause)) {
          // `export * as ns from x` -- the namespace object itself carries them.
          if (record(file, clause.name.text)) changed = true;
          continue;
        }
        for (const element of clause.elements) {
          if (element.isTypeOnly) continue;
          const source_name = (element.propertyName ?? element.name).text;
          if (!fromBarrel && !upstream?.has(source_name)) continue;
          if (record(file, element.name.text)) changed = true;
        }
      }
    }
  }
  return bearing;
}

/**
 * Exported names that can sit in a temporal dead zone.
 *
 * `function` declarations are hoisted and initialized before any module body
 * runs, so importing one from a half-evaluated module is always safe. Only
 * `const`, `let` and `class` bindings can be read before initialization, and
 * they are the only ones worth reporting on a deep import into the cycle.
 */
function tdzProneExportNames(source: ts.SourceFile): Set<string> {
  const prone = new Set<string>();
  const lexical = new Set<string>();
  for (const statement of source.statements) {
    const exported = hasModifier(statement, ts.SyntaxKind.ExportKeyword);
    if (ts.isVariableStatement(statement)) {
      const isLexical = Boolean(
        statement.declarationList.flags & (ts.NodeFlags.Const | ts.NodeFlags.Let),
      );
      if (!isLexical) continue;
      const names = new Set<string>();
      for (const d of statement.declarationList.declarations) collectBindingNames(d.name, names);
      for (const n of names) {
        lexical.add(n);
        if (exported) prone.add(n);
      }
    } else if (ts.isClassDeclaration(statement)) {
      if (statement.name) lexical.add(statement.name.text);
      if (!exported) continue;
      // `export default class K {}` publishes "default"; K is a local name a
      // consumer cannot import, so recording it there is what let a default
      // import of a half-initialized class through.
      if (hasModifier(statement, ts.SyntaxKind.DefaultKeyword)) prone.add("default");
      else if (statement.name) prone.add(statement.name.text);
    } else if (ts.isExportAssignment(statement) && !statement.isExportEquals) {
      // `export default <expression>` binds where the statement runs, so it is
      // in the dead zone until then. A default FUNCTION declaration is not an
      // ExportAssignment and stays hoisted, as above.
      prone.add("default");
    }
  }
  // `export { X }` publishes a lexical binding declared above.
  for (const statement of source.statements) {
    if (!ts.isExportDeclaration(statement) || statement.moduleSpecifier) continue;
    const clause = statement.exportClause;
    if (!clause || !ts.isNamedExports(clause)) continue;
    for (const element of clause.elements) {
      if (element.isTypeOnly) continue;
      if (lexical.has((element.propertyName ?? element.name).text)) prone.add(element.name.text);
    }
  }
  return prone;
}

/** What a module publishes that can still be uninitialized when a consumer reads it. */
interface ProneExports {
  names: Set<string>;
  /** True when the module may publish names this walk cannot enumerate. */
  star: boolean;
}

/**
 * Per module, the export names a cyclic consumer can catch in the dead zone.
 *
 * A re-export hands out the ORIGINAL binding, so what decides the answer is how
 * the declaring module writes it: a hoisted `function` is initialized before any
 * module body runs and can never be uninitialized, however many hops it travels.
 * Folded to a fixed point rather than recursively, because the barrel's own
 * re-export graph has cycles and a recursion would have to truncate them --
 * quietly dropping hazards on whichever module it happened to enter first.
 */
function proneExportSets(
  files: string[],
  sources: Map<string, string>,
  resolve: Resolver,
): Map<string, ProneExports> {
  interface Hop {
    target: string | null;
    clause: ts.NamedExports | undefined;
  }
  const out = new Map<string, ProneExports>();
  const hops = new Map<string, Hop[]>();
  for (const file of files) {
    const source = parse(file, sources.get(file) ?? "");
    const list: Hop[] = [];
    let edges = 0;
    for (const statement of source.statements) {
      const runtimeEdge =
        (ts.isImportDeclaration(statement) || ts.isExportDeclaration(statement)) &&
        statement.moduleSpecifier !== undefined &&
        !isErasedEdge(statement);
      if (runtimeEdge) edges += 1;
      if (!ts.isExportDeclaration(statement) || !runtimeEdge) continue;
      const specifier = statement.moduleSpecifier;
      if (!specifier || !ts.isStringLiteral(specifier)) continue;
      const clause = statement.exportClause;
      // `export * as ns from "./x"` binds a namespace OBJECT, which exists from
      // instantiation, so it is never in the dead zone.
      if (clause && !ts.isNamedExports(clause)) continue;
      list.push({ target: resolve(specifier.text, file), clause });
    }
    // A module with no runtime imports has nothing that can re-enter it, so its
    // body always finishes before any importer's does and none of its bindings
    // can be caught uninitialized. That is why prompt-queue-events.ts exists,
    // and it holds however many re-export hops away the reader sits.
    out.set(file, {
      names: edges === 0 ? new Set<string>() : tdzProneExportNames(source),
      star: false,
    });
    hops.set(file, list);
  }
  for (let changed = true; changed; ) {
    changed = false;
    for (const file of files) {
      const own = out.get(file) as ProneExports;
      for (const hop of hops.get(file) ?? []) {
        // An unresolved hop -- a package, or a path this resolver does not
        // model -- could publish anything, so it carries every name rather than
        // silently dropping the hazard.
        const inner: ProneExports | undefined =
          hop.target === null ? { names: new Set<string>(), star: true } : out.get(hop.target);
        if (!inner) continue;
        if (!hop.clause) {
          if (inner.star && !own.star) {
            own.star = true;
            changed = true;
          }
          for (const name of inner.names) {
            if (own.names.has(name)) continue;
            own.names.add(name);
            changed = true;
          }
          continue;
        }
        for (const element of hop.clause.elements) {
          if (element.isTypeOnly) continue;
          const from = (element.propertyName ?? element.name).text;
          if (!inner.star && !inner.names.has(from)) continue;
          if (own.names.has(element.name.text)) continue;
          own.names.add(element.name.text);
          changed = true;
        }
      }
    }
  }
  return out;
}

/** Functions a module exports by name, for resolving a cross-module call. */
function exportedFunctions(source: ts.SourceFile): Map<string, ts.FunctionLikeDeclaration> {
  const out = new Map<string, ts.FunctionLikeDeclaration>();
  for (const statement of source.statements) {
    if (ts.isExportAssignment(statement) && !statement.isExportEquals) {
      const expression = statement.expression;
      if (ts.isArrowFunction(expression) || ts.isFunctionExpression(expression)) {
        out.set("default", expression);
      }
      continue;
    }
    if (!hasModifier(statement, ts.SyntaxKind.ExportKeyword)) continue;
    if (
      ts.isFunctionDeclaration(statement) &&
      statement.body &&
      hasModifier(statement, ts.SyntaxKind.DefaultKeyword)
    ) {
      out.set("default", statement);
      continue;
    }
    if (ts.isFunctionDeclaration(statement) && statement.name && statement.body) {
      out.set(statement.name.text, statement);
      continue;
    }
    if (!ts.isVariableStatement(statement)) continue;
    // `const` only, for the same reason collectTargets takes only const.
    if (!(statement.declarationList.flags & ts.NodeFlags.Const)) continue;
    for (const declaration of statement.declarationList.declarations) {
      const initializer = declaration.initializer;
      if (!initializer || !ts.isIdentifier(declaration.name)) continue;
      if (ts.isArrowFunction(initializer) || ts.isFunctionExpression(initializer)) {
        out.set(declaration.name.text, initializer);
      }
    }
  }
  // `function read() {} export { read }` publishes a local callable without any
  // export modifier on the declaration, so the loop above never sees it and the
  // caller got no helper to follow. `export { read as default }` lands here too.
  const local = new Map<string, ts.FunctionLikeDeclaration>();
  for (const statement of source.statements) {
    if (ts.isFunctionDeclaration(statement) && statement.name && statement.body) {
      local.set(statement.name.text, statement);
      continue;
    }
    if (!ts.isVariableStatement(statement)) continue;
    if (!(statement.declarationList.flags & ts.NodeFlags.Const)) continue;
    for (const declaration of statement.declarationList.declarations) {
      const initializer = declaration.initializer;
      if (!initializer || !ts.isIdentifier(declaration.name)) continue;
      if (ts.isArrowFunction(initializer) || ts.isFunctionExpression(initializer)) {
        local.set(declaration.name.text, initializer);
      }
    }
  }
  for (const statement of source.statements) {
    if (!ts.isExportDeclaration(statement) || statement.moduleSpecifier) continue;
    const clause = statement.exportClause;
    if (!clause || !ts.isNamedExports(clause)) continue;
    for (const element of clause.elements) {
      if (element.isTypeOnly) continue;
      const fn = local.get((element.propertyName ?? element.name).text);
      if (fn) out.set(element.name.text, fn);
    }
  }
  return out;
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
      // An erased edge cannot drag a module into evaluation. Specifiers are
      // checked too: `import { type A }` sets no declaration-level flag.
      if (isErasedEdge(st)) continue;
      const target = resolve(spec.text, file);
      if (target) out.push(target);
    }
    return out;
  };

  // Resolved, not hard-coded. Moving the barrel to an equally importable
  // features/chat.ts would leave a hard-coded index.ts path with no edges, so
  // atRisk would come back empty and every offender would silently stop being
  // reported while the importer count still looked healthy.
  const entry = resolve(BARREL, path.join(SRC, "index.ts"));
  assert.ok(
    entry !== null,
    `${BARREL} does not resolve to a file under ${SRC}; the guard would scan nothing`,
  );

  const seen = new Set<string>();
  const stack = [entry as string];
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

/**
 * Module-scope reads that already existed when deep imports into the cycle
 * started being checked.
 *
 * Every one is the same hazard as the crash this file exists for: a lexical
 * binding read at module scope from a module the barrel's own initialization
 * can reach. They are allowed only because clearing them is a refactor of
 * unrelated features, not because they are safe.
 *
 * This list may SHRINK, never grow. A new entry means a new latent blank-page
 * bug, and the assertion below rejects it.
 *
 * Keyed by file and name with a COUNT, deliberately not by line: an unrelated
 * edit higher up the file moves every read below it, and keying on the line
 * turned that into a red guard on a branch that changed nothing. The count is
 * what the exception is really about -- how many of these reads a file still
 * has -- and it holds still while the lines move.
 */
const KNOWN_DEEP_CYCLE_READS = new Map<string, number>([
  ["components/assistant-ui/markdown-text.tsx: SEARCH_IMAGE_TAG", 2],
  ["components/assistant-ui/markdown-text.tsx: SearchImageElement", 1],
  ["components/assistant-ui/markdown-text.tsx: unslothDarkTheme", 2],
  ["components/assistant-ui/markdown-text.tsx: unslothLightTheme", 2],
  ["components/assistant-ui/thread.tsx: CodeExecutionToolUI", 1],
  ["components/assistant-ui/thread.tsx: ImageGenerationToolUI", 1],
  ["components/assistant-ui/thread.tsx: KnowledgeBaseToolUI", 1],
  ["components/assistant-ui/thread.tsx: MarkdownText", 1],
  ["components/assistant-ui/thread.tsx: PythonToolUI", 1],
  ["components/assistant-ui/thread.tsx: Reasoning", 1],
  ["components/assistant-ui/thread.tsx: ReasoningGroup", 1],
  ["components/assistant-ui/thread.tsx: RenderHtmlToolUI", 1],
  ["components/assistant-ui/thread.tsx: Sources", 1],
  ["components/assistant-ui/thread.tsx: TerminalToolUI", 1],
  ["components/assistant-ui/thread.tsx: ToolFallback", 1],
  ["components/assistant-ui/thread.tsx: ToolGroup", 1],
  ["components/assistant-ui/thread.tsx: WebSearchToolUI", 1],
  ["components/ui/sidebar.tsx: SIDEBAR_WIDTH_DEFAULT", 1],
  ["features/chat/artifacts/artifact-surface.tsx: unslothDarkTheme", 1],
  ["features/chat/artifacts/artifact-surface.tsx: unslothLightTheme", 1],
  ["features/chat/attachment-content.ts: MAX_OPEN_DOCUMENT_ARCHIVE_BYTES", 1],
  ["features/chat/chat-page.tsx: CONVERSATION_MARKDOWN_FORMAT", 1],
  ["features/chat/chat-page.tsx: CONVERSATION_MARKDOWN_LABEL", 1],
  ["features/chat/presets/preset-load-config.ts: KV_CACHE_DTYPES", 1],
  ["features/training/stores/training-config-store.ts: TRAINING_CONFIG_PERSISTENCE_NAME", 1],
  ["features/training/stores/training-config-store.ts: TRAINING_CONFIG_PERSISTENCE_VERSION", 1],
]);

test("no module-scope read of a chat barrel value", () => {
  const files = walkSources(SRC);
  // Only a module the barrel can reach during its own initialization can be
  // caught half-initialized. A leaf that merely imports from the barrel is
  // always evaluated after it, so an eager read there is safe.
  const atRisk = barrelInitClosure(files);
  const sources = readAll(files);
  const resolve = makeResolver(sources);
  const bearing = barrelBearingModules(sources, resolve);
  const runtimeEdges = (file: string): string[] => {
    const source = parse(file, sources.get(file) ?? "");
    const out: string[] = [];
    for (const st of source.statements) {
      const spec =
        (ts.isImportDeclaration(st) || ts.isExportDeclaration(st)) && st.moduleSpecifier;
      if (!spec || !ts.isStringLiteral(spec)) continue;
      if (isErasedEdge(st)) continue;
      out.push(spec.text);
    }
    return out;
  };

  const prone = proneExportSets(files, sources, resolve);
  const carriesProne = (target: string, exported?: string): boolean => {
    const set = prone.get(target);
    if (!set) return false;
    if (set.star) return true;
    return exported === undefined ? set.names.size > 0 : set.names.has(exported);
  };
  const barrelFile = resolve(BARREL, path.join(SRC, "index.ts"));

  const barrelNameFilter =
    (from: string) =>
    (specifier: string, exported?: string): boolean => {
      // A hoisted `function` the barrel re-exports is initialized before any
      // module body runs, so a cyclic consumer reading that name cannot crash;
      // only `const`, `let` and `class` bindings can. Following the re-export to
      // the declaring module is what tells the two apart.
      if (specifier === BARREL) {
        return barrelFile === null ? true : carriesProne(barrelFile, exported);
      }
      const target = resolve(specifier, from);
      if (target === null) return false;
      // A deep import into the cycle is the same hazard as the barrel. The
      // crash this guard exists for was a read of a const imported straight
      // from chat-runtime-store; #9852 later moved that import off the barrel,
      // which silently put the original defect outside a barrel-only filter.
      // Any module the barrel's own initialization can reach may itself be
      // mid-initialization when an at-risk module reads from it.
      if (atRisk.has(target)) {
        // A module with no runtime imports has nothing that can re-enter it, so
        // its bindings are always initialized before any importer body runs.
        // That is exactly why prompt-queue-events.ts exists, and reading from
        // such a leaf is safe however the barrel reaches it.
        if (runtimeEdges(target).length === 0) return false;
        return carriesProne(target, exported);
      }
      // A bridge that re-exports a barrel binding keeps the conservative
      // answer: bearing knows the name travelled, not which module declared it,
      // so an import-then-export hop stays reported rather than guessed at.
      const carried = bearing.get(target);
      if (!carried) return false;
      if (carried.has(STAR)) return true;
      return exported === undefined ? carried.size > 0 : carried.has(exported);
    };
  const offenders: string[] = [];
  const seenCounts = new Map<string, number>();
  let importers = 0;
  for (const file of files) {
    const text = sources.get(file) ?? "";
    // No text prefilter on BARREL: a module importing through a bridge never
    // spells the name, and skipping it was the hole.
    const source = parse(file, text);
    const names = barrelValueNames(source, barrelNameFilter(file));
    const helpers = new Map<string, ImportedHelper>();
    for (const statement of source.statements) {
      if (!ts.isImportDeclaration(statement)) continue;
      const specifier = statement.moduleSpecifier;
      if (!ts.isStringLiteral(specifier)) continue;
      const target = resolve(specifier.text, file);
      if (target === null) continue;
      const clause = statement.importClause;
      if (!clause || clause.isTypeOnly) continue;
      const bound = clause.namedBindings;
      if (!bound && !clause.name) continue;
      if (bound && ts.isNamespaceImport(bound)) continue;
      const helperSource = parse(target, sources.get(target) ?? "");
      const helperNames = barrelValueNames(helperSource, barrelNameFilter(target));
      if (helperNames.size === 0) continue;
      const exported = exportedFunctions(helperSource);
      // `import read from "./helper"` binds whatever that module default-exports.
      if (clause.name) {
        const fn = exported.get("default");
        if (fn) {
          helpers.set(clause.name.text, { file: target, source: helperSource, fn, names: helperNames });
        }
      }
      if (!bound || ts.isNamespaceImport(bound)) continue;
      for (const element of bound.elements) {
        if (element.isTypeOnly) continue;
        const fn = exported.get((element.propertyName ?? element.name).text);
        if (fn) helpers.set(element.name.text, { file: target, source: helperSource, fn, names: helperNames });
      }
    }
    if (names.size === 0 && helpers.size === 0) continue;
    if (names.size > 0) importers += 1;
    if (!atRisk.has(file)) continue;
    for (const hit of eagerReads(source, names, { helpers }, namespaceImportNames(source))) {
      const entry = `${relativeToSrc(file)}: ${hit}`;
      // The line is dropped from the key, and only from the key: the offender
      // message below still says where to look.
      const key = entry.replace(/ \(line \d+\)/g, "");
      if (KNOWN_DEEP_CYCLE_READS.has(key)) {
        seenCounts.set(key, (seenCounts.get(key) ?? 0) + 1);
        continue;
      }
      offenders.push(entry);
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
  // The list may only shrink, and nothing enforced that: an entry that stopped
  // matching stayed armed, ready to suppress a real regression that reappeared
  // under the same name later.
  const drifted = [...KNOWN_DEEP_CYCLE_READS]
    .map(([key, allowed]) => ({ key, allowed, seen: seenCounts.get(key) ?? 0 }))
    .filter(({ allowed, seen }) => allowed !== seen)
    .sort((a, b) => a.key.localeCompare(b.key));
  assert.deepEqual(
    drifted.map(({ key, allowed, seen }) => `${key} (allowed ${allowed}, found ${seen})`),
    [],
    `KNOWN_DEEP_CYCLE_READS is out of date. A count that dropped is progress: ` +
      `lower it, or delete the entry at zero, or it stays armed to mask a future ` +
      `regression. A count that rose is a new latent blank-page bug.`,
  );
  // Anti-vacuity: a barrel rename would otherwise make this pass by finding nothing.
  assert.ok(importers >= 5, `only ${importers} modules import values from ${BARREL}`);
});

test("the scan catches every shape the regex version missed", () => {
  const cases: Array<[string, string]> = [
    ["plain const", `import { K } from "${BARREL}";\nconst a = [K];\n`],
    [
      "default re-evaluated on a later argument-less call",
      `import { K } from "${BARREL}";\nfunction read(v = K) {}\nread(1);\nread();\n`,
    ],
    [
      "base constructor reached through a derived class",
      `import { K } from "${BARREL}";\nclass Base { constructor() { use(K); } }\nclass D extends Base {}\nnew D();\n`,
    ],
    [
      "static method invoked at module scope",
      `import { K } from "${BARREL}";\nclass C { static read() { return K; } }\nC.read();\n`,
    ],
    [
      "read after a short-circuited await",
      `import { K } from "${BARREL}";\nasync function r() { flag && await x; return K; }\nr();\n`,
    ],
    [
      "named callback handed to a synchronous method",
      `import { K } from "${BARREL}";\nfunction cb() { return K; }\n[0].map(cb);\n`,
    ],
    [
      "named function invoked through call",
      `import { K } from "${BARREL}";\nfunction f() { return K; }\nf.call(null);\n`,
    ],
    [
      "namespace read through a property",
      `import * as chat from "${BARREL}";\nconst v = chat.K;\n`,
    ],
    [
      "callback passed to a synchronous array method",
      `import { K } from "${BARREL}";\n[0].map(() => K);\n`,
    ],
    [
      "local object method invoked at module scope",
      `import { K } from "${BARREL}";\nconst obj = { read() { return K; } };\nobj.read();\n`,
    ],
    [
      "local function invoked as a template tag",
      `import { K } from "${BARREL}";\nfunction tag() { return K; }\nconst v = tag\`value\`;\n`,
    ],
    [
      "inline async IIFE before its first await",
      `import { K } from "${BARREL}";\nvoid (async () => { consume(K); await ready; })();\n`,
    ],
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
      "read after a conditional await still runs synchronously",
      `import { K } from "${BARREL}";\nasync function read() { if (skip) await x; return K; }\nread();\n`,
    ],
    [
      "default used because the call omits the argument",
      `import { K } from "${BARREL}";\nfunction read(v = K) {}\nread();\n`,
    ],
    [
      "default used because the call passes undefined",
      `import { K } from "${BARREL}";\nfunction read(v = K) {}\nread(undefined);\n`,
    ],
    [
      "await only inside a nested callback does not suspend",
      `import { K } from "${BARREL}";\nasync function read() { const g = async () => { await x; }; return K; }\nread();\n`,
    ],
    [
      "eager call two functions deep",
      `import { K } from "${BARREL}";\nfunction inner() { return K; }\nfunction outer() { return inner(); }\nconst v = outer();\n`,
    ],
    [
      // The block's own declarations were never collected, so the call to a
      // helper declared beside it resolved to nothing.
      "helper declared and called inside a nested block",
      `import { K } from "${BARREL}";\nif (ready) { const read = () => K; read(); }\n`,
    ],
    [
      "class declared and constructed inside a nested block",
      `import { K } from "${BARREL}";\n{ class C { constructor() { consume(K); } } new C(); }\n`,
    ],
    [
      "helper declared and called inside a switch case",
      `import { K } from "${BARREL}";\nswitch (v) { case 1: { function read() { return K; } read(); } }\n`,
    ],
    [
      // Destructuring performs the same property read `obj.value` does.
      "getter run by destructuring",
      `import { K } from "${BARREL}";\nconst obj = { get value() { return K; } };\nconst { value } = obj;\n`,
    ],
    [
      "getter run by a rest element",
      `import { K } from "${BARREL}";\nconst obj = { get value() { return K; } };\nconst { ...rest } = obj;\n`,
    ],
    [
      "getter run by object spread",
      `import { K } from "${BARREL}";\nconst obj = { get value() { return K; } };\nconst copy = { ...obj };\n`,
    ],
    [
      "setter run by an assignment",
      `import { K } from "${BARREL}";\nconst obj = { set value(v) { consume(K); } };\nobj.value = 1;\n`,
    ],
    [
      "static setter run by an assignment",
      `import { K } from "${BARREL}";\nclass C { static set value(v) { consume(K); } }\nC.value = 1;\n`,
    ],
    [
      "static getter run by a property read",
      `import { K } from "${BARREL}";\nclass C { static get value() { return K; } }\nconst v = C.value;\n`,
    ],
    [
      // The Promise constructor calls it before returning, named or not.
      "named promise executor",
      `import { K } from "${BARREL}";\nfunction read() { return K; }\nnew Promise(read);\n`,
    ],
    [
      // The base default runs because the derived constructor's `super()`
      // supplies nothing, whatever `new` was given.
      "base default reached through an argument-less super",
      `import { K } from "${BARREL}";\nclass Base { constructor(v = K) {} }\nclass D extends Base { constructor(v) { super(); } }\nnew D(1);\n`,
    ],
    [
      "local function applied as a class decorator",
      `import { K } from "${BARREL}";\nfunction deco() { consume(K); }\n@deco\nclass C {}\n`,
    ],
    [
      "local function applied as a method decorator",
      `import { K } from "${BARREL}";\nfunction deco() { consume(K); }\nclass C { @deco method() {} }\n`,
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
      "namespace object merely stored",
      `import * as chat from "${BARREL}";\nconst copy = chat;\n`,
    ],
    [
      "default skipped when every call supplies the argument",
      `import { K } from "${BARREL}";\nfunction read(v = K) {}\nread(1);\n`,
    ],
    [
      "named callback handed to an unknown function",
      `import { K } from "${BARREL}";\nfunction cb() { return K; }\nregisterLater(cb);\n`,
    ],
    [
      "instance method that is never invoked",
      `import { K } from "${BARREL}";\nclass C { read() { return K; } }\n`,
    ],
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
      "default skipped because the call supplies the argument",
      `import { K } from "${BARREL}";\nfunction read(v = K) {}\nread(1);\n`,
    ],
    [
      "inline async IIFE after its first await",
      `import { K } from "${BARREL}";\nvoid (async () => { await ready; consume(K); })();\n`,
    ],
    [
      "inline generator IIFE only builds an iterator",
      `import { K } from "${BARREL}";\nconst it = (function* () { yield K; })();\n`,
    ],
    [
      "callback passed to an unknown function stays deferred",
      `import { K } from "${BARREL}";\nregisterLater(() => K);\n`,
    ],
    [
      "a reassignable callee is not resolved to its declaration",
      `import { K } from "${BARREL}";\nlet read = () => K;\nread = () => 1;\nread();\n`,
    ],
    [
      "self-recursive function is not followed forever",
      `import { K } from "${BARREL}";\nfunction loop() { return loop(); }\nconst v = loop();\n`,
    ],
    [
      // The base default is skipped because `super(1)` supplies the argument,
      // which is the inverse of the positive case above: handing `new`'s own
      // arguments down the chain would have rejected this.
      "base default skipped because super supplies the argument",
      `import { K } from "${BARREL}";\nclass Base { constructor(v = K) {} }\nclass D extends Base { constructor() { super(1); } }\nnew D();\n`,
    ],
    [
      "a helper declared in a block nobody calls",
      `import { K } from "${BARREL}";\nif (ready) { const read = () => K; }\n`,
    ],
    [
      "destructuring a property that has no getter",
      `import { K } from "${BARREL}";\nconst obj = { get value() { return K; } };\nconst { other } = obj;\n`,
    ],
    [
      "a setter nobody assigns to",
      `import { K } from "${BARREL}";\nconst obj = { set value(v) { consume(K); } };\n`,
    ],
    [
      // Promise runs its executor; an arbitrary constructor does not.
      "named function handed to an unknown constructor",
      `import { K } from "${BARREL}";\nfunction read() { return K; }\nnew Registry(read);\n`,
    ],
    [
      "a decorator that is never applied",
      `import { K } from "${BARREL}";\nfunction deco() { consume(K); }\nexport { deco };\n`,
    ],
  ];
  for (const [label, code] of cases) {
    assert.deepEqual(analyse("t.ts", code), [], `${label} should not be flagged`);
  }
});

test("the barrel closure resolves imports that spell their extension", () => {
  // Invisible to the previous resolver, which appended a second extension.
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

  assert.deepEqual([...(bearing.get(bridge) ?? [])], ["K"], "a direct re-export carries K");
  assert.deepEqual([...(bearing.get(deeper) ?? [])], ["K"], "bearing propagates a second hop");
  assert.ok(!bearing.has(plain), "a module exporting its own value carries nothing");

  const names = (from: string, text: string): Set<string> =>
    barrelValueNames(parse(from, text), (specifier, exported) => {
      if (specifier === BARREL) return true;
      const target = resolve(specifier, from);
      if (target === null) return false;
      const carried = bearing.get(target);
      if (!carried) return false;
      if (carried.has(STAR)) return true;
      return exported === undefined ? carried.size > 0 : carried.has(exported);
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
  assert.deepEqual([...(bearing.get(bridge) ?? [])], ["K"], "import-then-export carries K");
  assert.deepEqual([...(bearing.get(aliased) ?? [])], ["J"], "the EXPORTED name is what consumers see");
  assert.ok(!bearing.has(copy), "export const is a copy, not the barrel's binding");
});

test("a bridge's own exports are not treated as barrel bindings", () => {
  const bridge = path.join(SRC, "fake", "mixed.ts");
  const consumer = path.join(SRC, "fake", "mixedconsumer.ts");
  const sources = new Map<string, string>([
    [bridge, `export { K } from "${BARREL}";\nexport const SAFE = 1;\n`],
    [consumer, `import { SAFE, K } from "./mixed";\nconst a = [SAFE, K];\n`],
  ]);
  const resolve = makeResolver(sources);
  const bearing = barrelBearingModules(sources, resolve);
  assert.deepEqual([...(bearing.get(bridge) ?? [])], ["K"], "only K comes from the barrel");

  const names = barrelValueNames(parse(consumer, sources.get(consumer) as string), (spec, exported) => {
    if (spec === BARREL) return true;
    const target = resolve(spec, consumer);
    const carried = target === null ? undefined : bearing.get(target);
    if (!carried) return false;
    return exported === undefined ? carried.size > 0 : carried.has(exported);
  });
  assert.deepEqual([...names], ["K"], "SAFE is the bridge's own value, not the barrel's");
});

test("an eager call into an imported helper is followed across modules", () => {
  const helper = path.join(SRC, "fake", "helper.ts");
  const safe = path.join(SRC, "fake", "safehelper.ts");
  const sources = new Map<string, string>([
    [helper, `import { K } from "${BARREL}";\nexport function read() { return K; }\n`],
    [safe, `import { K } from "${BARREL}";\nexport function read() { return 1; }\n`],
  ]);
  const build = (target: string) => {
    const source = parse(target, sources.get(target) as string);
    const fn = exportedFunctions(source).get("read") as ts.FunctionLikeDeclaration;
    return { file: target, source, fn, names: barrelValueNames(source) };
  };
  const consumerText = `import { read } from "./helper";\nconst v = read();\n`;
  const consumer = parse(path.join(SRC, "fake", "c.ts"), consumerText);

  const hits = eagerReads(consumer, new Set<string>(), {
    helpers: new Map([["read", build(helper)]]),
  });
  assert.equal(hits.length, 1, "the read inside the imported helper must be reported");
  assert.match(hits[0], /read\(\) \(line 2\) reaches .*helper\.ts: K/);

  const clean = eagerReads(consumer, new Set<string>(), {
    helpers: new Map([["read", build(safe)]]),
  });
  assert.deepEqual(clean, [], "a helper that does not touch the barrel is left alone");
});

test("the barrel entry is resolved, so moving it cannot silently empty the scan", () => {
  // The guard's own vacuity hazard: a hard-coded index.ts path would resolve to
  // nothing after a move, atRisk would be empty, and every offender would stop
  // being reported while `importers >= 5` still passed.
  const moved = path.join(SRC, "features", "chat.ts");
  const leaf = path.join(SRC, "leaf.ts");
  const sources = new Map<string, string>([
    [moved, `export * from "./chat/store";\n`],
    [path.join(SRC, "features", "chat", "store.ts"), `export const K = 1;\n`],
    [leaf, `import { K } from "${BARREL}";\n`],
  ]);
  const resolve = makeResolver(sources);
  assert.equal(resolve(BARREL, leaf), moved, "the barrel resolves to the moved file");
  assert.equal(
    resolve(BARREL, path.join(SRC, "index.ts")),
    moved,
    "and resolves the same way from the root the closure starts at",
  );
});

test("a star re-export of the barrel carries every name", () => {
  const bridge = path.join(SRC, "fake", "starbridge.ts");
  const sources = new Map<string, string>([[bridge, `export * from "${BARREL}";\n`]]);
  const resolve = makeResolver(sources);
  const bearing = barrelBearingModules(sources, resolve);
  const carried = bearing.get(bridge);
  assert.ok(carried, "the bridge must carry something");
  const names = barrelValueNames(
    parse(path.join(SRC, "fake", "c.ts"), `import { ANYTHING } from "./starbridge";\nconst a = [ANYTHING];\n`),
    (spec, exported) => {
      if (spec === BARREL) return true;
      const target = resolve(spec, path.join(SRC, "fake", "c.ts"));
      const c = target === null ? undefined : bearing.get(target);
      if (!c) return false;
      if (c.has("*")) return true;
      return exported === undefined ? c.size > 0 : c.has(exported);
    },
  );
  assert.deepEqual([...names], ["ANYTHING"], "any name from a star bridge is a barrel binding");
});

test("an imported helper is entered with the caller's arguments, default import included", () => {
  const named = path.join(SRC, "fake", "namedhelper.ts");
  const dflt = path.join(SRC, "fake", "defaulthelper.ts");
  const sources = new Map<string, string>([
    [named, `import { K } from "${BARREL}";\nexport function read(v = K) { return v; }\n`],
    [dflt, `import { K } from "${BARREL}";\nexport default function read() { return K; }\n`],
  ]);
  const build = (target: string, name: string) => {
    const source = parse(target, sources.get(target) as string);
    const fn = exportedFunctions(source).get(name) as ts.FunctionLikeDeclaration;
    assert.ok(fn, `${target} must export ${name}`);
    return { file: target, source, fn, names: barrelValueNames(source) };
  };
  const scan = (code: string, helpers: Map<string, ImportedHelper>) =>
    eagerReads(parse(path.join(SRC, "fake", "c.ts"), code), new Set<string>(), { helpers });

  const h = new Map([["read", build(named, "read")]]);
  assert.equal(scan(`import { read } from "./namedhelper";\nconst v = read();\n`, h).length, 1,
    "an omitted argument must let the helper's default be evaluated");
  assert.deepEqual(scan(`import { read } from "./namedhelper";\nconst v = read(1);\n`, h), [],
    "a supplied argument means the default never runs");

  const d = new Map([["read", build(dflt, "default")]]);
  assert.equal(scan(`import read from "./defaulthelper";\nconst v = read();\n`, d).length, 1,
    "a default-imported helper is followed like a named one");
});

test("allowlist keys are written the same way on every platform", () => {
  // A Windows runner reports path.relative with backslashes, which made every
  // entry in KNOWN_DEEP_CYCLE_READS miss and the guard fail there alone.
  // Written against a Windows-shaped path rather than this platform's, or the
  // assertion is an identity on Linux and proves nothing.
  assert.equal(toPosix("components\\assistant-ui\\thread.tsx"), "components/assistant-ui/thread.tsx");
  assert.equal(toPosix("features/chat/chat-page.tsx"), "features/chat/chat-page.tsx");
  assert.equal(relativeToSrc(path.join(SRC, "features", "chat", "x.ts")), "features/chat/x.ts");
  for (const entry of KNOWN_DEEP_CYCLE_READS.keys()) {
    assert.ok(!entry.includes("\\"), `${entry} is written with a Windows separator`);
  }
});

test("a default export is recorded under the name consumers import it by", () => {
  const prone = (code: string) => [...tdzProneExportNames(parse("t.ts", code))].sort();
  assert.deepEqual(prone(`export default class K {}\n`), ["default"],
    "K is a local name; consumers import this as default");
  assert.deepEqual(prone(`export default class {}\n`), ["default"]);
  assert.deepEqual(prone(`const v = 1;\nexport default v;\n`), ["default"],
    "an export assignment binds where it runs, so it can be read too early");
  assert.deepEqual(prone(`export default function read() {}\n`), [],
    "a default function declaration is hoisted, so it is never in the dead zone");
  assert.deepEqual(prone(`export function read() {}\nexport class C {}\n`), ["C"],
    "only the class binding can be uninitialized");
});

test("functions published through an export list are resolvable helpers", () => {
  const names = (code: string) => [...exportedFunctions(parse("t.ts", code)).keys()].sort();
  assert.deepEqual(names(`function read() {}\nexport { read };\n`), ["read"]);
  assert.deepEqual(names(`const read = () => 1;\nexport { read as run };\n`), ["run"],
    "the EXPORTED name is what the importer writes");
  assert.deepEqual(names(`function read() {}\nexport { read as default };\n`), ["default"]);
  assert.deepEqual(names(`function read() {}\n`), [],
    "a local function nobody exports is not a helper");
});

test("re-exported bindings are judged where they are declared", () => {
  const leaf = path.join(SRC, "fake", "leafconst.ts");
  const deep = path.join(SRC, "fake", "deepconst.ts");
  const bridge = path.join(SRC, "fake", "prone-bridge.ts");
  const sources = new Map<string, string>([
    // No runtime imports, so nothing can re-enter it mid-initialization.
    [leaf, `export const SAFE = 1;\n`],
    [deep, `import "./leafconst";\nexport const LATE = 1;\nexport function fn() {}\nexport class C {}\n`],
    [bridge, `export { SAFE } from "./leafconst";\nexport { LATE, fn, C } from "./deepconst";\n`],
  ]);
  const resolve = makeResolver(sources);
  const sets = proneExportSets([...sources.keys()], sources, resolve);
  assert.deepEqual([...(sets.get(leaf)?.names ?? [])], [],
    "a module with no runtime imports is always fully initialized first");
  assert.deepEqual([...(sets.get(deep)?.names ?? [])].sort(), ["C", "LATE"],
    "a hoisted function is not in the dead zone; a const and a class are");
  assert.deepEqual([...(sets.get(bridge)?.names ?? [])].sort(), ["C", "LATE"],
    "a re-export carries the declaring module's verdict, not a fresh one");
});
