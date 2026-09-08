// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Frontend CI runs on ubuntu only, so a test that is correct on POSIX and wrong on
// Windows stays green here forever. Thirteen of them accumulated that way, in two
// shapes, and both are decidable from the source without a Windows runner:
//
//   1. A native path used as a dynamic `import()` specifier. `fileURLToPath` gives
//      "/home/..." on POSIX, which node's ESM loader tolerates, and "D:\..." on
//      Windows, which it rejects with ERR_UNSUPPORTED_ESM_URL_SCHEME.
//   2. A `file:` URL's `pathname` used as a filesystem path. It is "/home/..." on
//      POSIX and "/D:/..." on Windows, which `readFile` reads as drive-relative and
//      opens as "D:\D:\...".
//
// Both APIs take a `URL` directly on every platform, so the fix in each case is to
// stop converting. This test is why the two shapes cannot come back before a Windows
// runner sees them: it fails on Linux.
//
// It is a rule about data flow, not a ban on the words. `fileURLToPath` reaching
// `existsSync` is correct and stays allowed; only `fileURLToPath` reaching `import()`
// is not.

import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import test from "node:test";

import ts from "typescript";

const TESTS_DIR = new URL("./", import.meta.url);

/** Every source file under tests/, as URLs. Never as paths: see the header. */
function collect(dir: URL, out: URL[] = []): URL[] {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const child = new URL(entry.name + (entry.isDirectory() ? "/" : ""), dir);
    if (entry.isDirectory()) {
      collect(child, out);
    } else if (
      /\.(?:m?ts|m?js)$/.test(entry.name) &&
      !/\.d\.m?ts$/.test(entry.name)
    ) {
      out.push(child);
    }
  }
  return out;
}

const FILES = collect(TESTS_DIR);

/**
 * `barrier` stops the descent at a node and everything under it. It is how a
 * sanitizer is modelled: the taint rules below ask "does this expression read
 * anything dangerous", and a converted value no longer does, however it was
 * built. Without it the rule cannot tell a repaired value from a raw one.
 */
type Barrier = (n: ts.Node) => boolean;

function walk(
  node: ts.Node,
  visit: (n: ts.Node) => void,
  barrier?: Barrier,
): void {
  if (barrier?.(node)) return;
  visit(node);
  node.forEachChild((child) => walk(child, visit, barrier));
}

function subtreeHas(
  node: ts.Node,
  predicate: (n: ts.Node) => boolean,
  barrier?: Barrier,
): boolean {
  let found = false;
  walk(
    node,
    (n) => {
      if (predicate(n)) found = true;
    },
    barrier,
  );
  return found;
}

/**
 * Identifiers *read* in `node`, so a taint on one reaches the whole expression.
 * Property names are skipped: `x.href` reads `x`, and treating `href` as a name
 * would let an unrelated local called `href` taint every member access in the file.
 */
function identifiersIn(
  node: ts.Node,
  barrier?: Barrier,
  out: ts.Identifier[] = [],
): ts.Identifier[] {
  if (barrier?.(node)) return out;
  if (ts.isIdentifier(node)) {
    out.push(node);
    return out;
  }
  if (ts.isPropertyAccessExpression(node)) {
    return identifiersIn(node.expression, barrier, out);
  }
  if (ts.isPropertyAssignment(node)) {
    return identifiersIn(node.initializer, barrier, out);
  }
  node.forEachChild((child) => {
    identifiersIn(child, barrier, out);
  });
  return out;
}

// A lexical resolver, so taint is carried by a BINDING and not by a name. Test
// files reuse `path`, `file` and `files` freely, and a file-wide set of names would
// let `const path = url.pathname` in one function reject an unrelated
// `const path = new URL(...); readFile(path)` in another.
const isScope = (n: ts.Node): boolean =>
  ts.isSourceFile(n) ||
  ts.isBlock(n) ||
  ts.isModuleBlock(n) ||
  ts.isCaseBlock(n) ||
  ts.isCatchClause(n) ||
  ts.isForStatement(n) ||
  ts.isForOfStatement(n) ||
  ts.isForInStatement(n) ||
  ts.isFunctionLike(n);

function enclosingScope(node: ts.Node): ts.Node | null {
  for (let n = node.parent; n; n = n.parent) {
    if (isScope(n)) return n;
  }
  return null;
}

/** name -> declaration, per scope. Only the forms that can carry a value. */
function declarationTable(
  source: ts.SourceFile,
): Map<ts.Node, Map<string, ts.Node>> {
  const table = new Map<ts.Node, Map<string, ts.Node>>();
  const put = (name: ts.Identifier, declaration: ts.Node): void => {
    const scope = enclosingScope(declaration);
    if (!scope) return;
    let inScope = table.get(scope);
    if (!inScope) {
      inScope = new Map<string, ts.Node>();
      table.set(scope, inScope);
    }
    // First declaration wins; a redeclaration of the same name in the same scope
    // is not something this suite does, and picking either is equally arbitrary.
    if (!inScope.has(name.text)) inScope.set(name.text, declaration);
  };
  /**
   * `const { pathname } = url` and `const [head] = parts` introduce bindings
   * exactly as `const x = ...` does. Each binding element is recorded as its own
   * declaration, so it can be tainted on its own: only `pathname` is dangerous in
   * `const { pathname, href } = url`, and `href` must stay clean.
   */
  const record = (name: ts.BindingName, declaration: ts.Node): void => {
    if (ts.isIdentifier(name)) {
      put(name, declaration);
      return;
    }
    for (const element of name.elements) {
      if (ts.isBindingElement(element)) record(element.name, element);
    }
  };
  walk(source, (n) => {
    if (ts.isVariableDeclaration(n) || ts.isParameter(n)) record(n.name, n);
    else if (ts.isFunctionDeclaration(n) && n.name) put(n.name, n);
    else if (ts.isImportSpecifier(n) || ts.isNamespaceImport(n)) put(n.name, n);
    else if (ts.isImportClause(n) && n.name) put(n.name, n);
  });
  return table;
}

/**
 * The declaration `use` refers to, by walking out through enclosing scopes.
 * Null when nothing in the file declares it, which is the safe answer: an
 * unresolved name is never treated as tainted, so the rules cannot fire on it.
 */
function resolve(
  use: ts.Identifier,
  table: Map<ts.Node, Map<string, ts.Node>>,
): ts.Node | null {
  for (let scope = enclosingScope(use); scope; scope = enclosingScope(scope)) {
    const found = table.get(scope)?.get(use.text);
    if (found) return found;
  }
  return null;
}

/** The module an import binding came from, so a rule can require the right one. */
function importModuleOf(declaration: ts.Node): string | null {
  for (let n: ts.Node | undefined = declaration; n; n = n.parent) {
    if (ts.isImportDeclaration(n)) {
      return ts.isStringLiteralLike(n.moduleSpecifier)
        ? n.moduleSpecifier.text
        : null;
    }
  }
  return null;
}

/**
 * What a call names on its module's side, and which module that is.
 *
 * Both halves matter. Without the first, `import { readFile as read }` hides an
 * fs call behind a local spelling. Without the second, an unrelated
 * `router.open(...)` or `dom.link(...)` is classified as one, and a rule that
 * fires on correct code is worse than no rule.
 */
function resolvedCallee(
  call: ts.CallExpression,
  table: Map<ts.Node, Map<string, ts.Node>>,
): { name: string; module: string | null } | null {
  if (ts.isPropertyAccessExpression(call.expression)) {
    const receiver = call.expression.expression;
    const declaration = ts.isIdentifier(receiver)
      ? resolve(receiver, table)
      : null;
    return {
      name: call.expression.name.text,
      module: declaration ? importModuleOf(declaration) : null,
    };
  }
  if (!ts.isIdentifier(call.expression)) return null;
  const declaration = resolve(call.expression, table);
  const module = declaration ? importModuleOf(declaration) : null;
  if (declaration && ts.isImportSpecifier(declaration)) {
    return {
      name: (declaration.propertyName ?? declaration.name).text,
      module,
    };
  }
  return { name: call.expression.text, module };
}

const FS_MODULES = new Set([
  "node:fs",
  "node:fs/promises",
  "fs",
  "fs/promises",
]);
const URL_MODULES = new Set(["node:url", "url"]);

/** `const { pathname } = url`, which reads the property without a member access. */
const isPathnameBinding = (n: ts.Node): boolean => {
  if (!ts.isBindingElement(n) || !ts.isObjectBindingPattern(n.parent)) {
    return false;
  }
  const property = n.propertyName ?? n.name;
  return ts.isIdentifier(property) && property.text === "pathname";
};

const isPathnameRead = (n: ts.Node): boolean =>
  (ts.isPropertyAccessExpression(n) && n.name.text === "pathname") ||
  (ts.isElementAccessExpression(n) &&
    n.argumentExpression !== undefined &&
    ts.isStringLiteralLike(n.argumentExpression) &&
    n.argumentExpression.text === "pathname") ||
  isPathnameBinding(n);

const isDynamicImport = (n: ts.Node): n is ts.CallExpression =>
  ts.isCallExpression(n) && n.expression.kind === ts.SyntaxKind.ImportKeyword;

// fs entry points that take a path or a URL, mapped to how many of their leading
// arguments are one. Every one of them breaks the same way when handed a Windows
// `file:` pathname, and a destination breaks exactly as a source does, so
// copyFile(sourceUrl, destination.pathname) has to be caught as well.
const FS_PATH_APIS = new Map([
  ["access", 1],
  ["accessSync", 1],
  ["appendFile", 1],
  ["appendFileSync", 1],
  ["chmod", 1],
  ["chmodSync", 1],
  ["chown", 1],
  ["chownSync", 1],
  ["copyFile", 2],
  ["copyFileSync", 2],
  ["cp", 2],
  ["cpSync", 2],
  ["createReadStream", 1],
  ["createWriteStream", 1],
  ["existsSync", 1],
  ["glob", 1],
  ["globSync", 1],
  ["link", 2],
  ["linkSync", 2],
  ["lstat", 1],
  ["lstatSync", 1],
  ["mkdir", 1],
  ["mkdirSync", 1],
  ["mkdtemp", 1],
  ["mkdtempSync", 1],
  ["open", 1],
  ["openAsBlob", 1],
  ["openSync", 1],
  ["opendir", 1],
  ["opendirSync", 1],
  ["readdir", 1],
  ["readdirSync", 1],
  ["readFile", 1],
  ["readFileSync", 1],
  ["readlink", 1],
  ["readlinkSync", 1],
  ["realpath", 1],
  ["realpathSync", 1],
  ["rename", 2],
  ["renameSync", 2],
  ["rm", 1],
  ["rmSync", 1],
  ["rmdir", 1],
  ["rmdirSync", 1],
  ["stat", 1],
  ["statSync", 1],
  ["truncate", 1],
  ["truncateSync", 1],
  ["unwatchFile", 1],
  ["utimes", 1],
  ["utimesSync", 1],
  ["watch", 1],
  ["watchFile", 1],
  ["symlink", 2],
  ["symlinkSync", 2],
  ["unlink", 1],
  ["unlinkSync", 1],
  ["writeFile", 1],
  ["writeFileSync", 1],
]);

/** How many leading arguments of `call` are a filesystem path, if any are. */
function fsPathArguments(
  call: ts.CallExpression,
  table: Map<ts.Node, Map<string, ts.Node>>,
): number | undefined {
  const callee = resolvedCallee(call, table);
  if (!callee?.module || !FS_MODULES.has(callee.module)) return undefined;
  return FS_PATH_APIS.get(callee.name);
}

/** The parameter list of a locally declared function, for the call-site step. */
function parametersOf(
  declaration: ts.Node | null,
): readonly ts.ParameterDeclaration[] | null {
  if (!declaration) return null;
  if (ts.isFunctionDeclaration(declaration)) return declaration.parameters;
  if (
    ts.isVariableDeclaration(declaration) &&
    declaration.initializer &&
    (ts.isArrowFunction(declaration.initializer) ||
      ts.isFunctionExpression(declaration.initializer))
  ) {
    return declaration.initializer.parameters;
  }
  return null;
}

/**
 * Bindings carrying a value derived from `seed`, propagated to a fixpoint.
 *
 * None of the steps is decoration; each is a way the shape has been written or
 * could plausibly be refactored into.
 *
 *   const x = <tainted>            initialization
 *   x = <tainted>                  a value built in steps
 *   const { pathname } = url       destructuring, which reads the property
 *                                  without a member access
 *   arr.push(<tainted>)            into a collection...
 *   for (const v of arr)           ...and back out of it. This pair is the
 *                                  marker-key defect exactly: it pushed
 *                                  url.pathname in one function and read the
 *                                  array with readFile in another.
 *   helper(<tainted>)              across a local helper boundary, onto the
 *                                  parameter the argument lands on
 *
 * The parameter step is an over-approximation: a helper called with a tainted
 * argument at one site and a clean one at another taints every use inside it.
 * That is the right direction here, because the tainted call site is itself a
 * defect, and this analysis exists to find those rather than to prove absence.
 *
 * `barrier` marks a sanitizer, whose result is clean whatever went into it.
 */
function taintedBindings(
  source: ts.SourceFile,
  table: Map<ts.Node, Map<string, ts.Node>>,
  seed: (n: ts.Node) => boolean,
  barrier?: Barrier,
): Set<ts.Node> {
  const tainted = new Set<ts.Node>();
  const isTainted = (expr: ts.Node): boolean =>
    subtreeHas(expr, seed, barrier) ||
    identifiersIn(expr, barrier).some((use) => {
      const declaration = resolve(use, table);
      return declaration !== null && tainted.has(declaration);
    });

  // `const x = <tainted>`.
  const assigned = (n: ts.Node): ts.Node[] =>
    ts.isVariableDeclaration(n) &&
    ts.isIdentifier(n.name) &&
    n.initializer !== undefined &&
    isTainted(n.initializer)
      ? [n]
      : [];

  // `x = <tainted>` after the fact. A value built in steps, `let path = "";
  // path = url.pathname;`, is otherwise invisible to `assigned` above.
  const reassigned = (n: ts.Node): ts.Node[] => {
    if (
      !ts.isBinaryExpression(n) ||
      n.operatorToken.kind !== ts.SyntaxKind.EqualsToken ||
      !ts.isIdentifier(n.left) ||
      !isTainted(n.right)
    ) {
      return [];
    }
    const declaration = resolve(n.left, table);
    return declaration ? [declaration] : [];
  };

  /**
   * A binding element, tainted either because the property it names is itself
   * the seed (`const { pathname } = url`) or because the whole right-hand side
   * was already tainted (`const [first] = taintedList`).
   */
  const destructured = (n: ts.Node): ts.Node[] => {
    if (!ts.isBindingElement(n)) return [];
    if (seed(n)) return [n];
    const declaration = n.parent?.parent;
    return declaration &&
      ts.isVariableDeclaration(declaration) &&
      declaration.initializer &&
      isTainted(declaration.initializer)
      ? [n]
      : [];
  };

  // `arr.push(<tainted>)`, which taints whichever `arr` is in scope here.
  const collected = (n: ts.Node): ts.Node[] => {
    if (
      !ts.isCallExpression(n) ||
      !ts.isPropertyAccessExpression(n.expression) ||
      (n.expression.name.text !== "push" &&
        n.expression.name.text !== "unshift") ||
      !ts.isIdentifier(n.expression.expression) ||
      !n.arguments.some((argument) => isTainted(argument))
    ) {
      return [];
    }
    const declaration = resolve(n.expression.expression, table);
    return declaration ? [declaration] : [];
  };

  // `for (const v of <tainted array>)`, which taints the element.
  const iterated = (n: ts.Node): ts.Node[] => {
    if (!ts.isForOfStatement(n) || !ts.isVariableDeclarationList(n.initializer))
      return [];
    const [declaration] = n.initializer.declarations;
    if (
      n.initializer.declarations.length !== 1 ||
      !ts.isIdentifier(declaration.name)
    ) {
      return [];
    }
    return isTainted(n.expression) ? [declaration] : [];
  };

  // `helper(<tainted>)` where `helper` is declared in this file: the parameter
  // the argument lands on carries the taint into the body. Without this a
  // pathname crossing one helper boundary is invisible, however it is used
  // inside.
  const passed = (n: ts.Node): ts.Node[] => {
    if (!ts.isCallExpression(n) || !ts.isIdentifier(n.expression)) return [];
    const parameters = parametersOf(resolve(n.expression, table));
    if (!parameters) return [];
    const out: ts.Node[] = [];
    n.arguments.forEach((argument, index) => {
      const parameter = parameters[index];
      if (parameter && isTainted(argument)) out.push(parameter);
    });
    return out;
  };

  let changed = true;
  let rounds = 0;
  while (changed && rounds < 12) {
    changed = false;
    rounds += 1;
    walk(source, (n) => {
      for (const binding of [
        ...assigned(n),
        ...reassigned(n),
        ...destructured(n),
        ...collected(n),
        ...iterated(n),
        ...passed(n),
      ]) {
        if (!tainted.has(binding)) {
          tainted.add(binding);
          changed = true;
        }
      }
    });
  }
  return tainted;
}

interface Scan {
  dynamicImports: number;
  fsCalls: number;
  nativePathImports: string[];
  pathnameToFs: string[];
}

function scanSource(source: ts.SourceFile, label: string): Scan {
  const result: Scan = {
    dynamicImports: 0,
    fsCalls: 0,
    nativePathImports: [],
    pathnameToFs: [],
  };
  const table = declarationTable(source);
  // Alias-aware and module-aware, like the fs rule: `import { fileURLToPath as
  // toPath }` must still seed the taint, and a same-named helper from elsewhere
  // must not.
  const isFileURLToPathCall = (n: ts.Node): boolean => {
    if (!ts.isCallExpression(n)) return false;
    const callee = resolvedCallee(n, table);
    return (
      callee?.name === "fileURLToPath" &&
      callee.module !== null &&
      URL_MODULES.has(callee.module)
    );
  };
  /**
   * `pathToFileURL` is the exact inverse of `fileURLToPath`, so a native path
   * put back through it is a legal specifier again and the taint must stop
   * there. That round trip is the standard conversion when a module location
   * genuinely starts life as a filesystem path, and both resolvers under
   * tests/helpers do it; a rule that rejected it would be pushing people to
   * weaken the rule rather than fix the code.
   *
   * Only a sanitizer for THIS rule. `pathToFileURL(url.pathname)` is still
   * wrong: the pathname is "/D:/..." on Windows, which is not a native path,
   * and converting it yields a URL for a path that does not exist. So the
   * pathname rule below is deliberately given no barrier.
   */
  const isPathToFileURLCall = (n: ts.Node): boolean => {
    if (!ts.isCallExpression(n)) return false;
    const callee = resolvedCallee(n, table);
    return (
      callee?.name === "pathToFileURL" &&
      callee.module !== null &&
      URL_MODULES.has(callee.module)
    );
  };
  const nativePaths = taintedBindings(
    source,
    table,
    isFileURLToPathCall,
    isPathToFileURLCall,
  );
  const urlPathnames = taintedBindings(source, table, isPathnameRead);
  const reaches = (
    expr: ts.Node,
    tainted: Set<ts.Node>,
    seed: (n: ts.Node) => boolean,
    barrier?: Barrier,
  ) =>
    subtreeHas(expr, seed, barrier) ||
    identifiersIn(expr, barrier).some((use) => {
      const declaration = resolve(use, table);
      return declaration !== null && tainted.has(declaration);
    });
  const at = (node: ts.Node): string =>
    `${label}:${source.getLineAndCharacterOfPosition(node.getStart(source)).line + 1}`;

  walk(source, (n) => {
    if (isDynamicImport(n)) {
      result.dynamicImports += 1;
      const specifier = n.arguments[0];
      if (
        specifier &&
        reaches(
          specifier,
          nativePaths,
          isFileURLToPathCall,
          isPathToFileURLCall,
        )
      ) {
        result.nativePathImports.push(at(n));
      }
      return;
    }
    if (ts.isCallExpression(n)) {
      const pathArguments = fsPathArguments(n, table);
      if (pathArguments !== undefined) {
        result.fsCalls += 1;
        // Every path-bearing position, not only the source: a destination
        // breaks on Windows exactly as a source does.
        for (const target of n.arguments.slice(0, pathArguments)) {
          if (reaches(target, urlPathnames, isPathnameRead)) {
            result.pathnameToFs.push(at(n));
            break;
          }
        }
      }
    }
  });
  return result;
}

function scan(file: URL): Scan {
  const text = readFileSync(file, "utf8");
  const label = decodeURIComponent(file.href.slice(TESTS_DIR.href.length));
  const kind = /\.m?ts$/.test(label) ? ts.ScriptKind.TS : ts.ScriptKind.JS;
  // The file name is a diagnostic label only; nothing resolves against it.
  return scanSource(
    ts.createSourceFile(label, text, ts.ScriptTarget.ESNext, true, kind),
    label,
  );
}

const SCANS = FILES.map(scan);

// The rules are only worth anything if the analysis reached code. A refactor that
// moved the suite, renamed the extensions or broke the parse would otherwise turn
// both of them into green no-ops.
test("the scan reads the whole suite", () => {
  assert.ok(
    FILES.length > 200,
    `only ${FILES.length} files found under tests/; the walk is not seeing the suite`,
  );
  const dynamicImports = SCANS.reduce((sum, s) => sum + s.dynamicImports, 0);
  const fsCalls = SCANS.reduce((sum, s) => sum + s.fsCalls, 0);
  assert.ok(
    dynamicImports > 50,
    `only ${dynamicImports} dynamic imports parsed; the import rule is not reaching code`,
  );
  assert.ok(
    fsCalls > 50,
    `only ${fsCalls} fs calls parsed; the pathname rule is not reaching code`,
  );
});

test("no test imports a module by native path", () => {
  assert.deepEqual(
    SCANS.flatMap((s) => s.nativePathImports),
    [],
    "a dynamic import() specifier is built from fileURLToPath. That is a native path, " +
      "which node's ESM loader rejects on Windows (ERR_UNSUPPORTED_ESM_URL_SCHEME). " +
      'Use new URL("...", import.meta.url).href, which import() accepts everywhere and ' +
      "which a ?bust= query can be appended to.",
  );
});

test("no test reads a file through a URL pathname", () => {
  assert.deepEqual(
    SCANS.flatMap((s) => s.pathnameToFs),
    [],
    'an fs call is given a file: URL pathname. That is "/D:/..." on Windows, which reads ' +
      'as drive-relative and opens "D:\\D:\\...". Pass the URL itself; every fs entry ' +
      "point accepts one. Use pathname only for display or for slicing, where it is / " +
      "separated on every platform.",
  );
});

// Both rules above pass on a suite with no violations, which is also what they would
// do if their detectors were gutted. Exercise the detectors on source known to be
// wrong, and on the corrected form of the same source.
test("the rules fire on the shapes they exist for, and only those", () => {
  const check = (code: string, label: string): Scan =>
    scanSource(
      ts.createSourceFile(
        label,
        code,
        ts.ScriptTarget.ESNext,
        true,
        ts.ScriptKind.TS,
      ),
      label,
    );

  const broken: [string, "nativePathImports" | "pathnameToFs"][] = [
    [
      `import { fileURLToPath } from "node:url";
       const M = fileURLToPath(new URL("../src/x.ts", import.meta.url));
       await import(\`\${M}?bust=1\`);`,
      "nativePathImports",
    ],
    [
      `import { fileURLToPath } from "node:url";
       await import(fileURLToPath(new URL("../src/x.ts", import.meta.url)));`,
      "nativePathImports",
    ],
    [
      `import { readFile } from "node:fs/promises";
       const files = [];
       files.push(new URL("./x.ts", import.meta.url).pathname);
       for (const f of files) await readFile(f, "utf8");`,
      "pathnameToFs",
    ],
    [
      `import { readFileSync } from "node:fs";
       readFileSync(new URL("./x.ts", import.meta.url).pathname, "utf8");`,
      "pathnameToFs",
    ],
    [
      // A destination is a path too, so a two-path API is scanned in both
      // positions. Only the second one is wrong here.
      `import { copyFile } from "node:fs/promises";
       const from = new URL("./a.ts", import.meta.url);
       const to = new URL("./b.ts", import.meta.url).pathname;
       await copyFile(from, to);`,
      "pathnameToFs",
    ],
    [
      // Built in steps, so the taint arrives by assignment and not by an
      // initializer.
      `import { readFile } from "node:fs/promises";
       let where = "";
       where = new URL("./x.ts", import.meta.url).pathname;
       await readFile(where, "utf8");`,
      "pathnameToFs",
    ],
    [
      // Imported under an alias, so the call site does not spell the fs name.
      `import { readFile as read } from "node:fs/promises";
       await read(new URL("./x.ts", import.meta.url).pathname, "utf8");`,
      "pathnameToFs",
    ],
    [
      // An fs entry point outside the read/write core is no more portable.
      `import { createReadStream } from "node:fs";
       createReadStream(new URL("./x.ts", import.meta.url).pathname);`,
      "pathnameToFs",
    ],
    [
      // fileURLToPath under an alias is still fileURLToPath.
      `import { fileURLToPath as toPath } from "node:url";
       const M = toPath(new URL("../src/x.ts", import.meta.url));
       await import(M);`,
      "nativePathImports",
    ],
    [
      // Reached through a namespace import rather than a named one.
      `import * as fs from "node:fs";
       fs.readFileSync(new URL("./x.ts", import.meta.url).pathname, "utf8");`,
      "pathnameToFs",
    ],
    [
      // Destructured, so the property is read without a member access.
      `import { readFile } from "node:fs/promises";
       const { pathname } = new URL("./x.ts", import.meta.url);
       await readFile(pathname, "utf8");`,
      "pathnameToFs",
    ],
    [
      // Destructured and renamed.
      `import { readFileSync } from "node:fs";
       const { pathname: where } = new URL("./x.ts", import.meta.url);
       readFileSync(where, "utf8");`,
      "pathnameToFs",
    ],
    [
      // Across a local helper boundary, which no single-scope rule can see.
      `import { readFile } from "node:fs/promises";
       function load(target) { return readFile(target, "utf8"); }
       await load(new URL("./x.ts", import.meta.url).pathname);`,
      "pathnameToFs",
    ],
    [
      // The same, for the import rule and through an arrow.
      `import { fileURLToPath } from "node:url";
       const load = (specifier) => import(specifier);
       await load(fileURLToPath(new URL("../src/x.ts", import.meta.url)));`,
      "nativePathImports",
    ],
  ];
  for (const [code, rule] of broken) {
    assert.ok(
      check(code, "broken.ts")[rule].length > 0,
      `${rule} did not fire on a case it exists for:\n${code}`,
    );
  }

  // The corrected shapes, which are what the fixed suite and its resolvers do.
  const clean = check(
    `import { existsSync } from "node:fs";
     import { readFile } from "node:fs/promises";
     import { fileURLToPath } from "node:url";
     const M = new URL("../src/x.ts", import.meta.url).href;
     await import(\`\${M}?bust=1\`);
     const SRC = fileURLToPath(new URL("../src/", import.meta.url));
     existsSync(SRC + "lib.ts");
     const files = [];
     files.push(new URL("./x.ts", import.meta.url));
     for (const f of files) { await readFile(f, "utf8"); f.pathname.slice(1); }
     function label() { const path = new URL("./y.ts", import.meta.url).pathname; return path; }
     async function read() { const path = new URL("./y.ts", import.meta.url); return readFile(path, "utf8"); }
     const router = { open(_: string) {}, link(_: string) {} };
     router.open(new URL("https://example.test/x").pathname);
     router.link(new URL("https://example.test/y").pathname);
     const { href, origin } = new URL("./z.ts", import.meta.url);
     await import(href + origin);`,
    "fixed.ts",
  );
  assert.deepEqual(
    clean.nativePathImports,
    [],
    "the import rule fired on a correct file",
  );
  assert.deepEqual(
    clean.pathnameToFs,
    [],
    "the pathname rule fired on a correct file",
  );
});

// A sanitizer is the one thing that can quietly turn a rule off, since silencing
// it and repairing it look identical from the outside. So assert both halves: the
// repaired form stops firing, and the unrepaired forms still do, in the same test.
test("pathToFileURL clears the native-path taint, and only it does", () => {
  const check = (code: string): Scan =>
    scanSource(
      ts.createSourceFile(
        "sanitizer.ts",
        code,
        ts.ScriptTarget.ESNext,
        true,
        ts.ScriptKind.TS,
      ),
      "sanitizer.ts",
    );

  // The round trip a module location genuinely starting life as a path needs.
  const repaired = check(
    `import { fileURLToPath, pathToFileURL } from "node:url";
     const native = fileURLToPath(new URL("../src/x.ts", import.meta.url));
     const specifier = pathToFileURL(native).href;
     await import(specifier);
     await import(pathToFileURL(native).href + "?bust=1");`,
  );
  assert.deepEqual(
    repaired.nativePathImports,
    [],
    "pathToFileURL is the exact inverse of fileURLToPath, so the specifier it " +
      "produces is legal on every platform and the rule must not reject it",
  );

  // Same file, same import of pathToFileURL in scope, but the path reaches
  // import() without going through it. Nothing about the fix blunts this.
  const stillBroken = check(
    `import { fileURLToPath, pathToFileURL } from "node:url";
     const native = fileURLToPath(new URL("../src/x.ts", import.meta.url));
     const unused = pathToFileURL(native).href;
     await import(native);`,
  );
  assert.equal(
    stillBroken.nativePathImports.length,
    1,
    "a native path still reaching import() must fire, whether or not " +
      "pathToFileURL appears elsewhere in the file",
  );

  // And it is a sanitizer for the native-path rule only. On Windows a URL
  // pathname is "/D:/...", which is not a native path, so putting it through
  // pathToFileURL yields a URL for a file that does not exist.
  const notSanitized = check(
    `import { pathToFileURL } from "node:url";
     import { readFile } from "node:fs/promises";
     const { pathname } = new URL("./x.ts", import.meta.url);
     await readFile(pathToFileURL(pathname), "utf8");`,
  );
  assert.equal(
    notSanitized.pathnameToFs.length,
    1,
    "pathToFileURL does not repair a URL pathname, so the pathname rule must " +
      "not treat it as a barrier",
  );
});
