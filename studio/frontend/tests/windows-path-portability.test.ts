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

function walk(node: ts.Node, visit: (n: ts.Node) => void): void {
  visit(node);
  node.forEachChild((child) => walk(child, visit));
}

function subtreeHas(
  node: ts.Node,
  predicate: (n: ts.Node) => boolean,
): boolean {
  let found = false;
  walk(node, (n) => {
    if (predicate(n)) found = true;
  });
  return found;
}

/**
 * Identifiers *read* in `node`, so a taint on one reaches the whole expression.
 * Property names are skipped: `x.href` reads `x`, and treating `href` as a name
 * would let an unrelated local called `href` taint every member access in the file.
 */
function identifiersIn(
  node: ts.Node,
  out: ts.Identifier[] = [],
): ts.Identifier[] {
  if (ts.isIdentifier(node)) {
    out.push(node);
    return out;
  }
  if (ts.isPropertyAccessExpression(node)) {
    return identifiersIn(node.expression, out);
  }
  if (ts.isPropertyAssignment(node)) {
    return identifiersIn(node.initializer, out);
  }
  node.forEachChild((child) => {
    identifiersIn(child, out);
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
  const record = (name: ts.Node, declaration: ts.Node): void => {
    if (!ts.isIdentifier(name)) return;
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
  walk(source, (n) => {
    if (ts.isVariableDeclaration(n) || ts.isParameter(n)) record(n.name, n);
    else if (ts.isFunctionDeclaration(n) && n.name) record(n.name, n);
    else if (ts.isImportSpecifier(n) || ts.isImportClause(n)) {
      if (n.name) record(n.name, n);
    }
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

const isFileURLToPathCall = (n: ts.Node): boolean =>
  ts.isCallExpression(n) &&
  ((ts.isIdentifier(n.expression) && n.expression.text === "fileURLToPath") ||
    (ts.isPropertyAccessExpression(n.expression) &&
      n.expression.name.text === "fileURLToPath"));

const isPathnameRead = (n: ts.Node): boolean =>
  (ts.isPropertyAccessExpression(n) && n.name.text === "pathname") ||
  (ts.isElementAccessExpression(n) &&
    n.argumentExpression !== undefined &&
    ts.isStringLiteralLike(n.argumentExpression) &&
    n.argumentExpression.text === "pathname");

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
  ["copyFile", 2],
  ["copyFileSync", 2],
  ["cp", 2],
  ["cpSync", 2],
  ["existsSync", 1],
  ["link", 2],
  ["linkSync", 2],
  ["lstat", 1],
  ["lstatSync", 1],
  ["mkdir", 1],
  ["mkdirSync", 1],
  ["open", 1],
  ["openSync", 1],
  ["opendir", 1],
  ["opendirSync", 1],
  ["readdir", 1],
  ["readdirSync", 1],
  ["readFile", 1],
  ["readFileSync", 1],
  ["realpath", 1],
  ["realpathSync", 1],
  ["rename", 2],
  ["renameSync", 2],
  ["rm", 1],
  ["rmSync", 1],
  ["stat", 1],
  ["statSync", 1],
  ["symlink", 2],
  ["symlinkSync", 2],
  ["unlink", 1],
  ["unlinkSync", 1],
  ["writeFile", 1],
  ["writeFileSync", 1],
]);

function calleeName(call: ts.CallExpression): string | null {
  if (ts.isIdentifier(call.expression)) return call.expression.text;
  if (ts.isPropertyAccessExpression(call.expression))
    return call.expression.name.text;
  return null;
}

/**
 * Bindings carrying a value derived from `seed`, propagated to a fixpoint through
 * `const x = <tainted>`, `arr.push(<tainted>)` and `for (const v of <tainted>)`.
 *
 * The array and for-of steps are not decoration. The failure this rule exists for
 * pushed `url.pathname` into an array in one function and read the array with
 * `readFile` in another, so a rule that only looked at the call site would see an
 * untainted local and pass.
 */
function taintedBindings(
  source: ts.SourceFile,
  table: Map<ts.Node, Map<string, ts.Node>>,
  seed: (n: ts.Node) => boolean,
): Set<ts.Node> {
  const tainted = new Set<ts.Node>();
  const isTainted = (expr: ts.Node): boolean =>
    subtreeHas(expr, seed) ||
    identifiersIn(expr).some((use) => {
      const declaration = resolve(use, table);
      return declaration !== null && tainted.has(declaration);
    });

  // `const x = <tainted>`.
  const assigned = (n: ts.Node): ts.Node | null =>
    ts.isVariableDeclaration(n) &&
    ts.isIdentifier(n.name) &&
    n.initializer !== undefined &&
    isTainted(n.initializer)
      ? n
      : null;

  // `arr.push(<tainted>)`, which taints whichever `arr` is in scope here.
  const collected = (n: ts.Node): ts.Node | null =>
    ts.isCallExpression(n) &&
    ts.isPropertyAccessExpression(n.expression) &&
    (n.expression.name.text === "push" ||
      n.expression.name.text === "unshift") &&
    ts.isIdentifier(n.expression.expression) &&
    n.arguments.some((argument) => isTainted(argument))
      ? resolve(n.expression.expression, table)
      : null;

  // `for (const v of <tainted array>)`, which taints the element.
  const iterated = (n: ts.Node): ts.Node | null => {
    if (!ts.isForOfStatement(n) || !ts.isVariableDeclarationList(n.initializer))
      return null;
    const [declaration] = n.initializer.declarations;
    if (
      n.initializer.declarations.length !== 1 ||
      !ts.isIdentifier(declaration.name)
    ) {
      return null;
    }
    return isTainted(n.expression) ? declaration : null;
  };

  let changed = true;
  let rounds = 0;
  while (changed && rounds < 10) {
    changed = false;
    rounds += 1;
    walk(source, (n) => {
      const binding = assigned(n) ?? collected(n) ?? iterated(n);
      if (binding !== null && !tainted.has(binding)) {
        tainted.add(binding);
        changed = true;
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
  const nativePaths = taintedBindings(source, table, isFileURLToPathCall);
  const urlPathnames = taintedBindings(source, table, isPathnameRead);
  const reaches = (
    expr: ts.Node,
    tainted: Set<ts.Node>,
    seed: (n: ts.Node) => boolean,
  ) =>
    subtreeHas(expr, seed) ||
    identifiersIn(expr).some((use) => {
      const declaration = resolve(use, table);
      return declaration !== null && tainted.has(declaration);
    });
  const at = (node: ts.Node): string =>
    `${label}:${source.getLineAndCharacterOfPosition(node.getStart(source)).line + 1}`;

  walk(source, (n) => {
    if (isDynamicImport(n)) {
      result.dynamicImports += 1;
      const specifier = n.arguments[0];
      if (specifier && reaches(specifier, nativePaths, isFileURLToPathCall)) {
        result.nativePathImports.push(at(n));
      }
      return;
    }
    if (ts.isCallExpression(n)) {
      const name = calleeName(n);
      const pathArguments = name === null ? undefined : FS_PATH_APIS.get(name);
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
     async function read() { const path = new URL("./y.ts", import.meta.url); return readFile(path, "utf8"); }`,
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
