// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Frontend CI is ubuntu-only, so two Windows-only breakages are caught from source:
// 1. a fileURLToPath result as an import() specifier (ERR_UNSUPPORTED_ESM_URL_SCHEME);
// 2. a file: URL's pathname as an fs path ("/D:/..." opens as "D:\D:\...").
// Both APIs accept a URL directly. fileURLToPath into existsSync is fine.

import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import test from "node:test";

import ts from "typescript";

const TESTS_DIR = new URL("./", import.meta.url);

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

/** `barrier` models a sanitizer: descent stops there, so a converted value is clean. */
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

/** Property names are skipped: `x.href` reads `x`, not a local called `href`. */
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

// Lexical resolver: taint is carried by a BINDING, not a name, since tests reuse names freely.
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
    if (!inScope.has(name.text)) inScope.set(name.text, declaration);
  };
  /** Each destructured element is its own declaration: only `pathname` is tainted, not `href`. */
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

/** Null when undeclared in the file; an unresolved name is never tainted. */
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

/** Resolves both import aliases and the source module, so `router.open` is not an fs call. */
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

// Mapped to how many leading arguments are paths; a destination breaks like a source.
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

function fsPathArguments(
  call: ts.CallExpression,
  table: Map<ts.Node, Map<string, ts.Node>>,
): number | undefined {
  const callee = resolvedCallee(call, table);
  if (!callee?.module || !FS_MODULES.has(callee.module)) return undefined;
  return FS_PATH_APIS.get(callee.name);
}

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
 * Bindings derived from `seed`, propagated to a fixpoint through initialization, reassignment,
 * destructuring, arr.push/for-of, and local helper parameters. The parameter step
 * over-approximates on purpose. `barrier` marks a sanitizer.
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

  const assigned = (n: ts.Node): ts.Node[] =>
    ts.isVariableDeclaration(n) &&
    ts.isIdentifier(n.name) &&
    n.initializer !== undefined &&
    isTainted(n.initializer)
      ? [n]
      : [];

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
  // Alias-aware and module-aware, like the fs rule.
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
   * pathToFileURL inverts fileURLToPath, so it sanitizes the import() rule only. The pathname
   * rule gets no barrier: on Windows a pathname is "/D:/...", not a native path.
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
  return scanSource(
    ts.createSourceFile(label, text, ts.ScriptTarget.ESNext, true, kind),
    label,
  );
}

const SCANS = FILES.map(scan);

// Guards against a refactor turning both rules into green no-ops.
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

// Exercise the detectors on known-bad source too, or gutted detectors would still pass.
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
      `import { copyFile } from "node:fs/promises";
       const from = new URL("./a.ts", import.meta.url);
       const to = new URL("./b.ts", import.meta.url).pathname;
       await copyFile(from, to);`,
      "pathnameToFs",
    ],
    [
      `import { readFile } from "node:fs/promises";
       let where = "";
       where = new URL("./x.ts", import.meta.url).pathname;
       await readFile(where, "utf8");`,
      "pathnameToFs",
    ],
    [
      `import { readFile as read } from "node:fs/promises";
       await read(new URL("./x.ts", import.meta.url).pathname, "utf8");`,
      "pathnameToFs",
    ],
    [
      `import { createReadStream } from "node:fs";
       createReadStream(new URL("./x.ts", import.meta.url).pathname);`,
      "pathnameToFs",
    ],
    [
      `import { fileURLToPath as toPath } from "node:url";
       const M = toPath(new URL("../src/x.ts", import.meta.url));
       await import(M);`,
      "nativePathImports",
    ],
    [
      `import * as fs from "node:fs";
       fs.readFileSync(new URL("./x.ts", import.meta.url).pathname, "utf8");`,
      "pathnameToFs",
    ],
    [
      `import { readFile } from "node:fs/promises";
       const { pathname } = new URL("./x.ts", import.meta.url);
       await readFile(pathname, "utf8");`,
      "pathnameToFs",
    ],
    [
      `import { readFileSync } from "node:fs";
       const { pathname: where } = new URL("./x.ts", import.meta.url);
       readFileSync(where, "utf8");`,
      "pathnameToFs",
    ],
    [
      `import { readFile } from "node:fs/promises";
       function load(target) { return readFile(target, "utf8"); }
       await load(new URL("./x.ts", import.meta.url).pathname);`,
      "pathnameToFs",
    ],
    [
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

// Silencing a sanitizer and repairing it look identical, so assert both halves.
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
