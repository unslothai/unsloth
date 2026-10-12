// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Compiles one chat React component for the preview frame: TSX/JSX to plain JS,
// then to the module-runner shape the frame's bootstrap links. The source is
// only ever parsed and printed here, never evaluated.

const DYNAMIC_IMPORT_MESSAGE =
  "Dynamic import() isn't supported in previews; import the module at the top of the file.";
const MAX_DIAGNOSTICS = 20;
// Linking rewrites every use of an imported binding to its imported name, so a long name
// under a short alias made 256 KB of source into 480 MB of output. Real exports are short.
const MAX_IMPORT_NAME_CHARS = 64;
// Under the backend's 4 MB stdout cap, so an oversized component reads as a compile error.
const MAX_CODE_CHARS = 3 * 1024 * 1024;
const TOO_LARGE_MESSAGE = "This component compiles to too much code to preview.";
const LINE_BREAK_RE = /\r\n|[\n\r\u2028\u2029]/;

function readStdin() {
  return new Promise((resolve, reject) => {
    let data = "";
    process.stdin.setEncoding("utf8");
    process.stdin.on("data", (chunk) => {
      data += chunk;
    });
    process.stdin.on("end", () => resolve(data));
    process.stdin.on("error", (error) => reject(error));
  });
}

function write(result) {
  process.stdout.write(`${JSON.stringify(result)}\n`);
}

// oxc reports label offsets in UTF-8 bytes; the UI counts columns in
// UTF-16 code units, like the editor and the browser do.
function lineColumn(bytes, offset) {
  if (!bytes || !Number.isInteger(offset) || offset < 0) {
    return { line: 0, column: 0 };
  }
  const before = bytes.subarray(0, Math.min(offset, bytes.length)).toString("utf8");
  const lines = before.split(LINE_BREAK_RE);
  return { line: lines.length, column: lines[lines.length - 1].length + 1 };
}

function isFatal(error) {
  return error?.severity !== "Warning" && error?.severity !== "Advice";
}

function toDiagnostic(error, bytes) {
  const message = String(error?.message || "").trim() || "Unknown compile error";
  const label = Array.isArray(error?.labels)
    ? error.labels.find((item) => Number.isInteger(item?.start))
    : undefined;
  return { message, ...lineColumn(bytes, label?.start) };
}

function errorResult(errors, bytes) {
  return {
    status: "error",
    diagnostics: errors
      .slice(0, MAX_DIAGNOSTICS)
      .map((error) => toDiagnostic(error, bytes)),
  };
}

function errorMessage(message) {
  return { status: "error", diagnostics: [{ message, line: 0, column: 0 }] };
}

// The module-level problems linking cannot report, as a message, or null.
function moduleProblem(parser, code) {
  const { module } = parser.parseSync("App.js", code, { sourceType: "module" });
  // Every import(), not just the string-literal ones linking lists in dynamicDeps.
  if (module.dynamicImports.length > 0) return DYNAMIC_IMPORT_MESSAGE;
  for (const entry of module.staticImports.flatMap((item) => item.entries)) {
    const name = entry.importName?.name;
    if (typeof name === "string" && name.length > MAX_IMPORT_NAME_CHARS) {
      return `Imported name "${name.slice(0, 40)}..." is longer than ${MAX_IMPORT_NAME_CHARS} characters.`;
    }
  }
  return null;
}

function transform(oxc, source, lang) {
  const transformed = oxc.transformSync(`App.${lang}`, source, {
    lang,
    sourceType: "module",
    jsx: { runtime: "automatic", development: false },
  });
  return { code: transformed.code, errors: (transformed.errors || []).filter(isFatal) };
}

function compile(oxc, parser, source, lang) {
  let transformed = transform(oxc, source, lang);
  if (transformed.errors.length > 0 && lang === "tsx") {
    // ```ts fences arrive as tsx, where `<T>(x: T) => x` and `<T>value` read as JSX.
    const asTs = transform(oxc, source, "ts");
    if (asTs.errors.length === 0) transformed = asTs;
  }
  if (transformed.errors.length > 0) {
    return errorResult(transformed.errors, Buffer.from(source, "utf8"));
  }

  const problem = moduleProblem(parser, transformed.code);
  if (problem !== null) return errorMessage(problem);

  // Offsets here point into the transformed code, not the user's source.
  const linked = oxc.moduleRunnerTransformSync("App.js", transformed.code);
  const linkErrors = (linked.errors || []).filter(isFatal);
  if (linkErrors.length > 0) {
    return errorResult(linkErrors, null);
  }
  if (linked.code.length > MAX_CODE_CHARS) {
    return errorMessage(TOO_LARGE_MESSAGE);
  }
  return { status: "ok", code: linked.code, deps: linked.deps || [] };
}

async function main() {
  const payload = JSON.parse((await readStdin()) || "{}");
  const source = payload?.source;
  if (typeof source !== "string") {
    throw new TypeError("source must be a string");
  }
  const lang = payload?.lang === "jsx" ? "jsx" : "tsx";

  let oxc;
  let parser;
  try {
    oxc = await import("oxc-transform");
    parser = await import("oxc-parser");
  } catch {
    write({ status: "unavailable", reason: "transform_missing" });
    return;
  }
  write(compile(oxc, parser, source, lang));
}

main().catch((error) => {
  process.stderr.write(String(error?.stack || error));
  process.exit(1);
});
