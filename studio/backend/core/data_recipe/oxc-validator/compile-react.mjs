// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Compiles one chat React component for the preview frame: TSX/JSX to plain JS,
// then to the module-runner shape the frame's bootstrap links. The source is
// only ever parsed and printed here, never evaluated.

const DYNAMIC_IMPORT_MESSAGE =
  "Dynamic import() isn't supported in previews; import the module at the top of the file.";
const MAX_DIAGNOSTICS = 20;
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

// oxc 0.131.0 reports label offsets in UTF-8 bytes; the UI counts columns in
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

function compile(oxc, source, lang) {
  const transformed = oxc.transformSync(`App.${lang}`, source, {
    lang,
    sourceType: "module",
    jsx: { runtime: "automatic", development: false },
  });
  const transformErrors = (transformed.errors || []).filter(isFatal);
  if (transformErrors.length > 0) {
    return errorResult(transformErrors, Buffer.from(source, "utf8"));
  }

  // Offsets here point into the transformed code, not the user's source.
  const linked = oxc.moduleRunnerTransformSync("App.js", transformed.code);
  const linkErrors = (linked.errors || []).filter(isFatal);
  if (linkErrors.length > 0) {
    return errorResult(linkErrors, null);
  }
  if (Array.isArray(linked.dynamicDeps) && linked.dynamicDeps.length > 0) {
    return {
      status: "error",
      diagnostics: [{ message: DYNAMIC_IMPORT_MESSAGE, line: 0, column: 0 }],
    };
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
  try {
    oxc = await import("oxc-transform");
  } catch {
    write({ status: "unavailable", reason: "transform_missing" });
    return;
  }
  write(compile(oxc, source, lang));
}

main().catch((error) => {
  process.stderr.write(String(error?.stack || error));
  process.exit(1);
});
