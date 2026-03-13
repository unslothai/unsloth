// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { spawnSync } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { parseSync } from "oxc-parser";

const LANG_TO_EXT = {
  js: "js",
  jsx: "jsx",
  ts: "ts",
  tsx: "tsx",
};

const VALIDATION_MODES = new Set(["syntax", "lint", "syntax+lint"]);
const CODE_SHAPES = new Set(["auto", "module", "snippet"]);
const SNIPPET_PREFIX = "(() => {\n";
const SNIPPET_SUFFIX = "\n})();\nexport {};\n";
const OXLINT_SUPPRESSED_RULES = ["no-unused-vars", "no-new-array"];
const TOOL_DIR = dirname(fileURLToPath(import.meta.url));

function mapLang(value) {
  const normalized = String(value || "").trim().toLowerCase();
  if (normalized === "javascript" || normalized === "js") {
    return "js";
  }
  if (normalized === "typescript" || normalized === "ts") {
    return "ts";
  }
  if (normalized === "jsx") {
    return "jsx";
  }
  if (normalized === "tsx") {
    return "tsx";
  }
  return "js";
}

function mapMode(value) {
  const normalized = String(value || "").trim().toLowerCase();
  if (VALIDATION_MODES.has(normalized)) {
    return normalized;
  }
  return "syntax";
}

function mapCodeShape(value) {
  const normalized = String(value || "").trim().toLowerCase();
  if (CODE_SHAPES.has(normalized)) {
    return normalized;
  }
  return "auto";
}

function parseFileIndex(filePath) {
  if (typeof filePath !== "string") {
    return null;
  }
  const match = basename(filePath).match(/^snippet_(\d+)\./);
  if (!match) {
    return null;
  }
  const parsed = Number.parseInt(match[1], 10);
  return Number.isFinite(parsed) ? parsed : null;
}

function toCodeString(code) {
  return typeof code === "string" ? code : String(code ?? "");
}

function makeValidationEntry({ code, index, lang, codeShape }) {
  const source = toCodeString(code);
  if (codeShape === "snippet") {
    return {
      index,
      lang,
      code: `${SNIPPET_PREFIX}${source}${SNIPPET_SUFFIX}`,
      offset: SNIPPET_PREFIX.length,
    };
  }
  return {
    index,
    lang,
    code: source,
    offset: 0,
  };
}

function shiftOffset(value, offset) {
  if (!Number.isInteger(value)) {
    return null;
  }
  const shifted = value - offset;
  return shifted >= 0 ? shifted : null;
}

function remapDiagnosticOffsets(diagnostic, offset) {
  if (!diagnostic || typeof diagnostic !== "object" || offset <= 0) {
    return diagnostic;
  }
  return {
    ...diagnostic,
    labels: Array.isArray(diagnostic.labels)
      ? diagnostic.labels.map((label) => ({
          ...label,
          start: shiftOffset(label.start, offset),
          end: shiftOffset(label.end, offset),
        }))
      : [],
  };
}

function normalizeParserError(error) {
  if (typeof error === "string") {
    return {
      code: null,
      message: error.trim() || "Unknown parser error",
      severity: null,
      labels: [],
      codeframe: null,
    };
  }
  if (!error || typeof error !== "object") {
    return {
      code: null,
      message: "Unknown parser error",
      severity: null,
      labels: [],
      codeframe: null,
    };
  }
  const code = typeof error.code === "string" ? error.code : null;
  const message = String(error.message || error.reason || "").trim() || "Unknown parser error";
  const severity = typeof error.severity === "string" ? error.severity : null;
  const labels = Array.isArray(error.labels)
    ? error.labels.map((label) => ({
        message:
          label && typeof label === "object" && typeof label.message === "string"
            ? label.message
            : null,
        start:
          label && typeof label === "object" && Number.isInteger(label.start)
            ? label.start
            : null,
        end:
          label && typeof label === "object" && Number.isInteger(label.end)
            ? label.end
            : null,
      }))
    : [];
  const codeframe = typeof error.codeframe === "string" ? error.codeframe : null;
  return {
    code,
    message,
    severity,
    labels,
    codeframe,
  };
}

function normalizeLintDiagnostic(diagnostic) {
  if (!diagnostic || typeof diagnostic !== "object") {
    return null;
  }

  const readString = (value) =>
    typeof value === "string" ? value : null;
  const readInt = (value) =>
    Number.isInteger(value) ? value : null;
  const asObject = (value) =>
    value && typeof value === "object" ? value : null;

  const message = String(diagnostic.message || "").trim();
  if (!message) {
    return null;
  }

  const severityRaw = String(diagnostic.severity || "").trim().toLowerCase();
  const severity = severityRaw === "error" ? "error" : "warning";

  const labels = [];
  if (Array.isArray(diagnostic.labels)) {
    for (const label of diagnostic.labels) {
      const labelObj = asObject(label);
      const span = asObject(labelObj?.span);
      const start = readInt(span?.offset);
      const length = readInt(span?.length);
      labels.push({
        message: readString(labelObj?.label),
        start,
        end: start !== null && length !== null ? start + length : null,
      });
    }
  }

  const code = typeof diagnostic.code === "string" ? diagnostic.code : null;
  return {
    code,
    message: code ? `${code}: ${message}` : message,
    severity,
    labels,
    codeframe: null,
  };
}

function makeResult({
  isValid,
  errorCount,
  warningCount = 0,
  message = "",
  severity = null,
  code = null,
  labels = [],
  codeframe = null,
}) {
  return {
    is_valid: Boolean(isValid),
    error_count: Number.isInteger(errorCount) ? errorCount : 0,
    warning_count: Number.isInteger(warningCount) ? warningCount : 0,
    error_message: String(message || ""),
    severity: typeof severity === "string" ? severity : null,
    code: typeof code === "string" ? code : null,
    labels: Array.isArray(labels) ? labels : [],
    codeframe: typeof codeframe === "string" ? codeframe : null,
  };
}

function syntaxResultFromErrors(errors) {
  const first = errors[0] ?? null;
  return makeResult({
    isValid: errors.length === 0,
    errorCount: errors.length,
    warningCount: 0,
    message: errors.slice(0, 3).map((error) => error.message).join(" | "),
    severity: first ? first.severity : null,
    code: first ? first.code : null,
    labels: first ? first.labels : [],
    codeframe: first ? first.codeframe : null,
  });
}

function runSyntaxParse(entry) {
  const ext = LANG_TO_EXT[entry.lang] ?? "js";
  const filename = `snippet_${entry.index}.${ext}`;
  try {
    const parsed = parseSync(filename, entry.code, {
      lang: entry.lang,
      sourceType: "module",
      showSemanticErrors: true,
    });
    const errors = Array.isArray(parsed?.errors)
      ? parsed.errors
          .map(normalizeParserError)
          .filter(Boolean)
          .map((error) => remapDiagnosticOffsets(error, entry.offset))
      : [];
    return errors;
  } catch (error) {
    return [
      remapDiagnosticOffsets(
        normalizeParserError(error),
        entry.offset,
      ),
    ];
  }
}

function pickPreferredErrorList(firstErrors, secondErrors) {
  if (secondErrors.length < firstErrors.length) {
    return secondErrors;
  }
  return firstErrors;
}

function validateSyntaxOne({ code, lang, index, codeShape }) {
  if (codeShape !== "auto") {
    const lintEntry = makeValidationEntry({
      code,
      index,
      lang,
      codeShape,
    });
    const errors = runSyntaxParse(lintEntry);
    return {
      result: syntaxResultFromErrors(errors),
      lintEntry,
    };
  }

  const moduleEntry = makeValidationEntry({
    code,
    index,
    lang,
    codeShape: "module",
  });
  const moduleErrors = runSyntaxParse(moduleEntry);
  if (moduleErrors.length === 0) {
    return {
      result: syntaxResultFromErrors(moduleErrors),
      lintEntry: moduleEntry,
    };
  }

  const snippetEntry = makeValidationEntry({
    code,
    index,
    lang,
    codeShape: "snippet",
  });
  const snippetErrors = runSyntaxParse(snippetEntry);
  if (snippetErrors.length === 0) {
    return {
      result: syntaxResultFromErrors(snippetErrors),
      lintEntry: snippetEntry,
    };
  }

  const chosenErrors = pickPreferredErrorList(moduleErrors, snippetErrors);
  const lintEntry = chosenErrors === snippetErrors ? snippetEntry : moduleEntry;
  return {
    result: syntaxResultFromErrors(chosenErrors),
    lintEntry,
  };
}

function resolveLintEntry({ code, lang, index, codeShape }) {
  if (codeShape !== "auto") {
    return makeValidationEntry({
      code,
      index,
      lang,
      codeShape,
    });
  }

  const moduleEntry = makeValidationEntry({
    code,
    index,
    lang,
    codeShape: "module",
  });
  if (runSyntaxParse(moduleEntry).length === 0) {
    return moduleEntry;
  }

  const snippetEntry = makeValidationEntry({
    code,
    index,
    lang,
    codeShape: "snippet",
  });
  if (runSyntaxParse(snippetEntry).length === 0) {
    return snippetEntry;
  }

  return moduleEntry;
}

function fallbackLintResults(entries, message) {
  return new Map(
    entries.map((entry) => [
      entry.index,
      makeResult({
        isValid: false,
        errorCount: 1,
        warningCount: 0,
        message,
        severity: "error",
      }),
    ]),
  );
}

function runLintBatch(entries) {
  if (entries.length === 0) {
    return new Map();
  }

  const entryByIndex = new Map(entries.map((entry) => [entry.index, entry]));
  const tempDir = mkdtempSync(join(tmpdir(), "oxlint-"));
  try {
    for (const entry of entries) {
      const ext = LANG_TO_EXT[entry.lang] ?? "js";
      const filePath = join(tempDir, `snippet_${entry.index}.${ext}`);
      writeFileSync(filePath, entry.code, "utf8");
    }

    const oxlintBin = join(TOOL_DIR, "node_modules", ".bin", "oxlint");
    const oxlintArgs = [
      ...OXLINT_SUPPRESSED_RULES.flatMap((rule) => ["-A", rule]),
      "--format",
      "json",
      tempDir,
    ];
    const exec = spawnSync(oxlintBin, oxlintArgs, {
      encoding: "utf8",
      cwd: TOOL_DIR,
    });
    if (exec.error) {
      return fallbackLintResults(
        entries,
        `oxlint execution failed: ${exec.error.message}`,
      );
    }
    const stdout = String(exec.stdout || "").trim();
    if (!stdout) {
      const stderr = String(exec.stderr || "").trim();
      return fallbackLintResults(
        entries,
        stderr || "oxlint returned empty output",
      );
    }

    let parsed;
    try {
      parsed = JSON.parse(stdout);
    } catch {
      return fallbackLintResults(entries, "oxlint JSON parse failed");
    }

    const rawDiagnostics = Array.isArray(parsed?.diagnostics)
      ? parsed.diagnostics
      : [];
    const byIndex = new Map();

    for (const diag of rawDiagnostics) {
      const filenameRaw =
        typeof diag?.filename === "string" ? diag.filename : "";
      const filename = filenameRaw.startsWith("file://")
        ? filenameRaw.replace("file://", "")
        : filenameRaw;
      const index = parseFileIndex(filename);
      if (index === null) {
        continue;
      }
      const normalized = normalizeLintDiagnostic(diag);
      if (!normalized) {
        continue;
      }
      const entry = entryByIndex.get(index);
      const remapped = remapDiagnosticOffsets(normalized, entry?.offset ?? 0);
      const list = byIndex.get(index) ?? [];
      list.push(remapped);
      byIndex.set(index, list);
    }

    const results = new Map();
    for (const entry of entries) {
      const diagnostics = byIndex.get(entry.index) ?? [];
      const errorDiagnostics = diagnostics.filter(
        (diag) => diag.severity === "error",
      );
      const warningDiagnostics = diagnostics.filter(
        (diag) => diag.severity !== "error",
      );
      const top = errorDiagnostics[0] ?? warningDiagnostics[0] ?? null;
      const messageSource =
        errorDiagnostics.length > 0 ? errorDiagnostics : warningDiagnostics;
      results.set(
        entry.index,
        makeResult({
          isValid: errorDiagnostics.length === 0,
          errorCount: errorDiagnostics.length,
          warningCount: warningDiagnostics.length,
          message: messageSource
            .slice(0, 3)
            .map((diag) => diag.message)
            .join(" | "),
          severity: top ? top.severity : null,
          code: top ? top.code : null,
          labels: top ? top.labels : [],
          codeframe: top ? top.codeframe : null,
        }),
      );
    }
    return results;
  } catch (error) {
    return fallbackLintResults(entries, `oxlint execution failed: ${error}`);
  } finally {
    rmSync(tempDir, { recursive: true, force: true });
  }
}

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

function runValidation({ codes, lang, mode, codeShape }) {
  if (mode === "syntax") {
    return codes.map((code, index) =>
      validateSyntaxOne({ code, lang, index, codeShape }).result,
    );
  }

  if (mode === "lint") {
    const entries = codes.map((code, index) =>
      resolveLintEntry({ code, lang, index, codeShape }),
    );
    const lintMap = runLintBatch(entries);
    return entries.map(
      (entry) =>
        lintMap.get(entry.index) ??
        makeResult({
          isValid: true,
          errorCount: 0,
          warningCount: 0,
        }),
    );
  }

  const syntaxRuns = codes.map((code, index) =>
    validateSyntaxOne({ code, lang, index, codeShape }),
  );
  const lintTargets = syntaxRuns
    .filter((run) => run.result.is_valid === true)
    .map((run) => run.lintEntry);
  const lintMap = runLintBatch(lintTargets);

  return syntaxRuns.map((run) => {
    if (run.result.is_valid !== true) {
      return run.result;
    }
    return (
      lintMap.get(run.lintEntry.index) ??
      makeResult({
        isValid: true,
        errorCount: 0,
        warningCount: 0,
      })
    );
  });
}

async function main() {
  const raw = await readStdin();
  let payload;
  try {
    payload = JSON.parse(raw || "{}");
  } catch {
    process.stdout.write(
      JSON.stringify([
        makeResult({
          isValid: false,
          errorCount: 1,
          warningCount: 0,
          message: "Invalid JSON payload",
          severity: "error",
        }),
      ]),
    );
    return;
  }

  const lang = mapLang(payload?.lang);
  const mode = mapMode(payload?.mode);
  const codeShape = mapCodeShape(payload?.code_shape);
  const codes = Array.isArray(payload?.codes) ? payload.codes : [];
  const out = runValidation({ codes, lang, mode, codeShape });
  process.stdout.write(JSON.stringify(out));
}

main().catch((error) => {
  process.stderr.write(String(error?.stack || error));
  process.exit(1);
});
