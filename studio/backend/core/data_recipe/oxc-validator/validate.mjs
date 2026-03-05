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

function normalizeParserError(error) {
  if (typeof error === "string") {
    return {
      message: error.trim() || "Unknown parser error",
      severity: null,
      labels: [],
      codeframe: null,
    };
  }
  if (!error || typeof error !== "object") {
    return {
      message: "Unknown parser error",
      severity: null,
      labels: [],
      codeframe: null,
    };
  }
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
  const message = String(diagnostic.message || "").trim();
  if (!message) {
    return null;
  }
  const severityRaw = String(diagnostic.severity || "").trim().toLowerCase();
  const severity = severityRaw === "error" ? "error" : "warning";
  const labels = Array.isArray(diagnostic.labels)
    ? diagnostic.labels.map((label) => {
        const span = label && typeof label === "object" ? label.span : null;
        const start =
          span && typeof span === "object" && Number.isInteger(span.offset)
            ? span.offset
            : null;
        const length =
          span && typeof span === "object" && Number.isInteger(span.length)
            ? span.length
            : null;
        return {
          message:
            label && typeof label === "object" && typeof label.label === "string"
              ? label.label
              : null,
          start,
          end:
            start !== null && length !== null
              ? start + length
              : null,
        };
      })
    : [];
  const code = typeof diagnostic.code === "string" ? diagnostic.code : null;
  return {
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
  labels = [],
  codeframe = null,
}) {
  return {
    is_valid: Boolean(isValid),
    error_count: Number.isInteger(errorCount) ? errorCount : 0,
    warning_count: Number.isInteger(warningCount) ? warningCount : 0,
    error_message: String(message || ""),
    severity: typeof severity === "string" ? severity : null,
    labels: Array.isArray(labels) ? labels : [],
    codeframe: typeof codeframe === "string" ? codeframe : null,
  };
}

function validateSyntaxOne({ code, lang, index }) {
  const ext = LANG_TO_EXT[lang] ?? "js";
  const filename = `snippet_${index}.${ext}`;
  const source = typeof code === "string" ? code : String(code ?? "");

  try {
    const parsed = parseSync(filename, source, {
      lang,
      sourceType: "module",
      showSemanticErrors: true,
    });
    const errors = Array.isArray(parsed?.errors)
      ? parsed.errors.map(normalizeParserError).filter(Boolean)
      : [];
    const first = errors[0] ?? null;
    return makeResult({
      isValid: errors.length === 0,
      errorCount: errors.length,
      warningCount: 0,
      message: errors.slice(0, 3).map((error) => error.message).join(" | "),
      severity: first ? first.severity : null,
      labels: first ? first.labels : [],
      codeframe: first ? first.codeframe : null,
    });
  } catch (error) {
    const normalized = normalizeParserError(error);
    return makeResult({
      isValid: false,
      errorCount: 1,
      warningCount: 0,
      message: normalized.message,
      severity: normalized.severity,
      labels: normalized.labels,
      codeframe: normalized.codeframe,
    });
  }
}

function fallbackLintResults(indexedCodes, message) {
  return new Map(
    indexedCodes.map((item) => [
      item.index,
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

function runLintBatch(indexedCodes, lang) {
  if (indexedCodes.length === 0) {
    return new Map();
  }

  const ext = LANG_TO_EXT[lang] ?? "js";
  const tempDir = mkdtempSync(join(tmpdir(), "oxlint-"));
  try {
    for (const item of indexedCodes) {
      const filePath = join(tempDir, `snippet_${item.index}.${ext}`);
      const source = typeof item.code === "string" ? item.code : String(item.code ?? "");
      writeFileSync(filePath, source, "utf8");
    }

    const oxlintBin = join(TOOL_DIR, "node_modules", ".bin", "oxlint");
    const exec = spawnSync(oxlintBin, ["--format", "json", tempDir], {
      encoding: "utf8",
      cwd: TOOL_DIR,
    });
    if (exec.error) {
      return fallbackLintResults(
        indexedCodes,
        `oxlint execution failed: ${exec.error.message}`,
      );
    }
    const stdout = String(exec.stdout || "").trim();
    if (!stdout) {
      const stderr = String(exec.stderr || "").trim();
      return fallbackLintResults(
        indexedCodes,
        stderr || "oxlint returned empty output",
      );
    }

    let parsed;
    try {
      parsed = JSON.parse(stdout);
    } catch {
      return fallbackLintResults(indexedCodes, "oxlint JSON parse failed");
    }

    const rawDiagnostics = Array.isArray(parsed?.diagnostics)
      ? parsed.diagnostics
      : [];
    const byIndex = new Map();

    for (const diag of rawDiagnostics) {
      const normalized = normalizeLintDiagnostic(diag);
      if (!normalized) {
        continue;
      }
      const filenameRaw =
        typeof diag?.filename === "string" ? diag.filename : "";
      const filename = filenameRaw.startsWith("file://")
        ? filenameRaw.replace("file://", "")
        : filenameRaw;
      const index = parseFileIndex(filename);
      if (index === null) {
        continue;
      }
      const list = byIndex.get(index) ?? [];
      list.push(normalized);
      byIndex.set(index, list);
    }

    const results = new Map();
    for (const item of indexedCodes) {
      const diagnostics = byIndex.get(item.index) ?? [];
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
        item.index,
        makeResult({
          isValid: errorDiagnostics.length === 0,
          errorCount: errorDiagnostics.length,
          warningCount: warningDiagnostics.length,
          message: messageSource
            .slice(0, 3)
            .map((diag) => diag.message)
            .join(" | "),
          severity: top ? top.severity : null,
          labels: top ? top.labels : [],
          codeframe: top ? top.codeframe : null,
        }),
      );
    }
    return results;
  } catch (error) {
    return fallbackLintResults(indexedCodes, `oxlint execution failed: ${error}`);
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

function runValidation({ codes, lang, mode }) {
  if (mode === "syntax") {
    return codes.map((code, index) => validateSyntaxOne({ code, lang, index }));
  }
  if (mode === "lint") {
    const indexedCodes = codes.map((code, index) => ({ index, code }));
    const lintMap = runLintBatch(indexedCodes, lang);
    return indexedCodes.map(
      (item) =>
        lintMap.get(item.index) ??
        makeResult({
          isValid: true,
          errorCount: 0,
          warningCount: 0,
        }),
    );
  }

  const syntaxResults = codes.map((code, index) =>
    validateSyntaxOne({ code, lang, index }),
  );
  const lintTargets = codes
    .map((code, index) => ({ index, code }))
    .filter((item) => syntaxResults[item.index]?.is_valid === true);
  const lintMap = runLintBatch(lintTargets, lang);

  return syntaxResults.map((syntaxResult, index) => {
    if (syntaxResult.is_valid !== true) {
      return syntaxResult;
    }
    return (
      lintMap.get(index) ??
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
  const codes = Array.isArray(payload?.codes) ? payload.codes : [];
  const out = runValidation({ codes, lang, mode });
  process.stdout.write(JSON.stringify(out));
}

main().catch((error) => {
  process.stderr.write(String(error?.stack || error));
  process.exit(1);
});
