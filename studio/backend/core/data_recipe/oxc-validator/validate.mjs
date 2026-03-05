import { parseSync } from "oxc-parser";

const LANG_TO_EXT = {
  js: "js",
  jsx: "jsx",
  ts: "ts",
  tsx: "tsx",
};

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

function normalizeError(error) {
  if (typeof error === "string") {
    return {
      message: error.trim() || "Unknown OXC error",
      severity: null,
      labels: [],
      codeframe: null,
    };
  }
  if (!error || typeof error !== "object") {
    return {
      message: "Unknown OXC error",
      severity: null,
      labels: [],
      codeframe: null,
    };
  }
  const message = String(error.message || error.reason || "").trim() || "Unknown OXC error";
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

function validateOne({ code, lang, index }) {
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
      ? parsed.errors.map(normalizeError).filter(Boolean)
      : [];
    const first = errors[0] ?? null;
    return {
      is_valid: errors.length === 0,
      error_count: errors.length,
      error_message: errors.slice(0, 3).map((error) => error.message).join(" | "),
      severity: first ? first.severity : null,
      labels: first ? first.labels : [],
      codeframe: first ? first.codeframe : null,
    };
  } catch (error) {
    const normalized = normalizeError(error);
    return {
      is_valid: false,
      error_count: 1,
      error_message: normalized.message,
      severity: normalized.severity,
      labels: normalized.labels,
      codeframe: normalized.codeframe,
    };
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

async function main() {
  const raw = await readStdin();
  let payload;
  try {
    payload = JSON.parse(raw || "{}");
  } catch {
    process.stdout.write(
      JSON.stringify([
        {
          is_valid: false,
          error_count: 1,
          error_message: "Invalid JSON payload",
          severity: null,
          labels: [],
          codeframe: null,
        },
      ]),
    );
    return;
  }

  const lang = mapLang(payload?.lang);
  const codes = Array.isArray(payload?.codes) ? payload.codes : [];
  const out = codes.map((code, index) => validateOne({ code, lang, index }));
  process.stdout.write(JSON.stringify(out));
}

main().catch((error) => {
  process.stderr.write(String(error?.stack || error));
  process.exit(1);
});
