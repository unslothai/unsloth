// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Builds the one HTML document a React preview runs as. It goes through the same sandboxed frame and
// strict CSP as an HTML page: every script is inline, so nothing here needs eval, blob: or a module.

export type RuntimeScript = { name: string; code: string };

export type CompileDiagnostic = { message: string; line: number; column: number };

// Only `</script` and `<!--` change the tokenizer's state inside a <script>; left as is, the page
// after them is swallowed with no error at all. `\x21`, not `\!`, which breaks /u regexes.
export function escapeInlineScript(code: string): string {
  return code.replace(/<\/(script)/gi, "<\\/$1").replace(/<!--/g, "<\\x21--");
}

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// JSON that is safe inside a <script>: no `<` can end it.
function scriptJson(value: unknown): string {
  return JSON.stringify(value).replace(/</g, "\\u003c");
}

export function usesTailwind(source: string): boolean {
  return /\bclassName\s*[=:]/.test(source);
}

// The module-runner helpers the compiled code calls, backed by the registry the runtime files fill.
// `importedNames` is absent for namespace, `export *` and side-effect imports.
export function previewBootstrap(available: readonly string[]): string {
  return `(() => {
  "use strict";
  const reg = (globalThis.__unslothModules ??= Object.create(null));
  const AVAILABLE = ${scriptJson([...available].sort())};
  const lookup = (id) => {
    if (!Object.prototype.hasOwnProperty.call(reg, id)) {
      throw new Error(JSON.stringify(String(id)) + " isn't available in previews. Available: " + AVAILABLE.join(", "));
    }
    return reg[id];
  };
  window.__unslothPreview = {
    import: async (id, meta) => {
      const mod = lookup(id);
      const names = meta && meta.importedNames;
      if (names) {
        for (const name of names) {
          if (!(name in mod)) {
            throw new SyntaxError("The requested module " + JSON.stringify(id) + " does not provide an export named " + JSON.stringify(name));
          }
        }
      }
      return mod;
    },
    dynamicImport: async () => {
      throw new Error("Dynamic import() isn't supported in previews.");
    },
    exportAll: (exports, source) => {
      if (source === exports || source == null || typeof source !== "object") return;
      for (const key in source) {
        if (key === "default" || key === "__esModule" || key in exports) continue;
        Object.defineProperty(exports, key, { enumerable: true, configurable: true, get: () => source[key] });
      }
    },
    importMeta: (file) => Object.freeze({ url: "preview:///" + file, env: Object.freeze({ MODE: "production", DEV: false, PROD: true, SSR: false }) }),
  };
})();`;
}

// Runs the compiled module, then mounts its default export (or an `App`). React 19's root hooks
// re-dispatch render errors as ErrorEvents: WebKit otherwise reports them as "Script error.".
export function componentScript(code: string, file: string): string {
  return `(async () => {
  "use strict";
  const P = window.__unslothPreview;
  const __vite_ssr_exports__ = Object.create(null);
  const __vite_ssr_import__ = P.import;
  const __vite_ssr_dynamic_import__ = P.dynamicImport;
  const __vite_ssr_exportAll__ = (source) => P.exportAll(__vite_ssr_exports__, source);
  const __vite_ssr_import_meta__ = P.importMeta(${scriptJson(file)});
  const __unsloth_local_app__ = await (async () => {
${code}
;return typeof App === "undefined" ? undefined : App;
  })();
  const Component = __vite_ssr_exports__.default ?? __vite_ssr_exports__.App ?? __unsloth_local_app__;
  const isComponent = typeof Component === "function" || (Component !== null && typeof Component === "object" && "$$typeof" in Component);
  if (!isComponent) {
    throw new Error("The preview needs a default export (or an App component) that is a React component.");
  }
  const React = globalThis.__unslothModules["react"];
  const { createRoot } = globalThis.__unslothModules["react-dom/client"];
  createRoot(document.getElementById("root"), {
    onUncaughtError: (error) => {
      const message = error && error.name ? error.name + ": " + error.message : String(error);
      window.dispatchEvent(new ErrorEvent("error", { error, message: "Uncaught " + message }));
    },
    onCaughtError: (error) => console.error(error),
  }).render(React.createElement(Component));
})();`;
}

const DOCUMENT_HEAD = `<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />`;

const inline = (code: string) => `<script>${escapeInlineScript(code)}</script>`;

export function buildReactPreviewHtml({
  code,
  runtimes,
  tailwind,
  available,
  title,
  file,
}: {
  /** Compiled module-runner code. */
  code: string;
  /** Library files in load order, React first. */
  runtimes: readonly RuntimeScript[];
  /** Tailwind's browser build, or null when the source has no className. */
  tailwind: string | null;
  /** Every specifier a preview can import, for the unknown-import message. */
  available: readonly string[];
  title: string;
  file: string;
}): string {
  return `${DOCUMENT_HEAD}
<title>${escapeHtml(title)}</title>
${tailwind === null ? "" : inline(tailwind)}
</head>
<body>
<div id="root"></div>
${runtimes.map((runtime) => inline(runtime.code)).join("\n")}
${inline(previewBootstrap(available))}
${inline(componentScript(code, file))}
</body>
</html>`;
}

// A document that reports each diagnostic as an error, so the frame's console, banner and Fix
// with model handle compile errors like any other.
export function buildCompileErrorHtml(diagnostics: readonly CompileDiagnostic[], title: string): string {
  const messages = diagnostics.map(({ message, line, column }) => {
    const where = line > 0 ? (column > 0 ? `line ${line}, column ${column}` : `line ${line}`) : "";
    return where ? `Compile error at ${where}: ${message}` : `Compile error: ${message}`;
  });
  return `${DOCUMENT_HEAD}
<title>${escapeHtml(title)}</title>
</head>
<body>
<script>
(() => {
  for (const message of ${scriptJson(messages)}) {
    window.dispatchEvent(new ErrorEvent("error", { message }));
  }
})();
</script>
</body>
</html>`;
}
