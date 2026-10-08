// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import vm from "node:vm";

import {
  buildCompileErrorHtml,
  buildReactPreviewHtml,
  componentScript,
  escapeInlineScript,
  previewBootstrap,
  usesTailwind,
} from "../src/features/chat/artifacts/react-preview/build-html.ts";

const AVAILABLE = ["react", "react/jsx-runtime", "lucide-react"];

class FakeErrorEvent {
  type: string;
  init: { message: string };
  constructor(type: string, init: { message: string }) {
    this.type = type;
    this.init = init;
  }
}

// A frame stand-in: the registry the runtime files fill, a root, and what the page reports.
function frame(registry: Record<string, unknown> = {}) {
  const errors: string[] = [];
  const rendered: unknown[] = [];
  const React = { createElement: (type: unknown) => ({ type }) };
  const context = vm.createContext({
    console: { error: (error: unknown) => errors.push(`console: ${String(error)}`) },
    ErrorEvent: FakeErrorEvent,
    __unslothModules: {
      react: { default: React, ...React },
      "react-dom/client": {
        createRoot: () => ({ render: (element: unknown) => rendered.push(element) }),
      },
      ...registry,
    },
    document: { getElementById: (id: string) => ({ id }) },
  });
  context.window = context;
  context.window.dispatchEvent = (event: { init: { message: string } }) => errors.push(event.init.message);
  vm.runInContext(previewBootstrap(AVAILABLE), context);
  const run = async (code: string) => {
    const promise = vm.runInContext(componentScript(code, "App.tsx"), context) as Promise<void>;
    await promise.catch((error: Error) => errors.push(`${error.name}: ${error.message}`));
  };
  return { run, errors, rendered };
}

// What moduleRunnerTransformSync 0.131.0 emits for `export default function App` with one import.
const EXPORT_DEFAULT = `Object.defineProperty(__vite_ssr_exports__, "default", { enumerable: true, configurable: true, get() { return App; } });
const __vite_ssr_import_0__ = await __vite_ssr_import__("react/jsx-runtime", { importedNames: ["jsx"] });
function App() { return __vite_ssr_import_0__.jsx("p", {}); }`;

test("the default export mounts", async () => {
  const page = frame({ "react/jsx-runtime": { jsx: () => null } });
  await page.run(EXPORT_DEFAULT);
  assert.deepEqual(page.errors, []);
  assert.equal(page.rendered.length, 1);
});

test("an App that isn't exported still mounts", async () => {
  const page = frame();
  await page.run("const App = () => null;");
  assert.deepEqual(page.errors, []);
  assert.equal(page.rendered.length, 1);
});

test("an import no library provides names the module and lists what is available", async () => {
  const page = frame();
  await page.run('const __vite_ssr_import_0__ = await __vite_ssr_import__("axios", { importedNames: ["default"] });');
  assert.deepEqual(page.errors, [
    'Error: "axios" isn\'t available in previews. Available: lucide-react, react, react/jsx-runtime',
  ]);
  assert.equal(page.rendered.length, 0);
});

test("a missing export (a renamed icon) names the export", async () => {
  const page = frame({ "lucide-react": { Heart: () => null } });
  await page.run('const __vite_ssr_import_0__ = await __vite_ssr_import__("lucide-react", { importedNames: ["Heart", "Hart"] });');
  assert.deepEqual(page.errors, [
    'SyntaxError: The requested module "lucide-react" does not provide an export named "Hart"',
  ]);
});

test("export * copies everything but default, and keeps what is already exported", async () => {
  const page = frame({ lib: { default: 1, a: 2, App: () => null } });
  await page.run(`Object.defineProperty(__vite_ssr_exports__, "a", { enumerable: true, configurable: true, get() { return 9; } });
__vite_ssr_exportAll__(await __vite_ssr_import__("lib"));`);
  assert.deepEqual(page.errors, []);
  // App came through export *, so it mounted.
  assert.equal(page.rendered.length, 1);
});

test("dynamic import is refused", async () => {
  const page = frame();
  await page.run('await __vite_ssr_dynamic_import__("react");');
  assert.deepEqual(page.errors, ["Error: Dynamic import() isn't supported in previews."]);
});

test("a module with nothing to mount says what the preview needs", async () => {
  const page = frame();
  await page.run("const x = 1;");
  assert.deepEqual(page.errors, [
    "Error: The preview needs a default export (or an App component) that is a React component.",
  ]);
});

test("only the given libraries are inlined, React first, and Tailwind only when asked", () => {
  const html = buildReactPreviewHtml({
    code: "/*component*/",
    runtimes: [
      { name: "react", code: "/*react-lib*/" },
      { name: "lucide-react", code: "/*lucide-lib*/" },
    ],
    tailwind: null,
    available: AVAILABLE,
    title: "Dash <board>",
    file: "App.tsx",
  });
  assert.ok(html.indexOf("/*react-lib*/") < html.indexOf("/*lucide-lib*/"));
  assert.ok(html.indexOf("/*lucide-lib*/") < html.indexOf("/*component*/"));
  assert.doesNotMatch(html, /recharts-lib|tailwind-lib/);
  assert.match(html, /<title>Dash &lt;board&gt;<\/title>/);
  const withTailwind = buildReactPreviewHtml({
    code: "",
    runtimes: [],
    tailwind: "/*tailwind-lib*/",
    available: AVAILABLE,
    title: "t",
    file: "App.tsx",
  });
  assert.ok(withTailwind.indexOf("/*tailwind-lib*/") < withTailwind.indexOf("</head>"));
});

test("usesTailwind looks for className", () => {
  assert.equal(usesTailwind('<div className="p-4" />'), true);
  assert.equal(usesTailwind("cn({ className: x })"), true);
  assert.equal(usesTailwind("<div style={{ padding: 4 }} />"), false);
});

test("code can't close its script or open an HTML comment", () => {
  const code = 'const s = "</script><script>alert(1)</script>"; const c = "<!-- x";';
  const escaped = escapeInlineScript(code);
  assert.doesNotMatch(escaped, /<\/script|<!--/i);
  // Still the same JavaScript.
  assert.equal(vm.runInNewContext(`${escaped}; s + c`), "</script><script>alert(1)</script><!-- x");
  const html = buildReactPreviewHtml({
    code,
    runtimes: [{ name: "react", code: "/*</SCRIPT>*/" }],
    tailwind: null,
    available: AVAILABLE,
    title: "t",
    file: "App.tsx",
  });
  // Three scripts (React, the bootstrap, the component), so three closing tags and no more.
  assert.equal(html.match(/<\/script/gi)?.length, 3);
  assert.doesNotMatch(html, /<!--/);
});

test("compile errors become error events with the original line and column", () => {
  const html = buildCompileErrorHtml(
    [
      { message: "Expected `;` but found `</script>`", line: 3, column: 7 },
      { message: "Unexpected token", line: 0, column: 0 },
    ],
    "App.tsx",
  );
  const errors: string[] = [];
  const script = /<script\b[^>]*>([\s\S]*?)<\/script\s*>/i.exec(html)?.[1] ?? "";
  vm.runInNewContext(script, {
    ErrorEvent: FakeErrorEvent,
    window: { dispatchEvent: (event: { init: { message: string } }) => errors.push(event.init.message) },
  });
  assert.deepEqual(errors, [
    "Compile error at line 3, column 7: Expected `;` but found `</script>`",
    "Compile error: Unexpected token",
  ]);
  assert.equal(html.match(/<\/script\s*>/gi)?.length, 1);
});
