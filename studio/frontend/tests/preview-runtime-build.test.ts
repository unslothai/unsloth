// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The React-preview runtime files, built for real from node_modules. These pin what the preview
 * loader relies on: the file and specifier names, sizes it can check a response against, versions
 * that match the lockfile, one React per page, and no string evaluation under the preview CSP.
 */

import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import vm from "node:vm";

import {
  MANIFEST_MODULE,
  buildPreviewRuntime,
  checkNoEval,
  previewRuntime,
} from "../vite-preview-runtime.ts";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const { files, manifest } = await buildPreviewRuntime(ROOT);
const code = (name: string): string => {
  const file = files.find((f) => f.name === name);
  assert.ok(file, `no ${name} file`);
  return file.code;
};

test("exactly the five runtime files, hashed, under assets/preview-runtime", () => {
  assert.deepEqual(
    files.map((f) => f.name),
    ["react", "lucide-react", "recharts", "motion", "tailwind"],
  );
  for (const f of files) {
    const hash = createHash("sha256").update(f.code).digest("hex").slice(0, 10);
    assert.equal(f.fileName, `assets/preview-runtime/${f.name}.${hash}.js`);
    assert.equal(manifest.files[f.name]?.url, f.fileName);
    // The loader compares this against the response body, so it is UTF-8 bytes, not UTF-16 units.
    assert.equal(manifest.files[f.name]?.bytes, Buffer.byteLength(f.code));
  }
  assert.deepEqual(Object.keys(manifest.files).sort(), [
    "lucide-react",
    "motion",
    "react",
    "recharts",
    "tailwind",
  ]);
  assert.equal(manifest.tailwind, "tailwind");
});

test("the files follow Vite's assetsDir", async () => {
  const other = await buildPreviewRuntime(ROOT, "static");
  for (const f of other.files) {
    assert.match(
      f.fileName,
      /^static\/preview-runtime\/[\w-]+\.[0-9a-f]{10}\.js$/,
    );
  }
});

test("every importable specifier maps to the file that registers it", () => {
  assert.deepEqual(manifest.modules, {
    react: "react",
    "react/jsx-runtime": "react",
    "react/jsx-dev-runtime": "react",
    "react-dom": "react",
    "react-dom/client": "react",
    "lucide-react": "lucide-react",
    recharts: "recharts",
    motion: "motion",
    "motion/react": "motion",
    "framer-motion": "motion",
  });
});

test("versions are the locked ones, and Tailwind's browser build matches the app's Tailwind", () => {
  const lock = JSON.parse(
    readFileSync(join(ROOT, "package-lock.json"), "utf8"),
  ) as { packages: Record<string, { version: string }> };
  const locked = (name: string) =>
    lock.packages[`node_modules/${name}`]?.version;
  assert.equal(manifest.files.react?.version, locked("react"));
  assert.equal(locked("react-dom"), locked("react"));
  assert.equal(manifest.files["lucide-react"]?.version, locked("lucide-react"));
  assert.equal(manifest.files.recharts?.version, locked("recharts"));
  assert.equal(manifest.files.motion?.version, locked("motion"));
  assert.equal(
    manifest.files.tailwind?.version,
    locked("@tailwindcss/browser"),
  );
  assert.equal(locked("@tailwindcss/browser"), locked("tailwindcss"));
});

test("each file carries one licence line naming what it bundles, and no other legal comments", () => {
  for (const f of files) {
    const [banner] = f.code.split("\n", 1);
    assert.match(banner ?? "", /^\/\*! .+@\d+\.\d+\.\d+ .*\*\/$/, f.name);
    assert.equal(f.code.split("/*!").length - 1, 1, f.name);
  }
  assert.ok(code("react").startsWith("/*! react-dom@"));
  assert.ok(code("recharts").includes("recharts@3."));
  assert.ok(code("tailwind").startsWith("/*! @tailwindcss/browser@"));
});

test("only the react file has React in it", () => {
  const internals =
    "__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE";
  assert.ok(code("react").includes(internals));
  for (const name of ["lucide-react", "recharts", "motion", "tailwind"]) {
    assert.ok(!code(name).includes(internals), `${name} bundles React`);
    assert.ok(!code(name).includes("unstable_scheduleCallback"), name);
  }
  // Built for production: no development-only React warnings.
  assert.ok(!code("react").includes("Invalid hook call"));
});

test("no file evaluates strings, beyond recharts' two global fallbacks", () => {
  for (const f of files) {
    assert.doesNotThrow(() => checkNoEval(f.name, f.code), f.name);
  }
  assert.equal(code("recharts").match(/Function\(`return this`\)/g)?.length, 2);
  assert.throws(() => checkNoEval("react", "a();eval(x)"), /evaluates code/);
  assert.throws(
    () => checkNoEval("react", "new Function('a')"),
    /evaluates code/,
  );
  assert.throws(
    () => checkNoEval("react", "x=Function ('a')"),
    /evaluates code/,
  );
  assert.throws(
    () => checkNoEval("react", "x=Function(`return this`)()"),
    /expected 0/,
  );
  assert.throws(
    () =>
      checkNoEval("recharts", `${code("recharts")};Function("return this")`),
    /expected 2 .* found 3/,
  );
  // Property access and longer names are not the global Function.
  assert.doesNotThrow(() =>
    checkNoEval("react", "a.Function(1);isFunction(2)"),
  );
});

/** Runs runtime files in order in a fresh global, as script tags would. */
function run(names: string[]): Record<string, Record<string, unknown>> {
  const context = vm.createContext({
    console,
    setTimeout,
    clearTimeout,
    queueMicrotask,
  });
  for (const name of names) {
    vm.runInContext(code(name), context, { filename: `${name}.js` });
  }
  return vm.runInContext("globalThis.__unslothModules", context);
}

test("the files register module namespaces, React first and shared", () => {
  const r = run(["react", "lucide-react", "recharts", "motion"]);
  assert.deepEqual(Object.keys(r).sort(), Object.keys(manifest.modules).sort());
  const react = r.react as Record<string, unknown>;
  const reactDefault = react.default as Record<string, unknown>;
  // CommonJS packages: `default` is the exports object and its keys are named exports too.
  assert.equal(typeof react.useState, "function");
  assert.equal(react.useState, reactDefault.useState);
  assert.equal(react.version, manifest.files.react?.version);
  assert.equal(typeof r["react-dom/client"]?.createRoot, "function");
  assert.equal(typeof r["react-dom"]?.createPortal, "function");
  assert.equal(typeof r["react/jsx-runtime"]?.jsx, "function");
  assert.equal(r["react/jsx-runtime"]?.Fragment, react.Fragment);
  // Icons and charts are forwardRef components, read through the one shared React.
  const forwardRef = (
    react.forwardRef as (f: () => null) => { $$typeof: symbol }
  )(() => null).$$typeof;
  assert.equal(
    (r["lucide-react"]?.Camera as { $$typeof: symbol }).$$typeof,
    forwardRef,
  );
  assert.equal(
    (r.recharts?.LineChart as { $$typeof: symbol }).$$typeof,
    forwardRef,
  );
  assert.equal(typeof r["motion/react"]?.motion, "function");
  assert.equal(r["framer-motion"]?.motion, r["motion/react"]?.motion);
  assert.equal(typeof r.motion?.animate, "function");
});

test("a library loaded before React says so", () => {
  assert.throws(() => run(["lucide-react"]), /"react" is not loaded/);
});

test("the plugin serves the manifest module and the files, and 404s anything else", async () => {
  const plugin = previewRuntime();
  type Hook = (...args: unknown[]) => unknown;
  const call = (hook: unknown, ...args: unknown[]) =>
    (typeof hook === "function" ? hook : (hook as { handler: Hook }).handler)(
      ...args,
    );
  call(plugin.configResolved, {
    root: ROOT,
    base: "/",
    command: "serve",
    build: { assetsDir: "assets" },
  });
  const resolved = call(plugin.resolveId, MANIFEST_MODULE) as string;
  assert.ok(resolved.startsWith("\0"));
  const source = (await call(plugin.load, resolved)) as string;
  assert.deepEqual(
    JSON.parse(source.replace(/^export default /, "").replace(/;$/, "")),
    manifest,
  );

  let prefix = "";
  let handler: Hook | undefined;
  call(plugin.configureServer, {
    middlewares: {
      use: (p: string, h: Hook) => {
        prefix = p;
        handler = h;
      },
    },
  });
  assert.equal(prefix, "/assets/preview-runtime/");
  const get = (url: string) =>
    new Promise<{ status: number; type: unknown; body: string }>(
      (resolve, reject) => {
        const headers: Record<string, unknown> = {};
        const res = {
          statusCode: 200,
          setHeader: (k: string, v: unknown) => {
            headers[k.toLowerCase()] = v;
          },
          end: (body = "") =>
            resolve({
              status: res.statusCode,
              type: headers["content-type"],
              body,
            }),
        };
        handler?.({ url }, res, reject);
      },
    );
  const react = manifest.files.react?.url ?? "";
  const ok = await get(`/${react.slice(prefix.length - 1)}?v=1`);
  assert.equal(ok.status, 200);
  assert.match(String(ok.type), /^text\/javascript/);
  assert.equal(ok.body, code("react"));
  assert.equal((await get("/react.0000000000.js")).status, 404);
  assert.equal((await get("/manifest.json")).status, 404);
  assert.equal((await get("/../index.html")).status, 404);
});
