// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// studiobench attribution build. Identical to `studio/frontend/vite.config.ts` in every respect
// that affects what the app DOES, differing only in what it lets us SEE: `react-dom/client`
// aliased to `react-dom/profiling` (the only build where React's `<Profiler>` `onRender` exists),
// hidden sourcemaps, and minifier name preservation.
// The resulting dist is injected into a REAL Unsloth install through `unsloth studio --frontend
// <dir>`, so backend, auth and storage stay real and only the frontend bundle is swapped: the
// failure this tool exists to correct was a fixture that did not run the real path.
// WHAT THIS BUILD CANNOT FIX: React's production and profiling dists are minified by React's own
// release pipeline before Vite sees them, so name preservation and sourcemaps recover OUR names
// and nothing inside `react-dom` (the original identifier really is `Zk`). That is the job of the
// symbol bridge in `analysis/symbols.py`.
// That is `tests/studio/studiobench/analysis/symbols.py`.

import { createRequire } from "node:module";
import path from "node:path";

const FRONTEND_ROOT = path.resolve(__dirname, "../../../../studio/frontend");

// The plugins are resolved from the FRONTEND package, not this directory. Vite bundles a
// `--config` file with bare imports left external and loads it under the config file's own path,
// so Node starts its `node_modules` walk here and climbs to the repository root, where nothing
// has a `node_modules`: only `studio/frontend/node_modules` does. With plain top-level imports
// this config could not be loaded from any working directory by any `vite` binary. Anchoring the
// lookup at the frontend package's `package.json` points at the same installed copies the
// shipping config uses.
// Vite's `bundleConfigFile` sets root = path.dirname(fileName).
const requireFromFrontend = createRequire(path.join(FRONTEND_ROOT, "package.json"));
const tailwindcss = requireFromFrontend("@tailwindcss/vite").default;
const react = requireFromFrontend("@vitejs/plugin-react").default;
const { defineConfig } = requireFromFrontend("vite");

export default defineConfig({
  root: FRONTEND_ROOT,
  plugins: [react(), tailwindcss()],
  // Keep an unrelated PostCSS config in an ancestor directory from leaking into Unsloth installs.
  // Tailwind is provided by its dedicated Vite plugin.
  css: {
    postcss: {
      plugins: [],
    },
  },
  optimizeDeps: {
    include: ["@dagrejs/dagre", "@dagrejs/graphlib"],
  },
  server: {
    host: "0.0.0.0",
    allowedHosts: true,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8888",
        changeOrigin: true,
      },
      "/v1": {
        target: "http://127.0.0.1:8888",
        changeOrigin: true,
      },
      "/seed/inspect": {
        target: "http://127.0.0.1:8004",
        changeOrigin: true,
      },
      "/seed/preview": {
        target: "http://127.0.0.1:8004",
        changeOrigin: true,
      },
      "/preview": {
        target: "http://127.0.0.1:8004",
        changeOrigin: true,
      },
      "/validate": {
        target: "http://127.0.0.1:8004",
        changeOrigin: true,
      },
      "/tools": {
        target: "http://127.0.0.1:8004",
        changeOrigin: true,
      },
    },
  },
  resolve: {
    // An ARRAY of regex aliases, not the object form, and the react-dom entry is deliberately
    // anchored. Aliasing the bare specifier `react-dom` recurses forever, because the profiling
    // bundle's own first lines are `require("react"), require("react-dom")`; and a plain string
    // `find: "react-dom/client"` is prefix matching in Vite, so it would also rewrite
    // `react-dom/profiling` into the same recursion.
    // `/^react-dom\/client$/` matches exactly the specifier the app imports. Bare `react-dom` is left
    // alone on purpose: it is a small shim dispatching through the injected internals object, so it
    // drives whichever renderer actually loaded.
    alias: [
      { find: /^react-dom\/client$/, replacement: "react-dom/profiling" },
      { find: /^@\//, replacement: `${path.resolve(FRONTEND_ROOT, "./src")}/` },
      {
        find: "@dagrejs/dagre",
        replacement: path.resolve(
          FRONTEND_ROOT,
          "./node_modules/@dagrejs/dagre/dist/dagre.cjs.js",
        ),
      },
    ],
  },
  build: {
    outDir: path.resolve(__dirname, "dist"),
    emptyOutDir: true,
    // `hidden` emits the .map files but suppresses the sourceMappingURL comment, so the bundle is
    // byte-comparable to a shipping one apart from the profiling renderer, and devtools does not
    // silently start resolving maps mid-measurement.
    sourcemap: "hidden",
    // Vite 8 minifies with oxc by default, and without a name-preservation knob the minifier renames
    // our own components too, leaving the symbol bridge with no anchors: of the 403 `export function
    // <Component>` names in `studio/frontend/src` only 37 survive an unconfigured build, against 390
    // with the option below.
    // The knob is the OXC MINIFIER's `mangle.keepNames`, reached through `output.minify`, and NOT the
    // bundler-level `output.keepNames`, which on the rolldown Vite 8.0.16 pins fails the build
    // outright with `[MISSING_EXPORT] "toString" is not exported by underscore/modules/_setup.js` and
    // 23 more. That is rolldown#9973: under `keepNames` the bundler splits a multi-declarator `export
    // var a = 1, b = 2` and drops the `export` keyword, so every name after the split stops being
    // exported. Underscore's `_setup.js` is written that way and we reach it through
    // `@dagrejs/graphlib`. Fixed upstream by rolldown#9974 in the 1.1.x line, unavailable without
    // overriding the rolldown Vite pins, which an attribution-only config does not get to do.
    // The pin is `rolldown@1.0.3`.
    // `mangle.keepNames` is the option that matters anyway: bundler-level renaming only appends
    // deconflicting suffixes, while the minifier's mangler is what turns `AppSidebar` into a letter.
    // `compress.keepNames` is set alongside it because oxc documents the pair as belonging together
    // (DCE has to stop treating the name-carrying binding as dead). Vite spreads the user `output`
    // last, so `minify` here replaces Vite's default rather than being clobbered; compression and
    // mangling stay on and the bundle grows only ~3.9%.
    // In `buildOutputOptions`.
    rolldownOptions: {
      output: {
        minify: {
          compress: { keepNames: { function: true, class: true } },
          mangle: { keepNames: { function: true, class: true } },
        },
        // Marks the bundle so the harness can prove it is measuring the attribution build and not a stale
        // shipping dist left in the directory handed to `unsloth studio --frontend`. Read by
        // `../analysis/bridge_build.py:assert_attribution_build`.
        // This is a BANNER and not a `define`: `define` performs identifier SUBSTITUTION in source, and no
        // Unsloth source file mentions `__STUDIOBENCH_ATTRIBUTION_BUILD__`, so a `define` entry
        // substitutes nothing and the marker is simply absent at runtime. Built that way first, the
        // assertion could never pass and the bundle was byte-identical with and without the option. A
        // banner is prepended unconditionally, so the global genuinely exists.
        banner: "globalThis.__STUDIOBENCH_ATTRIBUTION_BUILD__ = true;",
      },
    },
    commonjsOptions: {
      include: [/node_modules/, /@dagrejs\/dagre/, /@dagrejs\/graphlib/],
    },
  },
});
