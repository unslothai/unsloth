// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// studiobench attribution build.
//
// Identical to `studio/frontend/vite.config.ts` in every respect that affects
// what the app DOES, and different only in what it lets us SEE:
//
//   1. `react-dom/client` is aliased to `react-dom/profiling`, which is the
//      only build where React's `<Profiler>` `onRender` callback exists at all.
//   2. Hidden sourcemaps, so our own frames get real names without shipping a
//      `//# sourceMappingURL` comment.
//   3. `keepNames`, so the bundler and minifier do not rename our functions.
//
// The resulting dist is injected into a REAL Studio install through the
// existing `unsloth studio --frontend <dir>` flag (see `studio/backend/run.py`
// around line 2820), so the backend, auth and storage stay real and only the
// frontend bundle is swapped. That matters: the failure this whole tool exists
// to correct was a fixture that did not run the real path.
//
// WHAT THIS BUILD DOES NOT FIX, and cannot. React's production and profiling
// dists are minified by React's own release pipeline BEFORE Vite ever sees
// them. `keepNames` and sourcemaps therefore recover OUR component names and
// nothing whatsoever inside `react-dom`: the original identifier really is
// `Zk`. Recovering react-dom names is the job of the symbol bridge in
// `tests/studio/studiobench/analysis/symbols.py`, not of this config, and no
// build flag here can do it.

import path from "node:path";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const FRONTEND_ROOT = path.resolve(__dirname, "../studio/frontend");

export default defineConfig({
  root: FRONTEND_ROOT,
  plugins: [react(), tailwindcss()],
  // Keep an unrelated PostCSS config in an ancestor directory from leaking
  // into Studio installs. Tailwind is provided by its dedicated Vite plugin.
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
    // An ARRAY of regex aliases, not the object form, and the react-dom entry
    // is deliberately anchored. Two traps, both of which produce a build that
    // looks fine and is not:
    //
    //   - Aliasing the bare specifier `react-dom` recurses forever. The
    //     profiling bundle's own first lines are
    //     `require("react"), require("react-dom")`, so rewriting `react-dom`
    //     points it at itself.
    //   - A plain string `find: "react-dom/client"` is prefix matching in
    //     Vite, so it would also rewrite `react-dom/profiling` and produce the
    //     same recursion by a different route.
    //
    // `/^react-dom\/client$/` matches exactly the specifier the app imports and
    // nothing else. Bare `react-dom` is left alone on purpose: it is a small
    // shim that dispatches through the injected internals object, so it drives
    // whichever renderer actually loaded.
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
    // `hidden` emits the .map files but suppresses the sourceMappingURL
    // comment, so the bundle is byte-comparable to a shipping one apart from
    // the profiling renderer, and devtools does not silently start resolving
    // maps mid-measurement.
    sourcemap: "hidden",
    // Vite 8 minifies with oxc by default and the name-preservation knob is
    // Rolldown's `output.keepNames`, not a `terserOptions.keep_fnames`. Vite
    // spreads the user `output` last, so this is not clobbered. Without it the
    // minifier renames our own components too and even the app-side frames
    // become single letters, which would leave the symbol bridge with no
    // anchors to validate against.
    rolldownOptions: {
      output: {
        keepNames: true,
        // Marks the bundle so the harness can prove it is measuring the
        // attribution build and not a stale shipping dist left in the
        // directory handed to `unsloth studio --frontend`. Read by
        // `analysis/bridge_build.py:assert_attribution_build`.
        //
        // This is a BANNER and not a `define`, and the difference is not
        // cosmetic. `define` performs identifier SUBSTITUTION in source: it
        // rewrites occurrences of a token that already appear in the code. No
        // Studio source file mentions `__STUDIOBENCH_ATTRIBUTION_BUILD__`, so
        // a `define` entry substitutes nothing, emits nothing, and the marker
        // is simply absent at runtime. Built that way first; the assertion
        // could never pass and the bundle was byte-identical with and without
        // the option. A banner is prepended unconditionally, so the global
        // genuinely exists.
        banner: "globalThis.__STUDIOBENCH_ATTRIBUTION_BUILD__ = true;",
      },
    },
    commonjsOptions: {
      include: [/node_modules/, /@dagrejs\/dagre/, /@dagrejs\/graphlib/],
    },
  },
});
