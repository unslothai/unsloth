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
//   3. Minifier name preservation, so our own functions and classes keep
//      their source names instead of being mangled to single letters.
//
// The resulting dist is injected into a REAL Unsloth install through the
// existing `unsloth studio --frontend <dir>` flag (see `studio/backend/run.py`
// around line 2820), so the backend, auth and storage stay real and only the
// frontend bundle is swapped. That matters: the failure this whole tool exists
// to correct was a fixture that did not run the real path.
//
// WHAT THIS BUILD DOES NOT FIX, and cannot. React's production and profiling
// dists are minified by React's own release pipeline BEFORE Vite ever sees
// them. Name preservation and sourcemaps therefore recover OUR names and
// nothing whatsoever inside `react-dom`: the original identifier really is
// `Zk`. Recovering react-dom names is the job of the symbol bridge in
// `tests/studio/studiobench/analysis/symbols.py`, not of this config, and no
// build flag here can do it.

import { createRequire } from "node:module";
import path from "node:path";

const FRONTEND_ROOT = path.resolve(__dirname, "../studio/frontend");

// The plugins are resolved from the FRONTEND package, not from this directory,
// and that is not a stylistic preference. Vite bundles a `--config` file with
// its bare imports left external (`bundleConfigFile` sets `root =
// path.dirname(fileName)`) and then loads the bundle under the config file's
// own path: CJS via `module._compile(code, <config path>)`, ESM via a temp file
// in the nearest ancestor `node_modules`, of which this directory has none. So
// Node starts its `node_modules` walk at `attribution/` and climbs to the
// repository root, and NEITHER has a `node_modules` -- only
// `studio/frontend/node_modules` does. Written with plain top-level imports
// this config could not be loaded from any working directory, by any `vite`
// binary: `Cannot find module '@tailwindcss/vite'`, and the same for the React
// plugin and for `vite` itself. Anchoring the lookup at the frontend package's
// `package.json` is the whole fix, and it points at the same installed copies
// the shipping `studio/frontend/vite.config.ts` uses.
const requireFromFrontend = createRequire(path.join(FRONTEND_ROOT, "package.json"));
const tailwindcss = requireFromFrontend("@tailwindcss/vite").default;
const react = requireFromFrontend("@vitejs/plugin-react").default;
const { defineConfig } = requireFromFrontend("vite");

export default defineConfig({
  root: FRONTEND_ROOT,
  plugins: [react(), tailwindcss()],
  // Keep an unrelated PostCSS config in an ancestor directory from leaking
  // into Unsloth installs. Tailwind is provided by its dedicated Vite plugin.
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
    // Vite 8 minifies with oxc by default, and without a name-preservation
    // knob the minifier renames our own components too: the app-side frames
    // become single letters, which would leave the symbol bridge with no
    // anchors to validate against. Measured on this tree, of the 403
    // `export function <Component>` names in `studio/frontend/src` only 37
    // survive an unconfigured build, against 390 with the option below.
    //
    // The knob is the OXC MINIFIER's `mangle.keepNames`, reached through
    // `output.minify`, and NOT the bundler-level `output.keepNames`. That
    // distinction is the whole reason this block looks the way it does.
    // `output.keepNames: true` on the rolldown that Vite 8.0.16 pins
    // (`rolldown@1.0.3`, an exact pin in Vite's own `dependencies`) fails the
    // build outright:
    //
    //   [MISSING_EXPORT] "toString" is not exported by
    //   "node_modules/underscore/modules/_setup.js"
    //
    // and 23 more like it. That is rolldown#9973: under `keepNames` the
    // bundler splits a multi-declarator `export var a = 1, b = 2` into
    // separate declarations and drops the `export` keyword while doing it, so
    // every name after the split silently stops being exported. Underscore's
    // `_setup.js` is written almost entirely in that style
    // (`export var push = ..., slice = ..., toString = ...`) and we reach it
    // through `@dagrejs/graphlib`, so the build aborts. Fixed upstream by
    // rolldown#9974, first released in the 1.1.x line; unavailable to us
    // without overriding the rolldown that Vite pins, which would change the
    // bundler under the shipping UI and is not something an attribution-only
    // config gets to do.
    //
    // `mangle.keepNames` is the option that actually matters here anyway.
    // Bundler-level renaming only appends deconflicting suffixes and leaves
    // names readable; it is the minifier's mangler that turns `AppSidebar`
    // into a letter. `compress.keepNames` is set alongside it because oxc
    // documents the pair as belonging together -- DCE has to stop treating
    // the name-carrying binding as dead. Vite spreads the user `output` last
    // (`buildOutputOptions`), so `minify` here replaces Vite's default
    // `minify: true` rather than being clobbered by it; compression and
    // mangling both stay on, and the bundle grows only ~3.9% from carrying
    // the real names.
    rolldownOptions: {
      output: {
        minify: {
          compress: { keepNames: { function: true, class: true } },
          mangle: { keepNames: { function: true, class: true } },
        },
        // Marks the bundle so the harness can prove it is measuring the
        // attribution build and not a stale shipping dist left in the
        // directory handed to `unsloth studio --frontend`. Read by
        // `analysis/bridge_build.py:assert_attribution_build`.
        //
        // This is a BANNER and not a `define`, and the difference is not
        // cosmetic. `define` performs identifier SUBSTITUTION in source: it
        // rewrites occurrences of a token that already appear in the code. No
        // Unsloth source file mentions `__STUDIOBENCH_ATTRIBUTION_BUILD__`, so
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
