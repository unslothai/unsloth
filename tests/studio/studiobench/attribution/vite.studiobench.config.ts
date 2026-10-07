// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Attribution build: same behaviour as studio/frontend/vite.config.ts, but with react-dom/profiling,
// hidden sourcemaps and kept names. React internals stay minified; see analysis/symbols.py.

import { createRequire } from "node:module";
import path from "node:path";

const FRONTEND_ROOT = path.resolve(__dirname, "../../../../studio/frontend");

// Resolve plugins from the frontend package: Vite loads this config from its own dir,
// where no node_modules exists up the tree.
const requireFromFrontend = createRequire(path.join(FRONTEND_ROOT, "package.json"));
const tailwindcss = requireFromFrontend("@tailwindcss/vite").default;
const react = requireFromFrontend("@vitejs/plugin-react").default;
const { defineConfig } = requireFromFrontend("vite");

export default defineConfig({
  root: FRONTEND_ROOT,
  plugins: [react(), tailwindcss()],
  // Keep an unrelated ancestor PostCSS config from leaking in; Tailwind comes from its Vite plugin.
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
    // Anchored regex array: aliasing bare react-dom recurses (the profiling bundle requires it),
    // and a string find is prefix matching, which would also rewrite react-dom/profiling.
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
    // hidden: emit maps without sourceMappingURL so devtools does not resolve them mid-measurement.
    sourcemap: "hidden",
    // Use the oxc minifier's keepNames, not bundler output.keepNames, which breaks the build on the
    // pinned rolldown 1.0.3 (rolldown#9973). The mangler is what renames components anyway.
    rolldownOptions: {
      output: {
        minify: {
          compress: { keepNames: { function: true, class: true } },
          mangle: { keepNames: { function: true, class: true } },
        },
        // Marker checked by analysis/bridge_build.py:assert_attribution_build. A banner, not a define:
        // define only substitutes identifiers present in source, so the global would never exist.
        banner: "globalThis.__STUDIOBENCH_ATTRIBUTION_BUILD__ = true;",
      },
    },
    commonjsOptions: {
      include: [/node_modules/, /@dagrejs\/dagre/, /@dagrejs\/graphlib/],
    },
  },
});
