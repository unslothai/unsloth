// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Compile the SHIPPED module into a browser script, so the byte-for-byte proof runs against the
// TypeScript that is actually in the tree rather than against a hand-kept copy of it. Proving a
// reference implementation proves the reference implementation; what reaches a user is
// studio/frontend/src/components/assistant-ui/thread-fast-copy.ts, so that is what is built.
//
// Usage: node scripts/build-fast-copy-bundle.mjs <out-dir>, from studio/frontend.
// It lives here rather than beside its caller in tests/studio because node resolves a bare
// `import ... from "vite"` against the IMPORTING FILE's directory, and vite is installed
// under studio/frontend/node_modules.
// Writes <out-dir>/fastcopy.js, an IIFE bundle exposing the module's exports as `SBFastCopy`.
//
// `configFile: false` so studio/frontend/vite.config.ts (the app build: react plugin, chunking,
// asset pipeline) is not applied to a one-module library build. `copyPublicDir: false` because
// the app's public/ is tens of megabytes of images the proof never loads.

import path from "node:path";
import { fileURLToPath } from "node:url";
import { build } from "vite";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..");

const outDir = process.argv[2];
if (!outDir) {
  console.error("usage: node scripts/build-fast-copy-bundle.mjs <out-dir>");
  process.exit(2);
}

await build({
  root,
  configFile: false,
  logLevel: "error",
  build: {
    outDir: path.resolve(outDir),
    // The caller owns the directory and hands over a fresh one; emptying a path outside the
    // vite root is a footgun this has no need of.
    emptyOutDir: false,
    copyPublicDir: false,
    minify: false,
    lib: {
      entry: path.join(root, "src/components/assistant-ui/thread-fast-copy.ts"),
      name: "SBFastCopy",
      formats: ["iife"],
      fileName: () => "fastcopy.js",
    },
  },
});
