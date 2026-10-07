// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Bundles the shipped thread-fast-copy.ts into <out-dir>/fastcopy.js (IIFE `SBFastCopy`).
// Lives here so bare `vite` resolves from studio/frontend/node_modules.

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
    // The caller provides a fresh dir; never empty a path outside the vite root.
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
