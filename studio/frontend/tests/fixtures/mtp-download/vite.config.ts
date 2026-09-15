// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A/B config for tests/studio/playwright_mtp_download_visibility.py. The
// harness and dependencies stay identical while @ resolves to the checkout
// named by PW_SOURCE_FRONTEND_DIR, so base and fix compile separate sources.

// biome-ignore lint/correctness/noNodejsModules: Vite configs execute in Node.
import path from "node:path";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const frontendRoot = path.resolve(__dirname, "../../..");
const sourceFrontend = path.resolve(
  process.env.PW_SOURCE_FRONTEND_DIR ?? frontendRoot,
);

// biome-ignore lint/style/noDefaultExport: Vite loads its config as a default export.
export default defineConfig({
  root: __dirname,
  publicDir: path.resolve(frontendRoot, "public"),
  plugins: [react(), tailwindcss()],
  css: { postcss: { plugins: [] } },
  server: {
    host: "127.0.0.1",
    fs: { allow: [frontendRoot, sourceFrontend] },
  },
  resolve: {
    // Source files live in a second worktree during the negative control. Keep
    // one React runtime even when that checkout has its own dependency tree.
    dedupe: ["react", "react-dom"],
    alias: {
      "@": path.resolve(sourceFrontend, "src"),
      "@dagrejs/dagre": path.resolve(
        frontendRoot,
        "node_modules/@dagrejs/dagre/dist/dagre.cjs.js",
      ),
    },
  },
  optimizeDeps: { include: ["@dagrejs/dagre", "@dagrejs/graphlib"] },
});
