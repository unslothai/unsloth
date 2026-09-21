// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import path from "node:path";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const frontend = path.resolve(__dirname, "../../..");

export default defineConfig({
  root: __dirname,
  publicDir: path.join(frontend, "public"),
  plugins: [react(), tailwindcss()],
  css: { postcss: { plugins: [] } },
  resolve: {
    alias: {
      "@": path.join(frontend, "src"),
      "@dagrejs/dagre": path.join(
        frontend,
        "node_modules/@dagrejs/dagre/dist/dagre.cjs.js",
      ),
    },
  },
  optimizeDeps: { include: ["@dagrejs/dagre", "@dagrejs/graphlib"] },
});
