// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import path from "node:path";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { type Plugin, defineConfig } from "vite";
import { previewRuntime } from "./vite-preview-runtime.ts";

function smokeModuleDelay(): Plugin {
  const match = process.env.SMOKE_MODULE_DELAY_MATCH;
  const delayMs = Number(process.env.SMOKE_MODULE_DELAY_MS ?? "0");
  return {
    name: "smoke-module-delay",
    configureServer(server) {
      if (!match || !Number.isFinite(delayMs) || delayMs <= 0) return;
      server.middlewares.use((request, _response, next) => {
        if (!request.url?.includes(match)) {
          next();
          return;
        }
        setTimeout(next, delayMs);
      });
    },
  };
}

export default defineConfig({
  // concurrent browser checks use separate dependency optimizer caches.
  cacheDir: process.env.VITE_TEST_CACHE_DIR || "node_modules/.vite",
  // Reasoning's highlighter loads only the grammar it needs in its module worker.
  worker: { format: "es" },
  plugins: [react(), tailwindcss(), smokeModuleDelay(), previewRuntime()],
  // prevent ancestor PostCSS configs from leaking into installs; Tailwind uses its Vite plugin.
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
    alias: {
      "@": path.resolve(__dirname, "./src"),
      "@dagrejs/dagre": path.resolve(
        __dirname,
        "./node_modules/@dagrejs/dagre/dist/dagre.cjs.js",
      ),
    },
  },
  build: {
    commonjsOptions: {
      include: [/node_modules/, /@dagrejs\/dagre/, /@dagrejs\/graphlib/],
    },
    rolldownOptions: {
      // import() of a module the app already imports statically defers nothing, and it splits
      // that module's graph into extra startup chunks (#11588). Fail the build rather than warn.
      onLog(level, log, handler) {
        if (log.code === "INEFFECTIVE_DYNAMIC_IMPORT") {
          throw new Error(log.message);
        }
        handler(level, log);
      },
    },
  },
});
