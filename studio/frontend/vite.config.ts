// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import path from "node:path";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { type Plugin, defineConfig } from "vite";
import { chatPagePreload } from "./vite-plugin-chat-preload";
import { childSelectorOrder } from "./vite-plugin-child-selectors";
import { groupStateKeys } from "./vite-plugin-group-state";
import { hasAnchorOrder } from "./vite-plugin-has-anchor";
import { katexWoff2Only } from "./vite-plugin-katex-woff2";
import { preloadNonceInHead } from "./vite-plugin-preload-nonce";
import { viewportMediaGate } from "./vite-plugin-viewport-media";
import { previewRuntime } from "./vite-preview-runtime.ts";

// assistant-ui's tap re-rendered every message client on each streamed token (O(messages) per
// token); the stand-in skips unchanged children (src/lib/assistant-ui-tap-resources.ts). Not a
// resolve alias: on the dev server that also sends the app's own `@assistant-ui/tap` import to
// tap's raw files, a second copy beside the one prebundled into assistant-ui, and the thread never
// renders. The build resolves the stand-in with this plugin; on the dev server the dependency
// optimizer does, for assistant-ui's imports and for the app's `@assistant-ui/tap` entry, so both
// share the one prebundled copy.
const TAP_STAND_IN = new Map([
  ["@assistant-ui/tap", path.resolve(__dirname, "./src/lib/assistant-ui-tap.ts")],
  ["@assistant-ui/tap/react", path.resolve(__dirname, "./src/lib/assistant-ui-tap-react.ts")],
]);
const TAP_ENTRIES = new Map([
  [path.resolve(__dirname, "./node_modules/@assistant-ui/tap/dist/index.js"), TAP_STAND_IN.get("@assistant-ui/tap")],
  [path.resolve(__dirname, "./node_modules/@assistant-ui/tap/dist/react/index.js"), TAP_STAND_IN.get("@assistant-ui/tap/react")],
]);

function assistantUiTapStandIn() {
  return {
    name: "assistant-ui-tap-stand-in",
    resolveId(id: string, _importer?: string, options?: { isEntry?: boolean }) {
      return TAP_STAND_IN.get(id) ?? (options?.isEntry ? TAP_ENTRIES.get(id) : undefined) ?? null;
    },
  };
}

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
  plugins: [react(), tailwindcss(), childSelectorOrder(), hasAnchorOrder(), viewportMediaGate(), groupStateKeys(), { ...assistantUiTapStandIn(), apply: "build", enforce: "pre" }, smokeModuleDelay(), previewRuntime(), chatPagePreload(), katexWoff2Only(), preloadNonceInHead()],
  // prevent ancestor PostCSS configs from leaking into installs; Tailwind uses its Vite plugin.
  css: {
    postcss: {
      plugins: [],
    },
  },
  optimizeDeps: {
    // react-remove-scroll/UI is only imported by the aliased stand-in (src/lib/remove-scroll.tsx),
    // which the dev server's first scan does not see; found later, it forces a re-optimize and a
    // full reload that native Safari does not recover from.
    include: ["@dagrejs/dagre", "@dagrejs/graphlib", "react-remove-scroll/UI"],
    rolldownOptions: { plugins: [assistantUiTapStandIn()] },
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
      // MCP setup snippets use the page's origin.
      "/mcp": {
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
    alias: [
      { find: "@", replacement: path.resolve(__dirname, "./src") },
      {
        find: "@dagrejs/dagre",
        replacement: path.resolve(
          __dirname,
          "./node_modules/@dagrejs/dagre/dist/dagre.cjs.js",
        ),
      },
      // Radix's scroll lock without its document wheel / touch listeners, which restyle the
      // whole page in WebKit on every modal open and close (src/lib/remove-scroll.tsx).
      {
        find: /^react-remove-scroll$/,
        replacement: path.resolve(__dirname, "./src/lib/remove-scroll.tsx"),
      },
    ],
  },
  build: {
    commonjsOptions: {
      include: [/node_modules/, /@dagrejs\/dagre/, /@dagrejs\/graphlib/],
    },
    rolldownOptions: {
      // import() of a statically imported module defers nothing and splits extra startup chunks,
      // so fail the build rather than warn.
      onLog(level, log, handler) {
        if (log.code === "INEFFECTIVE_DYNAMIC_IMPORT") {
          throw new Error(log.message);
        }
        handler(level, log);
      },
    },
  },
});
