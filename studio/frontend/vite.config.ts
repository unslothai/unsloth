// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import path from "node:path";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import postcss, { type Plugin as PostcssPlugin } from "postcss";
import { type Plugin as VitePlugin, defineConfig } from "vite";

const UI_FONT_SCALE_VAR = "--ui-font-scale";
const CSS_WIDE_FONT_SIZE_PATTERN =
  /^(?:inherit|initial|revert(?:-layer)?|unset|smaller|larger)$/;
const RELATIVE_FONT_SIZE_PATTERN =
  /(?:^|[^\w.-])-?(?:\d*\.)?\d+(?:em|ex|cap|ch|ic|lh|rlh|%)\b/;
const ABSOLUTE_FONT_SIZE_PATTERN =
  /-?(?:\d*\.)?\d+(?:px|pt|pc|rem|cm|mm|q|in)\b/;
const FONT_SIZE_FUNCTION_PATTERN = /\b(?:calc|clamp|max|min|var)\(/;
const CSS_MODULE_ID_PATTERN = /\.css(?:$|\?)/;

// Relative sizes inherit the scale and must not compound through nested text.
function shouldScaleFontSize(value: string): boolean {
  const normalized = value.trim().toLowerCase();
  if (
    normalized.includes(UI_FONT_SCALE_VAR) ||
    normalized.includes("--custom-code-font-size") ||
    CSS_WIDE_FONT_SIZE_PATTERN.test(normalized)
  ) {
    return false;
  }

  // Root rem stays fixed, so rem typography still needs the multiplier.
  if (RELATIVE_FONT_SIZE_PATTERN.test(normalized)) {
    return false;
  }

  return (
    ABSOLUTE_FONT_SIZE_PATTERN.test(normalized) ||
    FONT_SIZE_FUNCTION_PATTERN.test(normalized)
  );
}

const scaleAbsoluteFontSizes: PostcssPlugin = {
  postcssPlugin: "unsloth-ui-font-scaling",
  // biome-ignore lint/style/useNamingConvention: PostCSS visitor API key.
  Declaration: {
    "font-size": (declaration) => {
      if (!shouldScaleFontSize(declaration.value)) {
        return;
      }
      declaration.value = `calc(${declaration.value} * var(${UI_FONT_SCALE_VAR}, 1))`;
    },
  },
};

// Scale named, arbitrary, and vendor font sizes in Tailwind's generated CSS.
function uiFontScalingPlugin(): VitePlugin {
  return {
    name: "unsloth-ui-font-scaling",
    // Run after Tailwind and before Vite converts CSS into a module.
    enforce: "pre",
    async transform(code, id) {
      if (!CSS_MODULE_ID_PATTERN.test(id)) {
        return null;
      }
      const result = await postcss([scaleAbsoluteFontSizes]).process(code, {
        from: id,
        map: false,
      });
      if (result.css === code) {
        return null;
      }
      return { code: result.css, map: null };
    },
  };
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss(), uiFontScalingPlugin()],
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
  },
});
