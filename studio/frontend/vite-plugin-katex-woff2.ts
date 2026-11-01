// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import postcss from "postcss";
import type { Plugin } from "vite";

// KaTeX's @font-face rules list woff2, then woff, then ttf. Every engine Studio runs in
// (WebView2, WKWebView, WebKitGTK 4.1, current Chromium / Firefox / Safari) takes woff2, the
// first entry, so the 40 woff / ttf files were emitted into dist and never fetched.
const KATEX_CSS = /[\\/]node_modules[\\/]katex[\\/]dist[\\/]katex(?:\.min)?\.css(?:$|\?)/;
const WOFF2_SOURCE = /url\(\s*(?:"[^"]*"|'[^']*'|[^)"']*)\s*\)\s*format\(\s*["']?woff2["']?\s*\)/gi;

export function keepWoff2Sources(css: string): { css: string; rules: number } {
  const root = postcss.parse(css);
  let rules = 0;
  root.walkAtRules("font-face", (rule) => {
    rule.walkDecls("src", (decl) => {
      const woff2 = decl.value.match(WOFF2_SOURCE);
      if (!woff2 || woff2.length !== 1) {
        throw new Error(`@font-face src "${decl.value}" has ${woff2?.length ?? 0} woff2 sources, expected 1`);
      }
      decl.value = woff2[0];
      rules += 1;
    });
  });
  return { css: root.toString(), rules };
}

export function katexWoff2Only(): Plugin {
  return {
    name: "katex-woff2-only",
    // Before vite:css resolves url() entries, which is where the font files get emitted.
    enforce: "pre",
    transform: {
      filter: { id: KATEX_CSS, code: "@font-face" },
      handler(code, id) {
        const { css, rules } = keepWoff2Sources(code);
        if (rules === 0) throw new Error(`no KaTeX @font-face src found in ${id}`);
        return { code: css, map: null };
      },
    },
  };
}
